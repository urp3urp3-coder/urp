import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__)

# ----------------------------------- 
# 1. Custom Conditioning Model
# ----------------------------------- 
class ConditioningMLP(nn.Module):
    """
    A simple MLP to project a scalar hemoglobin value to a high-dimensional embedding.
    This embedding will act as the "text" condition for the UNet.
    """
    def __init__(self, in_dim=1, out_dim=768, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # Input x is expected to be a scalar (or a batch of scalars)
        # Add a dimension if it's a 0-dim tensor or a batch of 0-dim tensors
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x)

# ----------------------------------- 
# 2. Custom Dataset
# ----------------------------------- 
class ConditionalNailDataset(Dataset):
    """Reads the pre-processed CSV and provides (image, hb_value) pairs."""
    def __init__(self, csv_file, image_size):
        self.data = pd.read_csv(csv_file)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1] for diffusion models
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        hb_value = row['hemoglobin']
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return {"image": image, "condition": torch.tensor(hb_value, dtype=torch.float32)}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with LoRA.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--dataset_csv", type=str, default="seyun/diffusion_data.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--output_dir", type=str, default="seyun/conditional-diffusion-lora", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate to use.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help=("The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']"))
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lora_rank", type=int, default=4, help="The dimension of the LoRA update matrices.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # ----------------------------------- 
    # 3. Load Models (VAE, UNet, Scheduler)
    # ----------------------------------- 
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # Freeze VAE as we are only training the UNet
    vae.requires_grad_(False)

    # ----------------------------------- 
    # 4. Setup LoRA and Conditioning MLP
    # ----------------------------------- 
    # Add LoRA layers to the UNet
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        # In older versions of diffusers, rank is the only argument needed.
        # We remove hidden_size and cross_attention_dim to ensure compatibility.
        lora_attn_procs[name] = LoRAAttnProcessor(rank=args.lora_rank)
    unet.set_attn_processors(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Initialize our custom conditioning model
    # The output dimension must match the UNet's cross_attention_dim
    cond_mlp = ConditioningMLP(out_dim=unet.config.cross_attention_dim)

    # ----------------------------------- 
    # 5. Optimizer and Dataloader
    # ----------------------------------- 
    # The optimizer should train both the LoRA layers and our conditioning MLP
    optimizer = torch.optim.AdamW(
        list(lora_layers.parameters()) + list(cond_mlp.parameters()),
        lr=args.learning_rate,
    )

    train_dataset = ConditionalNailDataset(args.dataset_csv, args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    # ----------------------------------- 
    # 6. Prepare for training with Accelerate
    # ----------------------------------- 
    unet, cond_mlp, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, cond_mlp, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device)

    # We need to initialize the trackers we use, and Accelerator will handle syncing them across processes
    if accelerator.is_main_process:
        accelerator.init_trackers("conditional-diffusion-training")

    # ----------------------------------- 
    # 7. Training Loop
    # ----------------------------------- 
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        pbar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        pbar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["image"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # --- THIS IS THE KEY PART ---
                # Get the conditional embedding from our MLP
                cond_embeddings = cond_mlp(batch["condition"])
                # The UNet expects a sequence, so we unsqueeze the time dimension
                cond_embeddings = cond_embeddings.unsqueeze(1)
                # ----------------------------

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond_embeddings).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes for logging.
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            
            pbar.set_postfix(Loss=loss.detach().item(), LR=lr_scheduler.get_last_lr()[0])

    # ----------------------------------- 
    # 8. Save the final model
    # ----------------------------------- 
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap the model before saving
        unet = accelerator.unwrap_model(unet)
        cond_mlp = accelerator.unwrap_model(cond_mlp)

        # Save the LoRA layers
        lora_layers.save_attn_procs(args.output_dir)
        
        # Save the conditioning MLP
        torch.save(cond_mlp.state_dict(), os.path.join(args.output_dir, "conditioning_mlp.pth"))
        
        print(f"\nTraining finished. LoRA weights saved to: {args.output_dir}")
        print(f"Conditioning MLP saved to: {os.path.join(args.output_dir, 'conditioning_mlp.pth')}")

if __name__ == "__main__":
    main()
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from PIL import Image
from tqdm.auto import tqdm

# ----------------------------------
# 1. Conditioning Model (from training script)
# ----------------------------------
class ConditioningMLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=768, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x)

# ----------------------------------
# 2. Generation Function
# ----------------------------------
def generate_images(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    # Load models
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # --- KEY CHANGE: Load LoRA weights using diffusers' method ---
    print(f"Loading LoRA weights from {args.model_dir}")
    unet.load_attn_procs(args.model_dir)
    # ----------------------------------------------------------

    # Load Conditioning MLP
    print(f"Loading Conditioning MLP from {os.path.join(args.model_dir, 'conditioning_mlp.pth')}")
    cond_mlp = ConditioningMLP(out_dim=unet.config.cross_attention_dim)
    # FIX: Added weights_only=True to address security warning
    cond_mlp.load_state_dict(torch.load(os.path.join(args.model_dir, 'conditioning_mlp.pth'), map_location=device, weights_only=True))

    # Move models to device
    vae.to(device)
    unet.to(device)
    cond_mlp.to(device)
    vae.eval()
    unet.eval()
    cond_mlp.eval()

    # Create scheduler
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Generation loop
    with torch.no_grad():
        for hb_value in args.hb_values:
            print(f"Generating {args.num_samples_per_hb} samples for Hb value: {hb_value}...")

            # 1. Get conditional and unconditional embeddings
            condition_tensor = torch.tensor([hb_value], dtype=torch.float32).to(device)
            cond_embedding = cond_mlp(condition_tensor)

            uncond_input = torch.zeros_like(condition_tensor)
            uncond_embedding = cond_mlp(uncond_input)

            # Repeat embeddings for each sample in the batch and add sequence dimension
            cond_embeddings = cond_embedding.repeat(args.num_samples_per_hb, 1).unsqueeze(1)
            uncond_embeddings = uncond_embedding.repeat(args.num_samples_per_hb, 1).unsqueeze(1)

            # Concatenate for classifier-free guidance
            encoder_hidden_states = torch.cat([uncond_embeddings, cond_embeddings])

            # 2. Prepare initial random noise (latents)
            latents = torch.randn(
                (args.num_samples_per_hb, unet.config.in_channels, args.resolution // 8, args.resolution // 8),
                generator=generator,
                device=device,
            )
            latents = latents * scheduler.init_noise_sigma

            # 3. Denoising loop
            for t in tqdm(scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # 4. Decode the latents to images
            latents = 1 / vae.config.scaling_factor * latents
            images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")

            pil_images = [Image.fromarray(image) for image in images]

            # 5. Save images
            for i, img in enumerate(pil_images):
                img_seed = np.random.randint(0, 100000) if args.seed is None else f"{args.seed}_{i}"
                # FIX: Corrected the f-string for the filename
                filename = f"hb_{hb_value:.2f}_sample_{i}_seed_{img_seed}.png"
                img.save(os.path.join(args.output_dir, filename))
                print(f"Saved: {filename}")

    print("Generation finished.")

# ----------------------------------
# 3. Main Execution
# ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate images using a trained conditional diffusion model with LoRA.")

    # Model and Path Arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model path.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the trained LoRA weights (pytorch_lora_weights.bin) and MLP model are saved.")
    parser.add_argument("--output_dir", type=str, default="seyun/generated_images", help="Directory to save the generated images.")

    # Generation Parameter Arguments
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")
    parser.add_argument("--hb_values", type=float, nargs='+', required=True, help="List of Hb values to generate images for (e.g., 10.5 12.0 15.8).")
    parser.add_argument("--num_samples_per_hb", type=int, default=5, help="Number of images to generate for each Hb value.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    generate_images(args)

if __name__ == "__main__":
    main()

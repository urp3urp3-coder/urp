
import subprocess
import itertools

# Define the grid of hyperparameters to search
param_grid = {
    'lr_head': [1e-4, 3e-4, 5e-4],
    'lr_backbone': [1e-5, 3e-5],
    'weight_decay': [5e-2, 1e-2],
    'batch_size': [8, 16],
}

# Get all combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Starting hyperparameter tuning for {len(param_combinations)} combinations...")

for i, params in enumerate(param_combinations):
    print(f"\n--- Combination {i+1}/{len(param_combinations)} ---")
    print(f"Parameters: {params}")
    
    # Construct the command
    cmd = [
        'python',
        'seyun/nail_hb_train.py',
        '--lr-head', str(params['lr_head']),
        '--lr-backbone', str(params['lr_backbone']),
        '--weight-decay', str(params['weight_decay']),
        '--batch-size', str(params['batch_size']),
        # You can add other fixed parameters here if needed
        # '--epochs-stage1', '10', 
    ]
    
    try:
        # Execute the training script
        subprocess.run(cmd, check=True)
        print(f"--- Finished Combination {i+1}/{len(param_combinations)} ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running combination {i+1}: {params} !!!")
        print(f"Error: {e}")
        # Decide if you want to stop or continue on error
        # continue
        break

print("\nHyperparameter tuning finished.")

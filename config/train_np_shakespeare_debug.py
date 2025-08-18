out_dir = 'out-np-shakespeare'
eval_interval = 100  # Evaluate frequently for debugging
eval_iters = 50
log_interval = 10
always_save_checkpoint = False  # Save space during debugging
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32  # Smaller batch for debugging
block_size = 128  # Shorter sequences for faster iteration

# Small transformer for quick debugging
n_layer = 4
n_head = 4  
n_embd = 256
dropout = 0.1
bias = True

# Fixed Neural Process parameters
use_global_latent = True  # Use the new global latent architecture
uncertainty_dim = 32  # Small latent space for debugging
context_aggregation = 'mean'  # Simple aggregation to start
free_bits = 0.2  # Prevent collapse but not too restrictive

# Very gentle KL annealing to prevent collapse
kl_weight_start = 0.0        # Start with no KL loss
kl_weight_end = 0.001        # Very low final weight (was 1.0!)
kl_anneal_steps = 2000       # Short annealing for Shakespeare
kl_anneal_start = 1000        # Start annealing after model stabilizes

# Training parameters
learning_rate = 3e-4  # Slightly lower for stability
max_iters = 3000      # Quick training for debugging
weight_decay = 1e-2
warmup_iters = 100
lr_decay_iters = 3000
min_lr = 3e-5
beta2 = 0.99
grad_clip = 1.0

# Debug settings
compile = False  # Disable for easier debugging
device = 'cuda'  # Change to 'cpu' if needed for debugging
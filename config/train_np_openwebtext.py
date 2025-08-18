# Neural Process GPT on OpenWebText  
out_dir = 'out-np-openwebtext'
eval_interval = 1000
log_interval = 100
eval_iters = 200
always_save_checkpoint = True
wandb_log = False

dataset = 'openwebtext'
gradient_accumulation_steps = 4
batch_size = 8  # Start smaller due to NP overhead
block_size = 512  # Smaller context for faster training
dtype = 'bfloat16'  # A30 supports bf16

# Medium-scale NP-GPT (GPT-2 small size)
n_layer = 8
n_head = 8  
n_embd = 512
dropout = 0.1

# NP parameters
uncertainty_dim = 256  # Larger for more expressiveness
n_function_samples = 5
# KL annealing parameters
kl_weight_start = 0.0        # Start with no KL loss
kl_weight_end = 0.5          # End KL weight
kl_anneal_steps = 200000     # Anneal over first 200k steps
kl_anneal_start = 50000      # Start annealing after 50k steps (let base model learn first)

# Training
learning_rate = 3e-4
max_iters = 500000
weight_decay = 1e-1
warmup_iters = 1000
lr_decay_iters = 500000
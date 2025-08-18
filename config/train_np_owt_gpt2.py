# Neural Process GPT at GPT-2 (124M) scale on OpenWebText
out_dir = 'out-np-gpt2-owt'
eval_interval = 1000
log_interval = 100
eval_iters = 200
always_save_checkpoint = True
wandb_log = True
wandb_project = 'owt-np-gpt2'
wandb_run_name = 'np-gpt2-124M-run1'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
dtype = 'bfloat16'

# --- Model Architecture ---
# GPT-2 (124M) standard parameters
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False

# --- NP Parameters ---
uncertainty_dim = 256
n_function_samples = 5

# --- KL Annealing Parameters ---
kl_weight_start = 0.0           # Start with no KL loss
kl_weight_end = 0.1             # Target a smaller final weight
kl_anneal_steps = 400000        # Anneal over a longer period
kl_anneal_start = 100000        # Start annealing later to let the bigger model stabilize first

# --- AdamW Optimizer ---
# Standard hyperparameters for GPT-2 from nanoGPT
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# --- Learning Rate Schedule ---
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
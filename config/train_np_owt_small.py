# config/train_np_owt_small.py
out_dir = 'out-np-owt-small'
dataset = 'openwebtext'
eval_interval = 5000
log_interval = 100
eval_iters = 200

# Small model for fast iteration
n_layer = 6
n_head = 8  
n_embd = 512
block_size = 512
batch_size = 16
gradient_accumulation_steps = 4

# Neural Process settings
use_global_latent = True
uncertainty_dim = 128
#context_aggregation = 'mean'
free_bits = 0.1

# Conservative regularization
kl_weight_start = 0.0
kl_weight_end = 0.005
kl_anneal_steps = 25000
kl_anneal_start = 25000

# Fast training
max_iters = 250000
learning_rate = 3e-4
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 250000
min_lr = 6e-5

compile = True        
# Neural Process GPT on Shakespeare
out_dir = 'out-np-shakespeare'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Baby NP-GPT model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# NP-specific parameters
uncertainty_dim = 128
n_function_samples = 5
kl_weight = 1  # tune this

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
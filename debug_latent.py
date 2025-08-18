"""
Debug script to check if the Neural Process latent mechanism is actually working
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

# Load model
out_dir = 'out-np-shakespeare'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Load tokenizer
meta_path = 'data/shakespeare_char/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

@torch.no_grad()
def debug_latent_path():
    """Check if latent actually affects outputs"""
    
    prompt = "ROMEO:"
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    print("="*60)
    print("DEBUGGING NEURAL PROCESS LATENT MECHANISM")
    print("="*60)
    
    # 1. Check if we can manually override the latent
    print("\n1. Manual latent override test...")
    
    # Get transformer hidden states
    x_hidden = model.get_transformer_hidden(x)
    print(f"Transformer hidden shape: {x_hidden.shape}")
    
    # Get the context
    global_context = model.aggregate_context(x_hidden)
    print(f"Global context shape: {global_context.shape}")
    
    # Get the learned latent distribution
    z_natural, mu, log_sigma = model.sample_global_latent(global_context, sample=True)
    print(f"Natural latent z shape: {z_natural.shape}")
    print(f"Natural mu mean: {mu.mean().item():.6f}, std: {mu.std().item():.6f}")
    print(f"Natural log_sigma mean: {log_sigma.mean().item():.6f}, std: {log_sigma.std().item():.6f}")
    
    # Test with natural latent
    z_emb_natural = model.latent_to_emb(z_natural).unsqueeze(1)
    x_conditioned_natural = x_hidden + z_emb_natural
    logits_natural = model.lm_head(x_conditioned_natural)
    probs_natural = F.softmax(logits_natural[0, -1, :].float(), dim=-1)
    
    print(f"\nNatural latent prediction:")
    top_natural = torch.topk(probs_natural, 5)
    for i, (prob, idx) in enumerate(zip(top_natural.values, top_natural.indices)):
        char = itos[idx.item()]
        print(f"  {i+1}. '{char}' ? {prob.item():.4f}")
    
    # Test with forced extreme latents
    print(f"\n2. Forced extreme latent test...")
    
    # Force very different latents
    z_positive = torch.ones_like(z_natural) * 5.0  # Large positive
    z_negative = torch.ones_like(z_natural) * -5.0  # Large negative
    z_zero = torch.zeros_like(z_natural)  # Zero latent
    
    latent_tests = [
        ("Positive extreme (+5)", z_positive),
        ("Negative extreme (-5)", z_negative), 
        ("Zero latent", z_zero)
    ]
    
    for name, z_test in latent_tests:
        z_emb_test = model.latent_to_emb(z_test).unsqueeze(1)
        x_conditioned_test = x_hidden + z_emb_test
        logits_test = model.lm_head(x_conditioned_test)
        probs_test = F.softmax(logits_test[0, -1, :].float(), dim=-1)
        
        print(f"\n{name} prediction:")
        top_test = torch.topk(probs_test, 5)
        for i, (prob, idx) in enumerate(zip(top_test.values, top_test.indices)):
            char = itos[idx.item()]
            print(f"  {i+1}. '{char}' ? {prob.item():.4f}")
        
        # Compare to natural
        kl_div = F.kl_div(torch.log(probs_test + 1e-8), probs_natural, reduction='sum')
        print(f"  KL divergence from natural: {kl_div.item():.6f}")
    
    # 3. Check if the latent projection layer is doing anything
    print(f"\n3. Latent projection analysis...")
    
    latent_to_emb_weight = model.latent_to_emb.weight
    latent_to_emb_bias = model.latent_to_emb.bias
    
    print(f"Latent-to-embedding weight shape: {latent_to_emb_weight.shape}")
    print(f"Weight mean: {latent_to_emb_weight.mean().item():.6f}, std: {latent_to_emb_weight.std().item():.6f}")
    print(f"Weight range: [{latent_to_emb_weight.min().item():.6f}, {latent_to_emb_weight.max().item():.6f}]")
    
    if latent_to_emb_bias is not None:
        print(f"Bias mean: {latent_to_emb_bias.mean().item():.6f}, std: {latent_to_emb_bias.std().item():.6f}")
    
    # 4. Check what happens with no latent conditioning
    print(f"\n4. No latent conditioning test...")
    
    # Bypass latent entirely
    logits_no_latent = model.lm_head(x_hidden)
    probs_no_latent = F.softmax(logits_no_latent[0, -1, :].float(), dim=-1)
    
    print(f"No latent conditioning prediction:")
    top_no_latent = torch.topk(probs_no_latent, 5)
    for i, (prob, idx) in enumerate(zip(top_no_latent.values, top_no_latent.indices)):
        char = itos[idx.item()]
        print(f"  {i+1}. '{char}' ? {prob.item():.4f}")
    
    # Compare natural latent vs no latent
    kl_natural_vs_none = F.kl_div(torch.log(probs_natural + 1e-8), probs_no_latent, reduction='sum')
    print(f"KL divergence (natural latent vs no latent): {kl_natural_vs_none.item():.6f}")
    
    # 5. Check if learned mu and sigma are meaningful
    print(f"\n5. Learned distribution analysis...")
    
    # Sample multiple times and see variation
    print("Sampling 10 different latents from learned distribution:")
    all_mus = []
    all_sigmas = []
    all_probs = []
    
    for i in range(10):
        z_sample, mu_sample, log_sigma_sample = model.sample_global_latent(global_context, sample=True)
        all_mus.append(mu_sample.cpu().numpy())
        all_sigmas.append(log_sigma_sample.exp().cpu().numpy())
        
        z_emb_sample = model.latent_to_emb(z_sample).unsqueeze(1)
        x_conditioned_sample = x_hidden + z_emb_sample
        logits_sample = model.lm_head(x_conditioned_sample)
        probs_sample = F.softmax(logits_sample[0, -1, :].float(), dim=-1)
        all_probs.append(probs_sample.cpu().numpy())
    
    mu_array = np.array(all_mus)
    sigma_array = np.array(all_sigmas)
    probs_array = np.array(all_probs)
    
    print(f"Mu variation across samples: mean={mu_array.mean():.6f}, std={mu_array.std():.6f}")
    print(f"Sigma variation across samples: mean={sigma_array.mean():.6f}, std={sigma_array.std():.6f}")
    print(f"Probability variation across samples: std={probs_array.std(axis=0).mean():.6f}")
    
    if probs_array.std(axis=0).mean() < 1e-6:
        print("? PROBLEM: Probability distributions are identical across samples!")
    else:
        print("? Good: Probability distributions vary across samples")
    
    return {
        'natural_probs': probs_natural.cpu().numpy(),
        'no_latent_probs': probs_no_latent.cpu().numpy(),
        'extreme_positive_probs': None,  # Will be filled in test
        'mu_std': mu_array.std(),
        'sigma_mean': sigma_array.mean(),
        'prob_variation': probs_array.std(axis=0).mean()
    }

if __name__ == "__main__":
    results = debug_latent_path()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results['prob_variation'] < 1e-6:
        print("?? ISSUE IDENTIFIED: Neural Process latent has no effect on predictions")
        print("\nPossible causes:")
        print("1. Latent-to-embedding projection learned to output zeros")
        print("2. KL regularization forced latent to be ignored")
        print("3. Training dynamics led to posterior collapse")
        print("4. Architecture issue in conditioning mechanism")
    else:
        print("? Neural Process latent is working correctly")
        print(f"Probability variation: {results['prob_variation']:.6f}")
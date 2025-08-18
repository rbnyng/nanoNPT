import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from model import GPTConfig, GPT

# Configuration
out_dir = 'out-np-shakespeare'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
seed = 1337

# Setup
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
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

print(f"Loaded model with {model.get_num_params()/1e6:.2f}M parameters")
print(f"Global latent: {model.config.use_global_latent}, Uncertainty dim: {model.config.uncertainty_dim}")

# Load tokenizer (Shakespeare character-level)
meta_path = 'data/shakespeare_char/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {len(itos)}")
print(f"Characters: {''.join(itos.values())}")

# Load validation data
val_data = np.memmap('data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')
print(f"Validation data: {len(val_data)} characters")

# ============================================================================
# SHAKESPEARE-SPECIFIC TEST PROMPTS
# ============================================================================

# Character names - should have different uncertainty patterns
CHARACTER_NAMES = [
    "ROMEO:",
    "JULIET:",
    "HAMLET:",
    "OTHELLO:",
    "KING:",
    "QUEEN:",
    "PRINCE:",
    "DUKE:",
    "LORD:",
    "LADY:",
]

# Famous quotes - should have LOW uncertainty for well-known continuations
FAMOUS_QUOTES = [
    "To be or not to ",  # "be"
    "Romeo, Romeo, wherefore art thou ",  # "Romeo"
    "A rose by any other name would smell as ",  # "sweet"
    "All the world's a stage, and all the men and women merely ",  # "players"
    "Friends, Romans, countrymen, lend me your ",  # "ears"
    "What light through yonder window ",  # "breaks"
]

# Stage directions - might have HIGH uncertainty (many possibilities)
STAGE_DIRECTIONS = [
    "Enter ",
    "Exit ",
    "[Enter ",
    "[Exit ",
    "[Exeunt ",
    "Scene ",
]

# Dialogue starters - various uncertainty levels
DIALOGUE_STARTERS = [
    "But soft! ",
    "Hark! ",
    "What say you, ",
    "My lord, ",
    "Good morrow, ",
    "Farewell, ",
]

# Incomplete words/phrases - should show uncertainty
INCOMPLETE_PHRASES = [
    "Thou ",
    "Thee ",
    "Thy ",
    "The ",
    "And ",
    "But ",
    "For ",
    "In ",
]

@torch.no_grad()
def analyze_prompt_uncertainty(model, prompt, n_samples=30, verbose=False):
    try:
        ids = encode(prompt)
        if len(ids) == 0:
            return None
        
        x = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Get transformer hidden states
        x_hidden = model.get_transformer_hidden(x)
        global_context = model.aggregate_context(x_hidden)
        
        # Sample multiple global latents and see their effects
        sample_logits = []
        sample_entropies = []
        
        for _ in range(n_samples):
            # Sample a different global latent each time
            z, mu, log_sigma = model.sample_global_latent(global_context, sample=True)
            
            # Apply this specific latent
            z_emb = model.latent_to_emb(z).unsqueeze(1)
            x_conditioned = x_hidden + z_emb
            logits = model.lm_head(x_conditioned)
            
            # Get last token prediction
            last_logits = logits[0, -1, :].float()
            probs = F.softmax(last_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            
            sample_logits.append(last_logits.cpu().numpy())
            sample_entropies.append(entropy.item())
        
        # Calculate uncertainty metrics
        logits_array = np.array(sample_logits)
        mean_logits = np.mean(logits_array, axis=0)
        var_logits = np.var(logits_array, axis=0)
        
        # Overall metrics
        total_variance = np.mean(var_logits)
        mean_entropy = np.mean(sample_entropies)
        std_entropy = np.std(sample_entropies)
        
        # Top predicted characters
        mean_probs = F.softmax(torch.tensor(mean_logits).float(), dim=-1).numpy()
        top_indices = np.argsort(mean_probs)[-5:][::-1]
        
        if verbose:
            print(f"\n--- Analyzing: '{prompt}' ---")
            print(f"Logit variance: {total_variance:.6f}")
            print(f"Entropy: {mean_entropy:.3f} ± {std_entropy:.3f}")
            print("Top predictions:")
            for i, idx in enumerate(top_indices):
                char = itos[idx] if idx < len(itos) else f"UNK_{idx}"
                prob = mean_probs[idx]
                var_prob = var_logits[idx]
                print(f"  {i+1}. '{char}' → prob: {prob:.3f}, var: {var_prob:.6f}")
        
        return {
            'prompt': prompt,
            'logit_variance': total_variance,
            'entropy_mean': mean_entropy,
            'entropy_std': std_entropy,
            'top_chars': [(itos[idx], mean_probs[idx], var_logits[idx]) for idx in top_indices],
            'n_samples': n_samples
        }
        
    except Exception as e:
        print(f"Error analyzing '{prompt}': {e}")
        return None

@torch.no_grad()
def sample_with_uncertainty(model, prompt, max_new_tokens=50, n_samples=5, temperature=0.8):
    print(f"\n=== UNCERTAINTY SAMPLING: '{prompt}' ===")
    
    try:
        ids = encode(prompt)
        if len(ids) == 0:
            print("Empty prompt!")
            return
        
        samples = model.generate_with_uncertainty(
            torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_samples=n_samples
        )
        
        for i, sample in enumerate(samples):
            sample_text = decode(sample[0].tolist())
            print(f"\n--- Sample {i+1} ---")
            print(sample_text)
            print("---")
            
    except Exception as e:
        print(f"Error sampling '{prompt}': {e}")

@torch.no_grad()
def compare_latent_effects(model, prompt, n_samples=3, max_tokens=30):
    print(f"\n=== GLOBAL LATENT COMPARISON: '{prompt}' ===")
    
    try:
        ids = encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Generate with different global latents
        for i in range(n_samples):
            # Get fresh global latent by doing a forward pass
            x_hidden = model.get_transformer_hidden(x)
            global_context = model.aggregate_context(x_hidden)
            z, _, _ = model.sample_global_latent(global_context, sample=True)
            
            # Generate with this fixed latent
            current_tokens = x.clone()
            for _ in range(max_tokens):
                if current_tokens.size(1) > model.config.block_size:
                    current_tokens = current_tokens[:, -model.config.block_size:]
                
                x_hidden = model.get_transformer_hidden(current_tokens)
                z_emb = model.latent_to_emb(z).unsqueeze(1)
                x_conditioned = x_hidden + z_emb
                logits = model.lm_head(x_conditioned)
                
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            print(f"\nLatent {i+1}: {decode(current_tokens[0].tolist())}")
            
    except Exception as e:
        print(f"Error in latent comparison '{prompt}': {e}")

@torch.no_grad()
def analyze_character_uncertainty():
    print("\n" + "="*70)
    print("CHARACTER UNCERTAINTY ANALYSIS")
    print("="*70)
    
    results = []
    for char in CHARACTER_NAMES:
        result = analyze_prompt_uncertainty(model, char, n_samples=20, verbose=False)
        if result:
            results.append(result)
            print(f"{char:15s} → Uncertainty: {result['logit_variance']:.6f}, Entropy: {result['entropy_mean']:.3f}")
    
    # Sort by uncertainty
    if results:
        results.sort(key=lambda x: x['logit_variance'], reverse=True)
        print(f"\nMost uncertain: {results[0]['prompt']} ({results[0]['logit_variance']:.6f})")
        print(f"Least uncertain: {results[-1]['prompt']} ({results[-1]['logit_variance']:.6f})")
    else:
        print("\nNo valid results obtained!")
    
    return results

@torch.no_grad()
def analyze_famous_quotes():
    print("\n" + "="*70)
    print("FAMOUS QUOTES ANALYSIS")
    print("="*70)
    
    results = []
    for quote in FAMOUS_QUOTES:
        result = analyze_prompt_uncertainty(model, quote, n_samples=20, verbose=False)
        if result:
            results.append(result)
            print(f"'{quote}' → Uncertainty: {result['logit_variance']:.6f}")
            # Show top prediction
            if result['top_chars']:
                top_char, top_prob, _ = result['top_chars'][0]
                print(f"  Top prediction: '{top_char}' (prob: {top_prob:.3f})")
    
    return results

@torch.no_grad()
def analyze_stage_directions():
    print("\n" + "="*70)
    print("STAGE DIRECTIONS ANALYSIS")
    print("="*70)
    
    results = []
    for stage in STAGE_DIRECTIONS:
        result = analyze_prompt_uncertainty(model, stage, n_samples=20, verbose=False)
        if result:
            results.append(result)
            print(f"'{stage}' → Uncertainty: {result['logit_variance']:.6f}")
    
    return results

@torch.no_grad()
def comprehensive_uncertainty_test():
    print("\n" + "="*70)
    print("COMPREHENSIVE UNCERTAINTY ANALYSIS")
    print("="*70)
    
    all_results = {
        'characters': [],
        'famous_quotes': [],
        'stage_directions': [],
        'dialogue_starters': [],
        'incomplete_phrases': []
    }
    
    # Test all categories
    categories = [
        ('characters', CHARACTER_NAMES),
        ('famous_quotes', FAMOUS_QUOTES),
        ('stage_directions', STAGE_DIRECTIONS),
        ('dialogue_starters', DIALOGUE_STARTERS),
        ('incomplete_phrases', INCOMPLETE_PHRASES)
    ]
    
    for category, prompts in categories:
        print(f"\n--- {category.upper().replace('_', ' ')} ---")
        for prompt in prompts:
            result = analyze_prompt_uncertainty(model, prompt, n_samples=15, verbose=False)
            if result:
                all_results[category].append(result)
                print(f"'{prompt:20s}' → {result['logit_variance']:.6f}")
    
    # Summary statistics
    print(f"\n--- SUMMARY STATISTICS ---")
    for category, results in all_results.items():
        if results:
            uncertainties = [r['logit_variance'] for r in results]
            print(f"{category:20s}: {np.mean(uncertainties):.6f} ± {np.std(uncertainties):.6f}")
    
    return all_results

if __name__ == "__main__":
    print("SHAKESPEARE NEURAL PROCESS UNCERTAINTY ANALYSIS")
    print("="*70)
    
    # Quick examples first
    sample_with_uncertainty(model, "ROMEO:", max_new_tokens=40, n_samples=3)
    sample_with_uncertainty(model, "To be or not to ", max_new_tokens=30, n_samples=3)
    
    # Global latent comparison
    compare_latent_effects(model, "JULIET:", n_samples=4, max_tokens=25)
    
    # Detailed analysis
    analyze_prompt_uncertainty(model, "ROMEO:", n_samples=30, verbose=True)
    analyze_prompt_uncertainty(model, "To be or not to ", n_samples=30, verbose=True)
    analyze_prompt_uncertainty(model, "Enter ", n_samples=30, verbose=True)
    
    # Character analysis
    char_results = analyze_character_uncertainty()
    
    # Famous quotes
    quote_results = analyze_famous_quotes()
    
    # Stage directions  
    stage_results = analyze_stage_directions()
    
    # Comprehensive test
    all_results = comprehensive_uncertainty_test()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    # Final insights
    if char_results:
        char_uncertainties = [r['logit_variance'] for r in char_results]
        print(f"Character uncertainty range: {min(char_uncertainties):.6f} - {max(char_uncertainties):.6f}")
    
    if quote_results:
        quote_uncertainties = [r['logit_variance'] for r in quote_results]
        print(f"Famous quotes uncertainty: {np.mean(quote_uncertainties):.6f} ± {np.std(quote_uncertainties):.6f}")
    
    if stage_results:
        stage_uncertainties = [r['logit_variance'] for r in stage_results]
        print(f"Stage directions uncertainty: {np.mean(stage_uncertainties):.6f} ± {np.std(stage_uncertainties):.6f}")
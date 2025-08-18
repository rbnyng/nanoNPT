import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from model import GPTConfig, GPT
from uncertainty_utils import analyze_prompt_uncertainty, sample_with_uncertainty, compare_latent_effects

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
def analyze_character_uncertainty():
    print("\n" + "="*70)
    print("CHARACTER UNCERTAINTY ANALYSIS")
    print("="*70)
    
    results = []
    for char in CHARACTER_NAMES:
        result = analyze_prompt_uncertainty(model, char, encode, decode, n_samples=20, verbose=False)
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
        result = analyze_prompt_uncertainty(model, quote, encode, decode, n_samples=20, verbose=False)
        if result:
            results.append(result)
            print(f"'{quote}' → Uncertainty: {result['logit_variance']:.6f}")
            # Show top prediction
            if result['top_tokens']:
                top_char, top_prob, _ = result['top_tokens'][0]
                print(f"  Top prediction: '{top_char}' (prob: {top_prob:.3f})")
    
    return results

@torch.no_grad()
def analyze_stage_directions():
    print("\n" + "="*70)
    print("STAGE DIRECTIONS ANALYSIS")
    print("="*70)
    
    results = []
    for stage in STAGE_DIRECTIONS:
        result = analyze_prompt_uncertainty(model, stage, encode, decode, n_samples=20, verbose=False)
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
            result = analyze_prompt_uncertainty(model, prompt, encode, decode, n_samples=15, verbose=False)
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
    sample_with_uncertainty(model, "ROMEO:", encode, decode, max_new_tokens=40, n_samples=3)
    sample_with_uncertainty(model, "To be or not to ", encode, decode, max_new_tokens=30, n_samples=3)
    
    # Global latent comparison
    compare_latent_effects(model, "JULIET:", encode, decode, n_samples=4, max_tokens=25)
    
    # Detailed analysis
    analyze_prompt_uncertainty(model, "ROMEO:", encode, decode, n_samples=30, verbose=True)
    analyze_prompt_uncertainty(model, "To be or not to ", encode, decode, n_samples=30, verbose=True)
    analyze_prompt_uncertainty(model, "Enter ", encode, decode, n_samples=30, verbose=True)
    
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
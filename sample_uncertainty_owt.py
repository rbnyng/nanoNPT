import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import tiktoken
from model import GPTConfig, GPT
from uncertainty_utils import analyze_prompt_uncertainty, sample_with_uncertainty, compare_latent_effects

# Configuration
out_dir = 'out-np-owt-small'
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
print(f"Training iteration: {checkpoint.get('iter_num', 'unknown')}")

# Load tokenizer (OpenWebText uses GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

print(f"Vocabulary size: {enc.n_vocab}")

# Load validation data
val_data = np.memmap('data/openwebtext/val.bin', dtype=np.uint16, mode='r')
print(f"Validation data: {len(val_data)} tokens")

# ============================================================================
# OPENWEBTEXT-SPECIFIC TEST PROMPTS
# ============================================================================

# Factual knowledge
FACTUAL_STATEMENTS = [
    "The capital of France is",
    "The president of the United States is",
    "Water boils at",
    "The sun rises in the",
    "Two plus two equals",
    "The largest planet in our solar system is",
]

# Famous quotes/completions - should have low uncertainty
FAMOUS_COMPLETIONS = [
    "To be or not to be, that is the",
    "Four score and seven years ago",
    "I have a dream that",
    "Ask not what your country can do for you",
    "The quick brown fox jumps over the",
    "Once upon a time",
]

# Opinion/subjective starters  
OPINION_STARTERS = [
    "In my opinion,",
    "I believe that",
    "The best movie ever made is",
    "My favorite food is", 
    "The most important thing in life is",
    "I think the future will",
]

# Technical/ambiguous contexts
TECHNICAL_CONTEXTS = [
    "The algorithm works by",
    "In machine learning,",
    "The economic impact of",
    "Scientists have discovered that",
    "According to recent studies,",
    "The main advantage of",
]

# News/current events style
NEWS_CONTEXTS = [
    "Breaking news:",
    "In a recent development,",
    "The company announced that",
    "Government officials stated that",
    "Following the announcement,",
    "Market analysts predict that",
]

# Short vs long contexts - test context length effects
SHORT_CONTEXTS = ["The", "In", "At", "For", "With", "By"]
LONG_CONTEXTS = [
    "After careful consideration of all the available evidence and expert opinions,",
    "In the rapidly evolving landscape of modern technology and digital innovation,",
    "Despite numerous challenges and setbacks throughout the development process,",
    "According to multiple independent sources and verified reports from experts,",
]

@torch.no_grad()
def run_category_analysis():
    print("\n" + "="*80)
    print("CATEGORICAL UNCERTAINTY ANALYSIS")
    print("="*80)
    
    categories = [
        ("Factual Statements", FACTUAL_STATEMENTS),
        ("Famous Completions", FAMOUS_COMPLETIONS), 
        ("Opinion Starters", OPINION_STARTERS),
        ("Technical Contexts", TECHNICAL_CONTEXTS),
        ("News Contexts", NEWS_CONTEXTS),
        ("Short Contexts", SHORT_CONTEXTS),
        ("Long Contexts", LONG_CONTEXTS),
    ]
    
    results = {}
    
    for category_name, prompts in categories:
        print(f"\n--- {category_name.upper()} ---")
        category_results = []
        
        for prompt in prompts:
            result = analyze_prompt_uncertainty(model, prompt, encode, decode, n_samples=20, verbose=False)
            if result:
                category_results.append(result)
                print(f"'{prompt[:40]:40s}' → {result['logit_variance']:.6f}")
        
        if category_results:
            uncertainties = [r['logit_variance'] for r in category_results]
            results[category_name] = {
                'mean': np.mean(uncertainties),
                'std': np.std(uncertainties),
                'results': category_results
            }
    
    # Summary comparison
    print(f"\n--- CATEGORY SUMMARY ---")
    for category_name, data in results.items():
        print(f"{category_name:20s}: {data['mean']:.6f} ± {data['std']:.6f}")
    
    return results

@torch.no_grad()
def test_context_length_effects():
    print(f"\n=== CONTEXT LENGTH EFFECTS ===")
    
    base_text = "The quick brown fox jumps over the lazy dog and then runs through the forest"
    words = base_text.split()
    
    results = []
    for length in [1, 2, 4, 8, 12, 16]:
        if length <= len(words):
            context = " ".join(words[:length])
            result = analyze_prompt_uncertainty(model, context, encode, decode, n_samples=15, verbose=False)
            if result:
                results.append((length, context, result['logit_variance'], result['entropy_mean']))
                print(f"Length {length:2d}: '{context[:40]:40s}' → Unc: {result['logit_variance']:.6f}")
    
    # Check correlation
    if len(results) > 3:
        lengths = [r[0] for r in results]
        uncertainties = [r[2] for r in results]
        correlation = np.corrcoef(lengths, uncertainties)[0, 1]
        print(f"\nCorrelation between context length and uncertainty: {correlation:.3f}")
    
    return results

@torch.no_grad()
def test_calibration_on_validation_data(model, n_samples=30):
    print(f"\n=== VALIDATION DATA CALIBRATION ===")
    
    uncertainties = []
    errors = []
    
    for i in range(n_samples):
        try:
            # Get a random sequence from the validation data
            start_idx = np.random.randint(0, len(val_data) - model.config.block_size - 2)
            context_len = min(100, model.config.block_size - 1)
            
            context = val_data[start_idx : start_idx + context_len]
            target = val_data[start_idx + context_len]
            
            x = torch.tensor([context], dtype=torch.long, device=model.parameters().__next__().device)
            
            global_context, initial_state = model.get_latent_context_and_intermediate_states(x)

            # Loop through the cheap sampling part
            sample_logits = []
            for _ in range(10): # Number of latents to sample per validation item
                z, _, _ = model.sample_global_latent(global_context, sample=True)
                
                film_params = model.latent_to_film(z)
                film_params = film_params.view(-1, model.config.n_layer, 2 * model.config.n_embd)
                gammas, betas = film_params.chunk(2, dim=-1)

                conditioned_x = initial_state
                for j, block in enumerate(model.transformer.h):
                    gamma_j = gammas[:, j, :].unsqueeze(1)
                    beta_j = betas[:, j, :].unsqueeze(1)
                    conditioned_x = block(conditioned_x, gamma_j, beta_j)
                
                conditioned_hidden = model.transformer.ln_f(conditioned_x)
                
                final_hidden = conditioned_hidden[:, 1:, :]
                logits = model.lm_head(final_hidden)
                sample_logits.append(logits[0, -1, :])
            
            logits_stack = torch.stack(sample_logits)
            uncertainty = torch.var(logits_stack, dim=0).mean().item()
            
            mean_logits = torch.mean(logits_stack, dim=0)
            target_tensor = torch.tensor([target], dtype=torch.long, device=model.parameters().__next__().device)
            error = F.cross_entropy(mean_logits.unsqueeze(0), target_tensor).item()
            
            uncertainties.append(uncertainty)
            errors.append(error)
            
            if i % 10 == 0:
                context_text = decode(context[-20:].tolist())
                target_text = decode([target])
                print(f"Sample {i}: uncertainty={uncertainty:.4f}, error={error:.3f}")
                print(f"  Context: '{context_text}' → '{target_text}'")
        
        except Exception as e:
            print(f"Error in sample {i}: {e}")
            continue
    
    if len(uncertainties) > 5:
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        print(f"\nCorrelation between uncertainty and prediction error: {correlation:.3f}")
    
    return uncertainties, errors
    
if __name__ == "__main__":
    print("OPENWEBTEXT NEURAL PROCESS UNCERTAINTY ANALYSIS")
    print("="*80)
    
    # Quick sampling examples
    sample_with_uncertainty(model, "The capital of France is", encode, decode, max_new_tokens=30, n_samples=3)
    sample_with_uncertainty(model, "In my opinion,", encode, decode, max_new_tokens=30, n_samples=3)
    
    # Global latent comparison
    compare_latent_effects(model, "The future of artificial intelligence", encode, decode, n_samples=4, max_tokens=25)
    
    # Detailed analysis of examples
    print("\n" + "="*80)
    print("DETAILED CASE STUDIES")
    print("="*80)
    
    key_examples = [
        "The capital of France is",
        "In my opinion,", 
        "Two plus two equals",
        "The best movie ever made is",
        "Scientists have discovered that",
    ]
    
    for example in key_examples:
        analyze_prompt_uncertainty(model, example, encode, decode, n_samples=30, verbose=True)
    
    # Category analysis
    category_results = run_category_analysis()
    
    # Context length effects
    length_results = test_context_length_effects()
    
    # Validation calibration
    uncertainties, errors = test_calibration_on_validation_data(model, n_samples=30)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Final summary
    if category_results:
        print("\nFindings:")
        factual_unc = category_results.get("Factual Statements", {}).get('mean', 0)
        opinion_unc = category_results.get("Opinion Starters", {}).get('mean', 0)
        
        if factual_unc > 0 and opinion_unc > 0:
            ratio = opinion_unc / factual_unc
            print(f"Opinion/Factual uncertainty ratio: {ratio:.2f}")
        
        if uncertainties and errors:
            correlation = np.corrcoef(uncertainties, errors)[0, 1]

"""
Uncertainty Analysis for Neural Process GPT trained on OpenWebText
Designed for BPE tokenized model with global latents
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import tiktoken
from model import GPTConfig, GPT

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

# Factual knowledge - should have LOW uncertainty for clear facts
FACTUAL_STATEMENTS = [
    "The capital of France is",
    "The president of the United States is",
    "Water boils at",
    "The sun rises in the",
    "Two plus two equals",
    "The largest planet in our solar system is",
]

# Famous quotes/completions - should have LOW uncertainty
FAMOUS_COMPLETIONS = [
    "To be or not to be, that is the",
    "Four score and seven years ago",
    "I have a dream that",
    "Ask not what your country can do for you",
    "The quick brown fox jumps over the",
    "Once upon a time",
]

# Opinion/subjective starters - should have HIGH uncertainty  
OPINION_STARTERS = [
    "In my opinion,",
    "I believe that",
    "The best movie ever made is",
    "My favorite food is", 
    "The most important thing in life is",
    "I think the future will",
]

# Technical/ambiguous contexts - varying uncertainty
TECHNICAL_CONTEXTS = [
    "The algorithm works by",
    "In machine learning,",
    "The economic impact of",
    "Scientists have discovered that",
    "According to recent studies,",
    "The main advantage of",
]

# News/current events style - medium uncertainty
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

# ============================================================================
# ANALYSIS FUNCTIONS  
# ============================================================================

@torch.no_grad()
def analyze_prompt_uncertainty(model, prompt, n_samples=30, verbose=False):
    """Analyze uncertainty for a single prompt using proper global latent sampling"""
    try:
        ids = encode(prompt)
        if len(ids) == 0:
            return None
        
        # Limit context length
        if len(ids) > model.config.block_size:
            ids = ids[-model.config.block_size:]
            
        x = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Get transformer hidden states ONCE
        x_hidden = model.get_transformer_hidden(x)
        global_context = model.aggregate_context(x_hidden)
        
        # Sample multiple DIFFERENT global latents and see their effects
        sample_logits = []
        sample_entropies = []
        
        for _ in range(n_samples):
            with ctx:
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
        
        # Top predicted tokens
        mean_probs = F.softmax(torch.tensor(mean_logits).float(), dim=-1).numpy()
        top_indices = np.argsort(mean_probs)[-5:][::-1]
        
        if verbose:
            print(f"\n--- Analyzing: '{prompt}' ---")
            print(f"Logit variance: {total_variance:.6f}")
            print(f"Entropy: {mean_entropy:.3f} ± {std_entropy:.3f}")
            print("Top predictions:")
            for i, idx in enumerate(top_indices):
                token = decode([idx])
                prob = mean_probs[idx]
                var_prob = var_logits[idx]
                print(f"  {i+1}. '{token}' → prob: {prob:.3f}, var: {var_prob:.6f}")
        
        return {
            'prompt': prompt,
            'logit_variance': total_variance,
            'entropy_mean': mean_entropy,
            'entropy_std': std_entropy,
            'top_tokens': [(decode([idx]), mean_probs[idx], var_logits[idx]) for idx in top_indices],
            'n_samples': n_samples
        }
        
    except Exception as e:
        print(f"Error analyzing '{prompt}': {e}")
        return None

@torch.no_grad()
def sample_with_uncertainty(model, prompt, max_new_tokens=50, n_samples=5, temperature=0.8):
    """Generate multiple samples showing uncertainty"""
    print(f"\n=== UNCERTAINTY SAMPLING: '{prompt}' ===")
    
    try:
        ids = encode(prompt)
        if len(ids) == 0:
            print("Empty prompt!")
            return
        
        if len(ids) > model.config.block_size:
            ids = ids[-model.config.block_size:]
        
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
def compare_global_latents(model, prompt, n_samples=4, max_tokens=30):
    """Show how different global latents affect the same prompt"""
    print(f"\n=== GLOBAL LATENT COMPARISON: '{prompt}' ===")
    
    try:
        ids = encode(prompt)
        if len(ids) > model.config.block_size:
            ids = ids[-model.config.block_size:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Generate with different global latents
        for i in range(n_samples):
            with ctx:
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
def run_category_analysis():
    """Run analysis across different prompt categories"""
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
            result = analyze_prompt_uncertainty(model, prompt, n_samples=20, verbose=False)
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
    """Test how uncertainty changes with context length"""
    print(f"\n=== CONTEXT LENGTH EFFECTS ===")
    
    base_text = "The quick brown fox jumps over the lazy dog and then runs through the forest"
    words = base_text.split()
    
    results = []
    for length in [1, 2, 4, 8, 12, 16]:
        if length <= len(words):
            context = " ".join(words[:length])
            result = analyze_prompt_uncertainty(model, context, n_samples=15, verbose=False)
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
def test_calibration_on_validation_data(n_samples=50):
    """Test uncertainty calibration using validation data"""
    print(f"\n=== VALIDATION DATA CALIBRATION ===")
    
    uncertainties = []
    errors = []
    
    for i in range(n_samples):
        try:
            # Random sequence from validation
            start_idx = np.random.randint(0, len(val_data) - model.config.block_size - 1)
            context_len = min(100, model.config.block_size - 1)
            
            context = val_data[start_idx:start_idx + context_len]
            target = val_data[start_idx + context_len]
            
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            # Get uncertainty for this context
            x_hidden = model.get_transformer_hidden(x)
            global_context = model.aggregate_context(x_hidden)
            
            # Sample multiple latents
            sample_logits = []
            for _ in range(10):
                z, _, _ = model.sample_global_latent(global_context, sample=True)
                z_emb = model.latent_to_emb(z).unsqueeze(1)
                x_conditioned = x_hidden + z_emb
                logits = model.lm_head(x_conditioned)
                sample_logits.append(logits[0, -1, :])
            
            # Calculate uncertainty and error
            logits_stack = torch.stack(sample_logits)
            uncertainty = torch.var(logits_stack, dim=0).mean().item()
            
            mean_logits = torch.mean(logits_stack, dim=0)
            error = F.cross_entropy(mean_logits.unsqueeze(0), torch.tensor([target], device=device)).item()
            
            uncertainties.append(uncertainty)
            errors.append(error)
            
            if i % 10 == 0:
                context_text = decode(context[-20:].tolist()) if len(context) >= 20 else decode(context.tolist())
                target_text = decode([target])
                print(f"Sample {i}: uncertainty={uncertainty:.4f}, error={error:.3f}")
                print(f"  Context: '{context_text}' → '{target_text}'")
        
        except Exception as e:
            print(f"Error in sample {i}: {e}")
            continue
    
    if len(uncertainties) > 5:
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        print(f"\nCorrelation between uncertainty and prediction error: {correlation:.3f}")
        print(f"Mean uncertainty: {np.mean(uncertainties):.4f} ± {np.std(uncertainties):.4f}")
        print(f"Mean error: {np.mean(errors):.3f} ± {np.std(errors):.3f}")
        
        # Show extremes
        sorted_indices = np.argsort(uncertainties)
        print(f"Lowest uncertainty: {uncertainties[sorted_indices[0]]:.4f}, error: {errors[sorted_indices[0]]:.3f}")
        print(f"Highest uncertainty: {uncertainties[sorted_indices[-1]]:.4f}, error: {errors[sorted_indices[-1]]:.3f}")
    
    return uncertainties, errors

if __name__ == "__main__":
    print("OPENWEBTEXT NEURAL PROCESS UNCERTAINTY ANALYSIS")
    print("="*80)
    
    # Quick sampling examples
    sample_with_uncertainty(model, "The capital of France is", max_new_tokens=30, n_samples=3)
    sample_with_uncertainty(model, "In my opinion,", max_new_tokens=30, n_samples=3)
    
    # Global latent comparison
    compare_global_latents(model, "The future of artificial intelligence", n_samples=4, max_tokens=25)
    
    # Detailed analysis of key examples
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
        analyze_prompt_uncertainty(model, example, n_samples=30, verbose=True)
    
    # Category analysis
    category_results = run_category_analysis()
    
    # Context length effects
    length_results = test_context_length_effects()
    
    # Validation calibration
    uncertainties, errors = test_calibration_on_validation_data(n_samples=30)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    # Final summary
    if category_results:
        print("\nKey Findings:")
        factual_unc = category_results.get("Factual Statements", {}).get('mean', 0)
        opinion_unc = category_results.get("Opinion Starters", {}).get('mean', 0)
        
        if factual_unc > 0 and opinion_unc > 0:
            ratio = opinion_unc / factual_unc
            print(f"Opinion/Factual uncertainty ratio: {ratio:.2f}")
            if ratio > 1.2:
                print("✅ Model shows higher uncertainty for subjective contexts!")
            elif ratio < 0.8:
                print("⚠️ Model shows higher uncertainty for factual contexts")
            else:
                print("📊 Uncertainty levels are similar across context types")
        
        if uncertainties and errors:
            correlation = np.corrcoef(uncertainties, errors)[0, 1]
            if correlation > 0.2:
                print("✅ Positive correlation: Higher uncertainty → Higher error")
            elif correlation < -0.2:
                print("🤔 Negative correlation: Higher uncertainty → Lower error")
            else:
                print("📊 No clear correlation between uncertainty and error")
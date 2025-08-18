"""
Analyze uncertainty and function space in Neural Process GPT on OpenWebText
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
out_dir = 'out-np-openwebtext'
device = 'cuda'
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

# Load tokenizer (OpenWebText uses GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Load validation data
val_data = np.memmap('data/openwebtext/val.bin', dtype=np.uint16, mode='r')

@torch.no_grad()
def sample_with_uncertainty(model, start_text, max_new_tokens=50, temperature=0.8, n_samples=5):
    """Sample multiple completions to show uncertainty"""
    model.eval()
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    print(f"=== UNCERTAINTY SAMPLING: '{start_text}' ===")
    
    for sample_idx in range(n_samples):
        current_idx = x.clone()
        
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = current_idx if current_idx.size(1) <= model.config.block_size else current_idx[:, -model.config.block_size:]
            
            # Forward through transformer
            pos = torch.arange(0, idx_cond.size(1), dtype=torch.long, device=device)
            tok_emb = model.transformer.wte(idx_cond)
            pos_emb = model.transformer.wpe(pos)
            x_hidden = model.transformer.drop(tok_emb + pos_emb)
            for block in model.transformer.h:
                x_hidden = block(x_hidden)
            x_hidden = model.transformer.ln_f(x_hidden)
            
            # Sample from function distribution
            function_repr = model.function_encoder(x_hidden[:, [-1], :])
            eps = torch.randn_like(function_repr) * 0.5
            function_sample = function_repr + eps
            logits = model.function_decoder_mean(function_sample)
            
            # Sample next token
            logits = logits.squeeze(1) / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            current_idx = torch.cat((current_idx, idx_next), dim=1)
        
        print(f"\n--- Sample {sample_idx + 1} ---")
        print(decode(current_idx[0].tolist()))
        print("---")

@torch.no_grad()
def analyze_function_space(model, sample_contexts):
    """Analyze the learned function representations"""
    model.eval()
    
    function_reprs = []
    valid_contexts = []
    
    print("\n=== FUNCTION SPACE ANALYSIS ===")
    
    for context in sample_contexts:
        try:
            ids = encode(context)
            if len(ids) > model.config.block_size:
                ids = ids[:model.config.block_size]
            if len(ids) == 0:
                continue
                
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            
            # Forward pass
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
            tok_emb = model.transformer.wte(x)
            pos_emb = model.transformer.wpe(pos)
            hidden = model.transformer.drop(tok_emb + pos_emb)
            for block in model.transformer.h:
                hidden = block(hidden)
            hidden = model.transformer.ln_f(hidden)
            
            # Get function representation
            func_repr = model.function_encoder(hidden[:, -1, :])
            function_reprs.append(func_repr.cpu().numpy())
            valid_contexts.append(context)
            
        except Exception as e:
            print(f"Skipping context due to error: {e}")
            continue
    
    if len(function_reprs) < 2:
        print("Not enough valid contexts for analysis")
        return
        
    function_reprs = np.array(function_reprs).squeeze()
    print(f"Function representation shape: {function_reprs.shape}")
    
    # Simple distance analysis
    print("\n=== PAIRWISE DISTANCES ===")
    for i in range(len(valid_contexts)):
        for j in range(i+1, len(valid_contexts)):
            dist = np.linalg.norm(function_reprs[i] - function_reprs[j])
            print(f"'{valid_contexts[i][:30]}...' vs '{valid_contexts[j][:30]}...': {dist:.3f}")

@torch.no_grad()
def measure_uncertainty_calibration(model, n_samples=100):
    """See if uncertainty correlates with prediction difficulty"""
    model.eval()
    
    uncertainties = []
    errors = []
    
    print("\n=== UNCERTAINTY CALIBRATION ===")
    print("Measuring correlation between uncertainty and prediction error...")
    
    for i in range(n_samples):
        try:
            # Get random sequence from validation data
            start_idx = np.random.randint(0, len(val_data) - model.config.block_size - 1)
            context_len = min(100, model.config.block_size)  # Use longer context for OpenWebText
            
            x = torch.tensor(val_data[start_idx:start_idx + context_len], dtype=torch.long, device=device)[None, ...]
            target = torch.tensor([val_data[start_idx + context_len]], dtype=torch.long, device=device)[None, ...]
            
            # Forward pass
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
            tok_emb = model.transformer.wte(x)
            pos_emb = model.transformer.wpe(pos)
            hidden = model.transformer.drop(tok_emb + pos_emb)
            for block in model.transformer.h:
                hidden = block(hidden)
            hidden = model.transformer.ln_f(hidden)
            
            # Sample multiple functions
            predictions = []
            for _ in range(20):  # More samples for better uncertainty estimate
                eps = torch.randn(1, model.config.uncertainty_dim, device=device) * 0.5
                func_repr = model.function_encoder(hidden[:, [-1], :]) + eps
                logits = model.function_decoder_mean(func_repr)
                predictions.append(logits)
            
            # Calculate uncertainty as variance
            logits_stack = torch.stack(predictions)
            uncertainty = torch.var(logits_stack, dim=0).mean().item()
            
            # Calculate error
            mean_logits = torch.mean(logits_stack, dim=0)
            error = F.cross_entropy(mean_logits.view(-1, mean_logits.size(-1)), target.view(-1)).item()
            
            uncertainties.append(uncertainty)
            errors.append(error)
            
            if i % 20 == 0:
                print(f"Sample {i}: uncertainty={uncertainty:.4f}, error={error:.3f}")
                print(f"  Context: '{decode(x[0][-20:].tolist())}'")
                print(f"  Target: '{decode(target[0].tolist())}'")
                
        except Exception as e:
            print(f"Error in sample {i}: {e}")
            continue
    
    if len(uncertainties) > 1:
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        print(f"\n🎯 UNCERTAINTY-ERROR CORRELATION: {correlation:.3f}")
        print(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
        print(f"Mean error: {np.mean(errors):.3f}")
        
        # Show some examples
        sorted_indices = np.argsort(uncertainties)
        print(f"\nLowest uncertainty: {uncertainties[sorted_indices[0]]:.4f}, error: {errors[sorted_indices[0]]:.3f}")
        print(f"Highest uncertainty: {uncertainties[sorted_indices[-1]]:.4f}, error: {errors[sorted_indices[-1]]:.3f}")
        
        # If correlation is good, celebrate!
        if correlation > 0.3:
            print("🚀 STRONG POSITIVE CORRELATION - This is breakthrough territory!")
        elif correlation > 0.1:
            print("✅ Positive correlation - Good progress!")
        else:
            print("⚠️  Weak/negative correlation - Needs work")
            
    else:
        print("Not enough valid samples for correlation analysis")

@torch.no_grad()
def test_predictable_vs_ambiguous(model):
    """Test uncertainty on predictable vs ambiguous contexts"""
    
    predictable_contexts = [
        "The capital of France is",
        "Two plus two equals",
        "The quick brown fox jumps over the",
        "Once upon a time",
        "In conclusion,",
        "The president of the United States is",
        "First, second, third,",
    ]
    
    ambiguous_contexts = [
        "The",
        "This",
        "After the meeting",
        "In my opinion",
        "The new",
        "Yesterday I",
        "The company",
    ]
    
    def get_context_uncertainty(context_text):
        try:
            ids = encode(context_text)
            if len(ids) == 0:
                return None
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            
            # Forward pass
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
            hidden = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
            for block in model.transformer.h:
                hidden = block(hidden)
            hidden = model.transformer.ln_f(hidden)
            
            # Sample multiple functions
            predictions = []
            for _ in range(30):  # More samples for OpenWebText
                eps = torch.randn(1, model.config.uncertainty_dim, device=device) * 0.5
                func_repr = model.function_encoder(hidden[:, [-1], :]) + eps
                logits = model.function_decoder_mean(func_repr)
                predictions.append(logits)
            
            # Calculate uncertainty as variance
            logits_stack = torch.stack(predictions)
            uncertainty = torch.var(logits_stack, dim=0).mean().item()
            return uncertainty
            
        except Exception as e:
            print(f"Error processing '{context_text}': {e}")
            return None
    
    print("\n=== PREDICTABLE vs AMBIGUOUS CONTEXTS ===")
    
    print("\n--- PREDICTABLE CONTEXTS (should have LOW uncertainty) ---")
    predictable_uncertainties = []
    for context in predictable_contexts:
        uncertainty = get_context_uncertainty(context)
        if uncertainty is not None:
            predictable_uncertainties.append(uncertainty)
            print(f"'{context}' → Uncertainty: {uncertainty:.4f}")
    
    print("\n--- AMBIGUOUS CONTEXTS (should have HIGH uncertainty) ---")
    ambiguous_uncertainties = []
    for context in ambiguous_contexts:
        uncertainty = get_context_uncertainty(context)
        if uncertainty is not None:
            ambiguous_uncertainties.append(uncertainty)
            print(f"'{context}' → Uncertainty: {uncertainty:.4f}")
    
    # Statistical comparison
    if predictable_uncertainties and ambiguous_uncertainties:
        pred_mean = np.mean(predictable_uncertainties)
        amb_mean = np.mean(ambiguous_uncertainties)
        
        print(f"\n--- SUMMARY ---")
        print(f"Predictable contexts - Mean uncertainty: {pred_mean:.4f}")
        print(f"Ambiguous contexts - Mean uncertainty: {amb_mean:.4f}")
        print(f"Ratio (ambiguous/predictable): {amb_mean/pred_mean:.2f}")
        
        if amb_mean > pred_mean:
            print("Ambiguous contexts have higher uncertainty!")
        else:
            print("Predictable contexts have higher uncertainty")
    
    return predictable_uncertainties, ambiguous_uncertainties

@torch.no_grad()
def analyze_next_token_predictions(model, context_text, n_samples=20):
    """Show what the model predicts for next token with uncertainty"""
    
    print(f"\n=== NEXT TOKEN ANALYSIS: '{context_text}' ===")
    
    ids = encode(context_text)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    
    # Forward pass
    pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
    hidden = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
    for block in model.transformer.h:
        hidden = block(hidden)
    hidden = model.transformer.ln_f(hidden)
    
    # Collect predictions from different function samples
    all_predictions = []
    for _ in range(n_samples):
        eps = torch.randn(1, model.config.uncertainty_dim, device=device) * 0.5
        func_repr = model.function_encoder(hidden[:, [-1], :]) + eps
        logits = model.function_decoder_mean(func_repr)
        probs = F.softmax(logits.squeeze(), dim=-1)
        all_predictions.append(probs.cpu().numpy())
    
    # Average predictions and find top tokens
    mean_probs = np.mean(all_predictions, axis=0)
    var_probs = np.var(all_predictions, axis=0)
    
    # Get top 10 most likely tokens
    top_indices = np.argsort(mean_probs)[-10:][::-1]
    
    print("Top predicted tokens:")
    for i, idx in enumerate(top_indices):
        token = decode([idx])
        mean_prob = mean_probs[idx]
        var_prob = var_probs[idx]
        print(f"{i+1:2d}. '{token}' → prob: {mean_prob:.3f}, variance: {var_prob:.6f}")
    
    # Overall uncertainty
    total_uncertainty = np.mean(var_probs)
    print(f"\nOverall uncertainty: {total_uncertainty:.6f}")

if __name__ == "__main__":
    # Test uncertainty sampling
    sample_with_uncertainty(model, "The capital of France is", max_new_tokens=30, n_samples=3)
    sample_with_uncertainty(model, "In my opinion,", max_new_tokens=30, n_samples=3)
    
    # Analyze function space
    sample_contexts = [
        "In the field of science,",
        "The political situation",
        "Once upon a time",
        "The financial markets",
        "In conclusion,",
        "Breaking news:",
        "From a technical perspective,"
    ]
    analyze_function_space(model, sample_contexts)
    
    # Check uncertainty calibration (THE BIG TEST!)
    measure_uncertainty_calibration(model, n_samples=50)
    
    # Test predictable vs ambiguous
    test_predictable_vs_ambiguous(model)
    
    # Analyze specific predictions
    analyze_next_token_predictions(model, "The capital of France is")
    analyze_next_token_predictions(model, "In my opinion,")
    analyze_next_token_predictions(model, "The")
    
    print("\n=== ANALYSIS COMPLETE ===")

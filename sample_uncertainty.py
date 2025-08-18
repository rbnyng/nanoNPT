"""
Analyze uncertainty and function space in Neural Process GPT
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from model import GPTConfig, GPT

# Configuration
out_dir = 'out-np-shakespeare'
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

# Load tokenizer
meta_path = 'data/shakespeare_char/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load validation data
val_data = np.memmap('data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')

@torch.no_grad()
def sample_with_uncertainty(model, start_text, max_new_tokens=100, temperature=0.8, n_samples=5):
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
            logits = logits.squeeze(1)  
            logits = logits / temperature
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
            print(f"'{valid_contexts[i][:20]}...' vs '{valid_contexts[j][:20]}...': {dist:.3f}")

@torch.no_grad()
def measure_uncertainty_calibration(model, n_samples=50):
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
            context_len = min(50, model.config.block_size)  # Use shorter context
            
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
            for _ in range(10):
                eps = torch.randn(1, model.config.uncertainty_dim, device=device) * 0.1
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
            
            if i % 10 == 0:
                print(f"Sample {i}: uncertainty={uncertainty:.3f}, error={error:.3f}")
                
        except Exception as e:
            print(f"Error in sample {i}: {e}")
            continue
    
    if len(uncertainties) > 1:
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        print(f"\nCorrelation between uncertainty and error: {correlation:.3f}")
        print(f"Mean uncertainty: {np.mean(uncertainties):.3f}")
        print(f"Mean error: {np.mean(errors):.3f}")
        
        # Show some examples
        sorted_indices = np.argsort(uncertainties)
        print(f"\nLowest uncertainty: {uncertainties[sorted_indices[0]]:.3f}, error: {errors[sorted_indices[0]]:.3f}")
        print(f"Highest uncertainty: {uncertainties[sorted_indices[-1]]:.3f}, error: {errors[sorted_indices[-1]]:.3f}")
    else:
        print("Not enough valid samples for correlation analysis")

@torch.no_grad() 
def diagnose_uncertainty_patterns(model, n_samples=20):
    """Check what contexts produce high vs low uncertainty"""
    model.eval()
    
    high_uncertainty_contexts = []
    low_uncertainty_contexts = []
    
    for i in range(n_samples):
        start_idx = np.random.randint(0, len(val_data) - 50)
        context = val_data[start_idx:start_idx + 30]
        
        x = torch.tensor(context, dtype=torch.long, device=device)[None, ...]
        
        # Get uncertainty
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
        hidden = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
        for block in model.transformer.h:
            hidden = block(hidden)
        hidden = model.transformer.ln_f(hidden)
        
        uncertainties = []
        for _ in range(10):
            eps = torch.randn(1, model.config.uncertainty_dim, device=device) * 0.5
            func_repr = model.function_encoder(hidden[:, [-1], :]) + eps
            logits = model.function_decoder_mean(func_repr)
            uncertainties.append(logits)
        
        uncertainty = torch.var(torch.stack(uncertainties), dim=0).mean().item()
        context_text = decode(context.tolist())
        
        if uncertainty > 0.12:
            high_uncertainty_contexts.append((uncertainty, context_text))
        elif uncertainty < 0.08:
            low_uncertainty_contexts.append((uncertainty, context_text))
    
    print("\n=== HIGH UNCERTAINTY CONTEXTS ===")
    for unc, text in sorted(high_uncertainty_contexts, reverse=True)[:3]:
        print(f"Uncertainty: {unc:.3f} | Text: '{text}'")
    
    print("\n=== LOW UNCERTAINTY CONTEXTS ===") 
    for unc, text in sorted(low_uncertainty_contexts)[:3]:
        print(f"Uncertainty: {unc:.3f} | Text: '{text}'")

@torch.no_grad()
def test_predictable_vs_ambiguous(model):
    """Test uncertainty on predictable vs ambiguous contexts"""
    
    predictable_contexts = [
        "ROMEO:\nMy name is ",
        "JULIET:\nGood night, good night! parting is such sweet ",
        "HAMLET:\nTo be or not to ",
        "The quick brown fox jumps over the ",
        "Once upon a ",
    ]
    
    ambiguous_contexts = [
        "ROMEO:\n",
        "JULIET:\n", 
        "Enter ",
        "KING:\n",
        "LADY:\n",
        "A ",
        "The ",
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
            for _ in range(20):  # More samples for better estimate
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
            print(f"'{context}' → Uncertainty: {uncertainty:.3f}")
    
    print("\n--- AMBIGUOUS CONTEXTS (should have HIGH uncertainty) ---")
    ambiguous_uncertainties = []
    for context in ambiguous_contexts:
        uncertainty = get_context_uncertainty(context)
        if uncertainty is not None:
            ambiguous_uncertainties.append(uncertainty)
            print(f"'{context}' → Uncertainty: {uncertainty:.3f}")
    
    # Statistical comparison
    if predictable_uncertainties and ambiguous_uncertainties:
        pred_mean = np.mean(predictable_uncertainties)
        amb_mean = np.mean(ambiguous_uncertainties)
        
        print(f"\n--- SUMMARY ---")
        print(f"Predictable contexts - Mean uncertainty: {pred_mean:.3f}")
        print(f"Ambiguous contexts - Mean uncertainty: {amb_mean:.3f}")
        print(f"Ratio (ambiguous/predictable): {amb_mean/pred_mean:.2f}")
        
        if amb_mean > pred_mean:
            print("Ambiguous contexts have higher uncertainty!")
        else:
            print("Predictable contexts have higher uncertainty")
    
    return predictable_uncertainties, ambiguous_uncertainties

@torch.no_grad()
def analyze_next_token_predictions(model, context_text, n_samples=10):
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
        char = itos[idx] if idx < len(itos) else f"UNK_{idx}"
        mean_prob = mean_probs[idx]
        var_prob = var_probs[idx]
        print(f"{i+1:2d}. '{char}' → prob: {mean_prob:.3f}, variance: {var_prob:.6f}")
    
    # Overall uncertainty
    total_uncertainty = np.mean(var_probs)
    print(f"\nOverall uncertainty: {total_uncertainty:.6f}")
    
if __name__ == "__main__":
    # Test uncertainty sampling
    sample_with_uncertainty(model, "ROMEO:\n", max_new_tokens=50, n_samples=3)
    
    # Analyze function space
    sample_contexts = [
        "ROMEO:",
        "JULIET:", 
        "HAMLET:",
        "KING:",
        "LADY:"
    ]
    analyze_function_space(model, sample_contexts)
    
    # Check uncertainty calibration
    measure_uncertainty_calibration(model, n_samples=30)
    
    diagnose_uncertainty_patterns(model)
    
    pred_unc, amb_unc = test_predictable_vs_ambiguous(model)
    
    # Analyze specific predictions
    analyze_next_token_predictions(model, "ROMEO:\nMy name is ")
    analyze_next_token_predictions(model, "ROMEO:\n")
    analyze_next_token_predictions(model, "Enter ")
    
    print("\n=== ANALYSIS COMPLETE ===")
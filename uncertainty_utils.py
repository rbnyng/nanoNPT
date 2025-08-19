import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

@torch.no_grad()
def analyze_prompt_uncertainty(model, prompt, encode, decode, n_samples=30, verbose=False):
    """
    Analyzes the model's epistemic uncertainty for the next token after a given prompt.

    This is done by sampling multiple global latents (z) for the same context and
    measuring the variance in the resulting logit predictions.

    Args:
        model (GPT): The trained Neural Process GPT model.
        prompt (str): The input text context.
        encode (function): The tokenizer's encoding function.
        decode (function): The tokenizer's decoding function.
        n_samples (int): The number of latent samples to draw for the analysis.
        verbose (bool): If True, prints a detailed breakdown of the results.

    Returns:
        dict or None: A dictionary containing uncertainty metrics, or None if an error occurs.
    """
    try:
        device = next(model.parameters()).device
        ids = encode(prompt)
        if not ids:
            return None

        if len(ids) > model.config.block_size:
            ids = ids[-model.config.block_size:]
        
        x = torch.tensor([ids], dtype=torch.long, device=device)

        # Run the expensive part of the model once to get the global context
        # and the initial hidden state before the conditioned blocks.
        global_context, initial_state = model.get_latent_context_and_intermediate_states(x)

        sample_logits = []
        sample_entropies = []
        
        for _ in range(n_samples):
            # a. Sample a different global latent each time
            z, _, _ = model.sample_global_latent(global_context, sample=True)
            
            # b. Generate per-layer FiLM parameters
            film_params = model.latent_to_film(z)
            film_params = film_params.view(-1, model.config.n_layer, 2 * model.config.n_embd)
            gammas, betas = film_params.chunk(2, dim=-1)

            # c. Run the conditioned forward pass using the cached initial_state
            conditioned_x = initial_state
            for i, block in enumerate(model.transformer.h):
                gamma_i = gammas[:, i, :].unsqueeze(1)
                beta_i = betas[:, i, :].unsqueeze(1)
                conditioned_x = block(conditioned_x, gamma_i, beta_i)
            
            conditioned_hidden = model.transformer.ln_f(conditioned_x)
            
            # d. Strip CLS and get logits
            final_hidden = conditioned_hidden[:, 1:, :] # (B, T, C)
            logits = model.lm_head(final_hidden)
            
            # Get last token prediction
            last_logits = logits[0, -1, :].float()
            probs = F.softmax(last_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))
            
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
        
        top_tokens_info = []
        for idx in top_indices:
            # Handle both BPE and char-level decoding
            token_str = decode([idx])
            prob = mean_probs[idx]
            var = var_logits[idx]
            top_tokens_info.append((token_str, prob, var))

        if verbose:
            print(f"\n--- Analyzing: '{prompt}' ---")
            print(f"Logit variance (Epistemic Uncertainty): {total_variance:.6f}")
            print(f"Mean Entropy (Aleatoric Uncertainty): {mean_entropy:.3f} ± {std_entropy:.3f}")
            print("Top predictions (from mean logits):")
            for i, (token, prob, var) in enumerate(top_tokens_info):
                print(f"  {i+1}. '{token}' → prob: {prob:.3f}, var: {var:.6f}")
        
        return {
            'prompt': prompt,
            'logit_variance': total_variance,
            'entropy_mean': mean_entropy,
            'entropy_std': std_entropy,
            'top_tokens': top_tokens_info,
            'n_samples': n_samples
        }
    except Exception as e:
        print(f"Error analyzing '{prompt}': {e}")
        return None

@torch.no_grad()
def sample_with_uncertainty(model, prompt, encode, decode, max_new_tokens=50, n_samples=5, temperature=0.8):
    """
    Generates multiple, diverse samples from a single prompt by using a different
    global latent for each sample.

    Args:
        model (GPT): The trained Neural Process GPT model.
        prompt (str): The input text context.
        encode (function): The tokenizer's encoding function.
        decode (function): The tokenizer's decoding function.
        max_new_tokens (int): The number of new tokens to generate for each sample.
        n_samples (int): The number of different samples to generate.
        temperature (float): The sampling temperature.
    """
    print(f"\n=== UNCERTAINTY SAMPLING: '{prompt}' ===")
    try:
        device = next(model.parameters()).device
        ids = encode(prompt)
        if not ids:
            print("Empty prompt!")
            return
        
        if len(ids) > model.config.block_size:
            ids = ids[-model.config.block_size:]
        
        x = torch.tensor([ids], dtype=torch.long, device=device)

        samples = model.generate_with_uncertainty(
            x,
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
def compare_latent_effects(model, prompt, encode, decode, n_samples=4, max_tokens=30, temperature=0.8):
    """
    Shows how different fixed global latents affect generation from the same prompt.

    Each generated sequence uses one fixed latent for all its generation steps,
    demonstrating how a single latent can impose a consistent style or topic.

    Args:
        model (GPT): The trained Neural Process GPT model.
        prompt (str): The input text context.
        encode (function): The tokenizer's encoding function.
        decode (function): The tokenizer's decoding function.
        n_samples (int): The number of different fixed latents to test.
        max_tokens (int): The number of tokens to generate for each latent.
        temperature (float): The sampling temperature.
    """
    print(f"\n=== GLOBAL LATENT COMPARISON: '{prompt}' ===")
    try:
        device = next(model.parameters()).device
        ids = encode(prompt)
        if len(ids) > model.config.block_size:
            ids = ids[-model.config.block_size:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Generate with different fixed global latents
        for i in range(n_samples):
            # Get a fresh global latent by doing one forward pass on the initial context
            hidden_states_with_cls = model.get_transformer_hidden(x)
            global_context = model.aggregate_context(hidden_states_with_cls)
            z_fixed, _, _ = model.sample_global_latent(global_context, sample=True)
            
            # Autoregressively generate with this fixed latent
            current_tokens = x.clone()
            for _ in range(max_tokens):
                idx_cond = current_tokens if current_tokens.size(1) <= model.config.block_size else current_tokens[:, -model.config.block_size:]
                
                h_cls = model.get_transformer_hidden(idx_cond)
                h = h_cls[:, 1:, :] # Strip CLS token

                # Condition with the SAME fixed latent at each step
                if hasattr(model.config, 'conditioning_method') and model.config.conditioning_method == 'film':
                    film_params = model.latent_to_film(z_fixed).unsqueeze(1)
                    gamma, beta = film_params.chunk(2, dim=-1)
                    h_conditioned = gamma * h + beta
                else:
                    z_emb = model.latent_to_emb(z_fixed).unsqueeze(1)
                    h_conditioned = h + z_emb
                
                logits = model.lm_head(h_conditioned)
                
                # Sample the next token
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            print(f"\nLatent {i+1}: {decode(current_tokens[0].tolist())}")
            
    except Exception as e:
        print(f"Error in latent comparison for '{prompt}': {e}")

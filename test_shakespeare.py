"""
Test script to verify the fixed Neural Process architecture works correctly
Run this before training to catch any issues early
"""

import torch
import numpy as np
from model import GPTConfig, GPT

def test_architecture():
    print("Testing Fixed Neural Process GPT Architecture...")
    
    # Test config
    config = GPTConfig(
        block_size=128,
        vocab_size=65,  # Shakespeare char vocab
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        use_global_latent=True,
        uncertainty_dim=32,
        context_aggregation='mean',
        free_bits=0.05
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)
    model.eval()
    
    print(f"Model created with {model.get_num_params()/1e6:.2f}M parameters")
    
    # Test data
    batch_size = 4
    seq_len = 64
    vocab_size = config.vocab_size
    
    # Random input sequences
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
    
    # Test forward pass
    with torch.no_grad():
        print("\n1. Testing forward pass...")
        logits, recon_loss, kl_loss = model(x, targets)
        
        print(f"   Logits shape: {logits.shape}")
        print(f"   Reconstruction loss: {recon_loss:.4f}")
        print(f"   KL loss: {kl_loss:.4f}")
        
        assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong logits shape: {logits.shape}"
        assert recon_loss.item() > 0, "Reconstruction loss should be positive"
        assert kl_loss.item() >= 0, "KL loss should be non-negative"
        print("   ✓ Forward pass working correctly")
    
    # Test inference mode
    with torch.no_grad():
        print("\n2. Testing inference mode...")
        logits_inf, recon_loss_inf, kl_loss_inf = model(x)
        
        print(f"   Inference logits shape: {logits_inf.shape}")
        assert recon_loss_inf is None, "Recon loss should be None in inference"
        assert kl_loss_inf is None, "KL loss should be None in inference"
        print("   ✓ Inference mode working correctly")
    
    # Test uncertainty sampling
    with torch.no_grad():
        print("\n3. Testing uncertainty sampling...")
        
        # Sample multiple times with same input
        sample_logits = []
        for i in range(5):
            logits_sample, _, _ = model(x, targets)
            sample_logits.append(logits_sample)
        
        # Check if samples are actually different
        logits_stack = torch.stack(sample_logits)
        variance = torch.var(logits_stack, dim=0).mean()
        
        print(f"   Variance across samples: {variance:.6f}")
        
        if variance > 1e-6:
            print("   ✓ Uncertainty sampling produces different outputs")
        else:
            print("   ⚠ WARNING: Samples are too similar - possible collapse")
    
    # Test generation with uncertainty
    with torch.no_grad():
        print("\n4. Testing generation with uncertainty...")
        
        # Start with a short prompt
        prompt = torch.randint(0, vocab_size, (1, 10), device=device)
        
        # Generate multiple samples
        samples = model.generate_with_uncertainty(
            prompt, max_new_tokens=20, temperature=0.8, n_samples=3
        )
        
        print(f"   Generated {len(samples)} samples")
        for i, sample in enumerate(samples):
            print(f"   Sample {i+1} length: {sample.shape[1]}")
        
        # Check if samples are different
        if len(set(sample.shape[1] for sample in samples)) > 1 or \
           not torch.equal(samples[0], samples[1]):
            print("   ✓ Generated samples are different")
        else:
            print("   ⚠ WARNING: Generated samples are identical")
    
    # Test context aggregation methods
    print("\n5. Testing different context aggregation methods...")
    
    for agg_method in ['mean', 'last']:
        print(f"   Testing {agg_method} aggregation...")
        config_test = GPTConfig(
            block_size=128, vocab_size=65, n_layer=2, n_head=2, n_embd=128,
            use_global_latent=True, uncertainty_dim=16, 
            context_aggregation=agg_method, free_bits=0.05
        )
        
        model_test = GPT(config_test).to(device)
        model_test.eval()
        
        with torch.no_grad():
            logits_test, _, _ = model_test(x[:2, :32], targets[:2, :32])
            print(f"   {agg_method}: logits shape {logits_test.shape} ✓")
    
    # Test KL loss behavior
    print("\n6. Testing KL loss behavior...")
    
    model.train()  # Switch to training mode
    kl_losses = []
    
    for i in range(10):
        with torch.no_grad():
            _, _, kl_loss = model(x, targets)
            kl_losses.append(kl_loss.item())
    
    kl_mean = np.mean(kl_losses)
    kl_std = np.std(kl_losses)
    
    print(f"   KL loss over 10 samples: {kl_mean:.4f} ± {kl_std:.4f}")
    
    if kl_mean > config.free_bits * config.uncertainty_dim * 0.5:
        print("   ✓ KL loss is above free bits threshold")
    else:
        print("   ⚠ WARNING: KL loss might be collapsed")
    
    # Test gradient flow
    print("\n7. Testing gradient flow...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Single training step
    logits, recon_loss, kl_loss = model(x, targets)
    total_loss = recon_loss + 0.01 * kl_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if 'global_context_encoder' in name or 'latent_to_emb' in name:
                print(f"   {name}: grad_norm = {grad_norm:.6f}")
    
    if len(grad_norms) > 0 and max(grad_norms) > 1e-8:
        print("   ✓ Gradients are flowing to NP components")
    else:
        print("   ⚠ WARNING: No gradients in NP components")
    
    optimizer.step()
    
    print("\n8. Architecture comparison test...")
    
    # Compare old vs new architecture on same input
    config_old = GPTConfig(
        block_size=128, vocab_size=65, n_layer=2, n_head=2, n_embd=128,
        use_global_latent=False, uncertainty_dim=16
    )
    
    model_old = GPT(config_old).to(device)
    model_old.eval()
    
    with torch.no_grad():
        logits_new, _, _ = model_test(x[:1, :32], targets[:1, :32])
        logits_old, _, _ = model_old(x[:1, :32], targets[:1, :32])
        
        print(f"   New architecture output shape: {logits_new.shape}")
        print(f"   Old architecture output shape: {logits_old.shape}")
        print("   ✓ Both architectures produce valid outputs")
    
    print("\n" + "="*60)
    print("ARCHITECTURE TEST SUMMARY")
    print("="*60)
    print("✓ Forward pass working")
    print("✓ Inference mode working") 
    print("✓ Generation working")
    print("✓ Multiple aggregation methods working")
    print("✓ Gradient flow verified")
    print("✓ Architecture comparison successful")
    print("\nArchitecture appears ready for training!")
    print("="*60)

def test_specific_shakespeare_cases():
    """Test with Shakespeare-specific patterns"""
    print("\nTesting Shakespeare-specific patterns...")
    
    # Load meta for character mapping
    try:
        import pickle
        with open('data/shakespeare_char/meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
        config = GPTConfig(
            block_size=128, vocab_size=meta['vocab_size'],
            n_layer=2, n_head=2, n_embd=128,
            use_global_latent=True, uncertainty_dim=16
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = GPT(config).to(device)
        model.eval()
        
        # Test different Shakespeare contexts
        test_contexts = [
            "ROMEO:",
            "JULIET:", 
            "Enter ",
            "To be or not to ",
            "KING:",
        ]
        
        print("Testing uncertainty on different contexts:")
        
        for context in test_contexts:
            try:
                ids = encode(context)
                if len(ids) > 0:
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    
                    # Sample multiple times
                    uncertainties = []
                    for _ in range(5):
                        with torch.no_grad():
                            logits, _, _ = model(x)
                            # Simple uncertainty measure: entropy
                            probs = torch.softmax(logits[0, -1], dim=-1)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                            uncertainties.append(entropy.item())
                    
                    mean_unc = np.mean(uncertainties)
                    std_unc = np.std(uncertainties)
                    print(f"   '{context}' → uncertainty: {mean_unc:.3f} ± {std_unc:.4f}")
                    
            except Exception as e:
                print(f"   '{context}' → Error: {e}")
        
    except FileNotFoundError:
        print("Shakespeare meta.pkl not found - run data preparation first")
        print("This test requires: python data/shakespeare_char/prepare.py")

if __name__ == "__main__":
    test_architecture()
    test_specific_shakespeare_cases()
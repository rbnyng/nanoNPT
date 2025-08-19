import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Neural Process parameters
    uncertainty_dim: int = 64  # Global latent dimension (much smaller)
    use_global_latent: bool = True  # Use global sequence-level latent
    context_aggregation: str = 'cls'  # 'mean', 'last', 'cls'
    conditioning_method: str = 'film' # 'add' or 'film'
    free_bits: float = 0.1  # Minimum KL loss per latent dimension to prevent collapse
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.latent_to_film = nn.Linear(config.uncertainty_dim, 2 * config.n_embd)
        # Standard transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size + 1, config.n_embd), # block_size + 1 for CLS
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Neural Process components
        if config.use_global_latent:
            # Global context encoder: sequence -> global latent distribution
            self.global_context_encoder = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd // 2),
                nn.ReLU(),
                nn.Linear(config.n_embd // 2, 2 * config.uncertainty_dim)
            )
            
            # Project latent back to embedding space for conditioning
            if config.conditioning_method == 'film':
                nn.init.zeros_(self.latent_to_film.weight)
                with torch.no_grad():
                    self.latent_to_film.bias[:config.n_embd].fill_(1.0)
                    self.latent_to_film.bias[config.n_embd:].zero_()
            else: # Default to 'add'
                self.latent_to_emb = nn.Linear(config.uncertainty_dim, config.n_embd)
                
        else:
            # Fallback to old per-token system (for comparison)
            self.function_encoder = nn.Linear(config.n_embd, 2 * config.uncertainty_dim)
            self.function_decoder_mean = nn.Linear(config.uncertainty_dim, config.vocab_size)

        # Standard language modeling head (tied weights)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (following GPT-2)
        self.lm_head.weight = self.transformer.wte.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_transformer_hidden(self, idx):
        """Get transformer hidden states without final projection"""
        device = idx.device
        b, t = idx.size()
        # we add 1 to t for the CLS token
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t + 1, dtype=torch.long, device=device) # t + 1
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # Prepend the CLS token to the sequence
        cls_token_emb = self.cls_token.expand(b, -1, -1) # (b, 1, n_embd)
        x = torch.cat((cls_token_emb, tok_emb), dim=1) # (b, t + 1, n_embd)
        
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t + 1, n_embd)
        x = self.transformer.drop(x + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x # returns shape (b, t + 1, n_embd)
        
    def aggregate_context(self, x):
        """Aggregate sequence into global context representation"""
        if self.config.context_aggregation == 'mean':
            return torch.mean(x, dim=1)  # (B, n_embd)
        elif self.config.context_aggregation == 'last':
            return x[:, -1, :]  # (B, n_embd)
        elif self.config.context_aggregation == 'cls':
            return x[:, 0, :] # Use the hidden state of the CLS token
        else:
            raise ValueError(f"Unknown context aggregation: {self.config.context_aggregation}")

    def sample_global_latent(self, context, sample=True):
        """Sample from global latent distribution"""
        latent_params = self.global_context_encoder(context)  # (B, 2*uncertainty_dim)
        mu, log_sigma = latent_params.chunk(2, dim=-1)  # Each (B, uncertainty_dim)
        
        if sample:
            eps = torch.randn_like(mu)
            z = mu + eps * log_sigma.exp()
        else:
            z = mu  # Use mean for deterministic inference
            
        return z, mu, log_sigma

    def compute_kl_loss(self, mu, log_sigma):
        """Compute KL divergence with free bits"""
        # Standard VAE KL: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        kl_per_dim = -0.5 * (1 + 2*log_sigma - mu.pow(2) - (2*log_sigma).exp())  # (B, latent_dim)
        
        # free-bits: subtract threshold, clamp below at 0
        if self.config.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim - self.config.free_bits, min=0.0)
        # sum over latent dimensions, mean over batch
        kl_loss = torch.sum(kl_per_dim, dim=-1).mean()
        
        return kl_loss

    def forward(self, idx, targets=None):
        # Get transformer hidden states (includes CLS token at position 0)
        hidden_states_with_cls = self.get_transformer_hidden(idx) # (B, T + 1, n_embd)
        
        if self.config.use_global_latent:
            if targets is not None:
                # Training: sample from global latent distribution
                global_context = self.aggregate_context(hidden_states_with_cls)  # (B, n_embd)
                z, mu, log_sigma = self.sample_global_latent(global_context, sample=True)
                
                # Strip the CLS token before conditioning and final projection
                x = hidden_states_with_cls[:, 1:, :] # (B, T, n_embd)
                
                # Condition transformer output with global latent
                if self.config.conditioning_method == 'film':
                    film_params = self.latent_to_film(z).unsqueeze(1) # (B, 1, 2 * n_embd)
                    gamma, beta = film_params.chunk(2, dim=-1) # Each is (B, 1, n_embd)
                    x_conditioned = gamma * x + beta # Apply FiLM
                else: # Default to 'add'
                    z_emb = self.latent_to_emb(z).unsqueeze(1)
                    x_conditioned = x + z_emb
                    
                # Standard language modeling
                logits = self.lm_head(x_conditioned)
                
                # Compute losses
                recon_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
                kl_loss = self.compute_kl_loss(mu, log_sigma)
                
                return logits, recon_loss, kl_loss
            else:
                # Inference: use mean latent (deterministic)
                global_context = self.aggregate_context(hidden_states_with_cls)
                z, _, _ = self.sample_global_latent(global_context, sample=False)
                
                # Strip the CLS token
                x = hidden_states_with_cls[:, 1:, :] # (B, T, n_embd)
                
                if self.config.conditioning_method == 'film':
                    film_params = self.latent_to_film(z).unsqueeze(1)  # (B, 1, 2*n_embd)
                    gamma, beta = film_params.chunk(2, dim=-1)         # (B, 1, n_embd)
                    x_conditioned = gamma * x + beta
                else:  # 'add'
                    z_emb = self.latent_to_emb(z).unsqueeze(1)         # (B, 1, n_embd)
                    x_conditioned = x + z_emb
                    
                logits = self.lm_head(x_conditioned)
                
                return logits, None, None
        else:
            # Fallback to old per-token system for comparison
            if targets is not None:
                function_output = self.function_encoder(x)
                mu, log_sigma = torch.chunk(function_output, 2, dim=-1)
                eps = torch.randn_like(mu)
                function_sample = mu + eps * log_sigma.exp()
                logits = self.function_decoder_mean(function_sample)
                
                recon_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
                kl_loss = self.compute_kl_loss(mu.view(-1, mu.size(-1)), log_sigma.view(-1, log_sigma.size(-1)))
                
                return logits, recon_loss, kl_loss
            else:
                function_output = self.function_encoder(x[:, [-1], :])
                mu, _ = torch.chunk(function_output, 2, dim=-1)
                logits = self.function_decoder_mean(mu)
                return logits, None, None

    @torch.no_grad()
    def generate_with_uncertainty(self, idx, max_new_tokens, temperature=1.0, top_k=None, n_samples=1):
        """
        Generate with uncertainty sampling using different global latents per sample
        """
        samples = []
        
        for _ in range(n_samples):
            current_idx = idx.clone()
            
            # Sample a global latent once for this entire sequence
            if self.config.use_global_latent:
                # Get initial context and sample latent
                x_init = self.get_transformer_hidden(current_idx)
                global_context = self.aggregate_context(x_init)
                z_fixed, _, _ = self.sample_global_latent(global_context, sample=True)
            
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = current_idx if current_idx.size(1) <= self.config.block_size else current_idx[:, -self.config.block_size:]
                
                # forward the model to get the logits for the index in the sequence
                hidden_states_with_cls = self.get_transformer_hidden(idx_cond)
                x = hidden_states_with_cls[:, 1:, :]  # strip CLS, consistent with forward()              
                
                if self.config.use_global_latent:
                    # Use the fixed global latent for entire sequence
                    if hasattr(self.config, 'conditioning_method') and self.config.conditioning_method == 'film':
                        film_params = self.latent_to_film(z_fixed).unsqueeze(1)
                        gamma, beta = film_params.chunk(2, dim=-1)
                        x_conditioned = gamma * x + beta
                    else: # Default to 'add'
                        z_emb = self.latent_to_emb(z_fixed).unsqueeze(1)
                        x_conditioned = x + z_emb                    
                    logits = self.lm_head(x_conditioned)
                else:
                    # Fallback per-token sampling
                    function_output = self.function_encoder(x[:, [-1], :])
                    mu, log_sigma = torch.chunk(function_output, 2, dim=-1)
                    eps = torch.randn_like(mu)
                    function_sample = mu + eps * log_sigma.exp()
                    logits = self.function_decoder_mean(function_sample)
                
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                current_idx = torch.cat((current_idx, idx_next), dim=1)

            samples.append(current_idx)
        
        return samples

    # Keep all the other methods from original (crop_block_size, from_pretrained, etc.)
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size+1])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Standard generation (uses mean latent if global, or deterministic if per-token)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
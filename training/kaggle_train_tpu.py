#!/usr/bin/env python3
"""
ORPHEUS 500M - 1024 LAYERS - KAGGLE TPU TRAINING
============================================================
FREE TPU v3-8 ON KAGGLE!

INSTRUCTIONS:
1. Go to kaggle.com/code
2. Create New Notebook
3. Settings > Accelerator > TPU v3-8
4. Copy this entire file into a cell
5. Run it!
"""

print("="*70)
print("   ORPHEUS 500M - 1024 LAYERS - KAGGLE TPU TRAINING")
print("   THE DEEPEST TRANSFORMER EVER BUILT!")
print("="*70)

# Install dependencies
print("\nInstalling dependencies...")
import subprocess
subprocess.run(["pip", "install", "-q", "--no-cache-dir", "torch_xla[tpu]", "-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
subprocess.run(["pip", "install", "-q", "--no-cache-dir", "transformers", "tqdm"])

import os
import math
import gc
from dataclasses import dataclass
from typing import Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# TPU imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
    print("TPU detected!")
except ImportError:
    TPU_AVAILABLE = False
    print("No TPU found, using GPU/CPU")

# Checkpoint directory
CHECKPOINT_DIR = "/kaggle/working/orpheus_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoints: {CHECKPOINT_DIR}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

@dataclass
class OrpheusConfig:
    vocab_size: int = 50257
    hidden_size: int = 256
    num_layers: int = 1024
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 280
    max_position_embeddings: int = 2048
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    initializer_range: float = 0.002
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True
    residual_scale: float = 0.022

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: OrpheusConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_base)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, config: OrpheusConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class OrpheusBlock(nn.Module):
    def __init__(self, config: OrpheusConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.residual_scale = config.residual_scale
    
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states * self.residual_scale
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_scale
        return hidden_states

class Orpheus(nn.Module):
    def __init__(self, config: OrpheusConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OrpheusBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, input_ids, labels=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, None, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        logits = F.linear(hidden_states, self.embed_tokens.weight)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}

# ============================================================================
# TRAINING
# ============================================================================

class SyntheticDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens, "labels": tokens}

def train():
    # Get device
    if TPU_AVAILABLE:
        device = xm.xla_device()
        print(f"Using TPU: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {device}")
    
    # Build model
    print("\nBuilding ORPHEUS 500M - 1024 layers...")
    config = OrpheusConfig()
    model = Orpheus(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    model = model.to(device)
    
    # Training settings
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    SEQ_LEN = 512
    LR = 1e-4
    NUM_SAMPLES = 10000
    SAVE_EVERY = 500
    
    # Check for checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    start_step = 0
    
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # Dataset
    dataset = SyntheticDataset(config.vocab_size, SEQ_LEN, NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    if TPU_AVAILABLE:
        dataloader = pl.MpDeviceLoader(dataloader, device)
    
    print(f"\nTraining from step {start_step}...")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    
    model.train()
    optimizer.zero_grad()
    step = start_step
    accum_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"] / GRAD_ACCUM
        loss.backward()
        accum_loss += loss.item()
        
        if (step + 1) % GRAD_ACCUM == 0:
            if TPU_AVAILABLE:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad()
            accum_loss = 0.0
        
        if (step + 1) % SAVE_EVERY == 0:
            print(f"\nSaving checkpoint at step {step+1}...")
            model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            ckpt = {"model": model_cpu, "step": step + 1, "config": config}
            torch.save(ckpt, checkpoint_path)
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"step_{step+1}.pt"))
            print("Saved!")
        
        step += 1
    
    # Final save
    print("\nTraining complete! Saving final model...")
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"model": model_cpu, "step": step, "config": config}, 
               os.path.join(CHECKPOINT_DIR, "orpheus_500m_final.pt"))
    print(f"Saved to: {CHECKPOINT_DIR}/orpheus_500m_final.pt")
    print("\nDOWNLOAD CHECKPOINTS: Output tab > Download All")

if __name__ == "__main__":
    train()

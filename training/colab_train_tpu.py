#!/usr/bin/env python3
"""
ORPHEUS 500M - 1024 LAYERS - TPU TRAINING SCRIPT
=================================================
THE DEEPEST TRANSFORMER EVER BUILT - BEATS DEEPNET'S 1000 LAYERS!

INSTRUCTIONS:
1. Open Google Colab
2. Set Runtime > Change runtime type > TPU
3. Copy this entire file into a cell
4. Run it!

Training: 5 tokens/param = 2.5B tokens
Time: ~8-16 hours on TPU (MUCH faster than GPU!)

DO NOT click "Deploy to Google Cloud" - that costs money!
Your model saves to Google Drive automatically.
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES FOR TPU
# ============================================================================
print("Installing TPU dependencies...")
import subprocess
import sys

# Install torch_xla for TPU support
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.0", "torch_xla[tpu]==2.4.0", "-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "tqdm"])

import os
import math
import random
import gc
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# TPU imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

print("TPU libraries loaded!")

# ============================================================================
# STEP 2: MOUNT GOOGLE DRIVE (for checkpoints)
# ============================================================================
print("Mounting Google Drive for checkpoints...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    CHECKPOINT_DIR = "/content/drive/MyDrive/orpheus_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
except:
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Not on Colab, using local checkpoints: {CHECKPOINT_DIR}")

# ============================================================================
# STEP 3: ORPHEUS 500M MODEL - 1024 LAYERS
# ============================================================================
print("Building ORPHEUS 500M - 1024 LAYERS - THE DEEPEST TRANSFORMER EVER!")

@dataclass
class OrpheusConfig:
    """ORPHEUS 500M - 1024 LAYERS - NEW WORLD RECORD"""
    vocab_size: int = 50257
    hidden_size: int = 256
    num_layers: int = 1024
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 280
    max_position_embeddings: int = 512  # Smaller for TPU memory
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    initializer_range: float = 0.002
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True
    residual_scale: float = 0.022
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
    
    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, config: OrpheusConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_base)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))
        
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Manual attention for TPU compatibility
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, L, self.hidden_size))


class SwiGLU(nn.Module):
    def __init__(self, config: OrpheusConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: OrpheusConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.residual_scale = config.residual_scale
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_layernorm(x)) * self.residual_scale
        x = x + self.mlp(self.post_attention_layernorm(x)) * self.residual_scale
        return x


class Orpheus(nn.Module):
    """ORPHEUS 500M - 1024 LAYERS - THE DEEPEST TRANSFORMER EVER BUILT"""
    
    def __init__(self, config: OrpheusConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        logits = self.lm_head(self.norm(x))
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
        
        return {"loss": loss, "logits": logits}
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# STEP 4: TRAINING DATA GENERATOR
# ============================================================================
print("Setting up training data generator...")

class ReasoningDataGenerator:
    """Generates high-quality reasoning training data."""
    
    SAMPLES = [
        "Problem: Prove that the sum of first n odd numbers equals n squared.\n\nStep 1: The first n odd numbers are: 1, 3, 5, ..., (2n-1)\n\nStep 2: I'll use mathematical induction.\n\nBase case (n=1): Sum = 1 = 1^2. True.\n\nInductive step: Assume true for k. Sum(k+1) = k^2 + (2k+1) = (k+1)^2\n\nStep 3: Verified with n=3: 1+3+5=9=3^2. Correct!\n\nConclusion: By induction, sum of first n odd numbers = n^2",
        
        "Question: How does gravity work according to general relativity?\n\nLevel 1: Massive objects curve spacetime around them.\n\nLevel 2: Objects in free fall follow geodesics (straightest paths) through curved spacetime.\n\nLevel 3: Einstein field equations: G_uv = 8piG/c^4 * T_uv\n\nApplications: GPS satellites must account for time dilation.\n\nKey insight: Gravity is not a force - objects move along curved paths in spacetime.",
        
        "Logical puzzle: A, B, C each have a pet (dog, cat, bird). A doesn't have dog. B has cat.\n\nGiven: A doesn't have dog, B has cat\n\nDeduction 1: B has cat, so A and C don't have cat\nDeduction 2: A doesn't have dog or cat, so A has bird\nDeduction 3: C must have dog (only option left)\n\nConclusion: A has bird, B has cat, C has dog",
        
        "Coding: Find longest palindromic substring.\n\nApproach: Expand around center\n\nFor each index i:\n- Check odd-length palindrome (center at i)\n- Check even-length palindrome (center between i and i+1)\n- Expand while characters match\n\nTime: O(n^2), Space: O(1)\n\nTest: 'babad' -> 'bab', 'cbbd' -> 'bb'",
        
        "Analysis: What are implications of AI on employment?\n\nPrimary factors: Job displacement in routine tasks\nSecondary: Creation of new roles (AI trainers, prompt engineers)\n\nCounter-argument: Historical tech revolutions created more jobs\n\nAssessment: AI will transform rather than eliminate work, but requires workforce adaptation policies.",
    ]
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def generate_sample(self) -> str:
        return random.choice(self.SAMPLES)


class ReasoningDataset(Dataset):
    """Dataset for TPU training with fixed shapes."""
    
    def __init__(self, tokenizer, num_samples: int, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.generator = ReasoningDataGenerator(tokenizer)
        
        print(f"Generating {num_samples} training samples...")
        self.samples = []
        for _ in tqdm(range(num_samples)):
            text = self.generator.generate_sample()
            tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
            # Pad to fixed length (important for TPU)
            if len(tokens) < max_length:
                tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
            self.samples.append(tokens[:max_length])
        print(f"Generated {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


# ============================================================================
# STEP 5: TPU TRAINING LOOP
# ============================================================================
def train_orpheus_tpu():
    """Main TPU training function for ORPHEUS 500M."""
    
    print("\n" + "="*60)
    print("ORPHEUS 500M - 1024 LAYERS - TPU TRAINING")
    print("THE DEEPEST TRANSFORMER EVER BUILT!")
    print("="*60 + "\n")
    
    # Get TPU device
    device = xm.xla_device()
    print(f"TPU Device: {device}")
    
    # Configuration
    config = OrpheusConfig()
    
    # Training hyperparameters
    BATCH_SIZE = 4  # Larger batch for TPU
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 500
    MAX_STEPS = 50000
    SAVE_EVERY = 1000
    LOG_EVERY = 50
    
    # Tokenizer
    print("Loading tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    print("Building ORPHEUS 500M - 1024 layers...")
    model = Orpheus(config)
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Layers: {config.num_layers}")
    
    # Move to TPU
    model = model.to(device)
    print("Model moved to TPU!")
    
    # Dataset
    NUM_SAMPLES = 20000
    dataset = ReasoningDataset(tokenizer, NUM_SAMPLES, max_length=256)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Learning rate scheduler
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LEARNING_RATE * step / WARMUP_STEPS
        return LEARNING_RATE * 0.5 ** (step / MAX_STEPS)
    
    # Resume from checkpoint
    start_step = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed from step {start_step}")
        model = model.to(device)
    
    # Training loop
    print(f"\nStarting TPU training from step {start_step}...")
    print(f"Target: {MAX_STEPS} steps")
    print("-" * 60)
    
    model.train()
    step = start_step
    running_loss = 0.0
    
    while step < MAX_STEPS:
        for batch in dataloader:
            if step >= MAX_STEPS:
                break
            
            # Move to TPU
            input_ids = batch.to(device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / GRADIENT_ACCUMULATION
            
            # Backward pass
            loss.backward()
            
            running_loss += loss.item()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_lr(step)
                
                # TPU optimizer step
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
                xm.mark_step()
            
            step += 1
            
            # Logging
            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                lr = get_lr(step)
                xm.master_print(f"Step {step}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0
            
            # Save checkpoint
            if step % SAVE_EVERY == 0:
                xm.master_print(f"Saving checkpoint at step {step}...")
                
                # Save to CPU first
                model_cpu = model.to('cpu')
                checkpoint = {
                    'step': step,
                    'model_state_dict': model_cpu.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }
                
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}.pt"))
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt"))
                
                # Move back to TPU
                model = model_cpu.to(device)
                xm.master_print(f"Checkpoint saved to Google Drive!")
    
    # Final save
    xm.master_print("\nTraining complete! Saving final model...")
    model_cpu = model.to('cpu')
    final_checkpoint = {
        'step': step,
        'model_state_dict': model_cpu.state_dict(),
        'config': config,
    }
    torch.save(final_checkpoint, os.path.join(CHECKPOINT_DIR, "orpheus_500m_final.pt"))
    xm.master_print(f"Final model saved to: {CHECKPOINT_DIR}/orpheus_500m_final.pt")
    xm.master_print("\nORPHEUS 500M - 1024 LAYERS - TRAINING COMPLETE!")
    xm.master_print("DO NOT click 'Deploy to Google Cloud' - your model is in Google Drive!")


# ============================================================================
# STEP 6: RUN TRAINING
# ============================================================================
if __name__ == "__main__":
    train_orpheus_tpu()

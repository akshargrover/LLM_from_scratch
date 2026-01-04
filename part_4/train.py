from __future__ import annotations
import argparse, time, signal
from pathlib import Path
import sys

import torch
import torch.nn as nn

# so we can import Part 3 model
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1] / 'part_3'))
from model_modern import GPTModern

from tokenizer_bpe import BPETokenizer
from dataset_bpe import make_loader
from lr_scheduler import WarmupCosineLR
from amp_accum import AmpGrad
from checkpointing import (
    load_checkpoint,
    _log_hparams_tb,
    _maybe_log_graph_tb,
    _is_tb,
    _log_model_stats,
    _maybe_log_attention,
    _log_samples_tb,
    _log_runtime,
    atomic_save_all,
)
from logger import init_logger


def run_cfg_from_args(args, vocab_size: int) -> dict:
    return dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_rmsnorm=True,
        use_swiglu=True,
        rope=True,
        max_pos=4096,
        sliding_window=None,
        attention_sink=0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='runs/part4')

    # tokenizer / model dims
    p.add_argument('--bpe', action='store_true', help='train and use a BPE tokenizer (recommended)')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)

    # train
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps', type=int, default=300, help='max optimizer steps for this run')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=20)
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--grad_accum_steps', type=int, default=4)

    # misc
    p.add_argument('--log', choices=['wandb', 'tensorboard', 'none'], default='tensorboard')
    p.add_argument('--save_every', type=int, default=50, help='save checkpoint every N optimizer steps')
    p.add_argument('--keep_last_k', type=int, default=2, help='keep last K step checkpoints (plus model_last.pt)')
    args = p.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # output dir and (possible) checkpoint
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model_last.pt"
    have_ckpt = ckpt_path.exists()

    # ---- load checkpoint meta if present ----
    ckpt = None
    saved_tok_dir = None
    if have_ckpt:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "config" not in ckpt:
            raise RuntimeError(
                "Checkpoint is missing 'config'."
                "Please re-save a checkpoint that includes the model config."
            )
        tok_file = ckpt_path.with_name("tokenizer_dir.txt")
        saved_tok_dir = tok_file.read_text().strip() if tok_file.exists() else None

    # ---- tokenizer ----
    tok = None
    tok_dir = None
    if have_ckpt:
        if not saved_tok_dir:
            raise RuntimeError(
                "Checkpoint was found but tokenizer_dir.txt is missing. "
                "Resume requires the original tokenizer."
            )
        tok = BPETokenizer(); tok.load(saved_tok_dir)
        tok_dir = saved_tok_dir
        vocab_size = tok.vocab_size
        print(f"[resume] Loaded tokenizer from {tok_dir} (vocab={vocab_size})")
    else:
        if args.bpe:
            tok = BPETokenizer(vocab_size=args.vocab_size)
            tok.train(args.data)
            tok_dir = str(out_dir / 'tokenizer'); Path(tok_dir).mkdir(parents=True, exist_ok=True)
            tok.save(tok_dir)
            vocab_size = tok.vocab_size
            print(f"[init] Trained tokenizer to {tok_dir} (vocab={vocab_size})")
        else:
            tok = None
            vocab_size = 256  # byte-level fallback (not recommended for Part 4)

#!/usr/bin/env python3
"""
TinyGPT v2 (NumPy edition)
A small, from-scratch character-level language model for your own input.txt.

Important: this is a GPT-inspired fixed-context neural language model (an
embedding + MLP), NOT a Transformer with self-attention. It is intentionally
small, dependency-light, and uses NumPy for fast matrix operations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

VERSION = 2


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class TinyGPT:
    def __init__(self, vocab_size: int, context: int = 16, emb: int = 32,
                 hidden: int = 64, seed: int = 42):
        self.vocab_size = int(vocab_size)
        self.context = int(context)
        self.emb = int(emb)
        self.hidden = int(hidden)
        rng = np.random.default_rng(seed)

        def weight(out_dim, in_dim):
            return (rng.standard_normal((out_dim, in_dim)) *
                    math.sqrt(2.0 / max(1, in_dim))).astype(np.float32)

        # Parameters: fixed context embeddings -> MLP -> next-character logits.
        self.p = {
            "E": (rng.standard_normal((vocab_size, emb)) * 0.05).astype(np.float32),
            "W1": weight(hidden, context * emb),
            "b1": np.zeros((hidden,), dtype=np.float32),
            "W2": weight(vocab_size, hidden),
            "b2": np.zeros((vocab_size,), dtype=np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.opt_step = 0

    def forward(self, x: np.ndarray, cache=False):
        # x: [batch, context]
        e = self.p["E"][x]                    # [B, C, D]
        flat = e.reshape(x.shape[0], -1)      # [B, C*D]
        z1 = flat @ self.p["W1"].T + self.p["b1"]
        h = np.tanh(z1)
        logits = h @ self.p["W2"].T + self.p["b2"]
        if cache:
            return logits, (x, e, flat, z1, h)
        return logits

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        logits, cache = self.forward(x, cache=True)
        x, e, flat, z1, h = cache
        probs = softmax(logits)
        n = x.shape[0]
        loss = -np.log(np.maximum(probs[np.arange(n), y], 1e-12)).mean()

        # Cross-entropy + softmax derivative.
        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads = {}
        grads["W2"] = dlogits.T @ h
        grads["b2"] = dlogits.sum(axis=0)
        dh = dlogits @ self.p["W2"]
        dz1 = dh * (1.0 - np.tanh(z1) ** 2)
        grads["W1"] = dz1.T @ flat
        grads["b1"] = dz1.sum(axis=0)
        dflat = dz1 @ self.p["W1"]
        de = dflat.reshape(n, self.context, self.emb)

        grads["E"] = np.zeros_like(self.p["E"])
        # Repeated character IDs must accumulate their gradients.
        np.add.at(grads["E"], x.reshape(-1), de.reshape(-1, self.emb))
        return float(loss), grads

    def update(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8,
               grad_clip=1.0):
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if norm > grad_clip:
            scale = grad_clip / (norm + 1e-12)
            grads = {k: g * scale for k, g in grads.items()}

        self.opt_step += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g * g)
            m_hat = self.m[k] / (1.0 - beta1 ** self.opt_step)
            v_hat = self.v[k] / (1.0 - beta2 ** self.opt_step)
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate(self, stoi, itos, seed_text="", length=200, temperature=0.8,
                 top_k=0, seed=123):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        rng = np.random.default_rng(seed)
        ids = [stoi[ch] for ch in seed_text if ch in stoi]
        if not ids:
            ids = [0]
        output = list(seed_text)
        for _ in range(length):
            ctx = ([0] * max(0, self.context - len(ids)) + ids[-self.context:])
            x = np.asarray([ctx], dtype=np.int64)
            logits = self.forward(x)[0] / temperature
            if top_k > 0 and top_k < self.vocab_size:
                keep = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.full_like(logits, -1e9)
                mask[keep] = logits[keep]
                logits = mask
            probs = softmax(logits)
            next_id = int(rng.choice(self.vocab_size, p=probs))
            ids.append(next_id)
            output.append(itos[next_id])
        return "".join(output)

    def save(self, path, chars, config):
        payload = {
            "version": np.array([VERSION], dtype=np.int32),
            "chars": np.asarray(chars),
            "config": np.asarray(json.dumps(config)),
            "opt_step": np.array([self.opt_step], dtype=np.int64),
        }
        for group, values in (("p", self.p), ("m", self.m), ("v", self.v)):
            for name, arr in values.items():
                payload[f"{group}_{name}"] = arr
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            chars = data["chars"].tolist()
            config = json.loads(str(data["config"].item()))
            model = cls(len(chars), **config)
            for group, values in (("p", model.p), ("m", model.m), ("v", model.v)):
                for name in values:
                    values[name][...] = data[f"{group}_{name}"]
            model.opt_step = int(data["opt_step"][0])
        return model, chars, config


def make_examples(ids: np.ndarray, context: int):
    # Predict each character from the preceding fixed-width context.
    if len(ids) <= context:
        raise ValueError(
            f"Dataset has {len(ids)} characters; it must contain more than "
            f"context={context}. Add more text or reduce --context."
        )
    x, y = [], []
    for i in range(context, len(ids)):
        x.append(ids[i-context:i])
        y.append(ids[i])
    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)


def batches(x, y, batch_size, rng):
    order = rng.permutation(len(x))
    for start in range(0, len(order), batch_size):
        ix = order[start:start + batch_size]
        yield x[ix], y[ix]


def evaluate(model, x, y, batch_size=256):
    total_loss, total = 0.0, 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = model.forward(xb)
        probs = softmax(logits)
        loss = -np.log(np.maximum(probs[np.arange(len(yb)), yb], 1e-12)).sum()
        total_loss += float(loss)
        total += len(yb)
    return total_loss / max(total, 1)


def train(args):
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(
            f"Dataset not found: {data_path}\n"
            "Create input.txt in this folder, or pass --data /path/to/input.txt."
        )
    text = data_path.read_text(encoding="utf-8")
    if not text:
        raise SystemExit(f"{data_path} is empty.")
    # Character-level vocabulary: works with Unicode code points, including Bengali.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.asarray([stoi[ch] for ch in text], dtype=np.int64)
    if len(chars) < 2:
        raise SystemExit("Dataset needs at least two distinct characters.")

    rng = np.random.default_rng(args.seed)
    split = int(len(ids) * (1.0 - args.val_fraction))
    split = max(args.context + 1, min(split, len(ids) - 1))
    train_ids, val_ids = ids[:split], ids[split:]
    # Validation examples can use characters immediately before the split as context.
    val_stream = ids[max(0, split - args.context):]
    x_train, y_train = make_examples(train_ids, args.context)
    if len(val_stream) > args.context:
        x_val, y_val = make_examples(val_stream, args.context)
    else:
        x_val, y_val = x_train[-min(100, len(x_train)):], y_train[-min(100, len(y_train)):]

    config = {
        "context": args.context,
        "emb": args.emb,
        "hidden": args.hidden,
        "seed": args.seed,
    }
    model = TinyGPT(len(chars), **config)
    print(f"Dataset       : {data_path} ({len(text):,} characters)")
    print(f"Vocabulary    : {len(chars)} unique characters")
    print(f"Train examples: {len(x_train):,}")
    print(f"Val examples  : {len(x_val):,}")
    print(f"Parameters    : {sum(a.size for a in model.p.values()):,}")
    print(f"Config        : context={args.context}, emb={args.emb}, hidden={args.hidden}")
    print("Training starts (Ctrl+C to stop; last checkpoint is saved on normal completion).")

    best_val = float("inf")
    start_time = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            losses = []
            for xb, yb in batches(x_train, y_train, args.batch_size, rng):
                lr = args.lr * max(0.1, 1.0 - (epoch - 1) / max(args.epochs, 1))
                loss, grads = model.loss_and_grads(xb, yb)
                model.update(grads, lr, grad_clip=args.grad_clip)
                losses.append(loss)
            train_loss = float(np.mean(losses))
            val_loss = evaluate(model, x_val, y_val)
            elapsed = time.time() - start_time
            print(f"epoch {epoch:4d}/{args.epochs} | train loss {train_loss:.4f} "
                  f"| val loss {val_loss:.4f} | {elapsed:.1f}s")
            if val_loss < best_val:
                best_val = val_loss
                model.save(args.out, chars, config)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving current model...")
        model.save(args.out, chars, config)

    print(f"\nModel saved to: {args.out}")
    print("\nSample:")
    print(model.generate(stoi, chars, seed_text=args.prompt, length=args.gen_length,
                         temperature=args.temperature, top_k=args.top_k, seed=args.seed))


def generate(args):
    model, chars, config = TinyGPT.load(args.model)
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(model.generate(stoi, chars, seed_text=args.prompt, length=args.length,
                         temperature=args.temperature, top_k=args.top_k, seed=args.seed))


def build_parser():
    parser = argparse.ArgumentParser(description="TinyGPT v2 - small NumPy character language model")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train from your own text file")
    t.add_argument("--data", default="input.txt")
    t.add_argument("--out", default="tinygpt_v2.npz")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--context", type=int, default=16)
    t.add_argument("--emb", type=int, default=32)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--lr", type=float, default=0.003)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--prompt", default="")
    t.add_argument("--gen-length", type=int, default=200)
    t.add_argument("--temperature", type=float, default=0.8)
    t.add_argument("--top-k", type=int, default=20)
    t.set_defaults(func=train)

    g = sub.add_parser("generate", help="generate text from a saved model")
    g.add_argument("--model", default="tinygpt_v2.npz")
    g.add_argument("--prompt", default="")
    g.add_argument("--length", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--seed", type=int, default=123)
    g.set_defaults(func=generate)
    return parser


def main():
    args = build_parser().parse_args()
    if getattr(args, "epochs", 1) < 1:
        raise SystemExit("--epochs must be >= 1")
    if getattr(args, "batch_size", 1) < 1:
        raise SystemExit("--batch-size must be >= 1")
    if getattr(args, "context", 1) < 1:
        raise SystemExit("--context must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()

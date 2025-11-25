import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model

# Focal loss (token-level)
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=3.0, weight=None, ignore_index=-100):
    """
    logits: (B, T, C)
    targets: (B, T) long
    Returns: scalar loss
    """
    B, T, C = logits.shape
    logits = logits.view(-1, C)            # (B*T, C)
    targets = targets.view(-1)             # (B*T,)
    mask = targets != ignore_index         # (B*T,)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_masked = logits[mask]
    targets_masked = targets[mask]

    logpt = -F.cross_entropy(logits_masked, targets_masked, reduction='none', weight=weight)
    pt = torch.exp(logpt)
    loss = ((1 - pt) ** gamma * -logpt).mean()
    return loss

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)  # shape (B, T)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, C)

            loss = focal_loss(logits, labels, gamma=args.focal_gamma, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")

if __name__ == "__main__":
    main()

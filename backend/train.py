"""Optional PyTorch training scaffold. Never loaded by the prototype website."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageOps

try:
    from .dataset import TARGETS, validate
except ImportError:
    from dataset import TARGETS, validate


class Cards(Dataset):
    def __init__(self, root, csv_path):
        self.root = Path(root)
        if Path(csv_path).suffix != ".jsonl":
            raise ValueError("Use the audited JSONL manifest; legacy CSV lacks provenance and missing-label handling")
        self.rows = validate([json.loads(line) for line in Path(csv_path).read_text().splitlines() if line.strip()], self.root)
        self.transform = transforms.Compose([transforms.Resize((352, 256)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            x = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        labels = row["labels"]
        values = [(float(labels[k])-1)/9 if labels.get(k) is not None else 0 for k in TARGETS]
        mask = [labels.get(k) is not None for k in TARGETS]
        return x, torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def main():
    p = argparse.ArgumentParser()
    for key in ("root", "train", "val", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--weights")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(max(1, args.threads))
    train, val = Cards(args.root, args.train), Cards(args.root, args.val)
    if {r["card_id"] for r in train.rows} & {r["card_id"] for r in val.rows}:
        raise ValueError("Physical-card overlap between training and validation")
    validate(train.rows + val.rows, args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, len(TARGETS)), nn.Sigmoid())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for x, y, mask in DataLoader(train, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            errors = nn.functional.smooth_l1_loss(model(x.to(device)), y.to(device), reduction="none")
            loss = errors[mask.to(device)].mean()
            loss.backward()
            optimizer.step()
        model.eval()
        error = torch.zeros(len(TARGETS), device=device)
        counts = torch.zeros_like(error)
        with torch.no_grad():
            for x, y, mask in DataLoader(val, batch_size=args.batch_size):
                mask = mask.to(device)
                error += ((model(x.to(device))-y.to(device)).abs()*mask).sum(0)*9
                counts += mask.sum(0)
        mae = (error/counts.clamp_min(1)).cpu()
        measured = counts.cpu() > 0
        metric = mae[measured].mean().item()
        metrics = {k: (mae[i].item() if measured[i] else None) for i,k in enumerate(TARGETS)}
        print(f"Epoch {epoch+1}: criterion MAE {metrics}", flush=True)
        if metric < best:
            best = metric
            torch.save({"state_dict": model.cpu().state_dict(), "targets": TARGETS, "input_size": [352,256], "validation_mae": {k: (mae[i].item() if measured[i] else None) for i,k in enumerate(TARGETS)}, "architecture":"resnet18-sigmoid5", "production_ready": False}, args.output)
            model.to(device)


if __name__ == "__main__":
    main()

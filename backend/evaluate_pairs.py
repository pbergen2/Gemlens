"""Evaluate a frozen paired checkpoint without training or model selection.

Use a reserved test manifest only after choosing the checkpoint on validation.
Repeated development against these results would turn this into a validation set.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from .dataset import validate
from .train_pairs import detail_crops, head, pairs


def metrics(records, baseline):
    if not records:
        raise ValueError('Cannot evaluate an empty set')
    errors = [abs(r['prediction'] - r['target']) for r in records]
    return {
        'cards': len(records),
        'card_mae': statistics.mean(errors),
        'baseline_card_mae': statistics.mean(abs(r['target'] - baseline) for r in records),
        'within_one_point': sum(e <= 1 for e in errors) / len(errors),
        'overgrade_by_more_than_one': sum(r['prediction'] - r['target'] > 1 for r in records) / len(records),
        'undergrade_by_more_than_one': sum(r['target'] - r['prediction'] > 1 for r in records) / len(records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'train', 'val', 'test', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--allow-automated-screening', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    rows = {s: [json.loads(line) for line in Path(getattr(args, s)).read_text().splitlines() if line.strip()]
            for s in ('train', 'val', 'test')}
    grouped = {s: pairs(items) for s, items in rows.items()}
    seen = set()
    for split, cards in grouped.items():
        ids = {c[0] for c in cards}
        if not ids or seen & ids:
            raise ValueError('Empty split or physical-card split leakage: ' + split)
        seen |= ids
    print('Checking all splits for invalid images and exact duplicate leakage...', flush=True)
    validate([r for items in rows.values() for r in items], root,
             allow_automated_screening=args.allow_automated_screening)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['view_order'] != ['front', 'back'] or checkpoint['target'] != 'overall_grade':
        raise ValueError('Unsupported checkpoint input/target')
    detail = checkpoint['detail_patches']
    size = 3072 if detail else 1024
    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval().requires_grad_(False)
    model = head(size)
    model.load_state_dict(checkpoint['head'])
    model.eval().requires_grad_(False)
    def transform(dimensions):
        return transforms.Compose([transforms.Resize(tuple(dimensions)), transforms.ToTensor(),
                                   transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    whole = transform(checkpoint['input_size'])
    patch = transform(checkpoint['detail_input_size'])
    cards = grouped['test']
    records = []
    with torch.inference_mode():
        for start in range(0, len(cards), 8):
            batch = cards[start:start + 8]
            images, patches = [], []
            for _, front, back, _ in batch:
                for path in (front, back):
                    with Image.open(root / path) as im:
                        oriented = ImageOps.exif_transpose(im).convert('RGB')
                        images.append(whole(oriented))
                        if detail:
                            patches.extend(patch(crop) for crop in detail_crops(oriented))
            encoded = encoder(torch.stack(images))
            if detail:
                features = torch.cat([encoder(torch.stack(patches[i:i + 16]))
                                      for i in range(0, len(patches), 16)]).reshape(len(images), 8, 512)
                encoded = torch.cat([encoded, features.mean(1), features.max(1).values], dim=1)
            normalized = (encoded.reshape(-1, size) - checkpoint['feature_mean']) / checkpoint['feature_std']
            predictions = (model(normalized) * 9 + 1).flatten().tolist()
            records.extend({'card_id': c[0], 'target': c[3], 'prediction': p} for c, p in zip(batch, predictions))
            if start % 64 == 0:
                print(f'Test: {min(start + 8, len(cards))}/{len(cards)} cards', flush=True)
    baseline = statistics.median(c[3] for c in grouped['train'])
    result = {
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'manifest_sha256': {s: hashlib.sha256(Path(getattr(args, s)).read_bytes()).hexdigest() for s in rows},
        'production_ready': False,
        'evaluation': 'Frozen checkpoint; no training, tuning, or test-set model selection',
        'constant_baseline_grade': baseline,
        'metrics': metrics(records, baseline),
        'per_grade': {str(g): metrics([r for r in records if r['target'] == g], baseline)
                      for g in sorted({r['target'] for r in records})},
        'limitations': ['Same-source scans, not a phone-photo or unseen-design benchmark',
                        'Automated screening may miss report overlays and incorrect pairs',
                        'Source-reported grades are not independently verified',
                        'No defect localization or supervised subgrades'],
        'predictions': records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()

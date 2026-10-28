"""Paired front/back grade baseline using frozen pretrained ResNet18 features.

This experiment trains only an overall-grade head, not defect localization.
The image encoder is frozen to keep initial experiments feasible on a CPU.
"""
from collections import Counter
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from .dataset import validate


def pairs(rows):
    grouped = {}
    for row in rows:
        card = grouped.setdefault(row['card_id'], {})
        if row['side'] in card:
            raise ValueError('Duplicate side for physical card')
        card[row['side']] = row
    result = []
    for key, card in sorted(grouped.items()):
        if set(card) != {'front', 'back'}:
            raise ValueError('Every card needs front and back')
        grade = card['front']['labels'].get('grade')
        if grade is None or card['back']['labels'].get('grade') != grade:
            raise ValueError('Pair must share an overall grade')
        result.append((key, card['front']['image'], card['back']['image'], grade))
    return result


def detail_crops(image):
    """Original-resolution squares at four corners and four edge midpoints.

    These are grading inputs, not predicted or confirmed defect locations.
    """
    w, h = image.size
    side = max(1, min(w, h)//4)
    positions = [(0, 0), (w-side, 0), (0, h-side), (w-side, h-side),
                 ((w-side)//2, 0), ((w-side)//2, h-side),
                 (0, (h-side)//2), (w-side, (h-side)//2)]
    return [image.crop((x, y, x+side, y+side)) for x, y in positions]


def head(input_features=1024):
    return nn.Sequential(nn.Linear(input_features, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 1), nn.Sigmoid())


def grade_weights(labels):
    """Inverse square-root frequency weights, computed from training labels only."""
    counts = Counter(labels.flatten().tolist())
    weights = torch.tensor([counts[v] ** -.5 for v in labels.flatten().tolist()], dtype=labels.dtype).reshape_as(labels)
    return weights / weights.mean()


def low_grade_weights(labels):
    """Balance training grades and give grades 1–6 three times the loss weight."""
    weights = grade_weights(labels) * torch.where(labels <= 6, 3., 1.)
    return weights / weights.mean()


def low_grade_error(predictions, labels):
    """Macro MAE over present grades 1–6 so grade 6 cannot dominate selection."""
    grades = torch.unique(labels[labels <= 6])
    if not len(grades):
        raise ValueError('Low-grade selection requires validation grades 1–6')
    return torch.stack([(predictions[labels == g] - labels[labels == g]).abs().mean()
                        for g in grades]).mean().item()


def range_target_error(predictions, labels, targets):
    """Worst MAE/target across five ranges; <=1 meets every MAE target.

    Targets must be fixed before training, never derived from test predictions.
    This prevents a strong common range from hiding a weak range in selection.
    """
    ratios = []
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        target = targets[f'{lo}-{hi}']
        if not isinstance(target, (float, int)) or not 0 < target < float('inf'):
            raise ValueError('Range targets must be finite and positive')
        mask = (labels >= lo) & (labels <= hi)
        if not mask.any():
            raise ValueError(f'Missing validation range {lo}-{hi}')
        ratios.append((predictions[mask]-labels[mask]).abs().mean().item()/target)
    return max(ratios)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'train', 'val', 'test', 'weights', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--allow-automated-screening', action='store_true', help='Experimental only: accept explicitly recorded automated screening without claiming manual review')
    p.add_argument('--detail-patches', action='store_true')
    p.add_argument('--balance-grades', action='store_true', help='Weight rare training grades by inverse square-root frequency')
    p.add_argument('--focus-low-grades', action='store_true', help='Balance grades, triple 1–6 loss weight, select checkpoint by validation macro MAE across grades 1–6')
    p.add_argument('--range-targets', help='JSON MAE targets for five ranges; select by worst relative target error')
    p.add_argument('--validation-only', action='store_true', help='Do not load or evaluate the test set')
    args = p.parse_args()
    if args.epochs < 1 or args.threads < 1:
        p.error('epochs and threads must be positive')
    if args.range_targets and args.focus_low_grades:
        p.error('Choose range targets or low-grade-only selection, not both')
    targets = json.loads(Path(args.range_targets).read_text()) if args.range_targets else None
    torch.manual_seed(42); random.seed(42); torch.set_num_threads(args.threads)
    root = Path(args.root)
    splits = ('train', 'val') if args.validation_only else ('train', 'val', 'test')
    rows = {k: [json.loads(s) for s in Path(getattr(args, k)).read_text().splitlines() if s.strip()] for k in splits}
    combined = [r for group in rows.values() for r in group]
    validate(combined, root, allow_automated_screening=args.allow_automated_screening)
    split_pairs = {k: pairs(v) for k, v in rows.items()}
    seen = set()
    for key, values in split_pairs.items():
        if not values: raise ValueError('Empty split: ' + key)
        ids = {v[0] for v in values}
        if seen & ids: raise ValueError('Physical-card split leakage')
        seen |= ids
    encoder = models.resnet18(weights=None)
    encoder.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    encoder.fc = nn.Identity(); encoder.eval(); encoder.requires_grad_(False)
    transform = transforms.Compose([transforms.Resize((352, 256)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    patch_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    feature_size = 3072 if args.detail_patches else 1024
    features = {}
    for split, cards in split_pairs.items():
        all_features = []
        with torch.inference_mode():
            for start in range(0, len(cards), 8):
                images, patches = [], []
                for _, front, back, _ in cards[start:start+8]:
                    for path in (front, back):
                        with Image.open(root/path) as im:
                            oriented = ImageOps.exif_transpose(im).convert('RGB')
                            images.append(transform(oriented))
                            if args.detail_patches:
                                patches.extend(patch_transform(crop) for crop in detail_crops(oriented))
                encoded = encoder(torch.stack(images))
                if args.detail_patches:
                    detail = torch.cat([encoder(torch.stack(patches[i:i+16])) for i in range(0,len(patches),16)]).reshape(len(images),8,512)
                    encoded = torch.cat([encoded,detail.mean(1),detail.max(1).values],dim=1)
                all_features.append(encoded.reshape(-1,feature_size))
                if start % 64 == 0: print(f'{split}: features {min(start+8,len(cards))}/{len(cards)} cards',flush=True)
        # Clone outside inference mode so tensors can be used in head backprop.
        features[split] = torch.cat(all_features).clone()
    mean = features['train'].mean(0); std = features['train'].std(0,unbiased=False).clamp_min(.01)
    features = {k:(v-mean)/std for k,v in features.items()}
    labels = {k:torch.tensor([v[3] for v in cards],dtype=torch.float32).reshape(-1,1) for k,cards in split_pairs.items()}
    weights = grade_weights(labels['train']) if args.balance_grades else torch.ones_like(labels['train'])
    if args.focus_low_grades:
        weights = low_grade_weights(labels['train'])
        low_grade_error(labels['val'], labels['val'])
    if targets is not None:
        range_target_error(labels['val'], labels['val'], targets)
    model=head(feature_size); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=float('inf'); history=[]; best_state=None
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(features['train'])).split(32):
            optimizer.zero_grad()
            prediction=model(features['train'][indices])
            loss=(nn.functional.smooth_l1_loss(prediction,(labels['train'][indices]-1)/9,reduction='none')*weights[indices]).mean()
            loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted = model(features['val'])*9+1
            overall_error=(predicted-labels['val']).abs().mean().item()
            error=low_grade_error(predicted, labels['val']) if args.focus_low_grades else overall_error
            if targets is not None:
                error=range_target_error(predicted, labels['val'], targets)
        history.append({'epoch':epoch+1,'validation_card_mae':overall_error,'selection_error':error})
        if error<best:
            best=error;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
        if epoch==0 or (epoch+1)%10==0:print(f'Epoch {epoch+1}: validation MAE {overall_error:.3f}; selection error {error:.3f}',flush=True)
    model.load_state_dict(best_state);model.eval()
    baseline=statistics.median(labels['train'].flatten().tolist())
    result={'architecture':'resnet18-frozen-paired-detail-mlp' if args.detail_patches else 'resnet18-frozen-paired-mlp','production_ready':False,'seed':42,
            'detail_patches':args.detail_patches,'validation_only':args.validation_only,
            'balance_grades':args.balance_grades,
            'focus_low_grades':args.focus_low_grades,
            'selection_metric':'worst_range_mae_target_ratio' if targets is not None else ('macro_mae_grades_1_to_6' if args.focus_low_grades else 'overall_mae'),
            'range_targets':targets,
            'all_range_mae_targets_met':best <= 1 if targets is not None else None,
            'automated_screening_allowed':args.allow_automated_screening,
            'weights_sha256':hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
            'split_cards':{k:len(v) for k,v in split_pairs.items()},'best_epoch':best_epoch,
            'history':history,'constant_baseline_grade':baseline,'metrics':{},
            'limitations':['Sampling scope and exclusions must be read from the accompanying dataset audit',
                           'Automated screening is not equivalent to human review; label leakage may remain' if args.allow_automated_screening else 'Manually reviewed inputs; review may still miss confounds',
                           'Frozen image features and downsampled scans; no phone-image evaluation',
                           'No subgrade supervision or defect localization',
                           'Source grade provenance is dataset-reported, not independently verified']}
    with torch.inference_mode():
        for split in (('val',) if args.validation_only else ('val','test')):
            predictions=(model(features[split])*9+1).flatten().tolist()
            records=[{'card_id':card[0],'target':card[3],'prediction':value} for card,value in zip(split_pairs[split],predictions)]
            errors=[abs(r['prediction']-r['target']) for r in records]
            groups={}
            for record in records:
                grade=str(record['target']);group=groups.setdefault(grade,[]);group.append(record)
            result['metrics'][split]={'card_mae':statistics.mean(errors),
                'baseline_card_mae':statistics.mean(abs(r['target']-baseline) for r in records),
                'within_one_point':sum(e<=1 for e in errors)/len(errors),
                'overgrade_by_more_than_one':sum(r['prediction']-r['target']>1 for r in records)/len(records),
                'per_grade':{k:{'cards':len(v),'mae':statistics.mean(abs(r['prediction']-r['target']) for r in v)} for k,v in groups.items()},
                'predictions':records}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'architecture':result['architecture'],'encoder':encoder.state_dict(),'head':best_state,
                'feature_mean':mean,'feature_std':std,'input_size':[352,256],
                'detail_patches':args.detail_patches,'detail_input_size':[224,224],
                'detail_pooling':'mean and max of 8 corner/edge crops per side',
                'view_order':['front','back'],'target':'overall_grade','production_ready':False},out)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{m:v for m,v in r.items() if m not in ('predictions','per_grade')} for k,r in result['metrics'].items()},indent=2),flush=True)

if __name__=='__main__':main()

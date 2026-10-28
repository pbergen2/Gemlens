"""Local experimental paired-grade inference. No uploads are persisted."""
import threading
from pathlib import Path

CHECKPOINT = Path(__file__).resolve().parents[1] / 'outputs/gemlens-balanced-ranges.pt'
_lock = threading.Lock()
_model = None


def predict(front, back):
    # Serialize CPU inference and lazy loading so concurrent uploads cannot
    # construct multiple encoders or exhaust memory with detail batches.
    global _model
    with _lock:
        import torch
        from torch import nn
        from torchvision import models, transforms
        from .train_pairs import head, detail_crops
        if _model is None:
            torch.set_num_threads(4)
            checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
            if checkpoint['view_order'] != ['front', 'back'] or not checkpoint['detail_patches']:
                raise RuntimeError('Unsupported grading checkpoint')
            encoder = models.resnet18(weights=None)
            encoder.fc = nn.Identity()
            encoder.load_state_dict(checkpoint['encoder'])
            encoder.eval().requires_grad_(False)
            regressor = head(3072)
            regressor.load_state_dict(checkpoint['head'])
            regressor.eval().requires_grad_(False)
            def transform(size):
                return transforms.Compose([transforms.Resize(tuple(size)), transforms.ToTensor(),
                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            _model = (encoder, regressor, checkpoint, transform(checkpoint['input_size']), transform(checkpoint['detail_input_size']))
        encoder, regressor, checkpoint, whole, detail = _model
        with torch.inference_mode():
            images = [front.convert('RGB'), back.convert('RGB')]
            global_features = encoder(torch.stack([whole(im) for im in images]))
            crops = [detail(crop) for im in images for crop in detail_crops(im)]
            patches = encoder(torch.stack(crops)).reshape(2,8,512)
            features = torch.cat([global_features, patches.mean(1), patches.max(1).values], dim=1).reshape(1,3072)
            normalized = (features-checkpoint['feature_mean'])/checkpoint['feature_std']
            grade = (regressor(normalized)*9+1).item()
        return {'predicted_grade': round(grade, 2), 'grade_range': None,
                'method': 'gemlens-balanced-ranges', 'model': CHECKPOINT.name,
                'confidence': 'Experimental · photo accuracy not independently established',
                'coverage': 'front and back', 'production_ready': False,
                'disclaimer': 'Learned overall grade estimate. Inspection markers are separate heuristic suggestions, not model explanations.'}

import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'),'optional training dependencies')
class PairTests(unittest.TestCase):
    def test_range_selection_cannot_hide_a_regression(self):
        import torch
        from backend.train_pairs import range_target_error
        labels = torch.tensor([[1.], [3.], [5.], [7.], [9.]])
        targets = {k: .5 for k in ('1-2', '3-4', '5-6', '7-8', '9-10')}
        self.assertEqual(range_target_error(labels+.5, labels, targets), 1.)
        predicted = labels.clone(); predicted[-1] += 1
        self.assertEqual(range_target_error(predicted, labels, targets), 2.)
        with self.assertRaises(ValueError):
            range_target_error(labels[:-1], labels[:-1], targets)
        with self.assertRaises(ValueError):
            range_target_error(labels, labels, dict(targets, **{'1-2': float('nan')}))

    def test_low_grade_focus_weights_and_selection(self):
        import torch
        from backend.train_pairs import low_grade_weights, low_grade_error
        weights = low_grade_weights(torch.tensor([[1.], [6.], [9.]]))
        self.assertAlmostEqual(weights.mean().item(), 1., places=6)
        self.assertAlmostEqual((weights[0]/weights[2]).item(), 3., places=6)
        labels = torch.tensor([[1.], [6.], [6.], [10.]])
        predictions = torch.tensor([[4.], [7.], [7.], [1.]])
        self.assertEqual(low_grade_error(predictions, labels), 2.)
        with self.assertRaises(ValueError):
            low_grade_error(torch.tensor([9.]), torch.tensor([10.]))

    def test_evaluation_counts_thresholds_and_error_direction(self):
        from backend.evaluate_pairs import metrics
        rows = [{'target': 5, 'prediction': value} for value in (5, 6, 7, 3)]
        result = metrics(rows, baseline=8)
        self.assertEqual(result['card_mae'], 1.25)
        self.assertEqual(result['baseline_card_mae'], 3)
        self.assertEqual(result['within_one_point'], .5)
        self.assertEqual(result['overgrade_by_more_than_one'], .25)
        self.assertEqual(result['undergrade_by_more_than_one'], .25)
        with self.assertRaises(ValueError):
            metrics([], baseline=8)

    def test_detail_crops_preserve_edge_pixels_and_native_size(self):
        from backend.train_pairs import detail_crops
        from PIL import Image
        image = Image.new('RGB',(400,600),'black')
        image.putpixel((0,0),(255,0,0))
        image.putpixel((399,599),(0,255,0))
        crops = detail_crops(image)
        self.assertEqual(len(crops),8)
        self.assertTrue(all(c.size==(100,100) for c in crops))
        self.assertEqual(crops[0].getpixel((0,0)),(255,0,0))
        self.assertEqual(crops[3].getpixel((99,99)),(0,255,0))

    def test_pairing_rejects_missing_duplicate_and_conflicting_sides(self):
        from backend.train_pairs import pairs
        front=dict(card_id='a',side='front',image='front.jpg',labels={'grade':8})
        back=dict(card_id='a',side='back',image='back.jpg',labels={'grade':8})
        self.assertEqual(pairs([back,front]),[('a','front.jpg','back.jpg',8)])
        for rows in ([front],[front,front,back],[front,dict(back,labels={'grade':9})]):
            with self.assertRaises(ValueError):pairs(rows)

    def test_head_learns_and_roundtrips(self):
        import torch
        from backend.train_pairs import head
        torch.manual_seed(42);torch.set_num_threads(2)
        model=head();x=torch.randn(4,1024);target=torch.ones(4,1)*.7
        initial=model[0].weight.detach().clone()
        optim=torch.optim.Adam(model.parameters(),lr=.001)
        loss=(model(x)-target).square().mean();loss.backward();optim.step()
        self.assertFalse(torch.equal(initial,model[0].weight))
        clone=head();clone.load_state_dict(model.state_dict());model.eval();clone.eval()
        self.assertTrue(torch.equal(model(x),clone(x)))

    def test_grade_weights_favor_rare_grades_and_normalize(self):
        import torch
        from backend.train_pairs import grade_weights
        labels=torch.tensor([[3.], [9.], [9.], [9.], [9.]])
        weights=grade_weights(labels)
        self.assertEqual(weights.shape, labels.shape)
        self.assertAlmostEqual(weights.mean().item(), 1., places=6)
        self.assertAlmostEqual((weights[0]/weights[1]).item(), 2., places=6)
        self.assertTrue(torch.equal(grade_weights(torch.tensor([[6.],[7.]])),torch.ones(2,1)))

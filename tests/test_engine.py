import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("engine", Path(__file__).resolve().parents[1]/"dist/engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def fixture(left=14, right=14, white=False, neutral=False):
    w, h = 252, 352
    data = bytearray([20, 24, 22, 255] * w*h)
    for y in range(22, 330):
        for x in range(16, 236):
            p = (32, 75, 180) if not neutral else (205, 205, 205)
            if 16+left <= x < 236-right and 40 <= y < 312:
                p = (160+(x%3)*15, 110+(y%3)*15, 60)
            if white and 19 <= x <= 22 and 140 <= y <= 185:
                p = (240, 241, 240)
            data[(y*w+x)*4:(y*w+x)*4+3] = bytes(p)
    return data, w, h


class EngineTests(unittest.TestCase):
    def test_patch_bounds_severity_and_separation(self):
        points = [(12, y) for y in range(12, 15)]
        points += [(208, y) for y in range(150, 162)]
        points += [(100, 100)]  # Isolated noise is not a patch.
        regions, count = engine.locate_whitening(points, 240, 340, (10, 10, 210, 310))
        self.assertEqual(count, 2)
        self.assertEqual(regions[0]['severity'], 'moderate')
        self.assertEqual(regions[0]['location'], 'right edge')
        self.assertEqual(regions[1]['severity'], 'light')
        self.assertEqual(regions[1]['location'], 'top left corner')
        self.assertEqual(regions[1]['bbox'], [12/240, 12/340, 13/240, 15/340])

    def test_centered(self):
        r = engine.analyze(*fixture(), original_height=1400)
        self.assertTrue(r['valid'])
        self.assertIsNotNone(r['centering'])
        self.assertLessEqual(abs(r['centering']['left_right'][0]-50), 5)
        self.assertEqual(r['edges'], 10)

    def test_off_center_lowers_score(self):
        clean = engine.analyze(*fixture())
        shifted = engine.analyze(*fixture(left=23, right=7))
        self.assertLess(shifted['centering']['score'], clean['centering']['score'])

    def test_whitening_lowers_edges_and_locates(self):
        clean = engine.analyze(*fixture())
        worn = engine.analyze(*fixture(white=True))
        self.assertLess(worn['edges'], clean['edges'])
        self.assertTrue(worn['regions'])
        self.assertTrue(any(.35 < p['y'] < .6 for p in worn['regions']))

    def test_neutral_border_abstains(self):
        r = engine.analyze(*fixture(neutral=True))
        self.assertIsNone(r['edges'])
        self.assertIsNone(r['corners'])

    def test_blank_rejected(self):
        self.assertFalse(engine.analyze(bytes([20,20,20,255])*10000,100,100)['valid'])

    def test_malformed_input(self):
        with self.assertRaises(ValueError):
            engine.analyze([],100,100)
        with self.assertRaises(ValueError):
            engine.analyze(*fixture(),crop=[0,0,float('nan'),1])

    def test_grade_abstains_for_poor_capture(self):
        r = engine.analyze(*fixture())
        r['poor_capture'] = True
        self.assertIsNone(engine.summarize([r])['grade_range'])

    def test_invalid_side_does_not_produce_grade(self):
        r = engine.analyze(*fixture())
        self.assertIsNone(engine.summarize([r,{'valid':False}])['grade_range'])

    def test_manual_crop(self):
        r = engine.analyze(*fixture(),crop=[16/252,22/352,236/252,330/352])
        self.assertTrue(r['valid'])
        self.assertEqual(r['boundary'],'manual')

    def test_grade_is_bounded(self):
        r = engine.analyze(*fixture())
        grade = engine.summarize([r])['grade_range']
        self.assertIsNotNone(grade)
        self.assertLessEqual(1, grade[0])
        self.assertLessEqual(grade[0],grade[1])
        self.assertLessEqual(grade[1],10)


if __name__ == '__main__':
    unittest.main()

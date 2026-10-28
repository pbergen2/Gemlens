import tempfile
import unittest
from pathlib import Path
from PIL import Image
from backend.dataset import validate, split_rows

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for i in range(6):
            Image.new('RGB', (180, 250), (i*30, 20, 30)).save(self.root/f'{i}.png')
            self.rows.append(dict(card_id=str(i), image=f'{i}.png', source='synthetic test only',
                rights_reference='test fixture', training_allowed=True, input_reviewed=True,
                grade_system='test', side='front', design='full_art', capture='phone',
                annotation_coverage='partial', labels={'grade':8, 'surface':None}, defects=[]))

    def test_missing_subgrades_preserved(self):
        self.assertIsNone(validate(self.rows,self.root)[0]['labels']['surface'])

    def test_split_keeps_all_views_together(self):
        rows=self.rows+[dict(self.rows[0],side='back')]
        splits=split_rows(validate(rows,self.root))
        groups=[{r['card_id'] for r in x} for x in splits.values()]
        self.assertTrue(all(not a & b for i,a in enumerate(groups) for b in groups[i+1:]))
        self.assertEqual(sum(map(len,splits.values())),7)

    def test_duplicate_under_new_id_rejected(self):
        self.rows[1]['image']='0.png'
        with self.assertRaisesRegex(ValueError,'duplicate'):validate(self.rows,self.root)

    def test_nan_rejected(self):
        self.rows[0]['labels']['grade']=float('nan')
        with self.assertRaisesRegex(ValueError,'1–10'):validate(self.rows,self.root)

    def test_permissions_and_boxes(self):
        self.rows[0]['training_allowed']=False
        with self.assertRaisesRegex(ValueError,'permission'):validate(self.rows,self.root)
        self.rows[0]['training_allowed']=True
        self.rows[0]['defects']=[{'type':'scratch','bbox':[0.5,0.1,0.2,0.3]}]
        with self.assertRaisesRegex(ValueError,'normalized'):validate(self.rows,self.root)

    def test_automated_screen_requires_explicit_opt_in(self):
        self.rows[0]['input_reviewed']=False
        self.rows[0]['screening']={'status':'passed','method':'5k-v1-automated','manual_review':False}
        with self.assertRaisesRegex(ValueError,'opt-in'):validate(self.rows,self.root)
        self.assertFalse(validate(self.rows,self.root,allow_automated_screening=True)[0]['input_reviewed'])
        self.rows[0]['screening']['status']='failed'
        with self.assertRaises(ValueError):validate(self.rows,self.root,allow_automated_screening=True)

import io
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from backend.app import app
from test_engine import fixture


class ApiTests(unittest.TestCase):
    def test_trained_grade_uses_original_pair_and_returns_model_provenance(self):
        data,w,h=fixture()
        im=Image.frombytes('RGBA',(w,h),bytes(data)).resize((w*4,h*4))
        buf=io.BytesIO(); im.save(buf,format='PNG'); raw=buf.getvalue()
        with patch('backend.inference.predict', return_value={'predicted_grade':7.25,'grade_range':None,'method':'test-model'}) as predict:
            result=self.client.post('/api/grade?front_bytes='+str(len(raw)),content=raw+raw)
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(result.json()['predicted_grade'],7.25)
        self.assertEqual([r['side'] for r in result.json()['reports']],['front','back'])
        self.assertGreater(predict.call_args.args[0].width,500)

    def test_trained_grade_rejects_missing_back_and_bad_crop(self):
        self.assertEqual(self.client.post('/api/grade?front_bytes=3',content=b'abc').status_code,422)
        with patch('backend.inference.predict') as predict:
            result=self.client.post('/api/grade?front_bytes=3',content=b'abcdef')
            self.assertEqual(result.status_code,422)
            predict.assert_not_called()

    def setUp(self):
        self.client = TestClient(app)

    def test_valid_image_and_summary(self):
        data, w, h = fixture()
        image = Image.frombytes('RGBA',(w,h),bytes(data))
        buf = io.BytesIO()
        image.save(buf,format='PNG')
        r = self.client.post('/api/analyze?side=front',content=buf.getvalue())
        self.assertEqual(r.status_code,200)
        self.assertTrue(r.json()['valid'])
        summary = self.client.post('/api/summarize',json=[r.json()])
        self.assertEqual(summary.status_code,200)
        self.assertIsNotNone(summary.json()['grade_range'])

    def test_nonimage_rejected(self):
        self.assertEqual(self.client.post('/api/analyze',content=b'not an image').status_code,422)

    def test_bad_summary(self):
        self.assertEqual(self.client.post('/api/summarize',json={}).status_code,422)

    def test_home_and_engine(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/engine.py').status_code,200)


if __name__ == '__main__':
    unittest.main()

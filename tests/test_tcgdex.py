import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from fastapi.testclient import TestClient
from backend.app import app
from backend import tcgdex

CARD = {'id': 'base1-1', 'name': 'Alakazam', 'localId': '1',
        'image': 'https://assets.tcgdex.net/en/base/base1/1', 'set': {'id': 'base1', 'name': 'Base Set'}}

class CatalogTests(unittest.TestCase):
    def setUp(self):
        tcgdex._cache.clear()
        self.client = TestClient(app)

    @patch('backend.tcgdex.urlopen')
    def test_live_contract_cache_and_reference_only(self, open_url):
        open_url.return_value = io.BytesIO(json.dumps(CARD).encode())
        a = self.client.get('/api/cards/base1-1')
        b = self.client.get('/api/cards/base1-1')
        self.assertEqual(a.status_code, 200)
        self.assertEqual(a.json(), b.json())
        self.assertEqual(a.json()['set'], 'Base Set')
        self.assertTrue(a.json()['reference_only'])
        self.assertNotIn('grade', a.json())
        self.assertTrue(a.json()['image'].endswith('/high.webp'))
        open_url.assert_called_once()

    @patch('backend.tcgdex.fetch')
    def test_search_filters_pocket_and_encodes_query(self, fetch):
        fetch.return_value = [CARD, {**CARD, 'id': 'tcgp-A1-1'}, {**CARD, 'id': 'A2b-031', 'image': 'https://assets.tcgdex.net/en/tcgp/A2b/031'}]
        result = self.client.get('/api/cards', params={'q': 'Mr. Mime & friends'})
        self.assertEqual(len(result.json()['cards']), 1)
        self.assertIn('%26', fetch.call_args.args[0])
        self.assertIn('pagination%3AitemsPerPage=20', fetch.call_args.args[0])

    def test_validation(self):
        for path in ['/api/cards?q=x', '/api/cards?q=ab&page=0', '/api/cards?q=ab&language=bad', '/api/cards/tcgp-A1-1']:
            self.assertEqual(self.client.get(path).status_code, 422)
        self.assertIsNone(tcgdex.card_view({**CARD, 'image': 'https://evil.example/card'})['image'])

    @patch('backend.tcgdex.urlopen')
    def test_failure_and_retry_not_cached(self, open_url):
        open_url.side_effect = URLError('offline')
        self.assertEqual(self.client.get('/api/cards/base1-1').status_code, 503)
        open_url.side_effect = None
        open_url.return_value = io.BytesIO(json.dumps(CARD).encode())
        self.assertEqual(self.client.get('/api/cards/base1-1').status_code, 200)

    @patch('backend.tcgdex.urlopen')
    def test_not_found_and_malformed(self, open_url):
        open_url.side_effect = HTTPError('url', 404, 'missing', {}, None)
        self.assertEqual(self.client.get('/api/cards/base1-1').status_code, 404)
        open_url.side_effect = None
        open_url.return_value = io.BytesIO(b'[null]')
        self.assertEqual(self.client.get('/api/cards?q=ab').status_code, 502)

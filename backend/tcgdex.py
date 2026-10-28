"""Bounded, read-only TCGdex catalog access; never supplies grading labels.

Only catalog searches leave this server. Uploaded photos are never sent to TCGdex.
Cache successful JSON responses for an hour, with at most 128 entries per process.
"""
import json
import time
from collections import OrderedDict
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote, urlsplit
from urllib.request import Request, urlopen

BASE = 'https://api.tcgdex.net/v2'
_cache = OrderedDict()
_lock = Lock()


class CatalogError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def fetch(path):
    with _lock:
        cached = _cache.get(path)
        if cached and time.monotonic() - cached[0] < 3600:
            _cache.move_to_end(path)
            return cached[1]
    try:
        request = Request(BASE + path, headers={'Accept': 'application/json', 'User-Agent': 'GemLens/0.1'})
        with urlopen(request, timeout=10) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError('Response too large')
        result = json.loads(data)
    except HTTPError as exc:
        if exc.code == 404:
            raise CatalogError(404, 'Reference card not found.') from exc
        raise CatalogError(503, 'TCGdex is unavailable. Try again shortly; inspection still works.') from exc
    except (URLError, OSError, ValueError) as exc:
        raise CatalogError(503, 'TCGdex is unavailable. Try again shortly; inspection still works.') from exc
    with _lock:
        _cache[path] = (time.monotonic(), result)
        _cache.move_to_end(path)
        while len(_cache) > 128:
            _cache.popitem(last=False)
    return result


def card_view(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('id'), str) or not isinstance(raw.get('name'), str):
        raise CatalogError(502, 'TCGdex returned an invalid card record.')
    image = raw.get('image')
    # Do not let upstream data insert arbitrary URLs into the page.
    if not isinstance(image, str) or urlsplit(image).scheme != 'https' or urlsplit(image).netloc != 'assets.tcgdex.net':
        image = None
    card_set = raw.get('set') if isinstance(raw.get('set'), dict) else {}
    return {
        'id': raw['id'], 'name': raw['name'], 'number': raw.get('localId'),
        'set': card_set.get('name'), 'set_id': card_set.get('id'),
        'image': image.rstrip('/') + '/high.webp' if image else None,
        'thumbnail': image.rstrip('/') + '/low.webp' if image else None,
        'source': 'TCGdex', 'reference_only': True,
    }


def is_pocket(raw):
    return isinstance(raw, dict) and (str(raw.get('id', '')).startswith('tcgp-') or '/tcgp/' in str(raw.get('image', '')))


def search(name, language='en', page=1):
    params = {'name': 'like:' + name, 'pagination:page': page, 'pagination:itemsPerPage': 20}
    raw = fetch('/' + language + '/cards?' + urlencode(params))
    if not isinstance(raw, list):
        raise CatalogError(502, 'TCGdex returned an invalid search response.')
    # Pocket is a digital game; its cards are not physical grading references.
    cards = [card_view(card) for card in raw[:20] if not is_pocket(card)]
    return {'cards': cards, 'page': page, 'has_more': len(raw) >= 20, 'source': 'TCGdex'}


def detail(card_id, language='en'):
    if card_id.startswith('tcgp-'):
        raise CatalogError(422, 'Digital Pocket cards are not physical grading references.')
    raw = fetch('/' + language + '/cards/' + quote(card_id, safe=''))
    if is_pocket(raw):
        raise CatalogError(422, 'Digital Pocket cards are not physical grading references.')
    return card_view(raw)

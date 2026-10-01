"""A text's vector depends on the text alone, never on what it is embedded with.

The int8 model quantizes activations with one scale per input tensor, so a
text embedded inside a shared batch moves with its batch mates and padding
(~0.98 cosine from its solo vector). embed_batch therefore runs one inference
per text; these tests pin that a list call returns exactly the vectors the
same texts get alone, for both kinds.
"""
from servers import embedder
from servers.brain import Brain
from tests.isolated_brain import _SHARED_MODEL_CACHE

# Mixed lengths on purpose: in a shared batch the short texts would be padded
# to the long one, which is what moved them.
TEXTS = [
    'Recall ranks nodes by field score.',
    'The embed queue drains recently written nodes every few seconds and '
    'backfills each missing vector view for them, one vector type at a time.',
    'Why did the eval regress?',
    ' '.join(['The daemon serves every hook over a local socket.'] * 12),
]


def _loaded():
    embedder.load_model({**Brain._EMBEDDER_DEFAULTS, 'cache_dir': _SHARED_MODEL_CACHE})
    assert embedder.is_ready(), embedder.stats.get('load_error')


def test_document_list_equals_each_text_alone():
    _loaded()
    together = embedder.embed_batch(TEXTS, kind='document')
    assert len(together) == len(TEXTS)
    for text, blob in zip(TEXTS, together):
        assert blob == embedder.embed_batch([text], kind='document')[0]


def test_query_list_equals_each_text_alone():
    _loaded()
    together = embedder.embed_batch(TEXTS, kind='query')
    for text, blob in zip(TEXTS, together):
        assert blob == embedder.embed_query(text)


def test_order_of_the_list_does_not_matter():
    _loaded()
    forward = embedder.embed_batch(TEXTS, kind='document')
    backward = embedder.embed_batch(list(reversed(TEXTS)), kind='document')
    assert forward == list(reversed(backward))

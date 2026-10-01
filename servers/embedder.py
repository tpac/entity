"""
brain — Embedding Engine

Thin wrapper around fastembed. No monkey-patches, no thread restrictions, no
custom-model registration. The plugin's isolated runtime pins
onnxruntime >= 1.20, which removes the old ARM64 threadpool race and makes
multi-threaded inference safe by default.

API:
  EMBEDDER_PIN                — the model artifact pin ([tool.brain.embedder]
                                in pyproject.toml, the one place external
                                versions live)
  load_model(config)          — initialize fastembed from a config
  embed_query(text)           — embed a query (with model's query prefix)
  embed_batch(texts, kind)    — list form, kind ∈ {"document","query"};
                                documents are always embedded through it
  cosine_similarity(a, b)     — float, for L2-normalized vectors
  is_ready / get_stats / ...  — introspection

Callers must pick document vs query explicitly — the model needs it to produce
vectors in matched geometry. Mixing them silently collapses recall quality.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import time
import struct
import tomllib
from typing import Optional, List, Dict, Any


def _load_pin() -> Dict[str, Any]:
    """[tool.brain.embedder] from the pyproject.toml beside servers/. A missing
    or malformed pin raises at import: an embedder with no pin would load
    whatever Hugging Face serves that day, and a silently refreshed artifact
    splits the vector space."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'pyproject.toml')
    with open(path, 'rb') as f:
        return dict(tomllib.load(f)['tool']['brain']['embedder'])


EMBEDDER_PIN = _load_pin()


# ─── fastembed session-option polyfill ───────────────────────────
# fastembed 0.8 accepts `extra_session_options={...}` but its
# EXPOSED_SESSION_OPTIONS allowlist hardcodes ("enable_cpu_mem_arena",) and
# its add_extra_session_options applies only that single key. We need three
# more session knobs:
#
#   session.intra_op.allow_spinning = 0     (config entry, prevents busy-wait)
#   session.inter_op.allow_spinning = 0     (config entry, prevents busy-wait)
#   enable_mem_pattern              = False (attribute, bounds ORT arena)
#
# Spin keys: without them, threads>1 pushes idle CPU to 250–290% on macOS
# ARM64 (onnxruntime#9313). Applied via add_session_config_entry().
#
# enable_mem_pattern: ORT's mem_pattern optimizer caches allocation patterns
# per distinct input tensor shape. With variable-length text inputs every
# recall produced a new (B, T) shape and the cache grew without bound (RSS
# 450 MB → 5+ GB observed). Disabling drops a small inference optimization
# (designed for fixed-shape models like CNNs) in exchange for a bounded
# arena. Set as a SessionOptions attribute, not a config entry.
#
# The polyfill extends EXPOSED_SESSION_OPTIONS so the new keys pass
# fastembed's allowlist assertion, and routes each key to the correct ORT
# API. Temporary — remove once fastembed exposes these upstream.
def _install_fastembed_spin_polyfill() -> None:
    try:
        from fastembed.common.onnx_model import OnnxModel
    except ImportError:
        return

    CONFIG_KEYS = (
        "session.intra_op.allow_spinning",
        "session.inter_op.allow_spinning",
    )
    ATTR_KEYS = (
        "enable_mem_pattern",
    )
    EXTRA_KEYS = CONFIG_KEYS + ATTR_KEYS
    exposed = set(OnnxModel.EXPOSED_SESSION_OPTIONS)
    if all(k in exposed for k in EXTRA_KEYS):
        return  # fastembed merged equivalent upstream — no polyfill needed

    OnnxModel.EXPOSED_SESSION_OPTIONS = (
        tuple(OnnxModel.EXPOSED_SESSION_OPTIONS) + EXTRA_KEYS
    )
    original_add = OnnxModel.add_extra_session_options  # bound classmethod

    def _patched_add(cls, session_options, extra_options):
        config = {k: v for k, v in extra_options.items() if k in CONFIG_KEYS}
        attrs = {k: v for k, v in extra_options.items() if k in ATTR_KEYS}
        rest = {k: v for k, v in extra_options.items()
                if k not in CONFIG_KEYS and k not in ATTR_KEYS}
        if rest:
            original_add.__func__(cls, session_options, rest)
        for k, v in config.items():
            session_options.add_session_config_entry(k, str(v))
        for k, v in attrs.items():
            setattr(session_options, k, v)

    OnnxModel.add_extra_session_options = classmethod(_patched_add)


_install_fastembed_spin_polyfill()


# ─── Runtime state ───────────────────────────────────────────────
_model = None
_config: Dict[str, Any] = {}
_doc_prefix: str = ""
_query_prefix: str = ""

stats = {
    'model_loaded': False,
    'model_name': None,
    'embedding_dim': None,
    'load_time_ms': 0,
    'load_error': None,
    'total_embeddings': 0,
    'total_embed_time_ms': 0,
    'errors': 0,
    'last_embed_ms': 0,
    'peak_embed_ms': 0,
}


# ─── Artifact pinning ────────────────────────────────────────────
# The model is a DEPENDENCY OF THE STORED VECTORS: ~8.4k node embeddings (plus
# edge/trace substrates) are only comparable to fresh query embeddings if the
# artifact that produced them is byte-identical to the one serving queries.
# fastembed follows the HF repo's `main` ref with no pin — a silently
# refreshed artifact splits the vector space in two (operator ruling
# 2026-08-07). load_model therefore verifies the LOADED snapshot against the
# configured pins and refuses to serve on mismatch: a loudly dead embedder
# beats a silently split space.

_HEX = set('0123456789abcdef')


def _looks_hex(s: str, n: int) -> bool:
    return len(s) == n and all(c in _HEX for c in s.lower())


def _verify_artifact_pin(model_obj, config) -> tuple:
    """Verify the loaded snapshot against pinned_revision / pinned_onnx_sha256.

    Returns (revision, onnx_sha256) — recorded in stats either way.
    Raises RuntimeError on mismatch. No-op (returns what it can) when no pin
    is configured.
    """
    pinned_rev = config.get('pinned_revision')
    pinned_sha = config.get('pinned_onnx_sha256')

    inner = getattr(model_obj, 'model', None)
    model_dir = getattr(inner, '_model_dir', None)
    if model_dir is None:
        if pinned_rev or pinned_sha:
            raise RuntimeError(
                'embedder pin check: cannot locate loaded model dir '
                '(fastembed internals changed?) — refusing to serve unverified artifact')
        return None, None
    model_dir = str(model_dir)

    # HF hub cache layout: .../models--org--repo/snapshots/<revision>/...
    revision = os.path.basename(model_dir.rstrip('/'))
    if not _looks_hex(revision, 40):
        revision = None

    onnx_sha = None
    for root, _dirs, files in os.walk(model_dir):
        for f in sorted(files):
            if not f.endswith('.onnx'):
                continue
            target = os.path.realpath(os.path.join(root, f))
            base = os.path.basename(target)
            if _looks_hex(base, 64) and base == pinned_sha:
                # hub<1.0 layout: the blob filename IS the content sha256.
                # hub>=1.0 (xet) also names blobs with 64-hex strings that are
                # NOT the content sha — a name is only trusted when it already
                # agrees with the pin; anything else earns a real hash before
                # it can be called a mismatch.
                onnx_sha = base
            else:                           # xet or non-hub layout — hash the bytes
                h = hashlib.sha256()
                with open(target, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b''):
                        h.update(chunk)
                onnx_sha = h.hexdigest()
            break
        if onnx_sha:
            break

    problems = []
    if pinned_rev and revision != pinned_rev:
        problems.append(f'revision {revision!r} != pinned {pinned_rev!r}')
    if pinned_sha and onnx_sha != pinned_sha:
        problems.append(f'onnx sha256 {onnx_sha!r} != pinned {pinned_sha!r}')
    if problems:
        raise RuntimeError(
            'EMBEDDER ARTIFACT PIN MISMATCH — %s. The upstream model changed '
            'or the cache was tampered with. Serving it would embed new '
            'writes in a different space than the stored vectors. Either '
            'restore the pinned artifact or DELIBERATELY update the pins '
            '(set_embedder_config) and re-embed the corpus.' % '; '.join(problems))
    return revision, onnx_sha


def _clone_file(src_real: str, dpath: str) -> None:
    """Materialize one file: hardlink when the volumes allow it (the common
    case — near-free, and the artifact is read-only content-addressed data),
    byte copy otherwise."""
    try:
        os.link(src_real, dpath)
    except OSError:
        shutil.copy2(src_real, dpath)


def _clone_link(src_root: str, spath: str, dpath: str) -> None:
    """Reproduce one symlink so the copy survives on its own: a link that
    resolves inside `src_root` is kept as a relative link (one blob payload;
    absolute in-tree links are rewritten relative — materializing them would
    double the payload and downgrade the pin check to a per-boot byte hash);
    a file link that points outside is materialized from its target. A
    dangling link or an out-of-tree DIRECTORY link (never produced by hub
    layouts; copying one would dereference an unbounded shared store) aborts
    the copy. `src_root` must be realpath'd by the caller."""
    if not os.path.exists(spath):   # follows the whole link chain
        raise OSError('dangling symlink %s -> %s'
                      % (spath, os.readlink(spath)))
    raw = os.readlink(spath)
    target = raw if os.path.isabs(raw) else os.path.join(os.path.dirname(spath), raw)
    # Resolve the PARENT only: macOS aliases $TMPDIR under /var → /private/var,
    # so an absolute hub link in the resolved spelling would misclassify as
    # out-of-tree against an unresolved root. Never realpath the final
    # component — the target's own link chain (an in-tree blob entry that
    # itself points at the outer store) must not decide in-tree-ness.
    target = os.path.join(os.path.realpath(os.path.dirname(target)),
                          os.path.basename(target))
    if os.path.commonpath([src_root, target]) == src_root:
        os.symlink(os.path.relpath(target, os.path.dirname(spath)), dpath)
    else:
        real = os.path.realpath(spath)
        if os.path.isdir(real):
            raise OSError('out-of-tree directory symlink %s -> %s'
                          % (spath, raw))
        _clone_file(real, dpath)


def _copytree_self_contained(src: str, dst: str) -> None:
    """copytree for a hub model tree whose copy must survive on its own.

    Newer huggingface_hub/xet cache layouts make the per-model
    blobs/<sha256> entry itself a symlink into an OUTER
    <cache>/blobs/<2-hex>/<hash> store — outside the models--<slug> tree
    this copies. A plain symlinks=True copy then ships a dangling blob
    link: TextEmbedding fails at load (ONNX NO_SUCHFILE), _model stays
    None, and the brain runs with dead semantic recall (observed
    2026-09-18 in a longmem corpus build).

    Per-entry rules are _clone_link/_clone_file's. Any failure — a dangling
    link, an unreadable subtree (os.walk's default onerror SWALLOWS scandir
    errors; a silently incomplete copy would shadow the re-download forever)
    — aborts the whole copy; the caller stages into a temp dir, so no
    partial tree is ever visible at dst.
    """
    src = os.path.realpath(src)   # /var vs /private/var must agree with links

    def _raise(err):
        raise err

    os.makedirs(dst)
    for root, dirs, files in os.walk(src, onerror=_raise):
        droot = os.path.join(dst, os.path.relpath(root, src)) if root != src else dst
        for name in dirs:
            spath = os.path.join(root, name)
            dpath = os.path.join(droot, name)
            if os.path.islink(spath):
                _clone_link(src, spath, dpath)
            else:
                # Default modes, deliberately no copystat: cloning a source
                # dir's mode before writing its children breaks on read-only
                # source dirs (0500 → children unwritable → abort + an
                # uncleanable stage). Files keep their metadata via
                # copy2/hardlink; dir modes are the umask's business.
                os.makedirs(dpath)
        for name in files:
            spath = os.path.join(root, name)
            dpath = os.path.join(droot, name)
            if os.path.islink(spath):
                _clone_link(src, spath, dpath)
            else:
                _clone_file(spath, dpath)


def _seed_cache_from_tmp(model_name: str, cache_dir: str) -> None:
    """One-time migration helper: if the durable cache doesn't hold this model
    yet but fastembed's default $TMPDIR cache does, copy the tree over instead
    of re-downloading 137MB (and instead of failing on an offline boot).
    Best-effort PER CANDIDATE slug — a broken candidate is skipped, not fatal
    to the other (the -Q name and the base repo it resolves to are separate
    trees, and either alone can carry the artifact); any failure falls through
    to fastembed's own download."""
    try:
        candidates = {model_name, model_name[:-2] if model_name.lower().endswith('-q') else model_name}
        tmp_cache = os.path.join(tempfile.gettempdir(), 'fastembed_cache')
        for cand in candidates:
            slug = 'models--' + cand.replace('/', '--')
            src = os.path.join(tmp_cache, slug)
            dst = os.path.join(cache_dir, slug)
            if not os.path.isdir(src) or os.path.isdir(dst):
                continue
            os.makedirs(cache_dir, exist_ok=True)
            # Reap stage dirs a hard-killed seeder left behind (the finally
            # below never runs on SIGKILL). Age-gated so a concurrent
            # seeder's LIVE stage is never touched — no seed runs for a day.
            for stale in os.listdir(cache_dir):
                p = os.path.join(cache_dir, stale)
                if (stale.startswith('.seed-')
                        # (category A: mtime is a persisted filesystem stamp,
                        # and monotonic has no shared origin to compare it to)
                        and time.time() - os.path.getmtime(p) > 86400):  # wall-clock-ok: mtime is wall time
                    shutil.rmtree(p, ignore_errors=True)
            # Stage into a private temp dir, then rename into place: no
            # partial dst is ever visible (a partial one would shadow the
            # re-download via the isdir guard above), and a concurrent
            # seeder's completed NON-EMPTY copy is never deleted — our
            # rename fails ENOTEMPTY against it. (POSIX rename does replace
            # an EMPTY dst dir; that just-created-nothing-written window is
            # accepted — the other writer is either a peer seeder using this
            # same protocol, where first-rename-wins is consistent, or a
            # downloader that immediately writes nested dirs.)
            stage_root = tempfile.mkdtemp(prefix='.seed-', dir=cache_dir)
            try:
                stage = os.path.join(stage_root, slug)
                _copytree_self_contained(src, stage)
                os.rename(stage, dst)
                print(f'[embedder] seeded durable cache from {src}', file=sys.stderr)
            except Exception as e:
                print(f'[embedder] cache seed of {slug} skipped ({e})',
                      file=sys.stderr)
            finally:
                shutil.rmtree(stage_root, ignore_errors=True)
    except Exception as e:
        print(f'[embedder] cache seed skipped ({e}) — fastembed will download',
              file=sys.stderr)


# ─── Prefix table ────────────────────────────────────────────────
# Models that require task-prefixed inputs. Keys are lowercased — lookup
# normalizes the configured model_name before checking. If your model isn't
# listed, prefixes default to empty strings (bge/e5/etc. don't need them).
_PREFIX_TABLE = {
    "nomic-ai/nomic-embed-text-v1.5":   ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1.5-q": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1":     ("search_document: ", "search_query: "),
}


def load_model(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize fastembed. Idempotent on repeat calls with same config.

    Config keys (Brain._get_embedder_config: EMBEDDER_PIN overlaid by brain_meta):
      model_name:  HuggingFace model ID (default EMBEDDER_PIN)
      dim:         Embedding dimensions (for stats — fastembed knows the real dim)
      cache_dir:   Optional override for fastembed's model cache
    """
    global _model, _config, _doc_prefix, _query_prefix

    if config is None:
        config = {}

    # Singleton guard — don't reload on same config
    if _model is not None and config == _config:
        return
    _config = dict(config)

    model_name = config.get('model_name', EMBEDDER_PIN['model_name'])
    dim = config.get('dim', EMBEDDER_PIN['dim'])
    cache_dir = config.get('cache_dir')

    stats['model_name'] = model_name
    stats['embedding_dim'] = dim

    # Resolve prefixes for this model (case-insensitive lookup)
    _doc_prefix, _query_prefix = _PREFIX_TABLE.get(model_name.lower(), ("", ""))

    t0 = time.time()
    try:
        from fastembed import TextEmbedding

        kwargs: Dict[str, Any] = {}
        if cache_dir:
            _seed_cache_from_tmp(model_name, cache_dir)
            kwargs['cache_dir'] = cache_dir

        # ORT session knobs. fastembed picks these up via
        # _select_exposed_session_options(kwargs) — i.e., flat kwargs
        # whose keys are in EXPOSED_SESSION_OPTIONS. Passing them inside
        # an `extra_session_options={...}` dict does NOT work — fastembed
        # has no such named param; the dict gets filtered out by name.
        # Our polyfill above extends EXPOSED_SESSION_OPTIONS for the spin
        # and mem_pattern keys.
        #
        #  - enable_cpu_mem_arena=False: ORT's arena pre-allocates per-shape
        #    slabs and never releases them — drives multi-GB RSS growth on
        #    long-running daemons with variable input shapes (fastembed
        #    issue #570, ORT issue #11627). This is THE leak fix; the
        #    other two are unrelated CPU-spin/memory hardening.
        #  - allow_spinning=0: prevent WorkerLoop busy-wait (idle CPU sink)
        #  - enable_mem_pattern=False: bound mem_pattern's per-shape cache
        #    (independent from the arena — both must be off)
        session_kwargs = {
            "enable_cpu_mem_arena": False,
            "session.intra_op.allow_spinning": "0",
            "session.inter_op.allow_spinning": "0",
            "enable_mem_pattern": False,
        }

        _model = TextEmbedding(model_name=model_name, **kwargs, **session_kwargs)

        # Artifact pin — raises on mismatch, failing the load LOUDLY. A dead
        # embedder is recoverable; a split vector space is silent corruption.
        revision, onnx_sha = _verify_artifact_pin(_model, config)
        stats['model_revision'] = revision
        stats['onnx_sha256'] = onnx_sha

        stats['load_time_ms'] = round((time.time() - t0) * 1000)
        stats['model_loaded'] = True
        stats['load_error'] = None
        print(
            f"[embedder] {model_name} ({dim}d) loaded in {stats['load_time_ms']}ms"
            + (f" [rev {revision[:8]}]" if revision else "")
            + (f" [prefixes: doc={_doc_prefix!r} query={_query_prefix!r}]" if _doc_prefix or _query_prefix else ""),
            file=sys.stderr,
        )

    except ImportError as e:
        stats['model_loaded'] = False
        stats['load_error'] = f"fastembed not installed: {e}"
        stats['errors'] += 1
        print(f"[embedder] CRITICAL: fastembed not installed — recall broken ({e})", file=sys.stderr)

    except Exception as e:
        _model = None    # never serve a model that failed load OR pin verify
        stats['model_loaded'] = False
        stats['load_error'] = str(e)
        stats['errors'] += 1
        print(f"[embedder] CRITICAL: model load failed — {e}", file=sys.stderr)


def is_ready() -> bool:
    return stats['model_loaded'] and _model is not None


def get_model_status() -> str:
    name = stats.get('model_name', '?')
    dim = stats.get('embedding_dim', '?')
    if is_ready():
        return f"READY: {name} ({dim}d, loaded in {stats['load_time_ms']}ms)"
    elif stats['load_error']:
        return f"FAILED: {stats['load_error']}"
    return "NOT LOADED: load_model() not called yet"


def get_config() -> Dict[str, Any]:
    return dict(_config)


# ─── Core inference ──────────────────────────────────────────────
#
# Variable-shape inputs are safe here because we set
# `enable_mem_pattern=False` at session creation (see load_model).
# With mem_pattern off, ORT does not cache allocation plans per
# distinct (batch, seq) shape and the arena stays bounded. Every call
# embeds one text (see embed_batch), so nothing is padded.

def embed_query(text: str) -> Optional[bytes]:
    """Embed text for SEARCH. Applies the model's query prefix.
    Mismatching document/query prefixes collapses cosine similarity — pick
    the right one for the call site.
    """
    return next(iter(embed_batch([text], kind='query')), None)


def embed_batch(texts: List[str], kind: str = "document") -> List[Optional[bytes]]:
    """Embed a list of texts. `kind` is 'document' or 'query'.

    One ORT inference call per text (batch_size=1), so a text's vector
    depends on the text alone. The int8 model quantizes activations with one
    scale per input tensor (DynamicQuantizeLinear), so inside a shared batch
    every text shifts with its batch mates and padding: ~0.98 cosine from
    its solo vector. Solo is also faster here (padding to the batch's
    longest text is wasted compute) and bounds peak memory to one text's
    attention instead of batch x longest-text.
    """
    if not texts:
        return []
    if not _model:
        stats['errors'] += 1
        return []
    prefix = _query_prefix if kind == "query" else _doc_prefix
    n = len(texts)

    t0 = time.time()
    try:
        results = [_vec_to_blob(v) for v in
                   _model.embed([prefix + t for t in texts], batch_size=1)]
        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += n
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = round(elapsed_ms / n)
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms
        return results
    except Exception as e:
        stats['errors'] += 1
        print(f"[embedder] embed error: {e}", file=sys.stderr)
        return []


# ─── Math / serialization ────────────────────────────────────────

def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two embedding blobs.
    For L2-normalized vectors, cosine = dot product. fastembed normalizes
    outputs, so dot product is correct and ~100× faster than full cosine.
    """
    if not a or not b:
        return 0.0
    import numpy as np
    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    if len(va) != len(vb):
        return 0.0
    return float(np.dot(va, vb))


def _vec_to_blob(vec) -> bytes:
    """Serialize embedding vector to float32 bytes. L2-normalizes so that
    cosine similarity reduces to a dot product downstream."""
    import numpy as np
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.tobytes()


def _blob_to_vec(blob: bytes) -> list:
    count = len(blob) // 4
    return list(struct.unpack(f'<{count}f', blob))


def get_stats() -> dict:
    out = {
        **stats,
        'avg_embed_ms': (
            round(stats['total_embed_time_ms'] / stats['total_embeddings'])
            if stats['total_embeddings'] > 0 else 0
        ),
    }
    # Surface both queue drain stats — makes "my write isn't indexed"
    # and "my access marks aren't landing" debuggable without digging
    # through logs. recall_write_queue is the bg_writer queue for
    # access marks (Phase 5, 2026-05-18).
    try:
        from . import embed_queue
        out['embed_queue'] = embed_queue.get_stats()
    except Exception as e:
        import sys as _sys
        print('[embedder] embed_queue.get_stats failed: %s' % e,
              file=_sys.stderr)
    try:
        from . import recall_write_queue
        out['recall_write_queue'] = recall_write_queue.get_stats()
    except Exception as e:
        import sys as _sys
        print('[embedder] recall_write_queue.get_stats failed: %s' % e,
              file=_sys.stderr)
    return out



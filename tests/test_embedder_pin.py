"""Embedder artifact pin — the anti-auto-update contract.

The model artifact is a dependency of every stored vector: if fastembed
silently pulls a refreshed artifact (its cache lives in macOS-purged $TMPDIR,
re-downloading whatever HF `main` serves), new embeddings land in a different
space than the ~8.4k stored ones. Operator ruling 2026-08-07: embedders must
never auto-update. These tests pin the enforcement mechanics WITHOUT loading
a model — pure filesystem contracts.
"""
import os
import shutil
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "servers"))
import embedder  # noqa: E402

GOOD_REV = "e" * 40
GOOD_SHA = "b" * 64


def _fake_model(model_dir):
    """Mimic fastembed's TextEmbedding.model._model_dir shape."""
    inner = types.SimpleNamespace(_model_dir=str(model_dir))
    return types.SimpleNamespace(model=inner)


def _hub_snapshot(base, rev=GOOD_REV, blob_sha=GOOD_SHA, *,
                  name="models--org--repo", outer_blob=False, dangling=False):
    """The one definition of the hub cache shape both test families use.

    Default: snapshots/<rev>/onnx/x.onnx -> blobs/<sha> (absolute in-tree
    link, as pathlib's symlink_to writes and hub sometimes does), blob bytes
    in the per-model tree. outer_blob=True builds the newer hub/xet shape:
    blobs/<sha> is itself a symlink into an OUTER <base>/blobs/<2-hex>/
    store (dangling=True leaves that outer target missing)."""
    repo = base / name
    blobs = repo / "blobs"
    snap = repo / "snapshots" / rev / "onnx"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    blob = blobs / blob_sha
    if outer_blob:
        outer = base / "blobs" / blob_sha[:2] / blob_sha
        if not dangling:
            outer.parent.mkdir(parents=True)
            outer.write_bytes(b"onnx-bytes")
        blob.symlink_to(outer)                       # absolute, out-of-tree
        (snap / "model_quantized.onnx").symlink_to(
            os.path.relpath(blob, snap))             # relative, in-tree
    else:
        blob.write_bytes(b"onnx-bytes")
        (snap / "model_quantized.onnx").symlink_to(blob)
    return repo / "snapshots" / rev


def _seed(tmp_path, monkeypatch, model="org/repo"):
    """Run _seed_cache_from_tmp against tmp_path/tmpdir as $TMPDIR; returns
    the durable cache dir. One preamble for every seed test."""
    monkeypatch.setattr(embedder.tempfile, "gettempdir",
                        lambda: str(tmp_path / "tmpdir"))
    durable = tmp_path / "durable"
    embedder._seed_cache_from_tmp(model, str(durable))
    return durable


class TestVerifyArtifactPin:
    def test_matching_pins_pass(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        rev, sha = embedder._verify_artifact_pin(
            _fake_model(snap),
            {"pinned_revision": GOOD_REV, "pinned_onnx_sha256": GOOD_SHA})
        assert rev == GOOD_REV and sha == GOOD_SHA

    def test_revision_mismatch_raises(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        with pytest.raises(RuntimeError, match="PIN MISMATCH"):
            embedder._verify_artifact_pin(
                _fake_model(snap),
                {"pinned_revision": "f" * 40, "pinned_onnx_sha256": GOOD_SHA})

    def test_sha_mismatch_raises(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        with pytest.raises(RuntimeError, match="PIN MISMATCH"):
            embedder._verify_artifact_pin(
                _fake_model(snap),
                {"pinned_revision": GOOD_REV, "pinned_onnx_sha256": "a" * 64})

    def test_no_pins_never_raises(self, tmp_path):
        import hashlib
        snap = _hub_snapshot(tmp_path)
        rev, sha = embedder._verify_artifact_pin(_fake_model(snap), {})
        # still reported for stats — the TRUE content hash, never the blob
        # filename (hub>=1.0 xet names are 64-hex but not the content sha)
        assert rev == GOOD_REV
        assert sha == hashlib.sha256(b"onnx-bytes").hexdigest()

    def test_xet_blob_name_is_not_trusted(self, tmp_path):
        # hub>=1.0 downloads via xet and names blobs with a 64-hex string
        # that is NOT the content sha256. A correct artifact under a foreign
        # name must PASS (hash the bytes), never report a false mismatch —
        # this exact false positive bricked fresh installs' embedders.
        import hashlib
        content_sha = hashlib.sha256(b"onnx-bytes").hexdigest()
        xet_name = "8" * 64  # 64-hex, deliberately != content sha
        snap = _hub_snapshot(tmp_path, blob_sha=xet_name)
        rev, sha = embedder._verify_artifact_pin(
            _fake_model(snap),
            {"pinned_revision": GOOD_REV, "pinned_onnx_sha256": content_sha})
        assert sha == content_sha

    def test_non_hub_layout_hashes_bytes(self, tmp_path):
        # plain dir, real file (no symlink, non-hex name) → sha256 of bytes
        import hashlib
        d = tmp_path / "local-model"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"raw-model-bytes")
        expected = hashlib.sha256(b"raw-model-bytes").hexdigest()
        rev, sha = embedder._verify_artifact_pin(
            _fake_model(d), {"pinned_onnx_sha256": expected})
        assert sha == expected
        assert rev is None  # dir name isn't a 40-hex revision

    def test_missing_model_dir_with_pin_raises(self):
        broken = types.SimpleNamespace(model=types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="cannot locate"):
            embedder._verify_artifact_pin(broken, {"pinned_revision": GOOD_REV})


class TestSeedCacheFromTmp:
    def test_seeds_q_variant_from_base_repo(self, tmp_path, monkeypatch):
        # nomic-Q resolves to the un-suffixed repo dir — seed must find it
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        src = tmp_cache / "models--org--repo"
        (src / "blobs").mkdir(parents=True)
        (src / "blobs" / "x").write_bytes(b"blob")
        monkeypatch.setattr(embedder.tempfile, "gettempdir",
                            lambda: str(tmp_path / "tmpdir"))
        durable = tmp_path / "durable"
        embedder._seed_cache_from_tmp("org/repo-Q", str(durable))
        assert (durable / "models--org--repo" / "blobs" / "x").exists()

    def test_existing_durable_copy_untouched(self, tmp_path, monkeypatch):
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        (tmp_cache / "models--org--repo" / "blobs").mkdir(parents=True)
        monkeypatch.setattr(embedder.tempfile, "gettempdir",
                            lambda: str(tmp_path / "tmpdir"))
        durable = tmp_path / "durable"
        marker = durable / "models--org--repo" / "KEEP"
        marker.parent.mkdir(parents=True)
        marker.write_text("existing")
        embedder._seed_cache_from_tmp("org/repo", str(durable))
        assert marker.read_text() == "existing"  # no clobber

    def test_seed_failure_is_nonfatal(self, monkeypatch):
        monkeypatch.setattr(embedder.tempfile, "gettempdir",
                            lambda: (_ for _ in ()).throw(OSError("boom")))
        embedder._seed_cache_from_tmp("org/repo", "/nonexistent/durable")  # no raise


class TestSeedCacheSelfContained:
    """The seeded copy must survive without the source cache: xet layouts
    (out-of-tree blob links) broke this — TextEmbedding failed with ONNX
    NO_SUCHFILE and eval brains built with dead semantic recall (2026-09-18)."""

    def test_out_of_tree_blob_is_materialized(self, tmp_path, monkeypatch):
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache, outer_blob=True)
        durable = _seed(tmp_path, monkeypatch)

        shutil.rmtree(tmp_cache)  # self-containment proof: source gone
        snap = durable / "models--org--repo" / "snapshots" / GOOD_REV
        onnx = snap / "onnx" / "model_quantized.onnx"
        assert onnx.read_bytes() == b"onnx-bytes"
        # One blob payload: the in-tree snapshot→blob link is preserved,
        # only the out-of-tree blob entry became a real file.
        assert onnx.is_symlink()
        blob = durable / "models--org--repo" / "blobs" / GOOD_SHA
        assert blob.is_file() and not blob.is_symlink()
        # The materialized name IS the content sha, so the pin's free
        # filename fast path still verifies the seeded copy.
        rev, sha = embedder._verify_artifact_pin(
            _fake_model(snap),
            {"pinned_revision": GOOD_REV, "pinned_onnx_sha256": GOOD_SHA})
        assert rev == GOOD_REV and sha == GOOD_SHA

    def test_materialized_blob_is_hardlinked_when_possible(self, tmp_path, monkeypatch):
        # Same volume here, as in the real $TMPDIR→durable case: the blob is
        # linked, not copied — a per-eval-item seed costs ~0 bytes, not 137MB.
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache, outer_blob=True)
        durable = _seed(tmp_path, monkeypatch)
        blob = durable / "models--org--repo" / "blobs" / GOOD_SHA
        assert blob.stat().st_nlink == 2

    def test_absolute_in_tree_link_stays_one_payload(self, tmp_path, monkeypatch):
        # hub writes absolute snapshot→blob links on some setups; the copy
        # rewrites them RELATIVE instead of materializing a second payload
        # (which would also downgrade the pin check from the free filename
        # read to hashing 137MB on every load_model).
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache)   # default shape: absolute in-tree link
        durable = _seed(tmp_path, monkeypatch)
        onnx = (durable / "models--org--repo" / "snapshots" / GOOD_REV
                / "onnx" / "model_quantized.onnx")
        assert onnx.is_symlink() and not os.path.isabs(os.readlink(onnx))
        shutil.rmtree(tmp_cache)
        assert onnx.read_bytes() == b"onnx-bytes"

    def test_dangling_blob_link_aborts_seed_entirely(self, tmp_path, monkeypatch):
        # A broken source must not seed a broken copy: no dst and no leftover
        # stage dirs — the absence lets fastembed fall through to a download.
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache, outer_blob=True, dangling=True)
        durable = _seed(tmp_path, monkeypatch)  # no raise
        assert not (durable / "models--org--repo").exists()
        assert list(durable.iterdir()) == []

    def test_broken_candidate_does_not_block_the_other(self, tmp_path, monkeypatch):
        # candidates is a set (nondeterministic order): a dangling -Q tree is
        # skipped per candidate, and the intact base tree still seeds.
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache, name="models--org--repo-Q",
                      outer_blob=True, dangling=True)
        _hub_snapshot(tmp_cache)
        durable = _seed(tmp_path, monkeypatch, model="org/repo-Q")
        assert (durable / "models--org--repo" / "blobs" / GOOD_SHA).exists()
        assert not (durable / "models--org--repo-Q").exists()

    def test_broken_candidate_does_not_block_the_other_mirrored(self, tmp_path, monkeypatch):
        # Mirror of the test above (broken BASE, intact -Q). Set iteration
        # order is fixed per process for the same two strings, so exactly one
        # of the pair discriminates a loop-level abort on every run — together
        # they make the per-candidate containment regression-deterministic.
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache, outer_blob=True, dangling=True)
        _hub_snapshot(tmp_cache, name="models--org--repo-Q")
        durable = _seed(tmp_path, monkeypatch, model="org/repo-Q")
        assert (durable / "models--org--repo-Q" / "blobs" / GOOD_SHA).exists()
        assert not (durable / "models--org--repo").exists()

    def test_unreadable_subtree_aborts_instead_of_partial_seed(self, tmp_path, monkeypatch):
        # os.walk's default onerror SWALLOWS scandir errors — a silently
        # partial copy would shadow the re-download forever via the
        # isdir(dst) guard in _seed_cache_from_tmp.
        if os.geteuid() == 0:
            pytest.skip("permission checks are void as root")
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        _hub_snapshot(tmp_cache)
        locked = tmp_cache / "models--org--repo" / "snapshots"
        locked.chmod(0)
        try:
            durable = _seed(tmp_path, monkeypatch)  # no raise (best-effort)
            assert not (durable / "models--org--repo").exists()
            # ... and the abort left nothing behind — no partial tree, no
            # stage skeleton (dir modes are NOT cloned onto the stage, so
            # cleanup can always descend).
            assert list(durable.iterdir()) == []
        finally:
            locked.chmod(0o755)

    def test_out_of_tree_directory_link_aborts(self, tmp_path, monkeypatch):
        # Never produced by hub layouts; copying one would dereference an
        # unbounded shared store into the seeded copy.
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        (tmp_cache / "shared-store").mkdir(parents=True)
        repo = tmp_cache / "models--org--repo"
        repo.mkdir(parents=True)
        (repo / "blobs").symlink_to(tmp_cache / "shared-store")
        durable = _seed(tmp_path, monkeypatch)
        assert not (durable / "models--org--repo").exists()

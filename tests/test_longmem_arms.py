"""Longmem on eval arms: the build drives a haystack through a tree's own
daemon (eval/longmem/build_corpus.py), the sweep asks on clone arms
(eval/longmem/sweep.py).

The helper tests are pure. The daemon tests boot keyless arms from this
checkout on seed brains, under a temp corpus root: nothing encodes, nothing
is paid, nothing reaches the operator's brain or corpora.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402

require_eval()  # D-8: eval/ is absent from the public tree

from eval.arms import tree_digest  # noqa: E402
from eval.longmem import build_corpus, corpus, sweep  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _session(n, evidence_at=()):
    out = []
    for k in range(n):
        out.append({'role': 'user', 'content': 'user turn %d' % k,
                    'has_answer': k in evidence_at})
        out.append({'role': 'assistant', 'content': 'reply %d' % k,
                    'has_answer': False})
    return out


def _item(qid='synthetic_arm1'):
    # Listed out of date order: the later session first.
    return {'question_id': qid, 'question_type': 'single-session-user',
            'question': 'What was said in the last turn?', 'answer': 'turn 5',
            'question_date': '2023/05/30 (Tue) 10:00',
            'haystack_dates': ['2023/05/29 (Mon) 09:00', '2023/05/20 (Sat) 08:00'],
            'haystack_sessions': [_session(3), _session(6, evidence_at=(5,))]}


class TestItemSessions(unittest.TestCase):

    def test_sessions_run_in_date_order(self):
        plan = build_corpus._item_sessions(_item())
        self.assertEqual([listed for listed, _, _ in plan], [1, 0])
        self.assertEqual([len(ex) for _, _, ex in plan], [6, 3])

    def test_same_timestamp_keeps_listed_order(self):
        item = _item()
        item['haystack_dates'] = ['2023/05/20 (Sat) 08:00'] * 2
        self.assertEqual([i for i, _, _ in build_corpus._item_sessions(item)], [0, 1])

    def test_a_missing_or_bad_date_fails_the_item(self):
        item = _item()
        item['haystack_dates'] = item['haystack_dates'][:1]
        with self.assertRaisesRegex(ValueError, 'one date per session'):
            build_corpus._item_sessions(item)
        item['haystack_dates'] = ['yesterday', '2023/05/20 (Sat) 08:00']
        with self.assertRaisesRegex(ValueError, 'unparseable'):
            build_corpus._item_sessions(item)

    def test_exchanges_pair_turns_and_mark_evidence(self):
        session = [{'role': 'user', 'content': 'a'},
                   {'role': 'assistant', 'content': 'b', 'has_answer': True},
                   {'role': 'user', 'content': 'c'}]
        self.assertEqual(build_corpus._exchanges(session),
                         [('a', 'b', True), ('c', '', False)])


class TestUnencodedLabels(unittest.TestCase):

    def _sessions(self):
        return [{'session_id': 's1', 'user_turns': 6, 'evidence_at': [5]},
                {'session_id': 's2', 'user_turns': 7, 'evidence_at': [1, 6]}]

    def test_the_tail_rule_strands_the_last_turns(self):
        sessions = self._sessions()
        build_corpus._label_unencoded(sessions, [1, 2], set())
        self.assertEqual([s['stranded'] for s in sessions], [1, 2])
        self.assertEqual([s['evidence_stranded'] for s in sessions], [1, 1])
        self.assertEqual([s['unencoded_after_failure'] for s in sessions], [0, 0])

    def test_a_failed_encode_is_not_the_tail_rules(self):
        sessions = self._sessions()
        build_corpus._label_unencoded(sessions, [1, 7], {'s2'})
        self.assertEqual([s['stranded'] for s in sessions], [1, 0])
        self.assertEqual([s['unencoded_after_failure'] for s in sessions], [0, 7])
        self.assertEqual(sessions[1]['evidence_stranded'], 0)


class TestOneCorpusPerCode(unittest.TestCase):
    """Two trees with the same code are one corpus, built once."""

    def test_same_code_trees_share_one_build(self):
        root = tempfile.mkdtemp(prefix='one-corpus-')
        self.addCleanup(shutil.rmtree, root, True)
        oracle = os.path.join(root, 'oracle.json')
        with open(oracle, 'w') as f:
            json.dump([_item()], f)
        trees = [os.path.join(root, t) for t in ('main', 'branch')]
        for t in trees:
            os.makedirs(t)
        built = []

        def fake_build(job):
            built.append(job['tree'])
            return {'qid': job['item']['question_id'], 'answerable': False}

        with mock.patch.object(corpus, 'CORPUS_ROOT', os.path.join(root, 'corpora')), \
                mock.patch('eval.arms.tree_digest', return_value='samecode'), \
                mock.patch('eval.arms.sync_runtime'), \
                mock.patch.object(build_corpus, '_build_item_on_arm', fake_build):
            hashes = build_corpus.build_arm_corpora(
                trees, 0, 42, oracle, 4, 'test', qids='synthetic_arm1')
        self.assertEqual(len(built), 1)
        self.assertEqual(len(set(hashes.values())), 1)
        self.assertEqual(sorted(hashes), sorted(os.path.realpath(t) for t in trees))


class TestHookContext(unittest.TestCase):

    def _ok(self, stdout):
        return [{'rc': 0, 'stdout': stdout, 'stderr': '', 'timed_out': False}]

    def test_context_from_the_hosts_json(self):
        out = json.dumps({'hookSpecificOutput': {
            'hookEventName': 'UserPromptSubmit', 'additionalContext': 'ctx'}})
        self.assertEqual(sweep._hook_context(self._ok(out)), ('ctx', ''))

    def test_nothing_surfaced_is_no_context_and_no_error(self):
        self.assertEqual(sweep._hook_context(self._ok('')), (None, ''))

    def test_a_failed_hook_is_the_harness_not_the_brain(self):
        failed = [{'rc': None, 'stdout': '', 'stderr': 'slow', 'timed_out': True}]
        context, error = sweep._hook_context(failed)
        self.assertIsNone(context)
        self.assertIn('prompt hook failed', error)
        for banner in ('[BRAIN]\n⚠️ RECALL FAILED: timeout',
                       '⚠️ Brain daemon is unreachable. Hook: recall.'):
            out = json.dumps({'hookSpecificOutput': {'additionalContext': banner}})
            context, error = sweep._hook_context(self._ok(out))
            self.assertIsNone(context)
            self.assertEqual(error, banner)

    def test_the_banners_are_the_hooks_own_words(self):
        scripts = os.path.join(TREE, 'hooks', 'scripts')
        text = ''.join(open(os.path.join(scripts, f)).read()
                       for f in ('pre_response_recall.py', 'hook_common.py'))
        for banner in sweep.HOOK_FAILURE_BANNERS:
            self.assertIn(banner, text)


class TestFailedAsk(unittest.TestCase):

    def test_a_question_that_could_not_be_asked_is_a_suspect_miss(self):
        reports = tempfile.mkdtemp(prefix='failed-ask-')
        self.addCleanup(shutil.rmtree, reports, True)
        it = {'qid': 'q1', 'axis': 'temporal', 'question': 'Q?', 'gold': 'A'}
        row = sweep._failed_rep(it, 'q1', None, 'ask failed: boom',
                                ['preflight:x'], reports, 'run', 'abc123')
        self.assertFalse(row['correct'])
        self.assertEqual(row['measurement_suspect'], ['preflight:x', 'harness_error'])
        self.assertTrue(os.path.isfile(
            os.path.join(reports, 'run', 'items', 'q1', 'result.json')))


class TestAskingTree(unittest.TestCase):

    def test_an_in_process_corpus_has_no_asking_tree(self):
        self.assertIsNone(sweep._asking_tree({'config': {'s1e': 'active'}}))
        with self.assertRaisesRegex(SystemExit, 'built in process'):
            sweep._asking_tree({'config': {'s1e': 'active'}}, tree=TREE)

    def test_the_building_tree_asks_only_while_its_code_is_unchanged(self):
        manifest = {'config': {'harness': build_corpus.ARM_HARNESS,
                               'tree': tree_digest(TREE)},
                    'tree_path': TREE}
        self.assertEqual(sweep._asking_tree(manifest), TREE)
        manifest['config']['tree'] = 'other-code'
        with self.assertRaisesRegex(SystemExit, 'code changed'):
            sweep._asking_tree(manifest)
        # Another tree on purpose: a recall A/B over one frozen corpus.
        self.assertEqual(sweep._asking_tree(manifest, tree=TREE + '/.'), TREE)


class TestCliKeepsArmsCodeOnly(unittest.TestCase):

    def test_in_process_overrides_are_refused_on_arms(self):
        for argv in (['--s1e', 'x.txt'], ['--interaction-off', 's1e_gist'],
                     ['--pooled'], ['--seed-pack', 'p.py']):
            with mock.patch.object(sys, 'argv', ['build_corpus.py'] + argv), \
                    mock.patch.object(build_corpus, 'build_arm_corpora') as built, \
                    mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    build_corpus.main()
                built.assert_not_called()

    def test_arm_flags_are_refused_in_process(self):
        with mock.patch.object(sys, 'argv', ['build_corpus.py', '--legacy-inprocess',
                                             '--tree', TREE]), \
                mock.patch.object(build_corpus, 'build_corpus') as built, \
                mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                build_corpus.main()
            built.assert_not_called()


class TestArmBuildAndAsk(unittest.TestCase):
    """One synthetic item through a keyless arm of this checkout, then its
    question on a clone. Keyless, nothing encodes: every turn stays pending,
    which is what the stranded labels must say."""

    @classmethod
    def setUpClass(cls):
        cls.env = dict(os.environ)
        cls.root = tempfile.mkdtemp(prefix='longmem-arms-test-')
        cls.oracle = os.path.join(cls.root, 'oracle.json')
        with open(cls.oracle, 'w') as f:
            json.dump([_item()], f)
        cls._corpus_root = mock.patch.object(
            corpus, 'CORPUS_ROOT', os.path.join(cls.root, 'corpora'))
        cls._corpus_root.start()
        try:
            cls.hashes = build_corpus.build_arm_corpora(
                [TREE], 0, 42, cls.oracle, 4, 'test', qids='synthetic_arm1',
                keyless=True)
        except BaseException:
            cls.tearDownClass()
            raise
        finally:
            # The build's measurement opens brains in process, which pins
            # the env; put it back before any test runs.
            os.environ.clear()
            os.environ.update(cls.env)
        cls.manifest = corpus.load_manifest(cls.hashes[TREE])
        cls.entry = cls.manifest['items'][0]

    @classmethod
    def tearDownClass(cls):
        cls._corpus_root.stop()
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_corpus_is_addressed_by_the_trees_code(self):
        config = self.manifest['config']
        self.assertEqual(config['harness'], build_corpus.ARM_HARNESS)
        self.assertEqual(config['tree'], tree_digest(TREE))
        self.assertTrue(config['keyless'])
        self.assertEqual(self.manifest['tree_path'], TREE)

    def test_one_brain_session_per_haystack_session_in_date_order(self):
        sessions = self.entry['sessions']
        self.assertEqual([s['listed_index'] for s in sessions], [1, 0])
        self.assertEqual([s['user_turns'] for s in sessions], [6, 3])
        self.assertEqual(len({s['session_id'] for s in sessions}), 2)
        self.assertEqual(self.entry['hook_failures'], 0)
        db = os.path.join(self.entry['brain_dir'], 'brain_logs.db')
        conn = sqlite3.connect('file:%s?mode=ro&immutable=1' % db, uri=True)
        rows = conn.execute(
            "SELECT session_id, summary FROM trace_events WHERE scale='s0' "
            "AND ref_type='user_message' ORDER BY created_at").fetchall()
        conn.close()
        self.assertEqual([r[0] for r in rows],
                         [sessions[0]['session_id']] * 6 + [sessions[1]['session_id']] * 3)
        framed = corpus.frame_user_msg('user turn 0', '2023/05/20 (Sat) 08:00',
                                       self.entry['counterpart'])
        self.assertTrue(rows[0][1].startswith(framed[:60]))

    def test_unencoded_turns_are_labelled_not_dropped(self):
        self.assertEqual([s['stranded'] for s in self.entry['sessions']], [6, 3])
        self.assertEqual(self.entry['stranded_turns'], 9)
        self.assertEqual(self.entry['evidence_turns'], 1)
        self.assertTrue(self.entry['evidence_only_stranded'])

    def test_a_question_is_asked_on_a_clone_never_the_frozen_item(self):
        frozen = os.path.join(self.entry['brain_dir'], 'brain.db')
        before = os.stat(frozen).st_mtime_ns
        answer = {'hypothesis': 'turn 5', 'abstained': False,
                  'has_context': False, 'elapsed_ms': 0}
        root = os.path.join(self.root, 'ask')
        with mock.patch.object(sweep, 'answer_question', return_value=answer):
            qr, ar, _, data = sweep._ask_on_clone(TREE, self.entry, root, None)
        self.assertEqual(os.stat(frozen).st_mtime_ns, before)
        self.assertNotEqual(os.path.realpath(data),
                            os.path.realpath(self.entry['brain_dir']))
        self.assertEqual(qr['harness_error'], '')
        conn = sqlite3.connect('file:%s?mode=ro&immutable=1'
                               % os.path.join(data, 'brain_logs.db'), uri=True)
        said = conn.execute(
            "SELECT ref_type, summary FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type IN ('user_message', "
            "'assistant_message') ORDER BY created_at",
            (qr['query_session_id'],)).fetchall()
        conn.close()
        self.assertEqual([r[0] for r in said], ['user_message', 'assistant_message'])
        self.assertIn('What was said in the last turn?', said[0][1])
        self.assertEqual(said[1][1], 'turn 5')


if __name__ == '__main__':
    unittest.main()

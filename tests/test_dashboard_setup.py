"""Dashboard /setup backend — write_api_key / api_key_present.

The setup form is the no-terminal onboarding path for the API key
(keyless boot notices point at http://localhost:47303/setup). Contract:
one user-config file (~/.config/brain/env), mode 600, atomic replace,
value never echoed; the dashboard's passive-observer invariant (never
writes the DBs) is untouched.
"""
import os
import unittest
import tempfile

from dashboard.queries.setup import write_api_key, api_key_present


class TestWriteApiKey(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.env = os.path.join(self._tmp.name, 'brain', 'env')

    def tearDown(self):
        self._tmp.cleanup()

    def test_fresh_write_creates_file_mode_600(self):
        ok, msg = write_api_key(self.env, 'sk-ant-test-abc123')
        self.assertTrue(ok)
        self.assertNotIn('sk-ant', msg)          # never echo the key
        with open(self.env) as f:
            self.assertIn('ANTHROPIC_API_KEY=sk-ant-test-abc123\n', f.read())
        self.assertEqual(os.stat(self.env).st_mode & 0o777, 0o600)

    def test_replaces_existing_key_preserves_other_vars(self):
        os.makedirs(os.path.dirname(self.env))
        with open(self.env, 'w') as f:
            f.write('BRAIN_OPERATOR_NAME=Ada\n'
                    'ANTHROPIC_API_KEY=sk-ant-OLD\n'
                    'DASHBOARD_PORT=47303\n')
        ok, _ = write_api_key(self.env, 'sk-ant-NEW-key-value')
        self.assertTrue(ok)
        content = open(self.env).read()
        self.assertIn('BRAIN_OPERATOR_NAME=Ada\n', content)
        self.assertIn('DASHBOARD_PORT=47303\n', content)
        self.assertIn('ANTHROPIC_API_KEY=sk-ant-NEW-key-value\n', content)
        self.assertNotIn('sk-ant-OLD', content)
        self.assertEqual(content.count('ANTHROPIC_API_KEY='), 1)

    def test_rejects_non_sk_and_whitespace_and_short(self):
        for bad in ('changeme', 'sk-a', 'sk-ant with space', '',
                    'sk-ant-tab\tchar'):
            ok, msg = write_api_key(self.env, bad)
            self.assertFalse(ok, 'accepted %r' % bad)
            if bad:  # '' is a substring of everything
                self.assertNotIn(bad, msg)       # never echo the attempt
        self.assertFalse(os.path.exists(self.env))  # nothing written

    def test_api_key_present(self):
        self.assertFalse(api_key_present(self.env))
        write_api_key(self.env, 'sk-ant-test-abc123')
        self.assertTrue(api_key_present(self.env))


if __name__ == '__main__':
    unittest.main()

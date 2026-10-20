"""loud_truncation — the shared loud-cap primitives, tested directly. The
channel render tests exercise them only through their callers; these pin the
algorithm itself, so a caller can't quietly re-inline a variant."""
import unittest

from servers.loud_truncation import (cap_list_loud, cap_text_at_boundary,
                                     cap_text_loud, compose_block_loud,
                                     looks_truncated)


class TestComposeBlockLoud(unittest.TestCase):

    def test_all_fit(self):
        self.assertEqual(compose_block_loud(['a', 'bb', 'ccc'], str, cap=100),
                         ('a\n\nbb\n\nccc', 3, 0))

    def test_overflow_counts_every_item_past_the_cut(self):
        items = ['x' * 30] * 6                      # 32 per item incl. separator
        body, kept, dropped = compose_block_loud(items, str, cap=100)
        self.assertEqual((kept, dropped), (3, 3))   # 96 fits, the 4th would be 128
        self.assertLess(len(body), 100)

    def test_single_oversize_item_is_kept(self):
        body, kept, dropped = compose_block_loud(['y' * 500], str, cap=10)
        self.assertEqual((body, kept, dropped), ('y' * 500, 1, 0))

    def test_empty(self):
        self.assertEqual(compose_block_loud([], str, cap=10), ('', 0, 0))

    def test_reserved_spends_the_budget(self):
        items = ['x' * 30] * 3
        _, kept_free, _ = compose_block_loud(items, str, cap=100)
        _, kept, dropped = compose_block_loud(items, str, cap=100, reserved=30)
        self.assertEqual(kept_free, 3)
        self.assertEqual((kept, dropped), (2, 1))   # 30+32+32 fits, +32 doesn't

    def test_items_past_the_cut_are_never_rendered(self):
        calls = []

        def render(s):
            calls.append(s)
            return s
        compose_block_loud(['x' * 30] * 6, render, cap=100)
        self.assertEqual(len(calls), 4)             # 3 kept + the one that overflowed

    def test_rendered_items_are_stripped(self):
        body, _, _ = compose_block_loud(['  a  ', '\nb\n'], str, cap=100)
        self.assertEqual(body, 'a\n\nb')


class TestCapPrimitives(unittest.TestCase):

    def test_cap_text_loud_marker_names_dropped_count(self):
        self.assertEqual(cap_text_loud('abcdef', 3), 'abc …[+3 chars truncated]')
        self.assertEqual(cap_text_loud('abc', 3), 'abc')
        self.assertEqual(cap_text_loud(None, 3), '')

    def test_cap_text_loud_bare_marker(self):
        # A hard cut at the limit, marked without a count.
        self.assertEqual(cap_text_loud('abcdef', 3, marker='…'), 'abc …')
        self.assertTrue(looks_truncated(cap_text_loud('abcdef', 3, marker='…')))

    def test_cap_list_loud_stays_a_list(self):
        self.assertEqual(cap_list_loud([1, 2, 3], 2), [1, 2, '…[+1 more truncated]'])
        self.assertEqual(cap_list_loud([1], 2), [1])


class TestBoundaryCutBareMarker(unittest.TestCase):
    """A marker without %d is appended as is — the tool moment's bare '…'."""

    def test_bare_marker_names_no_count(self):
        out = cap_text_at_boundary('Sentence one. ' * 40, 300, marker='…')
        self.assertTrue(out.endswith('. …'))
        self.assertNotIn('+', out)
        self.assertTrue(looks_truncated(out))


class TestLooksTruncated(unittest.TestCase):
    """The contract read backwards — what a reader copied out of a cut view
    looks like. Consumed by the swap write boundary (contract.apply_swaps),
    where a marker tail explains a miss the generic advice misdiagnoses."""

    def test_every_family_is_recognized(self):
        from servers.contract import CONTENT_CUT_MARKER, truncate_marked
        self.assertTrue(looks_truncated(truncate_marked('x' * 400, 300)))
        self.assertTrue(looks_truncated(cap_text_loud('y' * 400, 300)))
        self.assertTrue(looks_truncated(cap_text_at_boundary(
            'Sentence one. ' * 40, 300, marker=CONTENT_CUT_MARKER)))

    def test_uncut_text_is_not_flagged(self):
        self.assertFalse(looks_truncated('a whole value, nothing dropped'))
        self.assertFalse(looks_truncated(''))
        self.assertFalse(looks_truncated(None))

    def test_ellipsis_inside_prose_is_not_a_marker(self):
        """Only a tail counts — mid-text ellipsis is the author's own."""
        self.assertFalse(looks_truncated('he trailed off… then finished'))


if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.trie_grammar_backend import TrieGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")


class TestTrieGrammar(CustomTestCase):
    def setUp(self):
        self.backend = TrieGrammarBackend(vocab_size=16)

    def tearDown(self):
        self.backend.executor.shutdown(wait=True)

    def test_shared_prefix_termination_and_rollback(self):
        grammar = self.backend.dispatch_trie("[[1,2],[1,3],[4]]")
        self.assertEqual(grammar.allowed_tokens, [1, 4])
        grammar.accept_token(1)
        self.assertEqual(grammar.allowed_tokens, [2, 3])
        grammar.accept_token(2)
        self.assertTrue(grammar.is_terminated())
        grammar.rollback(1)
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar.allowed_tokens, [2, 3])

    def test_copy_and_fork_keep_state_isolated(self):
        grammar = self.backend.dispatch_trie("[[1,2],[1,3]]")
        grammar.accept_token(1)
        branch = grammar.fork()
        branch.accept_token(2)
        self.assertTrue(branch.is_terminated())
        self.assertEqual(grammar.allowed_tokens, [2, 3])
        fresh = grammar.copy()
        self.assertEqual(fresh.allowed_tokens, [1])

    def test_cpu_mask_allows_only_children(self):
        grammar = self.backend.dispatch_trie("[[1,2],[3]]")
        mask = grammar.allocate_vocab_mask(vocab_size=8, batch_size=1, device="cpu")
        grammar.fill_vocab_mask(mask, 0)
        logits = torch.arange(8, dtype=torch.float).unsqueeze(0)
        grammar.apply_vocab_mask(logits, mask)
        self.assertTrue(torch.isneginf(logits[0, 0]))
        self.assertEqual(logits[0, 1].item(), 1.0)
        self.assertEqual(logits[0, 3].item(), 3.0)
        self.assertTrue(torch.isneginf(logits[0, 7]))

    def test_invalid_path_returns_invalid_grammar(self):
        grammar = self.backend.dispatch_trie("[[16]]")
        self.assertIsInstance(grammar, InvalidGrammarObject)

    def test_prefix_path_is_rejected(self):
        grammar = self.backend.dispatch_trie("[[1],[1,2]]")
        self.assertIsInstance(grammar, InvalidGrammarObject)


if __name__ == "__main__":
    unittest.main()

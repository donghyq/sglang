import unittest

import torch

from sglang.srt.constrained.trie_beam_search import (
    TrieBeam,
    TrieBeamGroup,
    build_trie_beam_candidate_layout,
    constrained_beam_search_step,
    trie_constrained_beam_topk,
    trie_constrained_beam_topk_from_layout,
)
from sglang.srt.constrained.trie_grammar_backend import TrieGrammar, TrieGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")


class TestTrieBeamSearch(CustomTestCase):
    def setUp(self):
        self.backend = TrieGrammarBackend(vocab_size=8)

    def tearDown(self):
        self.backend.executor.shutdown(wait=True)

    def test_beam_expands_only_legal_children_and_ranks_paths(self):
        root = self.backend.dispatch_trie("[[1,2],[1,3],[4]]")
        beams = [TrieBeam([], 0.0, root)]
        first = constrained_beam_search_step(
            beams, torch.tensor([[0.0, 4.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0]]), 2
        )
        self.assertEqual([beam.tokens for beam in first], [[1], [4]])
        self.assertTrue(first[1].finished)
        second = constrained_beam_search_step(
            first,
            torch.tensor([[0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 8]),
            2,
        )
        self.assertEqual(second[0].tokens, [1, 3])
        self.assertTrue(second[0].finished)

    def test_group_tracks_parent_mapping_and_completed_results(self):
        root = self.backend.dispatch_trie("[[1,2],[1,3],[4]]")
        group = TrieBeamGroup(root, width=2, num_return_sequences=2)

        active = group.advance(
            torch.tensor([[0.0, 4.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0]])
        )
        self.assertEqual([beam.tokens for beam in active], [[1]])
        self.assertEqual(active[0].parent_id, 0)
        self.assertEqual(group.last_pruned_parent_ids, [])
        self.assertEqual([beam.tokens for beam in group.results], [[4]])

        active = group.advance(
            torch.tensor([[0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0]])
        )
        self.assertEqual(active, [])
        self.assertTrue(group.is_finished)
        self.assertEqual([beam.tokens for beam in group.results], [[1, 3], [4]])
        self.assertEqual([beam.parent_id for beam in group.results], [1, 0])

    def test_group_globally_prunes_multiple_parents_with_stable_ties(self):
        root = self.backend.dispatch_trie(
            "[[1,3,7],[1,4,7],[2,5,7],[2,6,7]]"
        )
        group = TrieBeamGroup(root, width=2)
        group.advance(torch.tensor([[0.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

        active = group.advance(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            )
        )
        self.assertEqual([beam.tokens for beam in active], [[1, 3], [1, 4]])
        self.assertEqual([beam.parent_id for beam in active], [1, 1])
        self.assertEqual(
            [beam.tokens for beam in group.last_pruned], [[2, 5], [2, 6]]
        )
        self.assertEqual(group.last_pruned_parent_ids, [2])
        self.assertEqual(group.results, [])
        self.assertEqual(group.last_slot_transition.slot_reuses, {3: 1})
        self.assertEqual(group.last_slot_transition.slot_forks, {4: 1})
        self.assertEqual(group.last_slot_transition.parent_ids_to_release, [2])

    def test_device_topk_group_path_matches_reference_branch_lifecycle(self):
        trie = "[[1,3,7],[1,4,7],[2,5,7],[2,6,7]]"
        reference = TrieBeamGroup(
            self.backend.dispatch_trie(trie), width=2, num_return_sequences=2
        )
        accelerated = TrieBeamGroup(
            self.backend.dispatch_trie(trie), width=2, num_return_sequences=2
        )
        first_logits = torch.tensor(
            [[0.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        second_logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        reference.advance(first_logits)
        accelerated.advance_with_constrained_topk(first_logits)
        reference.advance(second_logits)
        accelerated.advance_with_constrained_topk(second_logits)

        self.assertEqual(
            [beam.tokens for beam in accelerated.active],
            [beam.tokens for beam in reference.active],
        )
        self.assertEqual(
            [beam.parent_id for beam in accelerated.active],
            [beam.parent_id for beam in reference.active],
        )
        self.assertEqual(
            [beam.tokens for beam in accelerated.last_pruned],
            [beam.tokens for beam in reference.last_pruned],
        )
        self.assertEqual(
            accelerated.last_slot_transition, reference.last_slot_transition
        )

    def test_device_topk_group_path_accepts_prebuilt_layout(self):
        group = TrieBeamGroup(self.backend.dispatch_trie("[[1,2],[1,3],[4]]"), width=2)
        layout = build_trie_beam_candidate_layout(
            [[1, 4]], [[False, True]], device="cpu"
        )

        group.advance_with_constrained_topk(
            torch.tensor([[0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]]),
            layout,
        )

        self.assertEqual([beam.tokens for beam in group.active], [[1]])
        self.assertEqual([beam.tokens for beam in group.results], [[4]])

    def test_prepared_candidate_layout_matches_late_preparation(self):
        trie = "[[1,3],[1,4],[2,5]]"
        prepared = TrieBeamGroup(self.backend.dispatch_trie(trie), width=2)
        late = TrieBeamGroup(self.backend.dispatch_trie(trie), width=2)
        logits = torch.tensor(
            [[0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        layout = prepared.prepare_candidate_layout("cpu")
        prepared.advance_with_constrained_topk(logits, layout)
        late.advance_with_constrained_topk(logits)

        self.assertEqual(
            [beam.tokens for beam in prepared.active],
            [beam.tokens for beam in late.active],
        )
        self.assertEqual(
            [beam.tokens for beam in prepared.results],
            [beam.tokens for beam in late.results],
        )

    def test_group_keeps_only_requested_completed_results_and_releases_parent(self):
        root = self.backend.dispatch_trie("[[1],[2],[3]]")
        group = TrieBeamGroup(root, width=2, num_return_sequences=2)

        active = group.advance(
            torch.tensor([[0.0, 1.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
        )

        self.assertEqual(active, [])
        self.assertEqual([beam.tokens for beam in group.last_completed], [[1], [2], [3]])
        self.assertEqual([beam.tokens for beam in group.results], [[2], [3]])
        self.assertEqual(len(group.completed), 2)
        self.assertEqual(group.last_slot_transition.slot_reuses, {})
        self.assertEqual(group.last_slot_transition.slot_forks, {})
        self.assertEqual(group.last_slot_transition.parent_ids_to_release, [0])

    def test_group_finishes_when_no_active_branch_has_a_child(self):
        grammar = TrieGrammar(({},), (False,))
        group = TrieBeamGroup(grammar, width=1)
        self.assertEqual(group.advance(torch.zeros((1, 8))), [])
        self.assertTrue(group.is_finished)
        self.assertEqual(group.results, [])

    def test_decode_side_handoff_reconstructs_root_beam_state(self):
        group = TrieBeamGroup(
            self.backend.dispatch_trie("[[1,2],[3]]"), width=2
        )

        active = group.advance_from_handoff_candidates(
            [(1, -0.2, False), (3, -0.4, True)]
        )

        self.assertEqual([beam.tokens for beam in active], [[1]])
        self.assertEqual(active[0].parent_id, 0)
        self.assertEqual([beam.tokens for beam in group.results], [[3]])
        self.assertEqual(group.last_slot_transition.slot_reuses, {1: 0})

    def test_decode_side_handoff_rejects_inconsistent_terminal_flag(self):
        group = TrieBeamGroup(self.backend.dispatch_trie("[[1]]"), width=1)

        with self.assertRaisesRegex(ValueError, "terminal flag"):
            group.advance_from_handoff_candidates([(1, 0.0, False)])

    def test_decode_side_handoff_releases_root_when_all_candidates_terminate(self):
        group = TrieBeamGroup(
            self.backend.dispatch_trie("[[1],[2]]"),
            width=2,
            num_return_sequences=1,
        )

        active = group.advance_from_handoff_candidates(
            [(1, -0.1, True), (2, -0.2, True)]
        )

        self.assertEqual(active, [])
        self.assertTrue(group.is_finished)
        self.assertEqual([beam.tokens for beam in group.results], [[1]])
        self.assertEqual(group.last_slot_transition.slot_reuses, {})
        self.assertEqual(group.last_slot_transition.slot_forks, {})
        self.assertEqual(group.last_slot_transition.parent_ids_to_release, [0])

    def test_group_rejects_invalid_return_count(self):
        root = self.backend.dispatch_trie("[[1]]")
        with self.assertRaisesRegex(ValueError, "num_return_sequences"):
            TrieBeamGroup(root, width=2, num_return_sequences=3)

    def test_gpu_style_child_gather_keeps_terminal_and_active_paths_separate(self):
        selection = trie_constrained_beam_topk(
            logits=torch.tensor(
                [
                    [0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 4.0],
                ]
            ),
            parent_scores=torch.tensor([0.0, 1.0]),
            allowed_tokens=[[1, 4], [3, 6]],
            terminal_children=[[False, True], [False, False]],
            width=2,
            num_completed=1,
        )

        self.assertEqual(selection.active_parent_indices.tolist(), [1, 1])
        self.assertEqual(selection.active_token_ids.tolist(), [3, 6])
        self.assertEqual(selection.completed_parent_indices.tolist(), [0])
        self.assertEqual(selection.completed_token_ids.tolist(), [4])
        self.assertEqual(selection.pruned_parent_indices.tolist(), [0])
        self.assertEqual(selection.pruned_token_ids.tolist(), [1])

    def test_gpu_style_child_gather_keeps_equal_scores_in_trie_order(self):
        selection = trie_constrained_beam_topk(
            logits=torch.zeros((1, 5)),
            parent_scores=torch.zeros(1),
            allowed_tokens=[[3, 1, 2]],
            terminal_children=[[False, False, False]],
            width=2,
            num_completed=1,
        )

        self.assertEqual(selection.active_token_ids.tolist(), [3, 1])
        self.assertEqual(selection.pruned_token_ids.tolist(), [2])

    def test_prebuilt_candidate_layout_matches_one_shot_selection(self):
        logits = torch.tensor(
            [[0.0, 4.0, 0.0, 2.0], [0.0, 0.0, 3.0, 5.0]]
        )
        parent_scores = torch.tensor([0.0, 1.0])
        layout = build_trie_beam_candidate_layout(
            [[1, 3], [2, 3]], [[False, True], [False, False]], device="cpu"
        )
        from_layout = trie_constrained_beam_topk_from_layout(
            logits, parent_scores, layout, width=2, num_completed=1
        )
        one_shot = trie_constrained_beam_topk(
            logits,
            parent_scores,
            [[1, 3], [2, 3]],
            [[False, True], [False, False]],
            width=2,
            num_completed=1,
        )

        self.assertEqual(
            from_layout.active_parent_indices.tolist(),
            one_shot.active_parent_indices.tolist(),
        )
        self.assertEqual(
            from_layout.active_token_ids.tolist(), one_shot.active_token_ids.tolist()
        )
        self.assertEqual(
            from_layout.completed_token_ids.tolist(),
            one_shot.completed_token_ids.tolist(),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_child_gather_and_global_sort_stay_on_device(self):
        selection = trie_constrained_beam_topk(
            logits=torch.tensor(
                [
                    [0.0, 2.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 4.0, 3.0],
                ],
                device="cuda",
            ),
            parent_scores=torch.tensor([0.0, 1.0], device="cuda"),
            allowed_tokens=[[1, 4], [3, 4]],
            terminal_children=[[False, True], [False, False]],
            width=2,
            num_completed=1,
        )

        self.assertEqual(selection.active_parent_indices.device.type, "cuda")
        self.assertEqual(selection.active_scores.device.type, "cuda")
        self.assertEqual(selection.active_token_ids.cpu().tolist(), [3, 4])
        self.assertEqual(selection.completed_token_ids.cpu().tolist(), [4])

if __name__ == "__main__":
    unittest.main()

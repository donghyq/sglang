# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fixed-width constrained beam primitives for token-id tries.

This module deliberately keeps beam expansion independent from the scheduler. It
provides a deterministic reference implementation for the future batched GPU
beam runner. KV page ownership lives in the paged allocator so runtime code has
one authoritative source of truth for allocation and reclamation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import torch

from sglang.srt.constrained.trie_grammar_backend import TrieGrammar


@dataclass
class TrieBeam:
    tokens: List[int]
    score: float
    grammar: TrieGrammar
    finished: bool = False
    branch_id: int = 0
    parent_id: int | None = None

    def fork(self) -> "TrieBeam":
        return TrieBeam(
            tokens=list(self.tokens),
            score=self.score,
            grammar=self.grammar.fork(),
            finished=self.finished,
            branch_id=self.branch_id,
            parent_id=self.parent_id,
        )


@dataclass(frozen=True)
class TrieBeamSlotTransition:
    """Describe the request-slot actions required after one beam step.

    This is intentionally a scheduler-independent plan.  The scheduler owns
    the actual request slots and KV pages, while this plan identifies which
    selected child may reuse its parent's slot, which children need a forked
    slot, and which parent slots have no active descendant.
    """

    slot_reuses: Dict[int, int]
    slot_forks: Dict[int, int]
    parent_ids_to_release: List[int]


@dataclass(frozen=True)
class TrieBeamTopK:
    """GPU-resident result of one trie-constrained beam selection.

    The tensors identify a child by its row in the input beam batch and its
    selected token.  Keeping this representation tensor-based makes the
    expensive work -- child-logit gathering and global score ordering -- run
    on the device.  The scheduler may materialize only the surviving branches
    when it performs request-slot and KV-page lifecycle transitions.
    """

    active_parent_indices: torch.Tensor
    active_token_ids: torch.Tensor
    active_scores: torch.Tensor
    completed_parent_indices: torch.Tensor
    completed_token_ids: torch.Tensor
    completed_scores: torch.Tensor
    pruned_parent_indices: torch.Tensor
    pruned_token_ids: torch.Tensor
    pruned_scores: torch.Tensor


@dataclass(frozen=True)
class TrieBeamCandidateLayout:
    """Compressed Trie child metadata for one active Beam set.

    The layout may be prepared while the scheduler constructs the next decode
    batch, then reused on the forward device without rebuilding a dense
    vocabulary mask.  Each element describes one legal ``(parent, token)``
    child edge.
    """

    parent_indices: torch.Tensor
    token_ids: torch.Tensor
    is_terminal: torch.Tensor


@dataclass
class TrieBeamRuntimeBranch:
    """Scheduler-owned resources associated with one active beam branch."""

    req: Any
    kv_indices: torch.Tensor
    # ``kv_indices`` is the complete logical attention mapping.  Its leading
    # part can be protected by the radix cache, so it must never be registered
    # with (or released by) the Beam ownership tracker.
    beam_owned_kv_indices: torch.Tensor


class TrieBeamRuntime:
    """Apply Beam slot transitions to request-slot and KV-page ownership.

    The scheduler remains responsible for creating child requests and placing
    them in decode batches. Only page-aligned prefixes may be shared; any
    partial tail must already be private before this coordinator is called.
    """

    def __init__(self, req_to_token_pool: Any, kv_allocator: Any, page_size: int) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be positive, got {page_size}.")
        self.req_to_token_pool = req_to_token_pool
        self.kv_allocator = kv_allocator
        self.page_size = page_size
        self.branches: Dict[int, TrieBeamRuntimeBranch] = {}

    def register_root(
        self,
        branch_id: int,
        req: Any,
        kv_indices: torch.Tensor,
        beam_owned_kv_indices: torch.Tensor | None = None,
    ) -> None:
        """Register the root branch after its prefill KV pages are allocated.

        A prefix-cache hit remains owned by the cache.  The Beam runtime only
        takes ownership of the uncached suffix, while retaining the complete
        mapping for attention and copy-on-write decisions.
        """
        if branch_id in self.branches:
            raise ValueError(f"Beam branch {branch_id} is already registered.")
        if beam_owned_kv_indices is None:
            beam_owned_kv_indices = kv_indices
        self.kv_allocator.register_beam_pages(beam_owned_kv_indices)
        self.branches[branch_id] = TrieBeamRuntimeBranch(
            req, kv_indices, beam_owned_kv_indices
        )

    def apply_transition(
        self,
        transition: TrieBeamSlotTransition,
        create_child_req: Callable[[TrieBeam], Any],
        active_children: Sequence[TrieBeam],
        shared_prefixes: Dict[int, torch.Tensor],
        create_child_kv_indices: Callable[
            [TrieBeamRuntimeBranch, TrieBeam, torch.Tensor], torch.Tensor
        ],
        release_unregistered_child_kv_indices: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        """Apply one selected Beam transition to request and KV resources.

        ``create_child_kv_indices`` must return the complete child mapping. It
        shares the supplied page-aligned prefix and provides freshly allocated,
        physically copied private pages for the remaining tail.  The runtime
        verifies this ownership boundary before publishing the child branch.

        Before a child mapping is registered, its private tail is owned by
        ``create_child_kv_indices``.  If any later operation fails,
        ``release_unregistered_child_kv_indices`` releases those prepared but
        still unregistered tails.  Registered tails are instead released by
        the allocator, preserving a single ownership rule for every page.
        """
        children_by_id = {child.branch_id: child for child in active_children}
        expected_ids = set(transition.slot_reuses) | set(transition.slot_forks)
        if set(children_by_id) != expected_ids:
            raise ValueError("Active children do not match the slot transition.")
        source = dict(self.branches)
        parent_ids = set(transition.slot_reuses.values()) | set(
            transition.slot_forks.values()
        ) | set(transition.parent_ids_to_release)
        missing = parent_ids - set(source)
        if missing:
            raise ValueError(f"Unknown parent beam branches: {sorted(missing)}.")

        next_branches: Dict[int, TrieBeamRuntimeBranch] = {}
        for child_id, parent_id in transition.slot_reuses.items():
            next_branches[child_id] = source[parent_id]

        forked_by_parent: Dict[int, List[int]] = {}
        for child_id, parent_id in transition.slot_forks.items():
            forked_by_parent.setdefault(parent_id, []).append(child_id)

        for parent_id, child_ids in forked_by_parent.items():
            prefix = shared_prefixes.get(parent_id)
            if prefix is None:
                raise ValueError(f"Missing shared KV prefix for branch {parent_id}.")
            if prefix.numel() % self.page_size:
                raise ValueError("A shared beam KV prefix must end at a page boundary.")
            parent = source[parent_id]
            if prefix.numel() and not torch.equal(
                parent.kv_indices[: prefix.numel()], prefix
            ):
                raise ValueError("The shared KV prefix is not a parent mapping prefix.")
            child_beams = [children_by_id[child_id] for child_id in child_ids]
            child_reqs = []
            child_kv_indices = []
            slots_forked = False
            shared_prefix_forked = False
            registered_private_tails = []
            try:
                for child in child_beams:
                    child_reqs.append(create_child_req(child))
                for child in child_beams:
                    child_kv_indices.append(
                        create_child_kv_indices(parent, child, prefix)
                    )
                child_owned_kv_indices = []
                owned_shared_prefix = parent.beam_owned_kv_indices[
                    : max(0, prefix.numel() - (parent.kv_indices.numel() - parent.beam_owned_kv_indices.numel()))
                ]
                for child_kv in child_kv_indices:
                    child_owned_kv_indices.append(
                        torch.cat((owned_shared_prefix, child_kv[prefix.numel() :]))
                    )
                for child_kv in child_kv_indices:
                    self._validate_child_kv_indices(
                        parent.kv_indices, prefix, child_kv
                    )
                self.req_to_token_pool.fork_beam_slots_from_prefix(
                    parent.req, child_reqs, prefix.numel(), self.page_size
                )
                slots_forked = True
                self.kv_allocator.fork_shared_prefix(
                    owned_shared_prefix, len(child_ids)
                )
                shared_prefix_forked = True
                for child_owned_kv in child_owned_kv_indices:
                    private_tail = child_owned_kv[owned_shared_prefix.numel() :]
                    self.kv_allocator.register_beam_pages(private_tail)
                    registered_private_tails.append(private_tail)
            except Exception:
                for private_tail in reversed(registered_private_tails):
                    self.kv_allocator.release_beam_suffix(private_tail)
                if shared_prefix_forked:
                    for _ in child_ids:
                        self.kv_allocator.release_beam_suffix(owned_shared_prefix)
                if release_unregistered_child_kv_indices is not None:
                    for child_kv in child_kv_indices[len(registered_private_tails) :]:
                        # The shared prefix can contain radix-cache-owned
                        # pages.  Only the just-created private tail has not
                        # been registered anywhere and may be returned here.
                        release_unregistered_child_kv_indices(
                            child_kv[prefix.numel() :]
                        )
                if slots_forked:
                    for child_req in child_reqs:
                        self.req_to_token_pool.free(child_req)
                raise
            for child_id, child_req, child_kv, child_owned_kv in zip(
                child_ids, child_reqs, child_kv_indices, child_owned_kv_indices
            ):
                self.req_to_token_pool.req_to_token[
                    child_req.req_pool_idx, : child_kv.numel()
                ] = child_kv
                next_branches[child_id] = TrieBeamRuntimeBranch(
                    child_req, child_kv, child_owned_kv
                )

        for parent_id in transition.parent_ids_to_release:
            parent = source[parent_id]
            self.kv_allocator.release_beam_suffix(parent.beam_owned_kv_indices)
            self.req_to_token_pool.free(parent.req)
        self.branches = next_branches

    def _validate_child_kv_indices(
        self,
        parent_kv_indices: torch.Tensor,
        shared_prefix: torch.Tensor,
        child_kv_indices: torch.Tensor,
    ) -> None:
        """Reject incomplete mappings or a shared writable tail."""
        if child_kv_indices.ndim != 1:
            raise ValueError("A child beam KV mapping must be a rank-1 tensor.")
        if child_kv_indices.numel() != parent_kv_indices.numel():
            raise ValueError(
                "A forked beam must preserve the parent's complete KV mapping."
            )
        prefix_len = shared_prefix.numel()
        if prefix_len and not torch.equal(child_kv_indices[:prefix_len], shared_prefix):
            raise ValueError("A child KV mapping must start with the shared prefix.")
        parent_tail_pages = set(
            (parent_kv_indices[prefix_len:] // self.page_size).cpu().tolist()
        )
        child_tail_pages = set(
            (child_kv_indices[prefix_len:] // self.page_size).cpu().tolist()
        )
        if parent_tail_pages & child_tail_pages:
            raise ValueError(
                "A forked beam tail must use private KV pages; copy the tail before sharing."
            )

    def append_decode_kv_locations(
        self, branch_ids: Sequence[int], kv_locations: torch.Tensor
    ) -> None:
        """Record KV locations allocated for the next Decode write.

        ``prepare_for_decode`` allocates one physical KV location for every
        active branch.  These locations must become part of the branch mapping
        before the next transition, otherwise completion can only release the
        prefill mapping and leaks the Decode allocation.
        """
        if kv_locations.ndim != 1:
            raise ValueError("Decode KV locations must be a rank-1 tensor.")
        if len(branch_ids) != kv_locations.numel():
            raise ValueError(
                "The number of Decode KV locations must match the live branches."
            )
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("A Decode batch cannot contain a beam branch twice.")
        unknown_branch_ids = set(branch_ids) - set(self.branches)
        if unknown_branch_ids:
            raise ValueError(
                f"Unknown beam branches: {sorted(unknown_branch_ids)}."
            )

        locations_to_register = []
        for branch_id, kv_location in zip(branch_ids, kv_locations):
            branch = self.branches[branch_id]
            page_id = int(kv_location.item()) // self.page_size
            if page_id == 0:
                raise ValueError("KV page 0 cannot be owned by a beam branch.")
            branch_page_ids = set(
                (branch.beam_owned_kv_indices // self.page_size).cpu().tolist()
            )
            if page_id not in branch_page_ids:
                locations_to_register.append(kv_location)

        if locations_to_register:
            self.kv_allocator.register_beam_pages(torch.stack(locations_to_register))

        for branch_id, kv_location in zip(branch_ids, kv_locations):
            branch = self.branches[branch_id]
            branch.kv_indices = torch.cat((branch.kv_indices, kv_location.view(1)))
            if int(kv_location.item()) // self.page_size not in set(
                (branch.beam_owned_kv_indices // self.page_size).cpu().tolist()
            ):
                branch.beam_owned_kv_indices = torch.cat(
                    (branch.beam_owned_kv_indices, kv_location.view(1))
                )

    def release_all(self) -> None:
        """Release remaining active branches after completion or abort."""
        for branch in self.branches.values():
            self.kv_allocator.release_beam_suffix(branch.beam_owned_kv_indices)
            self.req_to_token_pool.free(branch.req)
        self.branches.clear()


class TrieBeamExecution:
    """Coordinate one Trie Beam group with its request and KV resources.

    This is the scheduler-facing boundary for the single-machine Beam path.
    It deliberately does not construct a ``ScheduleBatch``: the scheduler
    retains that responsibility, while this class makes one logits result map
    atomically to Beam selection and request/KV ownership transitions.
    """

    def __init__(
        self,
        group: "TrieBeamGroup",
        runtime: TrieBeamRuntime,
    ) -> None:
        self.group = group
        self.runtime = runtime

    @property
    def is_finished(self) -> bool:
        return self.group.is_finished

    @property
    def results(self) -> List[TrieBeam]:
        return self.group.results

    def register_root(
        self,
        req: Any,
        kv_indices: torch.Tensor,
        beam_owned_kv_indices: torch.Tensor | None = None,
    ) -> None:
        """Bind the root Beam branch to its completed prefill KV mapping."""
        if len(self.group.active) != 1 or self.group.active[0].branch_id != 0:
            raise ValueError(
                "A Trie Beam execution root must be registered before its first advance."
            )
        self.runtime.register_root(0, req, kv_indices, beam_owned_kv_indices)

    def advance(
        self,
        logits: torch.Tensor,
        create_child_req: Callable[[TrieBeam], Any],
        shared_prefixes: Dict[int, torch.Tensor],
        create_child_kv_indices: Callable[
            [TrieBeamRuntimeBranch, TrieBeam, torch.Tensor], torch.Tensor
        ],
        release_unregistered_child_kv_indices: Callable[[torch.Tensor], None] | None = None,
        candidate_layout: TrieBeamCandidateLayout | None = None,
    ) -> List[TrieBeam]:
        """Consume one logits matrix and commit the corresponding Beam transition.

        ``shared_prefixes`` is only required for parents that actually fork a
        surviving child.  A terminal or pruned child owns no request slot, so
        it never requires a separate KV mapping.
        """
        group_state = (
            self.group.active,
            self.group.completed,
            self.group.last_completed,
            self.group.last_pruned,
            self.group.last_pruned_parent_ids,
            self.group.last_slot_transition,
            self.group._next_branch_id,
        )
        active = self.group.advance_with_constrained_topk(logits, candidate_layout)
        try:
            self.runtime.apply_transition(
                self.group.last_slot_transition,
                create_child_req,
                active,
                shared_prefixes,
                create_child_kv_indices,
                release_unregistered_child_kv_indices,
            )
        except Exception:
            (
                self.group.active,
                self.group.completed,
                self.group.last_completed,
                self.group.last_pruned,
                self.group.last_pruned_parent_ids,
                self.group.last_slot_transition,
                self.group._next_branch_id,
            ) = group_state
            raise
        return active

    def release_all(self) -> None:
        """Release all active branches when a root request is aborted."""
        self.runtime.release_all()


def build_trie_beam_candidate_layout(
    allowed_tokens: Sequence[Sequence[int]],
    terminal_children: Sequence[Sequence[bool]],
    *,
    device: torch.device | str,
) -> TrieBeamCandidateLayout:
    """Pack variable-width Trie children into device-resident tensors.

    This host-side preparation is intentionally separated from score selection
    so future scheduler code can cache or overlap the transfer with a model
    forward.
    """
    if len(terminal_children) != len(allowed_tokens):
        raise ValueError("terminal_children must align with allowed_tokens.")

    flat_tokens: List[int] = []
    flat_parent_indices: List[int] = []
    flat_terminal: List[bool] = []
    for parent_index, (children, child_terminals) in enumerate(
        zip(allowed_tokens, terminal_children)
    ):
        if len(children) != len(child_terminals):
            raise ValueError(
                "Each terminal-child list must match its allowed-token list."
            )
        flat_tokens.extend(children)
        flat_parent_indices.extend([parent_index] * len(children))
        flat_terminal.extend(child_terminals)

    return TrieBeamCandidateLayout(
        parent_indices=torch.tensor(
            flat_parent_indices, dtype=torch.long, device=device
        ),
        token_ids=torch.tensor(flat_tokens, dtype=torch.long, device=device),
        is_terminal=torch.tensor(flat_terminal, dtype=torch.bool, device=device),
    )


def trie_constrained_beam_topk(
    logits: torch.Tensor,
    parent_scores: torch.Tensor,
    allowed_tokens: Sequence[Sequence[int]],
    terminal_children: Sequence[Sequence[bool]],
    width: int,
    num_completed: int,
) -> TrieBeamTopK:
    """Gather Trie children and select active/completed paths on the GPU.

    ``allowed_tokens`` and ``terminal_children`` are compact per-parent Trie
    child lists.  They are transferred as one flattened tensor instead of
    creating a vocabulary-sized mask for every branch.  Scores use the full
    vocabulary log-softmax, so hard constraint selection does not alter model
    likelihoods.  Equal scores retain Trie child order deterministically.
    """
    if width < 1:
        raise ValueError(f"width must be positive, got {width}.")
    if num_completed < 1:
        raise ValueError(
            f"num_completed must be positive, got {num_completed}."
        )
    if logits.ndim != 2 or logits.shape[0] != len(allowed_tokens):
        raise ValueError(
            "logits must have one row per allowed-token list; got "
            f"{tuple(logits.shape)} for {len(allowed_tokens)} lists."
        )
    if parent_scores.ndim != 1 or parent_scores.shape[0] != logits.shape[0]:
        raise ValueError(
            "parent_scores must have one value per logits row; got "
            f"{tuple(parent_scores.shape)} for {logits.shape[0]} rows."
        )

    layout = build_trie_beam_candidate_layout(
        allowed_tokens, terminal_children, device=logits.device
    )
    return trie_constrained_beam_topk_from_layout(
        logits, parent_scores, layout, width, num_completed
    )


def trie_constrained_beam_topk_from_layout(
    logits: torch.Tensor,
    parent_scores: torch.Tensor,
    layout: TrieBeamCandidateLayout,
    width: int,
    num_completed: int,
) -> TrieBeamTopK:
    """Select constrained Beam children from a prebuilt device layout."""
    if width < 1:
        raise ValueError(f"width must be positive, got {width}.")
    if num_completed < 1:
        raise ValueError(
            f"num_completed must be positive, got {num_completed}."
        )
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2, got {tuple(logits.shape)}.")
    if parent_scores.ndim != 1 or parent_scores.shape[0] != logits.shape[0]:
        raise ValueError(
            "parent_scores must have one value per logits row; got "
            f"{tuple(parent_scores.shape)} for {logits.shape[0]} rows."
        )
    if (
        layout.parent_indices.ndim != 1
        or layout.token_ids.ndim != 1
        or layout.is_terminal.ndim != 1
        or len(layout.parent_indices) != len(layout.token_ids)
        or len(layout.token_ids) != len(layout.is_terminal)
    ):
        raise ValueError("Trie candidate layout tensors must be aligned vectors.")
    if layout.parent_indices.device != logits.device:
        raise ValueError("Trie candidate layout must be on the logits device.")
    if len(layout.parent_indices) and (
        layout.parent_indices.min() < 0
        or layout.parent_indices.max() >= logits.shape[0]
        or layout.token_ids.min() < 0
        or layout.token_ids.max() >= logits.shape[1]
    ):
        raise ValueError("Trie candidate layout contains an out-of-range index.")

    device = logits.device
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_score = torch.empty(0, dtype=logits.dtype, device=device)
    if layout.token_ids.numel() == 0:
        return TrieBeamTopK(
            empty_long, empty_long, empty_score, empty_long, empty_long,
            empty_score, empty_long, empty_long, empty_score
        )

    parent_indices = layout.parent_indices
    token_ids = layout.token_ids
    is_terminal = layout.is_terminal
    # We only need the legal child logits.  Subtracting row-wise logsumexp is
    # mathematically equivalent to gathering from full log-softmax while
    # avoiding a materialized ``[active_beams, vocab_size]`` log-probability
    # tensor.
    scores = (
        parent_scores.to(device=device, dtype=logits.dtype)[parent_indices]
        + logits[parent_indices, token_ids]
        - torch.logsumexp(logits, dim=-1)[parent_indices]
    )
    # ``stable=True`` preserves parent/Trie insertion order for equal scores.
    ranked = torch.argsort(scores, descending=True, stable=True)

    def select(
        mask: torch.Tensor, limit: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = ranked[mask[ranked]][:limit]
        return parent_indices[indices], token_ids[indices], scores[indices]

    active_parent_indices, active_token_ids, active_scores = select(
        ~is_terminal, width
    )
    completed_parent_indices, completed_token_ids, completed_scores = select(
        is_terminal, num_completed
    )
    nonterminal_ranked = ranked[(~is_terminal)[ranked]]
    pruned_indices = nonterminal_ranked[width:]
    pruned_parent_indices = parent_indices[pruned_indices]
    pruned_token_ids = token_ids[pruned_indices]
    pruned_scores = scores[pruned_indices]
    return TrieBeamTopK(
        active_parent_indices, active_token_ids, active_scores,
        completed_parent_indices, completed_token_ids, completed_scores,
        pruned_parent_indices, pruned_token_ids, pruned_scores
    )


class TrieBeamGroup:
    """Own the lifecycle of one fixed-width, trie-constrained beam search.

    The group is deliberately scheduler-independent. It gives the future
    runtime a single source of truth for branch identities, parent relations,
    active slots and completed results before those branches own request-pool
    slots or KV pages. Terminal branches leave active immediately, so they
    cannot consume a padded decode slot or be expanded again.
    """

    def __init__(
        self, root_grammar: TrieGrammar, width: int, num_return_sequences: int = 1
    ) -> None:
        if width < 1:
            raise ValueError(f"width must be positive, got {width}.")
        if not 1 <= num_return_sequences <= width:
            raise ValueError(
                "num_return_sequences must be in [1, beam_width], got "
                f"{num_return_sequences} for beam_width={width}."
            )
        self.width = width
        self.num_return_sequences = num_return_sequences
        self.active = [TrieBeam([], 0.0, root_grammar, branch_id=0)]
        self.completed: List[TrieBeam] = []
        self.last_completed: List[TrieBeam] = []
        self.last_pruned: List[TrieBeam] = []
        self.last_pruned_parent_ids: List[int] = []
        self.last_slot_transition = TrieBeamSlotTransition({}, {}, [])
        self._next_branch_id = 1

    @property
    def is_finished(self) -> bool:
        return not self.active

    @property
    def results(self) -> List[TrieBeam]:
        """Return deterministic top completed paths without mutating state."""
        return sorted(self.completed, key=_beam_rank_key)[: self.num_return_sequences]

    def _record_completed(self, beam: TrieBeam) -> None:
        """Retain only output-relevant terminal paths.

        Completed paths no longer need a decode slot.  Keeping more than the
        requested return count would only retain Python-side result state; a
        lower-scoring completed path can never become better in a later step.
        """
        self.completed.append(beam)
        self.completed.sort(key=_beam_rank_key)
        del self.completed[self.num_return_sequences :]

    def advance_from_handoff_candidates(
        self, candidates: Sequence[tuple[int, float, bool]]
    ) -> List[TrieBeam]:
        """Commit the first Beam expansion prepared by a prefill worker.

        P/D disaggregation transfers prompt KV, not the prompt's full-vocabulary
        logits.  The prefill worker therefore performs the compact Trie child
        gather and transfers only selected legal children.  Decode reconstructs
        the normal Beam state from those candidates before its first Decode
        forward.  This keeps the transferred metadata bounded by Beam width.
        """
        if len(self.active) != 1 or self.active[0].branch_id != 0:
            raise ValueError(
                "Trie Beam handoff candidates can only initialize the root branch."
            )
        root = self.active[0]
        seen_tokens = set()
        active: List[TrieBeam] = []
        self.last_completed = []
        self.last_pruned = []
        self.last_pruned_parent_ids = []
        for token, score, terminal in candidates:
            if token in seen_tokens:
                raise ValueError("Trie Beam handoff contains duplicate token IDs.")
            seen_tokens.add(token)
            if token not in root.grammar.allowed_tokens:
                raise ValueError(
                    f"Trie Beam handoff token {token} is not legal at the root."
                )
            grammar = root.grammar.fork()
            grammar.accept_token(token)
            if bool(terminal) != grammar.is_terminated():
                raise ValueError(
                    "Trie Beam handoff terminal flag does not match the Trie edge."
                )
            child = TrieBeam(
                [token],
                float(score),
                grammar,
                finished=grammar.is_terminated(),
                branch_id=self._next_branch_id,
                parent_id=root.branch_id,
            )
            self._next_branch_id += 1
            if terminal:
                self._record_completed(child)
                self.last_completed.append(child)
            elif len(active) < self.width:
                active.append(child)
            else:
                self.last_pruned.append(child)
                self.last_pruned_parent_ids.append(root.branch_id)
        if not candidates:
            raise ValueError("Trie Beam handoff must contain at least one candidate.")
        self.active = active
        if active:
            self.last_slot_transition = TrieBeamSlotTransition(
                slot_reuses={active[0].branch_id: root.branch_id},
                slot_forks={child.branch_id: root.branch_id for child in active[1:]},
                parent_ids_to_release=[],
            )
        else:
            self.last_slot_transition = TrieBeamSlotTransition({}, {}, [root.branch_id])
        return active

    def advance(self, logits: torch.Tensor) -> List[TrieBeam]:
        """Select the next active branches from logits aligned with active.

        Each selected child records the source branch in parent_id; the
        scheduler will later use this mapping to fork or COW
        request-to-token slots.
        """
        if logits.ndim != 2 or logits.shape[0] != len(self.active):
            raise ValueError(
                "logits must have shape "
                f"({len(self.active)}, vocab_size), got {tuple(logits.shape)}."
            )

        parent_ids = {beam.branch_id for beam in self.active}
        candidates: List[TrieBeam] = []
        self.last_completed = []
        for beam_index, beam in enumerate(self.active):
            allowed = beam.grammar.allowed_tokens
            if not allowed:
                continue
            log_probs = torch.log_softmax(logits[beam_index], dim=-1)
            for token in allowed:
                child = beam.fork()
                child.branch_id = self._next_branch_id
                self._next_branch_id += 1
                child.parent_id = beam.branch_id
                child.grammar.accept_token(token)
                child.tokens.append(token)
                child.score += float(log_probs[token])
                child.finished = child.grammar.is_terminated()
                if child.finished:
                    self.last_completed.append(child)
                    self._record_completed(child)
                else:
                    candidates.append(child)

        candidates.sort(key=_beam_rank_key)
        self.active = candidates[: self.width]
        self.last_pruned = candidates[self.width :]
        selected_parent_ids = {beam.parent_id for beam in self.active}
        self.last_pruned_parent_ids = sorted(parent_ids - selected_parent_ids)
        self.last_slot_transition = plan_trie_beam_slot_transition(
            parent_ids, self.active
        )
        return self.active

    def advance_with_constrained_topk(
        self,
        logits: torch.Tensor,
        candidate_layout: TrieBeamCandidateLayout | None = None,
    ) -> List[TrieBeam]:
        """Advance with device-side Trie candidate gathering and global ranking.

        Unlike the reference advance path, this method does not materialize one
        Python candidate per legal Trie edge. Child-logit gathering and global
        constrained Top-K remain on the forward device. Only selected active
        branches and output-relevant terminal branches are materialized for the
        scheduler's request-slot transition.

        The optional candidate layout can be constructed while the decode batch
        is assembled and reused here after model forward.
        """
        if logits.ndim != 2 or logits.shape[0] != len(self.active):
            raise ValueError(
                "logits must have shape "
                f"({len(self.active)}, vocab_size), got {tuple(logits.shape)}."
            )

        previous_active = self.active
        parent_ids = {beam.branch_id for beam in previous_active}
        allowed_tokens = [beam.grammar.allowed_tokens for beam in previous_active]
        terminal_children = []
        for beam, allowed in zip(previous_active, allowed_tokens):
            flags = []
            for token in allowed:
                grammar = beam.grammar.fork()
                grammar.accept_token(token)
                flags.append(grammar.is_terminated())
            terminal_children.append(flags)

        parent_scores = torch.tensor(
            [beam.score for beam in previous_active],
            device=logits.device,
            dtype=logits.dtype,
        )
        if candidate_layout is None:
            candidate_layout = build_trie_beam_candidate_layout(
                allowed_tokens, terminal_children, device=logits.device
            )
        selection = trie_constrained_beam_topk_from_layout(
            logits,
            parent_scores,
            candidate_layout,
            self.width,
            self.num_return_sequences,
        )

        def materialize(
            parent_indices: torch.Tensor,
            token_ids: torch.Tensor,
            scores: torch.Tensor,
        ) -> List[TrieBeam]:
            children: List[TrieBeam] = []
            for parent_index, token, score in zip(
                parent_indices.tolist(), token_ids.tolist(), scores.tolist()
            ):
                parent = previous_active[parent_index]
                child = parent.fork()
                child.branch_id = self._next_branch_id
                self._next_branch_id += 1
                child.parent_id = parent.branch_id
                child.grammar.accept_token(token)
                child.tokens.append(token)
                child.score = float(score)
                child.finished = child.grammar.is_terminated()
                children.append(child)
            return children

        self.active = materialize(
            selection.active_parent_indices,
            selection.active_token_ids,
            selection.active_scores,
        )
        self.last_completed = materialize(
            selection.completed_parent_indices,
            selection.completed_token_ids,
            selection.completed_scores,
        )
        self.last_pruned = materialize(
            selection.pruned_parent_indices,
            selection.pruned_token_ids,
            selection.pruned_scores,
        )
        for completed in self.last_completed:
            self._record_completed(completed)
        selected_parent_ids = {beam.parent_id for beam in self.active}
        self.last_pruned_parent_ids = sorted(parent_ids - selected_parent_ids)
        self.last_slot_transition = plan_trie_beam_slot_transition(
            parent_ids, self.active
        )
        return self.active


def plan_trie_beam_slot_transition(
    previous_parent_ids: set[int], selected_children: Sequence[TrieBeam]
) -> TrieBeamSlotTransition:
    """Build a deterministic active-branch request-slot transition plan.

    Each parent can donate its existing request slot to one surviving child.
    Additional children of the same parent require independently allocated
    slots whose page-aligned KV prefix is shared through the paged allocator.
    Parents without an active descendant can release their slots after their
    terminal and pruned children have been recorded.
    """
    slot_reuses: Dict[int, int] = {}
    slot_forks: Dict[int, int] = {}
    active_parent_ids = set()

    for child in selected_children:
        if child.parent_id is None or child.parent_id not in previous_parent_ids:
            raise ValueError(
                "Each selected child must reference an active parent; got "
                f"child branch {child.branch_id} with parent {child.parent_id}."
            )
        active_parent_ids.add(child.parent_id)
        if child.parent_id not in slot_reuses.values():
            slot_reuses[child.branch_id] = child.parent_id
        else:
            slot_forks[child.branch_id] = child.parent_id

    return TrieBeamSlotTransition(
        slot_reuses=slot_reuses,
        slot_forks=slot_forks,
        parent_ids_to_release=sorted(previous_parent_ids - active_parent_ids),
    )


def _beam_rank_key(beam: TrieBeam) -> tuple[float, int]:
    """Rank high scores first and make equal-score selection reproducible."""
    return (-beam.score, beam.branch_id)


def constrained_beam_search_step(
    beams: Sequence[TrieBeam], logits: torch.Tensor, width: int
) -> List[TrieBeam]:
    """Advance a fixed-width beam using only the trie children of each beam.

    ``logits`` contains one row per input beam.  Scores are accumulated from
    the full-vocabulary log-softmax, which is equivalent to applying a hard
    trie mask before sampling while preserving model likelihoods.
    """
    if width < 1:
        raise ValueError(f"width must be positive, got {width}.")
    if logits.ndim != 2 or logits.shape[0] != len(beams):
        raise ValueError(
            f"logits must have shape ({len(beams)}, vocab_size), got {tuple(logits.shape)}."
        )

    candidates: List[TrieBeam] = []
    for beam_index, beam in enumerate(beams):
        if beam.finished or beam.grammar.is_terminated():
            retained = beam.fork()
            retained.finished = True
            candidates.append(retained)
            continue

        allowed = beam.grammar.allowed_tokens
        if not allowed:
            continue
        log_probs = torch.log_softmax(logits[beam_index], dim=-1)
        for token in allowed:
            child = beam.fork()
            child.grammar.accept_token(token)
            child.tokens.append(token)
            child.score += float(log_probs[token])
            child.finished = child.grammar.is_terminated()
            candidates.append(child)

    # Python's sort is stable: equal-score branches remain deterministic in
    # trie insertion order and then in their input-beam order.
    candidates.sort(key=_beam_rank_key)
    return candidates[:width]

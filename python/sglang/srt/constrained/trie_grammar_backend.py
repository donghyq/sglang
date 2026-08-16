# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Token-id Trie constraints for generative retrieval.

The immutable trie is shared by request-local grammar objects.  A grammar object
contains only its current node and ancestry, making cache hits and beam forks safe.
"""

import dataclasses
import json
from typing import Dict, List, Sequence, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarStats,
    InvalidGrammarObject,
)
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)


class TrieGrammar(BaseGrammarObject):
    def __init__(
        self,
        children: Tuple[Dict[int, int], ...],
        terminal: Tuple[bool, ...],
        grammar_stats: GrammarStats | None = None,
        node_id: int = 0,
        node_history: Sequence[int] = (),
        accepted_tokens: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.children = children
        self.terminal = terminal
        self.node_id = node_id
        self.node_history = list(node_history)
        self.accepted_tokens = list(accepted_tokens)
        self.grammar_stats = grammar_stats

    @property
    def allowed_tokens(self) -> List[int]:
        return list(self.children[self.node_id])

    def allowed_tokens_with_terminal(self) -> tuple[List[int], List[bool]]:
        """Return legal next Tokens together with their terminal status.

        Beam scheduling needs both pieces of information before model forward.
        Looking them up directly from the immutable Trie avoids forking one
        grammar object per child merely to determine whether that child ends a
        SID path.
        """
        children = self.children[self.node_id]
        tokens = list(children)
        return tokens, [self.terminal[children[token]] for token in tokens]

    def accept_token(self, token: int) -> None:
        if self.is_terminated():
            return
        next_node = self.children[self.node_id].get(token)
        if next_node is None:
            raise ValueError(
                f"Token {token} is not a valid child of trie node {self.node_id}."
            )
        self.node_history.append(self.node_id)
        self.node_id = next_node
        self.accepted_tokens.append(token)
        self.current_token = token

    def rollback(self, k: int) -> None:
        if k < 0 or k > len(self.node_history):
            raise ValueError(
                f"Cannot rollback {k} trie tokens after {len(self.node_history)} accepted tokens."
            )
        if k == 0:
            return
        self.node_id = self.node_history[-k]
        del self.node_history[-k:]
        del self.accepted_tokens[-k:]
        self.current_token = self.accepted_tokens[-1] if self.accepted_tokens else None

    def is_terminated(self):
        return self.terminal[self.node_id]

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(
            (batch_size, (vocab_size + 31) // 32), dtype=torch.int32, device=device
        )

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        set_token_filter_torch(
            vocab_mask, self.allowed_tokens, idx, is_allowed=True, reset_vocab_mask=True
        )

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "xpu", "musa"}:
            from sglang.srt.utils import is_hip

            if is_hip():
                from sgl_kernel import apply_token_bitmask_inplace_cuda

                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
                return
            from sglang.kernels.ops.grammar.bitmask_ops import (
                apply_token_bitmask_inplace_triton,
            )

            apply_token_bitmask_inplace_triton(logits, vocab_mask)
            return
        if logits.device.type == "npu":
            import sgl_kernel_npu  # noqa: F401

            torch.ops.npu.apply_token_bitmask(logits, vocab_mask)
            return

        bit_offsets = torch.arange(32, device=logits.device, dtype=torch.int64)
        allowed = (
            (vocab_mask.to(torch.int64).unsqueeze(-1) >> bit_offsets) & 1
        ).to(torch.bool).flatten(1)[:, : logits.shape[-1]]
        logits.masked_fill_(~allowed, float("-inf"))

    def copy(self) -> "TrieGrammar":
        stats = (
            dataclasses.replace(
                self.grammar_stats, is_cache_hit=True, tree_traversal_time=[]
            )
            if self.grammar_stats is not None
            else None
        )
        return TrieGrammar(self.children, self.terminal, stats)

    def fork(self) -> "TrieGrammar":
        stats = (
            dataclasses.replace(self.grammar_stats, tree_traversal_time=[])
            if self.grammar_stats is not None
            else None
        )
        return TrieGrammar(
            self.children,
            self.terminal,
            stats,
            self.node_id,
            self.node_history,
            self.accepted_tokens,
        )

    def try_jump_forward(self, tokenizer):
        # A trie can branch at every node, so there is no tokenizer-string
        # jump-forward optimization that preserves all valid continuations.
        return None


class TrieGrammarBackend(BaseGrammarBackend):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    @property
    def is_support_token_filter(self):
        return True

    def dispatch_trie(self, key_string: str) -> BaseGrammarObject:
        try:
            paths = json.loads(key_string)
            children, terminal = self._build_trie(paths)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return InvalidGrammarObject(str(exc))
        return TrieGrammar(
            tuple(children), tuple(terminal), GrammarStats(dispatch_type="trie")
        )

    def _build_trie(self, paths: object) -> Tuple[List[Dict[int, int]], List[bool]]:
        if not isinstance(paths, list) or not paths:
            raise ValueError("trie must contain at least one token-id path.")
        children: List[Dict[int, int]] = [{}]
        terminal = [False]
        for path_index, path in enumerate(paths):
            if not isinstance(path, list) or not path:
                raise ValueError(f"trie path at index {path_index} must not be empty.")
            node = 0
            for token in path:
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("trie token IDs must be integers.")
                if not 0 <= token < self.vocab_size:
                    raise ValueError(
                        f"trie token IDs must be in [0, {self.vocab_size - 1}], got {token}."
                    )
                if terminal[node]:
                    raise ValueError(
                        "trie paths must not use one accepted path as the prefix of another."
                    )
                child = children[node].get(token)
                if child is None:
                    child = len(children)
                    children[node][token] = child
                    children.append({})
                    terminal.append(False)
                node = child
            if children[node]:
                raise ValueError(
                    "trie paths must not use one accepted path as the prefix of another."
                )
            terminal[node] = True
        return children, terminal

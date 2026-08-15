"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        # A token is the smallest independently addressable KV unit when
        # page_size is one. Keep Beam ownership separate from normal allocator
        # bookkeeping so shared-prefix references cannot be freed normally.
        self.beam_token_refcounts: dict[int, int] = {}
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.beam_token_refcounts.clear()
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def _beam_token_ids(self, kv_indices: torch.Tensor) -> list[int]:
        if kv_indices.numel() == 0:
            return []
        token_ids = torch.unique(kv_indices).cpu().tolist()
        if 0 in token_ids:
            raise ValueError(
                "KV token 0 is reserved for padded outputs and cannot be beam-owned."
            )
        return token_ids

    def register_beam_pages(self, kv_indices: torch.Tensor) -> None:
        """Register KV tokens newly owned by a root Beam or private suffix.

        The method name matches the paged allocator interface. With a page
        size of one, each token index is itself a safely shareable KV page.
        """
        token_ids = self._beam_token_ids(kv_indices)
        duplicate_token_ids = [
            token_id for token_id in token_ids if token_id in self.beam_token_refcounts
        ]
        if duplicate_token_ids:
            raise ValueError(
                f"KV token {duplicate_token_ids[0]} is already beam-owned."
            )
        for token_id in token_ids:
            self.beam_token_refcounts[token_id] = 1

    def fork_shared_prefix(
        self, kv_indices: torch.Tensor, child_count: int = 1
    ) -> None:
        """Add child references to the complete shared prefix.

        Non-paged KV has no partial writable page: every committed token has a
        distinct physical KV slot, so any prefix length is safe to share.
        """
        if child_count < 1:
            raise ValueError(f"child_count must be positive, got {child_count}.")
        token_ids = self._beam_token_ids(kv_indices)
        missing_token_ids = [
            token_id for token_id in token_ids if token_id not in self.beam_token_refcounts
        ]
        if missing_token_ids:
            raise ValueError(
                f"Cannot fork unregistered KV token {missing_token_ids[0]}."
            )
        for token_id in token_ids:
            self.beam_token_refcounts[token_id] += child_count

    def release_beam_suffix(self, kv_indices: torch.Tensor) -> None:
        """Release one Beam's KV references and recycle final references."""
        token_ids = self._beam_token_ids(kv_indices)
        missing_token_ids = [
            token_id for token_id in token_ids if token_id not in self.beam_token_refcounts
        ]
        if missing_token_ids:
            raise ValueError(
                f"Cannot release unregistered KV token {missing_token_ids[0]}."
            )

        released_token_ids = []
        for token_id in token_ids:
            refcount = self.beam_token_refcounts[token_id]
            if refcount == 1:
                del self.beam_token_refcounts[token_id]
                released_token_ids.append(token_id)
            else:
                self.beam_token_refcounts[token_id] = refcount - 1
        if released_token_ids:
            self._free_unshared(
                torch.tensor(
                    released_token_ids, dtype=torch.int64, device=self.device
                )
            )

    def free(self, free_index: torch.Tensor):
        beam_owned = set(self._beam_token_ids(free_index)) & set(
            self.beam_token_refcounts
        )
        if beam_owned:
            raise ValueError(
                "Use release_beam_suffix() to free beam-owned KV tokens; "
                f"direct free would violate shared ownership for tokens {sorted(beam_owned)}."
            )
        self._free_unshared(free_index)

    def _free_unshared(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )

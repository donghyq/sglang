from __future__ import annotations

import heapq
import hashlib
import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import SessionLifecycleState

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        super().__init__(cache, params)
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None
        # Lazy bind eviction strategy since tree core is initialized after component init.
        self.session_ref_eviction_strategy = (
            self._session_ref_eviction_strategy
            if cache.enable_session_radix_cache
            else None
        )

    def _ensure_eviction_strategy(self) -> None:
        if self.session_ref_eviction_strategy is None:
            self.session_ref_eviction_strategy = (
                self.tree_core.eviction_strategy.get_priority
            )

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        node = leaf
        while node is not None and node is not self.tree_core.root_node:
            cd = node.component_data[self.component_type]
            assert cd.session_ref > 0
            cd.session_ref -= 1
            node = node.parent

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        stop = old_ancestor if old_ancestor is not None else self.tree_core.root_node
        node = leaf
        while (
            node is not None
            and node is not stop
            and node is not self.tree_core.root_node
        ):
            node.component_data[self.component_type].session_ref += 1
            node = node.parent

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        stop = fallback if fallback is not None else self.tree_core.root_node
        node = leaf
        while (
            node is not None
            and node is not stop
            and node is not self.tree_core.root_node
        ):
            cd = node.component_data[self.component_type]
            assert cd.session_ref > 0
            cd.session_ref -= 1
            node = node.parent

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        if match_device_only:
            return lambda node: (
                node.component_data[self.component_type].value is not None
            )

        # HiCache: evicted + backuped nodes are valid match boundaries.
        return lambda node: (
            node.component_data[self.component_type].value is not None or node.backuped
        )

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # Compute Full KV host hit length: walk from last_host_node up to
        # last_device_node, summing host_value lengths of evicted nodes.
        ct = self.component_type
        kv_host_hit = 0
        node = result.best_match_node
        root_node = self.tree_core.root_node
        while node is not result.last_device_node and node is not root_node:
            full_host = node.component_data[ct].host_value
            if full_host is not None:
                kv_host_hit += len(full_host)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        new_parent.component_data[ct].session_ref = child.component_data[ct].session_ref
        child_cd = child.component_data[ct]
        assert new_parent.component_data[ct].session_ids is None
        split_len = len(new_parent.key)
        if child_cd.value is not None:
            new_parent.component_data[ct].value = child_cd.value[:split_len].clone()
            child_cd.value = child_cd.value[split_len:].clone()
        if child_cd.host_value is not None:
            new_parent.component_data[ct].host_value = child_cd.host_value[
                :split_len
            ].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[self.component_type].append(cd.value)
            policy = getattr(self.cache, "agent_lifecycle", None)
            if policy is not None and any(a.get("gpu_indices") for a in policy.attempts.values()):
                policy.evicted(cd.value)
            if policy is not None and policy.consume_native_victim_log_budget():
                for sid in cd.session_ids or ():
                    policy.emit(sid, "DEVICE_RELEASED", node_id=node.id, physical_bytes=len(cd.value)*policy.bytes_per_token,
                                host_copy_present=cd.host_value is not None, native_reference_protection=True)
                start, end = self._dag_node_range(node)
                record = dict(
                    node_id=node.id,
                    freed_tokens=len(cd.value),
                    physical_bytes=len(cd.value) * policy.bytes_per_token,
                    logical_prefix_tokens=end - start,
                    root_prefix_tokens=end,
                    stable_key=self._dag_stable_key(node),
                    session_ids=sorted(cd.session_ids or ()),
                    dag_mode=policy.dag_mode,
                )
                pending = getattr(self.cache, "_pending_native_victim_frees", None)
                if pending is not None:
                    pending.append(record)
                else:
                    # Direct component-level callers outside UnifiedRadixCache
                    # have no deferred-free boundary; keep the event observable.
                    policy.emit(None, "NATIVE_VICTIM_FREED", **record)
            freed = len(cd.value)
            self.tree_core.component_evictable_size_[self.component_type] -= freed
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            # because SWA's free_swa still needs to read Full.value.
            # cd.value = None

        # Host layer
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[self.component_type].append(cd.host_value)
            cd.host_value = None
        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    @staticmethod
    def _dag_stable_key(node: UnifiedTreeNode) -> int:
        """A stable logical prefix identity for TP tie-breaking.

        The value is independent of object addresses, node IDs, and creation
        order. It is computed only while building a bounded eviction heap, not
        in the Decode token path.
        """
        key = node.key
        if key is None:
            return 0
        last_hash = node.get_last_hash_value()
        if last_hash is not None:
            # Storage-enabled production trees already carry namespace-aware
            # page hashes. Use the terminal page identity instead of copying
            # every token along the root path.
            parts = [(last_hash, key.extra_key, key.cache_salt)]
        else:
            # Dependency-light unit trees may not compute page hashes; walk the
            # logical path only on that fallback path.
            parts = []
            current = node
            while current is not None and current.key is not None:
                current_key = current.key
                parts.append(
                    (tuple(current_key.token_ids), current_key.extra_key, current_key.cache_salt)
                )
                current = current.parent
        digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    @staticmethod
    def _dag_node_range(node: UnifiedTreeNode) -> tuple[int, int]:
        end = 0
        current = node
        while current is not None and current.key is not None:
            end += len(current.key)
            current = current.parent
        node_key_len = len(node.key) if node.key is not None else 0
        return max(0, end - node_key_len), end

    @staticmethod
    def _dag_is_ancestor_or_self(node: UnifiedTreeNode, leaf: UnifiedTreeNode) -> bool:
        current = leaf
        while current is not None:
            if current is node:
                return True
            current = current.parent
        return False

    def _dag_raw_hints_for_node(
        self, node: UnifiedTreeNode, grouped_hints: dict | None = None
    ) -> tuple[int | None, int]:
        policy = getattr(self.cache, "agent_lifecycle", None)
        if policy is None or policy.dag_mode == "off":
            return None, 0
        if grouped_hints is None:
            grouped_hints = policy.dag_hints_by_session()
        session_ids = set(node.component_data[self.component_type].session_ids or ())
        for session_id in grouped_hints:
            leaves = self._session_leaves.get(session_id, ())
            if any(self._dag_is_ancestor_or_self(node, leaf) for leaf in leaves):
                session_ids.add(session_id)
        start, end = self._dag_node_range(node)
        hints = []
        for session_id in session_ids:
            for hint in grouped_hints.get(session_id, ()):
                # Protect only nodes wholly inside the confirmed stable
                # prefix. A stable 8K prefix landing inside one 12K node does
                # not protect the private 4K suffix.
                if end <= hint.prefix_tokens and start < hint.prefix_tokens:
                    hints.append(hint)
        if not hints:
            return None, 0
        distance = min((h.next_use_distance for h in hints if h.next_use_distance is not None), default=None)
        return distance, len(hints)

    def _dag_hints_for_node(
        self, node: UnifiedTreeNode, grouped_hints: dict | None = None
    ) -> tuple[int, int, int]:
        policy = getattr(self.cache, "agent_lifecycle", None)
        if policy is None or policy.dag_mode == "off":
            return 0, 0, 0
        distance, consumer_count = self._dag_raw_hints_for_node(node, grouped_hints)
        if distance is None:
            foreground_protected = node in getattr(
                self, "_dag_foreground_protected_nodes", set()
            )
            return (1 if foreground_protected else 0), 0, 0
        near = distance <= policy.dag_near_use_distance
        protected = node in getattr(self, "_dag_protected_nodes", set())
        rank = 2 if near and protected else 1
        # Min-heap: a more negative key evicts first, so farther future use is
        # reclaimed before nearer future use.
        return rank, -distance, consumer_count

    def _dag_try_protect_node(self, node: UnifiedTreeNode, grouped_hints: dict) -> bool:
        policy = getattr(self.cache, "agent_lifecycle", None)
        if policy is None or node in getattr(self, "_dag_protected_nodes", set()):
            return False
        distance, _count = self._dag_raw_hints_for_node(node, grouped_hints)
        if distance is None or distance > policy.dag_near_use_distance:
            return False
        value = node.component_data[self.component_type].value
        tokens = len(value) if value is not None else 0
        if tokens <= 0:
            return False
        if self._dag_protected_tokens_used + tokens > self._dag_protected_token_budget:
            return False
        self._dag_protected_nodes.add(node)
        self._dag_protected_tokens_used += tokens
        return True

    def _dag_try_protect_foreground_node(
        self, node: UnifiedTreeNode, grouped_hints: dict
    ) -> bool:
        policy = getattr(self.cache, "agent_lifecycle", None)
        if policy is None or node in getattr(
            self, "_dag_foreground_protected_nodes", set()
        ):
            return False
        distance, _count = self._dag_raw_hints_for_node(node, grouped_hints)
        if distance is not None:
            return False
        value = node.component_data[self.component_type].value
        tokens = len(value) if value is not None else 0
        if tokens <= 0:
            return False
        if (
            self._dag_foreground_tokens_used + tokens
            > self._dag_foreground_token_budget
        ):
            return False
        self._dag_foreground_protected_nodes.add(node)
        self._dag_foreground_tokens_used += tokens
        return True

    def _session_ref_eviction_strategy(self, node: UnifiedTreeNode):
        policy = getattr(self.cache, "agent_lifecycle", None)
        ref = self.session_ref(node)
        session_ids = node.component_data[self.component_type].session_ids or ()
        lifecycle_states = {
            self.cache.session_refs.get_session_lifecycle(session_id)
            for session_id in session_ids
        }
        tool_waiting_only = bool(
            lifecycle_states
            and SessionLifecycleState.TOOL_WAITING in lifecycle_states
            and SessionLifecycleState.ACTIVE not in lifecycle_states
            and SessionLifecycleState.READY_SOON not in lifecycle_states
        )
        if policy is not None:
            if policy.mode in {"native", "v1"}:
                # V1 uses guarded explicit demote; ordinary victims keep LRU.
                tool_waiting_only = False
        if policy is None or policy.dag_mode == "off":
            # Bit-for-bit policy-equivalent to bed4 when DAG is disabled.
            return (
                not tool_waiting_only,
                ref > 0,
                ref,
                self.tree_core.eviction_strategy.get_priority(node),
            )
        dag_rank, dag_distance, _consumer_count = self._dag_hints_for_node(
            node, getattr(self, "_dag_grouped_hints", None)
        )
        stable_key = self._dag_stable_key(node)
        return (
            dag_rank,
            dag_distance,
            not tool_waiting_only,
            ref > 0,
            ref,
            self.tree_core.eviction_strategy.get_priority(node),
            stable_key,
        )

    def _native_session_ref_eviction_strategy(self, node: UnifiedTreeNode):
        """The bed4 comparison key, retained for B-mode observation."""
        ref = self.session_ref(node)
        session_ids = node.component_data[self.component_type].session_ids or ()
        lifecycle_states = {
            self.cache.session_refs.get_session_lifecycle(session_id)
            for session_id in session_ids
        }
        tool_waiting_only = bool(
            lifecycle_states
            and SessionLifecycleState.TOOL_WAITING in lifecycle_states
            and SessionLifecycleState.ACTIVE not in lifecycle_states
            and SessionLifecycleState.READY_SOON not in lifecycle_states
        )
        policy = getattr(self.cache, "agent_lifecycle", None)
        if policy is not None and policy.mode in {"native", "v1"}:
            tool_waiting_only = False
        return (
            not tool_waiting_only,
            ref > 0,
            ref,
            self.tree_core.eviction_strategy.get_priority(node),
        )

    def _evict_device_start(self, request_cnt: int) -> None:
        self._ensure_eviction_strategy()
        policy = getattr(self.cache, "agent_lifecycle", None)
        started = time.perf_counter()
        self._dag_grouped_hints = None
        self._dag_protected_nodes = set()
        self._dag_protected_tokens_used = 0
        self._dag_protected_token_budget = 0
        self._dag_foreground_protected_nodes = set()
        self._dag_foreground_tokens_used = 0
        self._dag_foreground_token_budget = 0
        self._dag_candidate_tokens = 0
        self._dag_counted_nodes = set()
        leaves = list(self.tree_core.evictable_device_leaves)
        if policy is not None and policy.dag_mode != "off":
            self._dag_grouped_hints = policy.dag_hints_by_session()
            self._dag_candidate_tokens = sum(
                len(n.component_data[self.component_type].value)
                if n.component_data[self.component_type].value is not None
                else 0
                for n in leaves
            )
            self._dag_counted_nodes.update(leaves)
            self._dag_protected_token_budget = int(
                self._dag_candidate_tokens * policy.dag_protected_token_fraction
            )
            self._dag_foreground_token_budget = int(
                self._dag_candidate_tokens * policy.dag_foreground_token_fraction
            )
            for node in leaves:
                self._dag_try_protect_node(node, self._dag_grouped_hints)
            ordinary = [
                node
                for node in leaves
                if self._dag_raw_hints_for_node(node, self._dag_grouped_hints)[0] is None
            ]
            # Protect the most recently used ordinary foreground leaves up to a
            # finite token budget, so a stream of DAG hints cannot starve them.
            ordinary.sort(key=lambda node: node.last_access_time, reverse=True)
            for node in ordinary:
                self._dag_try_protect_foreground_node(node, self._dag_grouped_hints)
        self._evict_device_request_cnt = request_cnt
        self._evict_device_last_node = None
        self._evict_device_heap = [
            (self.session_ref_eviction_strategy(n), n)
            for n in leaves
        ]
        heapq.heapify(self._evict_device_heap)
        if policy is not None:
            native_order = heapq.nsmallest(
                32,
                leaves,
                key=self._native_session_ref_eviction_strategy,
            )
            native_snapshot = [
                (self._dag_stable_key(node), self.session_ref(node))
                for node in native_order
            ]
            # Compute both orders in every group so B/E/EP carry identical
            # observation overhead; in off mode the two orders are equal.
            dag_order = heapq.nsmallest(
                32,
                leaves,
                key=self._session_ref_eviction_strategy,
            )
            dag_snapshot = [
                (self._dag_stable_key(node), self.session_ref(node))
                for node in dag_order
            ]
            elapsed_us = int((time.perf_counter() - started) * 1_000_000)
            policy.dag_stats["eviction_candidate_checks"] += len(self._evict_device_heap)
            policy.dag_stats["eviction_cpu_us"] += elapsed_us
            policy.dag_stats["metadata_entries"] = policy.dag_metadata_size()
            if policy.consume_native_eviction_log_budget():
                policy.emit(
                    None,
                    "NATIVE_EVICTION_PLAN",
                    candidates=len(self._evict_device_heap),
                    cpu_us=elapsed_us,
                    dag_mode=policy.dag_mode,
                    metadata_entries=policy.dag_metadata_size(),
                    protected_token_budget=self._dag_protected_token_budget,
                    foreground_protected_token_budget=self._dag_foreground_token_budget,
                    native_candidate_order=native_snapshot,
                    dag_candidate_order=dag_snapshot,
                )

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        ct = self.component_type
        lv = self._evict_device_last_node
        if (
            lv is not None
            and lv.parent is not None
            and lv.parent in self.tree_core.evictable_device_leaves
        ):
            if self._dag_grouped_hints is not None:
                parent_value = lv.parent.component_data[self.component_type].value
                if parent_value is not None and lv.parent not in self._dag_counted_nodes:
                    self._dag_counted_nodes.add(lv.parent)
                    self._dag_candidate_tokens += len(parent_value)
                    policy = getattr(self.cache, "agent_lifecycle", None)
                    if policy is not None:
                        self._dag_protected_token_budget = int(
                            self._dag_candidate_tokens
                            * policy.dag_protected_token_fraction
                        )
                        self._dag_foreground_token_budget = int(
                            self._dag_candidate_tokens
                            * policy.dag_foreground_token_fraction
                        )
                self._dag_try_protect_node(lv.parent, self._dag_grouped_hints)
                self._dag_try_protect_foreground_node(
                    lv.parent, self._dag_grouped_hints
                )
            heapq.heappush(
                self._evict_device_heap,
                (self.session_ref_eviction_strategy(lv.parent), lv.parent),
            )
        self._evict_device_last_node = None
        while tracker[ct] < self._evict_device_request_cnt and self._evict_device_heap:
            _, x = heapq.heappop(self._evict_device_heap)
            if x not in self.tree_core.evictable_device_leaves:
                continue
            policy = getattr(self.cache, "agent_lifecycle", None)
            if (policy is not None and policy.dag_mode != "off"
                    and (any(cd.lock_ref > 0 for cd in x.component_data)
                         or x.load_back_pending_id is not None)):
                continue
            if policy is not None and policy.consume_native_victim_log_budget():
                start, end = self._dag_node_range(x)
                dag_rank, dag_distance, _consumer_count = self._dag_hints_for_node(x)
                policy.emit(
                    None,
                    "NATIVE_VICTIM_SELECTED",
                    node_id=x.id,
                    logical_prefix_tokens=end - start,
                    root_prefix_tokens=end,
                    stable_key=self._dag_stable_key(x),
                    session_ids=sorted(x.component_data[ct].session_ids or ()),
                    dag_mode=policy.dag_mode,
                    dag_rank=dag_rank,
                    dag_distance=dag_distance,
                    reason=(
                        "dag_near_future_protected"
                        if dag_rank == 2
                        else "dag_future_or_native_lru"
                    ),
                )
            self._evict_device_last_node = x
            return x.id
        return None

    def _evict_device_end(self) -> None:
        self._evict_device_heap = []
        self._evict_device_last_node = None

    def _host_eviction_strategy(self, node):
        policy = getattr(self.cache, "agent_lifecycle", None)
        if policy is None or policy.mode == "v0":
            return self.session_ref_eviction_strategy(node)
        ref = self.session_ref(node)
        return (True, ref > 0, ref, self.tree_core.eviction_strategy.get_priority(node))

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict host leaves to free KV host pool space."""
        self._ensure_eviction_strategy()
        heap = [
            (self._host_eviction_strategy(n), n)
            for n in self.tree_core.evictable_host_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.tree_core.evictable_host_leaves:
                continue
            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
            if (
                x.parent is not None
                and x.parent in self.tree_core.evictable_host_leaves
            ):
                heapq.heappush(
                    heap,
                    (self._host_eviction_strategy(x.parent), x.parent),
                )

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type

        # Only the last host node needs to be protected.
        if lock_host:
            cd = node.component_data[ct]
            # write_back mode: the anchor may be device-only (no host_value); pin it anyway.
            if cd.host_value is None and not self.tree_core.is_write_back:
                return result
            cd.host_lock_ref += 1
            self.tree_core._update_evictable_leaf_sets(node)
            return result

        root = self.tree_core.root_node
        cur = node

        # The bottom device-evicted segment is locked too (no ledger move —
        # nothing is on device); a load-back that materializes a value under
        # lock credits protected directly.
        while cur is not root and cur.component_data[ct].value is None:
            cur.component_data[ct].lock_ref += 1
            cur = cur.parent

        # Lock the device-on segment up to root
        delta = 0
        while cur is not root:
            cd = cur.component_data[ct]
            assert cd.value is not None, (
                f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
            )
            if cd.lock_ref == 0:
                key_len = len(cd.value)
                self.tree_core.component_evictable_size_[ct] -= key_len
                self.tree_core.component_protected_size_[ct] += key_len
                delta += key_len
            cd.lock_ref += 1
            self.tree_core.evictable_device_leaves.discard(cur)
            cur = cur.parent
        result.delta = delta
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams,
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if lock_host:
            cd = node.component_data[ct]
            if cd.host_lock_ref == 0:
                return
            if cd.host_value is None and not self.tree_core.is_write_back:
                return
            cd.host_lock_ref -= 1
            self.tree_core._update_evictable_leaf_sets(node)
            return

        root = self.tree_core.root_node
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            assert cd.lock_ref > 0, (
                f"FULL segment release hit lock_ref=0 on node {cur.id}"
            )
            if cd.lock_ref == 1 and cd.value is not None:
                key_len = len(cd.value)
                self.tree_core.component_evictable_size_[ct] += key_len
                self.tree_core.component_protected_size_[ct] -= key_len
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.tree_core._update_evictable_leaf_sets(cur)
            cur = cur.parent

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        staging_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Full KV backup is handled by the main flow
            # (cache_controller.write on host_value directly).
            # No extra PoolTransfer needed.
            return None

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node. FULL device evict only from leaves,
            # so once we hit a device-on node, everything above is also device-on
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur.evicted:
                cd = cur.component_data[ct]
                assert cd.host_value is not None
                backed_up.append(cd.host_value)
                nodes.append(cur)
                cur = cur.parent
            backed_up.reverse()
            nodes.reverse()
            return [
                PoolTransfer(
                    name=PoolName.KV,
                    host_indices=(
                        torch.cat(backed_up)
                        if backed_up
                        else torch.empty((0,), dtype=torch.int64, device="cpu")
                    ),
                    device_indices=None,
                    nodes_to_load=[n.id for n in nodes],
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                node.component_data[ct].host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                self.tree_core._update_evictable_leaf_sets(node)
                return

            xfer = transfers[0]
            device_indices = xfer.device_indices
            offset = 0
            for nid in xfer.nodes_to_load or []:
                n = self.tree_core.node_by_id(nid)
                cd = n.component_data[ct]
                n_len = len(cd.host_value)
                cd.value = device_indices[offset : offset + n_len].clone()
                offset += n_len
                # Full uses leaf sets, not LRU. A value materialized under
                # lock is protected; the last release moves it to evictable.
                if cd.lock_ref > 0:
                    self.tree_core.component_protected_size_[ct] += n_len
                else:
                    self.tree_core.component_evictable_size_[ct] += n_len
                self.tree_core._update_evictable_leaf_sets(n)

            self.tree_core._update_evictable_leaf_sets(node)

    def _full_allocator(self):
        """The allocator that owns the full-attention pool alone."""
        allocator = self.cache.token_to_kv_pool_allocator
        return allocator.full_attn_allocator if self.cache.is_swa_enabled else allocator

    def build_external_linker_transfer(
        self,
        phase: LinkerTransferPhase,
        node: Optional[UnifiedTreeNode],
        keys: Optional[Sequence[str]],
    ) -> Optional[PoolTransfer]:
        if phase == LinkerTransferPhase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            value = node.component_data[self.component_type].value
            if value is None:
                return None
            return PoolTransfer(
                name=PoolName.KV,
                device_indices=value.to(torch.int64),
                keys=list(node.hash_value),
            )

        if not keys:
            return None

        if phase == LinkerTransferPhase.LOOKUP:
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

        if phase == LinkerTransferPhase.LOAD:
            allocator = self._full_allocator()
            num_tokens = len(keys) * self.cache.page_size
            shortfall = max(0, num_tokens - allocator.available_size())
            if shortfall:
                self.cache.evict(EvictParams(num_tokens=shortfall))
            slots = allocator.alloc(num_tokens)
            if slots is None:
                return None

            return PoolTransfer(
                name=PoolName.KV,
                device_indices=slots.to(torch.int64),
                keys=list(keys),
            )

    def update_external_linker_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        full_transfer: PoolTransfer,
        transfer: PoolTransfer,
        prefix_len: int,
        *,
        insert_result: Optional[InsertResult] = None,
        canonical_full: Optional[torch.Tensor] = None,
    ) -> Optional[PoolTransfer]:
        if phase == ExternalLinkerLoadPhase.ABORT:
            self._full_allocator().free(transfer.device_indices)
            return None
        if phase == ExternalLinkerLoadPhase.PREPARE:
            return transfer

        assert phase == ExternalLinkerLoadPhase.COMMIT
        return transfer

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._full_kv_pool_host is None:
            return
        for host_value in host_values:
            self.cache.host_pool_group.free(host_value, pool=PoolName.KV)

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            alloc = self.cache.token_to_kv_pool_allocator
            for indices in action.indices:
                # tree values are page-aligned copies of a kv row: page-exact segments
                if self.cache.is_swa_enabled:
                    alloc.full_attn_allocator.free_segment(indices, start_pos=0)
                else:
                    alloc.free_segment(indices, start_pos=0)
            return
        raise AssertionError(
            f"FullComponent: unhandled ComponentAction {type(action).__name__}"
        )

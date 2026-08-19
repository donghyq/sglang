from collections import defaultdict
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlSenderFailureCleanup(unittest.TestCase):
    def test_failure_exception_cleans_room_state_before_raising(self):
        room = 7
        expected_exc = RuntimeError("transfer failed")
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender._send_failed = False
        sender._send_error = None
        staging_ctx = SimpleNamespace(
            prefetched_rooms={room, 8},
            prefetch_requested={(room, 0, "session-a"), (8, 0, "session-b")},
        )
        sender.kv_mgr = SimpleNamespace(
            enable_staging=True,
            _staging_ctx=staging_ctx,
            request_status={room: object()},
            req_to_decode_prefix_len={room: 3},
            transfer_infos={room: object()},
            exceptions={room: expected_exc},
            failure_records={room: "transfer failed"},
            failure_lock=threading.Lock(),
        )

        with self.assertRaises(RuntimeError) as cm:
            sender.failure_exception()

        self.assertIs(cm.exception, expected_exc)
        self.assertTrue(sender._send_failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(room, sender.kv_mgr.request_status)
        self.assertNotIn(room, sender.kv_mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, sender.kv_mgr.transfer_infos)
        self.assertNotIn(room, sender.kv_mgr.exceptions)
        self.assertNotIn(room, sender.kv_mgr.failure_records)
        self.assertNotIn(room, staging_ctx.prefetched_rooms)
        self.assertNotIn((room, 0, "session-a"), staging_ctx.prefetch_requested)
        self.assertIn(8, staging_ctx.prefetched_rooms)
        self.assertIn((8, 0, "session-b"), staging_ctx.prefetch_requested)


class TestMooncakeSenderFailureCleanup(unittest.TestCase):
    def test_failure_exception_cleans_room_state_before_raising(self):
        room = 8
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender.kv_mgr = SimpleNamespace(
            request_status={room: KVPoll.Failed},
            req_to_decode_prefix_len={room: 5},
            transfer_infos={room: object()},
            failure_records={room: "RDMA transfer failed"},
            failure_lock=threading.Lock(),
        )

        with self.assertRaises(KVTransferError) as cm:
            sender.failure_exception()

        self.assertEqual(cm.exception.bootstrap_room, room)
        self.assertIn("bootstrap_room=8", str(cm.exception))
        self.assertIn("RDMA transfer failed", str(cm.exception))
        self.assertFalse(cm.exception.is_aborted_by_request)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(room, sender.kv_mgr.request_status)
        self.assertNotIn(room, sender.kv_mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, sender.kv_mgr.transfer_infos)
        self.assertNotIn(room, sender.kv_mgr.failure_records)

    def test_abort_error_is_classified_without_counting_as_transfer_failure(self):
        error = KVTransferError(10, "Aborted by AbortReq.")

        self.assertTrue(error.is_aborted_by_request)

    def test_transfer_worker_failure_is_cleaned_by_sender(self):
        room = 9
        session_id = "decode-session"
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.enable_trace = False
        mgr.enable_staging = False
        mgr.request_status = {room: KVPoll.Transferring}
        mgr.transfer_infos = {
            room: {
                session_id: SimpleNamespace(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=31010,
                    mooncake_session_id=session_id,
                    dst_kv_indices=np.array([4], dtype=np.int32),
                    required_dst_info_num=1,
                    is_dummy=False,
                )
            }
        }
        mgr.req_to_decode_prefix_len = {room: 5}
        mgr.decode_kv_args_table = {
            session_id: SimpleNamespace(
                dst_kv_ptrs=[0],
                dst_attn_tp_size=1,
            )
        }
        mgr.attn_tp_rank = 0
        mgr.attn_cp_rank = 0
        mgr.attn_dp_rank = 0
        mgr.attn_cp_size = 1
        mgr.pp_rank = 0
        mgr.pp_size = 1
        mgr.attn_tp_size = 1
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.session_lock = threading.Lock()
        mgr.session_failures = defaultdict(int)
        mgr.failed_sessions = set()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.send_kvcache = MagicMock(return_value=-1)
        mgr.sync_status_to_decode_endpoint = MagicMock()

        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([3], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
        )
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue, executor=MagicMock())

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertIn(session_id, mgr.failed_sessions)
        self.assertIn("Failed to send kv chunk", mgr.failure_records[room])
        mgr.sync_status_to_decode_endpoint.assert_called_once_with(
            "127.0.0.1", 31010, room, KVPoll.Failed, 0
        )

        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender.kv_mgr = mgr
        with self.assertRaises(KVTransferError):
            sender.failure_exception()

        self.assertNotIn(room, mgr.request_status)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.failure_records)


if __name__ == "__main__":
    unittest.main()

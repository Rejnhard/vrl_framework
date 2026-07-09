# Copyright 2026 Jacek Rejnhard.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import logging
import os
import pickle
import queue
import threading
import time
import typing
import uuid
import weakref
import zlib
from dataclasses import dataclass

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vrl_framework.core.contracts import (
    PlannerValidator,
    PolicySequenceBatch,
    RecurrentStateSnapshot,
    ReplaySegmentMeta,
    RetrievedMemoryBatch,
)
from vrl_framework.core.settings import MEM_CFG
from vrl_framework.math_ops.geometry import LorentzGeometry


class MemoryQualityBand(int):
    GREEN = 0
    YELLOW = 1
    RED = 2


class MemoryTierState(int):
    COLD_ON_SSD = 0
    WARM_IN_RAM = 1
    HOT_IN_VRAM = 2
    EVICTING = 3


@dataclass
class MemoryEntityDescriptor:
    object_id: str
    tier_state: int
    segment_id: int
    byte_offset: int
    length: int
    dtype: torch.dtype
    quant_scheme: float
    quality_score: float
    last_access_step: int
    predicted_next_use: float
    true_quant_error: float
    quality_band: int


class GlobalExperienceReplay:
    """Implements an LMDB-backed out-of-core experience replay.

    Handles asynchronous serialization of latent frames and TD errors to SSD.
    Uses lock-guarded buffering to prevent IO blocking on the main training thread.
    """

    def __init__(self, db_path: str):
        self.max_records = getattr(MEM_CFG, "segment_size", 1024) * 100
        self.active_records = 0
        self.write_ptr = 0

        self.buffer_td = torch.empty((self.max_records, 1), dtype=torch.float32, pin_memory=torch.cuda.is_available())
        self.buffer_qe = torch.empty((self.max_records, 1), dtype=torch.float32, pin_memory=torch.cuda.is_available())
        self.buffer_quality_band = torch.empty(
            (self.max_records, 1), dtype=torch.uint8, pin_memory=torch.cuda.is_available()
        )
        self.buffer_signatures = torch.empty(
            (self.max_records, 64), dtype=torch.bool, pin_memory=torch.cuda.is_available()
        )

        self.qe_rolling_window: typing.Deque[float] = collections.deque(maxlen=10000)
        self.qe_p50 = 0.5
        self.qe_p90 = 1.5
        self.qe_p99 = 2.0

        self._buffer_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.active_segment_id = 0
        self.segment_buffer_ep: typing.List[bytes] = []
        self.segment_buffer_td: typing.List[bytes] = []

        os.makedirs(db_path, exist_ok=True)
        self.max_map_size = 4 * 1024 * 1024
        self.env = lmdb.open(
            db_path, map_size=self.max_map_size, subdir=True, max_dbs=5, lock=True, readahead=False, meminit=False
        )
        self.segment_db = self.env.open_db(b"segments")
        self.meta_db = self.env.open_db(b"metadata")
        self.lsh_index_db = self.env.open_db(b"lsh_index")
        self.segproto_cache: typing.Dict[int, bytes] = {}
        self.write_queue: "queue.Queue[typing.Any]" = queue.Queue(maxsize=100)
        self._segment_id_lock = threading.Lock()

        self._bg_writer = threading.Thread(target=self._atomic_segment_writer, daemon=True)
        self._bg_writer.start()
        self._bg_compactor = threading.Thread(target=self._background_compaction_loop, daemon=True)
        self._bg_compactor.start()

    def close(self):
        """Terminates background threads and closes the LMDB environment."""
        self._shutdown_event.set()
        if hasattr(self, "write_queue"):
            try:
                self.write_queue.put_nowait(None)
            except queue.Full:
                pass
        if hasattr(self, "_bg_writer"):
            self._bg_writer.join(timeout=2.0)
        if hasattr(self, "_bg_compactor"):
            self._bg_compactor.join(timeout=2.0)
        if hasattr(self, "env"):
            try:
                self.env.sync()
                self.env.close()
            except Exception as e:
                logging.error(f"Error during LMDB environment closure: {e}")

    def _atomic_segment_writer(self):
        """Async LMDB writer.

        Isolating LSH signatures enables fast Hamming lookups without loading full records.
        """
        while not self._shutdown_event.is_set():
            try:
                task = self.write_queue.get(timeout=0.5)
                if task is None:
                    break
                seg_id, payload_ep, payload_td, metadata, sig_list = task

                success = False
                while not success:
                    try:
                        with self.env.begin(write=True) as txn:
                            txn.put(f"seg_ep_{seg_id}".encode("ascii"), payload_ep, db=self.segment_db)
                            txn.put(f"seg_td_{seg_id}".encode("ascii"), payload_td, db=self.segment_db)
                            txn.put(f"segmeta:{seg_id}".encode("ascii"), pickle.dumps(metadata), db=self.meta_db)

                            if sig_list:
                                sig_arrays = [np.frombuffer(s, dtype=bool) for s in sig_list]
                                majority_sig = (np.mean(sig_arrays, axis=0) >= 0.5).astype(bool).tobytes()

                                txn.put(f"segproto:{seg_id}".encode("ascii"), majority_sig, db=self.lsh_index_db)
                                self.segproto_cache[seg_id] = majority_sig

                                for local_idx, sig_bytes in enumerate(sig_list):
                                    txn.put(
                                        f"segrec:{seg_id}:{local_idx}".encode("ascii"), sig_bytes, db=self.lsh_index_db
                                    )

                        success = True
                    except lmdb.MapFullError:
                        current_size = self.env.info()["map_size"]
                        new_size = min(current_size * 2, self.max_map_size)

                        if new_size == current_size:
                            success = True
                            break

                        self.env.set_mapsize(new_size)
                        time.sleep(0.01)
                    except Exception as e:
                        import logging

                        logging.error(f"LMDB write error: {e}")
                        success = True
                        break

                self.write_queue.task_done()

            except queue.Empty:
                continue
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Background writer loop failed")

    def _background_compaction_loop(self):
        """Aggregates and filters LMDB records to reclaim space.

        Tombstones records with quantization errors exceeding the 90th percentile.
        Preserves extreme outliers (> p99) as priority samples if they survived initial compaction.
        """
        while not self._shutdown_event.is_set():
            time.sleep(60)
            try:
                segments_to_compact = []
                with self.env.begin() as txn:
                    cursor = txn.cursor(self.meta_db)
                    for key, value in cursor:
                        if not key.startswith(b"segmeta:"):
                            continue
                        meta = pickle.loads(value)

                        is_red = meta.quality_band == MemoryQualityBand.RED
                        is_high_error = meta.true_quant_error > self.qe_p90
                        is_tombstone = getattr(meta, "tombstone", False)

                        is_novelty_anchor = (meta.true_quant_error > self.qe_p99) and (meta.last_compacted_step > 5)

                        if (is_red or is_high_error) and not is_tombstone and not is_novelty_anchor:
                            seg_id = meta.segment_id
                            segments_to_compact.append(seg_id)

                if segments_to_compact:
                    with self.env.begin(write=True) as txn:
                        for seg_id in segments_to_compact:
                            meta_bytes = txn.get(f"segmeta:{seg_id}".encode("ascii"), db=self.meta_db)
                            if meta_bytes:
                                meta = pickle.loads(meta_bytes)
                                meta.tombstone = True
                                txn.put(f"segmeta:{seg_id}".encode("ascii"), pickle.dumps(meta), db=self.meta_db)

                    for seg_id in segments_to_compact:
                        with self.env.begin() as txn:
                            meta_bytes = txn.get(f"segmeta:{seg_id}".encode("ascii"), db=self.meta_db)
                            ep_bytes = txn.get(f"seg_ep_{seg_id}".encode("ascii"), db=self.segment_db)
                            td_bytes = txn.get(f"seg_td_{seg_id}".encode("ascii"), db=self.segment_db)

                        if not meta_bytes or not ep_bytes:
                            continue

                        meta = pickle.loads(meta_bytes)
                        record_count = meta.record_count

                        valid_ep = []
                        valid_td = []
                        valid_sig = []

                        dim_bytes = meta.record_stride_bytes
                        dim_td_bytes = meta.td_stride_bytes

                        with self.env.begin() as txn:
                            for idx in range(record_count):
                                ep_chunk = ep_bytes[idx * dim_bytes : (idx + 1) * dim_bytes]
                                td_chunk = td_bytes[idx * dim_td_bytes : (idx + 1) * dim_td_bytes]
                                sig_chunk = txn.get(f"segrec:{seg_id}:{idx}".encode("ascii"), db=self.lsh_index_db)

                                if sig_chunk:
                                    valid_ep.append(ep_chunk)
                                    valid_td.append(td_chunk)
                                    valid_sig.append(sig_chunk)

                        if valid_ep:
                            joined_ep = b"".join(valid_ep)
                            joined_td = b"".join(valid_td)

                            new_meta = ReplaySegmentMeta(
                                segment_id=self.active_segment_id,
                                record_count=len(valid_ep),
                                quality_band=MemoryQualityBand.GREEN,
                                true_quant_error=self.qe_p50,
                                scale=meta.scale,
                                record_dim=meta.record_dim,
                                td_dim=meta.td_dim,
                                record_stride_bytes=meta.record_stride_bytes,
                                td_stride_bytes=meta.td_stride_bytes,
                                payload_version=meta.payload_version,
                                retrieval_hits=meta.retrieval_hits,
                                utility_score=meta.utility_score,
                                last_compacted_step=meta.last_compacted_step + 1,
                                prototype_popcount=meta.prototype_popcount,
                                format_signature=b"VRLX",
                                checksum=zlib.crc32(joined_ep),
                                tombstone=False,
                            )
                            try:
                                with self._segment_id_lock:
                                    target_segment_id = self.active_segment_id
                                    new_meta.segment_id = target_segment_id
                                    self.write_queue.put(
                                        (target_segment_id, joined_ep, joined_td, new_meta, valid_sig), timeout=5.0
                                    )
                                    self.active_segment_id += 1

                                with self.env.begin(write=True) as txn:
                                    txn.delete(f"seg_ep_{seg_id}".encode("ascii"), db=self.segment_db)
                                    txn.delete(f"seg_td_{seg_id}".encode("ascii"), db=self.segment_db)
                                    txn.delete(f"segmeta:{seg_id}".encode("ascii"), db=self.meta_db)
                                    txn.delete(f"segproto:{seg_id}".encode("ascii"), db=self.lsh_index_db)
                                    for idx in range(record_count):
                                        txn.delete(f"segrec:{seg_id}:{idx}".encode("ascii"), db=self.lsh_index_db)
                            except queue.Full:
                                logging.warning(f"Write queue saturated. Backing off compaction for Segment {seg_id}.")

            except Exception as e:
                logging.error(f"LMDB Compaction error: {e}")

    def batch_write(
        self,
        keys: list,
        full_records_bytes: list,
        td_tensors: torch.Tensor,
        quant_errors: torch.Tensor,
        signatures: typing.Optional[torch.Tensor] = None,
    ):
        if not full_records_bytes:
            return

        with self._buffer_lock:
            batch_len = td_tensors.size(0)

            self.qe_rolling_window.extend(quant_errors.detach().cpu().tolist())
            if len(self.qe_rolling_window) > 100:
                qe_array = np.array(self.qe_rolling_window, dtype=np.float32)
                self.qe_p50 = float(np.percentile(qe_array, 50))
                self.qe_p90 = float(np.percentile(qe_array, 90))
                self.qe_p99 = float(np.percentile(qe_array, 99))

            bands = torch.where(
                quant_errors < self.qe_p50,
                int(MemoryQualityBand.GREEN),
                torch.where(quant_errors < self.qe_p90, int(MemoryQualityBand.YELLOW), int(MemoryQualityBand.RED)),
            ).byte()

            if self.write_ptr + batch_len > self.max_records:
                self.write_ptr = 0

            end_ptr = self.write_ptr + batch_len

            self.buffer_td[self.write_ptr : end_ptr].copy_(td_tensors.view(batch_len, -1), non_blocking=True)
            self.buffer_qe[self.write_ptr : end_ptr].copy_(quant_errors.view(batch_len, -1), non_blocking=True)
            self.buffer_quality_band[self.write_ptr : end_ptr].copy_(bands.view(batch_len, -1), non_blocking=True)

            if signatures is not None:
                self.buffer_signatures[self.write_ptr : end_ptr].copy_(
                    signatures.view(batch_len, -1), non_blocking=True
                )

            self.write_ptr = end_ptr
            self.active_records = min(self.max_records, self.active_records + batch_len)

            joined_ep = b"".join(full_records_bytes)
            joined_td = td_tensors.cpu().numpy().astype(np.float16).tobytes()

            new_meta = ReplaySegmentMeta(
                segment_id=self.active_segment_id,
                record_count=batch_len,
                quality_band=0,
                true_quant_error=self.qe_p50,
                scale=1.0,
                record_dim=256,
                td_dim=1,
                record_stride_bytes=512,
                td_stride_bytes=2,
                payload_version=1,
                retrieval_hits=0,
                utility_score=0.0,
                last_compacted_step=0,
                prototype_popcount=0,
                format_signature=b"VRLX",
                checksum=zlib.crc32(joined_ep),
                tombstone=False,
            )

            sig_list = []
            if signatures is not None:
                sig_cpu = signatures.cpu().numpy()
                sig_list = [sig_cpu[i].tobytes() for i in range(batch_len)]

            with self._segment_id_lock:
                target_segment_id = self.active_segment_id
                self.active_segment_id += 1
                try:
                    self.write_queue.put((target_segment_id, joined_ep, joined_td, new_meta, sig_list), timeout=2.0)
                except queue.Full:
                    import logging

                    logging.warning(
                        f"LMDB write queue saturated. Dropping segment {target_segment_id} to prevent thread lock."
                    )

    def batch_read_sync(self, segment_ids: list):
        """Maps LMDB byte buffers to PyTorch tensors synchronously."""
        ep_tensors = []
        td_tensors = []
        meta_list = []
        sig_list: typing.List[typing.Optional[torch.Tensor]] = []

        for seg_id in segment_ids:
            if seg_id == self.active_segment_id and len(self.segment_buffer_ep) > 0:
                ep_tensors.append(torch.from_numpy(np.frombuffer(b"".join(self.segment_buffer_ep), dtype=np.int8)))
                td_tensors.append(torch.from_numpy(np.frombuffer(b"".join(self.segment_buffer_td), dtype=np.float16)))
                from vrl_framework.core.contracts import ReplaySegmentMeta

                meta_list.append(
                    ReplaySegmentMeta(
                        segment_id=seg_id,
                        record_count=len(self.segment_buffer_ep),
                        quality_band=0,
                        true_quant_error=0.5,
                        scale=1.0,
                        record_dim=256,
                        td_dim=1,
                        record_stride_bytes=256,
                        td_stride_bytes=2,
                        payload_version=2,
                        retrieval_hits=0,
                        utility_score=0.0,
                        last_compacted_step=0,
                        prototype_popcount=0,
                        format_signature=b"VRLX",
                        checksum=0,
                        tombstone=False,
                    )
                )
                sig_list.append(None)

        with self.env.begin(buffers=True) as txn:
            for seg_id in segment_ids:
                if seg_id == self.active_segment_id:
                    continue
                meta_bytes = txn.get(f"segmeta:{seg_id}".encode("ascii"), db=self.meta_db)
                if not meta_bytes:
                    continue
                metadata = pickle.loads(meta_bytes)

                raw_bytes = txn.get(f"seg_ep_{seg_id}".encode("ascii"), db=self.segment_db)
                raw_td_bytes = txn.get(f"seg_td_{seg_id}".encode("ascii"), db=self.segment_db)

                if raw_bytes and raw_td_bytes:
                    assert (
                        len(raw_bytes) % metadata.record_stride_bytes == 0
                    ), "Corrupted LMDB segment: Epoch record byte misalignment."
                    assert (
                        len(raw_td_bytes) % metadata.td_stride_bytes == 0
                    ), "Corrupted LMDB segment: Temporal Difference byte misalignment."

                    np_arr = np.frombuffer(raw_bytes, dtype=np.int8)
                    ep_tensors.append(torch.from_numpy(np_arr.copy()))

                    np_td_arr = np.frombuffer(raw_td_bytes, dtype=np.float16)
                    td_tensors.append(torch.from_numpy(np_td_arr.copy()))

                    meta_list.append(metadata)

                    record_count = metadata.record_count
                    seg_sigs = []
                    for idx in range(record_count):
                        sig_bytes = txn.get(f"segrec:{seg_id}:{idx}".encode("ascii"), db=self.lsh_index_db)
                        if sig_bytes:
                            seg_sigs.append(torch.from_numpy(np.frombuffer(sig_bytes, dtype=np.bool_)))

                    sig_list.append(torch.stack(seg_sigs) if seg_sigs else None)
        return ep_tensors, td_tensors, meta_list, sig_list

    def flush(self) -> None:
        """Synchronously flushes buffers and saves offset pointers to disk."""
        if self.active_records > 0:
            try:
                old_size = getattr(self, "segment_size", 0)
                self.segment_size = self.active_records
                self.batch_write([], [], torch.tensor([]), torch.tensor([]))
                self.segment_size = old_size
            except Exception as e:
                logging.error(f"LMDB Queue drain error: {e}")

        if hasattr(self, "write_queue"):
            self.write_queue.join()

        if hasattr(self, "env"):
            self.env.sync(True)

        try:
            metadata_path = os.path.join(self.env.path(), "offsets_snapshot.json")
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "segment_counter": getattr(self, "segment_counter", 0),
                        "active_records": getattr(self, "active_records", 0),
                        "timestamp": time.time(),
                    },
                    f,
                )
        except Exception as e:
            logging.error(f"Error saving offsets snapshot: {e}")


class SemanticConsolidationEngine(nn.Module):
    """Executes VRAM to RAM migration via symmetric min-max int8 quantization."""

    def __init__(self, buffer_ref):
        super().__init__()
        self.buffer_ref = weakref.ref(buffer_ref)

    def execute_consolidation(self):
        """
        Applies symmetric min-max int8 quantization for VRAM->RAM offload.
        Drops records exceeding true_quant_error_threshold.
        """
        buffer = self.buffer_ref()
        if buffer is None:
            return

        ptr = buffer.bank_ptr.item()
        if ptr == 0:
            return

        with torch.no_grad():
            raw_memory = buffer.episodic_bank[:ptr]
            raw_td = buffer.td_error_profiles[:ptr]

            v_bound = MEM_CFG.quantization_v_bound
            scale = (2.0 * v_bound) / 255.0

            outlier_threshold = torch.quantile(torch.abs(raw_memory), 0.95)
            outlier_mask = torch.abs(raw_memory) > outlier_threshold

            clipped_memory = torch.clamp(raw_memory, min=-v_bound, max=v_bound)
            quantized_mem = ((clipped_memory + v_bound) / scale - 128).to(torch.int8)

            restored_base = ((quantized_mem.float() + 128.0) * scale - v_bound).half()
            restored = torch.where(outlier_mask, raw_memory, restored_base)

            true_quant_error = F.mse_loss(restored.float(), raw_memory.float(), reduction="none").mean(dim=-1)
            survival_mask = true_quant_error < MEM_CFG.true_quant_error_threshold

            valid_memory = quantized_mem[survival_mask]
            valid_td = raw_td[survival_mask]
            valid_qe = true_quant_error[survival_mask]
            survivors = valid_memory.size(0)

            if survivors == 0:
                buffer.bank_ptr.copy_(torch.tensor(0, dtype=torch.long))
                return

            valid_memory_fp32 = raw_memory[survival_mask].float()
            hash_planes_gpu = buffer.lsh_hash_planes_cpu.to(valid_memory_fp32.device)
            precomputed_signatures = (torch.matmul(valid_memory_fp32, hash_planes_gpu) > 0.0).cpu()

            written = 0
            while written < survivors:
                available_space = buffer.ring_buffer_capacity - buffer.ring_ptr
                insert_count = min(survivors - written, available_space)

                start_idx = buffer.ring_ptr
                end_idx = buffer.ring_ptr + insert_count
                w_end = written + insert_count

                buffer.ram_ring_buffer_ep[start_idx:end_idx].copy_(valid_memory[written:w_end], non_blocking=True)
                buffer.ram_ring_buffer_td[start_idx:end_idx].copy_(valid_td[written:w_end], non_blocking=True)

                buffer.ram_ring_buffer_next_states[start_idx:end_idx].copy_(
                    buffer.next_states[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_actions[start_idx:end_idx].copy_(
                    buffer.actions[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_old_logprobs[start_idx:end_idx].copy_(
                    buffer.old_logprobs[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_returns[start_idx:end_idx].copy_(
                    buffer.returns[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_advantages[start_idx:end_idx].copy_(
                    buffer.advantages[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_costs[start_idx:end_idx].copy_(
                    buffer.costs[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_dones[start_idx:end_idx].copy_(
                    buffer.dones[:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_buffer_episode_ids[start_idx:end_idx].copy_(
                    buffer.episode_ids[:ptr][survival_mask][written:w_end], non_blocking=True
                )

                buffer.ram_ring_snapshot_metagru[start_idx:end_idx].copy_(
                    buffer.recurrent_state_snapshot["metagru_h"][:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_snapshot_pondergru[start_idx:end_idx].copy_(
                    buffer.recurrent_state_snapshot["pondergru_h"][:ptr][survival_mask][written:w_end],
                    non_blocking=True,
                )
                buffer.ram_ring_snapshot_gradstm[start_idx:end_idx].copy_(
                    buffer.recurrent_state_snapshot["gradientstm_h"][:ptr][survival_mask][written:w_end],
                    non_blocking=True,
                )
                buffer.ram_ring_snapshot_stmptr[start_idx:end_idx].copy_(
                    buffer.recurrent_state_snapshot["stmptr"][:ptr][survival_mask][written:w_end], non_blocking=True
                )
                buffer.ram_ring_snapshot_stmtensor[start_idx:end_idx].copy_(
                    buffer.recurrent_state_snapshot["stmtensor_k"][:ptr][survival_mask][written:w_end],
                    non_blocking=True,
                )

                buffer.lsh_signatures_ram[start_idx:end_idx].copy_(precomputed_signatures[written:w_end])
                buffer.segment_id_ram[start_idx:end_idx] = buffer.runtime_context.lmdb_bank.active_segment_id

                buffer.ring_ptr += insert_count
                written += insert_count

                if buffer.ring_ptr >= int(buffer.ring_buffer_capacity * 0.9):
                    dump_size = buffer.ring_ptr
                    packed_records = []

                    latent_fp32 = buffer.ram_ring_buffer_ep[:dump_size].float().view(dump_size, -1)
                    nxt_lat = buffer.ram_ring_buffer_next_states[:dump_size].float().view(dump_size, -1)
                    act = buffer.ram_ring_buffer_actions[:dump_size].float().view(dump_size, -1)
                    old_lp = buffer.ram_ring_buffer_old_logprobs[:dump_size].float().view(dump_size, -1)
                    ret = buffer.ram_ring_buffer_returns[:dump_size].float().view(dump_size, -1)
                    adv = buffer.ram_ring_buffer_advantages[:dump_size].float().view(dump_size, -1)
                    cst = buffer.ram_ring_buffer_costs[:dump_size].float().view(dump_size, -1)
                    dne = buffer.ram_ring_buffer_dones[:dump_size].float().view(dump_size, -1)
                    ep_id = buffer.ram_ring_buffer_episode_ids[:dump_size].float().view(dump_size, -1)

                    snap_metagru = buffer.ram_ring_snapshot_metagru[:dump_size].float().view(dump_size, -1)
                    snap_pondergru = buffer.ram_ring_snapshot_pondergru[:dump_size].float().view(dump_size, -1)
                    snap_gradstm = buffer.ram_ring_snapshot_gradstm[:dump_size].float().view(dump_size, -1)
                    snap_stmptr = buffer.ram_ring_snapshot_stmptr[:dump_size].float().view(dump_size, -1)
                    snap_stmtensor = buffer.ram_ring_snapshot_stmtensor[:dump_size].float().view(dump_size, -1)

                    full_record_tensor = torch.cat(
                        [
                            latent_fp32,
                            nxt_lat,
                            act,
                            old_lp,
                            ret,
                            adv,
                            cst,
                            dne,
                            ep_id,
                            snap_metagru,
                            snap_pondergru,
                            snap_gradstm,
                            snap_stmptr,
                            snap_stmtensor,
                        ],
                        dim=-1,
                    ).numpy()

                    packed_records = [record.tobytes() for record in full_record_tensor]

                    td_dump = buffer.ram_ring_buffer_td[:dump_size].clone()
                    qe_dump = valid_qe[:dump_size].clone()
                    keys = [f"{buffer.agent_uuid}_{buffer.global_seq_id + i}" for i in range(dump_size)]
                    buffer.global_seq_id += dump_size
                    sig_dump = buffer.lsh_signatures_ram[:dump_size].clone()

                    def _lmdb_flush(k_list, packed_list, td_tensor, q_error_tensor, sig_tensor):
                        buffer.runtime_context.lmdb_bank.batch_write(
                            k_list, packed_list, td_tensor, q_error_tensor, sig_tensor
                        )

                    buffer.runtime_context.io_worker.submit(
                        _lmdb_flush, keys, packed_records, td_dump, qe_dump, sig_dump
                    )
                    buffer.ring_ptr = 0

            buffer.episodic_bank.fill_(0.0)
            buffer.td_error_profiles.fill_(0.0)
            buffer.bank_ptr.copy_(torch.tensor(0, dtype=torch.long))


class ArchivalPrefetchEngine(nn.Module):
    """Manages asynchronous I/O retrieval of LMDB segments into the VRAM core."""

    def __init__(self, buffer_ref):
        super().__init__()
        self.buffer_ref = weakref.ref(buffer_ref)

    def execute_prefetch(self, current_context=None):
        """Decodes queued SSD segments into VRAM."""
        buffer = self.buffer_ref()
        if buffer is None or buffer.runtime_context.lmdb_bank.prefetch_queue.empty():
            return

        segs_to_fetch = set()
        queue_limit = getattr(MEM_CFG, "prefetch_queue_limit", 64)
        fetched_count = 0

        while not buffer.runtime_context.lmdb_bank.prefetch_queue.empty() and fetched_count < queue_limit:
            segs_to_fetch.add(buffer.runtime_context.lmdb_bank.prefetch_queue.get())
            fetched_count += 1

        if not segs_to_fetch:
            return
        retrieved_tensors, td_tensors, meta_list, sig_list = buffer.runtime_context.lmdb_bank.batch_read_sync(
            list(segs_to_fetch)
        )
        if not retrieved_tensors:
            return

        if current_context is not None:
            # Project context to 32 bits for Hamming distance computation: [B, D] x [D, 32] -> [B, 32]
            q_sig = torch.matmul(current_context.float().cpu(), buffer.lsh_hash_planes_cpu) > 0.0
            if q_sig.dim() == 1:
                q_sig = q_sig.unsqueeze(0)

        with torch.no_grad():
            valid_tensors = []
            valid_seg_ids = []
            valid_rec_indices = []
            valid_q_bands = []

            for ep_t, td_t, meta, sigs in zip(retrieved_tensors, td_tensors, meta_list, sig_list):
                if meta.quality_band == MemoryQualityBand.RED:
                    continue

                stride = meta.record_stride_bytes // 4
                if ep_t.shape[0] % stride == 0:
                    t_reshaped = ep_t.view(-1, stride)
                    t_latent = t_reshaped[:, : buffer.dim]

                    td_stride = meta.td_stride_bytes // 2
                    td_reshaped = td_t.view(-1, td_stride)
                    td_signal = td_reshaped.mean(dim=-1)

                    if current_context is not None and sigs is not None and sigs.size(0) == t_reshaped.size(0):
                        hamming_dists = torch.sum(q_sig.unsqueeze(1) ^ sigs.unsqueeze(0), dim=-1).mean(dim=0)
                        td_ranking_score = hamming_dists.float() - (td_signal.float() * 0.1)
                        top_m = min(getattr(MEM_CFG, "retrieval_top_k_limit", 16), t_reshaped.size(0))
                        _, top_idx = torch.topk(td_ranking_score, top_m, largest=False)
                        t_latent = t_latent[top_idx]

                        valid_seg_ids.append(
                            torch.full((top_m,), meta.segment_id, dtype=torch.long, device=buffer.core.device)
                        )
                        valid_rec_indices.append(top_idx.to(buffer.core.device))
                        valid_q_bands.append(
                            torch.full((top_m,), meta.quality_band, dtype=torch.long, device=buffer.core.device)
                        )
                    else:
                        rec_count = t_latent.size(0)
                        valid_seg_ids.append(
                            torch.full((rec_count,), meta.segment_id, dtype=torch.long, device=buffer.core.device)
                        )
                        valid_rec_indices.append(torch.arange(rec_count, dtype=torch.long, device=buffer.core.device))
                        valid_q_bands.append(
                            torch.full((rec_count,), meta.quality_band, dtype=torch.long, device=buffer.core.device)
                        )

                    scale = meta.scale
                    v_bound = scale * 127.0
                    restored = (t_latent.float().to(buffer.episodic_bank.device) + 128.0) * scale - v_bound
                    restored_fp16 = restored.half()
                    corrected_tensor = restored_fp16 + buffer.dequantization_corrector(restored_fp16)

                    PlannerValidator.assert_no_nan_inf(corrected_tensor, "LMDB_Prefetch_Reconstruction")
                    valid_tensors.append(corrected_tensor)

                    meta.retrieval_hits += 1
                    buffer.runtime_context.lmdb_bank.segmeta_cache[meta.segment_id] = meta

                    updated_sigs = (
                        (
                            torch.matmul(
                                corrected_tensor.float(), buffer.lsh_hash_planes_cpu.to(corrected_tensor.device)
                            )
                            > 0.0
                        )
                        .cpu()
                        .numpy()
                    )

                    with buffer.runtime_context.lmdb_bank.env.begin(write=True) as write_txn:
                        write_txn.put(
                            f"segmeta:{meta.segment_id}".encode("ascii"),
                            pickle.dumps(meta),
                            db=buffer.runtime_context.lmdb_bank.meta_db,
                        )
                        for local_idx in range(corrected_tensor.size(0)):
                            write_txn.put(
                                f"segrec:{meta.segment_id}:{local_idx}".encode("ascii"),
                                updated_sigs[local_idx].tobytes(),
                                db=buffer.runtime_context.lmdb_bank.lsh_index_db,
                            )

            if valid_tensors:
                new_core_blocks = torch.cat(valid_tensors, dim=0)
                new_seg_ids = torch.cat(valid_seg_ids, dim=0)
                new_rec_indices = torch.cat(valid_rec_indices, dim=0)
                new_q_bands = torch.cat(valid_q_bands, dim=0)

                insert_size = min(new_core_blocks.size(0), buffer.core.size(0))

                buffer.core.copy_(torch.roll(buffer.core, shifts=-insert_size, dims=0))
                buffer.core[-insert_size:] = new_core_blocks[:insert_size]

                buffer.core_source_segmentids.copy_(
                    torch.roll(buffer.core_source_segmentids, shifts=-insert_size, dims=0)
                )
                buffer.core_source_segmentids[-insert_size:] = new_seg_ids[:insert_size]

                buffer.core_source_recordindices.copy_(
                    torch.roll(buffer.core_source_recordindices, shifts=-insert_size, dims=0)
                )
                buffer.core_source_recordindices[-insert_size:] = new_rec_indices[:insert_size]

                buffer.core_source_qualitybands.copy_(
                    torch.roll(buffer.core_source_qualitybands, shifts=-insert_size, dims=0)
                )
                buffer.core_source_qualitybands[-insert_size:] = new_q_bands[:insert_size]


class SemanticRetrievalInterface:
    """Retrieves memory segments based on Lorentz projected distances."""

    def __init__(self, q_episodic: nn.Module, q_alarm: nn.Module, q_procedural: nn.Module, dim: int):
        self.q_episodic = q_episodic
        self.q_alarm = q_alarm
        self.q_procedural = q_procedural
        self.dim = dim

    def execute_tri_path_retrieval(
        self,
        current_context: torch.Tensor,
        valid_memory: torch.Tensor,
        valid_td: torch.Tensor,
        combined_seg_ids: torch.Tensor,
        combined_rec_idx: torch.Tensor,
        combined_q_bands: torch.Tensor,
        top_k: int,
    ) -> RetrievedMemoryBatch:
        """Computes top-k memory aggregations using projected similarity spaces."""
        if valid_memory.size(0) < top_k:
            return RetrievedMemoryBatch(
                episodic_context=torch.zeros_like(current_context),
                alarm_context=torch.zeros_like(current_context),
                procedural_context=torch.zeros_like(current_context),
                episodic_source_segment_ids=torch.zeros(0, dtype=torch.long, device=current_context.device),
                episodic_source_record_indices=torch.zeros(0, dtype=torch.long, device=current_context.device),
                episodic_source_qualitybands=torch.zeros(0, dtype=torch.long, device=current_context.device),
                context_confidence=torch.zeros(0, device=current_context.device),
                is_empty=True,
            )

        q_ep = LorentzGeometry.project(self.q_episodic(current_context))
        q_al = LorentzGeometry.project(self.q_alarm(current_context))
        q_pr = LorentzGeometry.project(self.q_procedural(current_context))

        valid_memory_proj = LorentzGeometry.project(valid_memory)

        q_ep_exp = q_ep.unsqueeze(1)
        q_al_exp = q_al.unsqueeze(1)
        q_pr_exp = q_pr.unsqueeze(1)
        mem_exp = valid_memory_proj.unsqueeze(0)

        raw_dist_ep = LorentzGeometry.distance(q_ep_exp, mem_exp)
        raw_dist_al = LorentzGeometry.distance(q_al_exp, mem_exp)
        raw_dist_pr = LorentzGeometry.distance(q_pr_exp, mem_exp)

        historical_surprise = torch.abs(valid_td.t()) + 1e-4
        surprise_modulator = torch.log1p(historical_surprise)

        sim_ep = -raw_dist_ep * surprise_modulator
        ep_scores, top_ep_idx = torch.topk(sim_ep, top_k, dim=-1)
        episodic_context = valid_memory[top_ep_idx].mean(dim=1)
        self.last_ecq_error = F.mse_loss(episodic_context.detach(), current_context.detach())
        if hasattr(self, "ram_ring_buffer_ep"):
            safe_k = min(top_k, top_ep_idx.size(1))
            safe_idx = top_ep_idx[:, :safe_k]
            valid_mask = safe_idx < self.ram_ring_buffer_ep.size(0)

            if valid_mask.any():
                cpu_idx = safe_idx[valid_mask].cpu()
                b_indices = (
                    torch.arange(current_context.size(0), device=current_context.device).view(-1, 1).expand(-1, safe_k)
                )
                cpu_ctx = current_context[b_indices[valid_mask]].half().cpu()
                self.ram_ring_buffer_ep[cpu_idx] = self.ram_ring_buffer_ep[cpu_idx] * 0.99 + cpu_ctx * 0.01

        alarm_logits = -raw_dist_al * valid_td.t()
        al_scores, top_al_idx = torch.topk(alarm_logits, top_k, dim=-1)
        alarm_context = valid_memory[top_al_idx].mean(dim=1)

        sim_pr = -raw_dist_pr
        pr_scores, top_pr_idx = torch.topk(sim_pr, top_k, dim=-1)
        procedural_context = valid_memory[top_pr_idx].mean(dim=1)

        def _calc_confidence_margin(scores: torch.Tensor) -> torch.Tensor:
            if scores.size(-1) < 2:
                return torch.zeros(scores.size(0), device=scores.device)
            top2 = torch.topk(scores, 2, dim=-1).values
            return top2[:, 0] - top2[:, 1]

        ep_margin = _calc_confidence_margin(sim_ep)
        al_margin = _calc_confidence_margin(alarm_logits)
        pr_margin = _calc_confidence_margin(sim_pr)

        confidence = (ep_margin + al_margin + pr_margin) / 3.0

        return RetrievedMemoryBatch(
            episodic_context=episodic_context,
            alarm_context=alarm_context,
            procedural_context=procedural_context,
            episodic_source_segment_ids=combined_seg_ids[top_ep_idx],
            episodic_source_record_indices=combined_rec_idx[top_ep_idx],
            episodic_source_qualitybands=combined_q_bands[top_ep_idx],
            context_confidence=confidence,
            is_empty=False,
        )


class EpisodicReplayBuffer(nn.Module):
    """Implements a 3-tier replay buffer across VRAM, RAM, and SSD.

    Maintains recent transitions in VRAM. Quantizes and offloads older records
    to host RAM and SSD asynchronously. Preserves temporal ordering for
    Backpropagation Through Time (BPTT).
    """

    def __init__(self, runtime_context, dim=256):
        super().__init__()
        self.runtime_context = runtime_context
        self.dim = dim
        self.hot_capacity = MEM_CFG.hot_capacity

        self.consolidator = SemanticConsolidationEngine(self)
        self.prefetcher = ArchivalPrefetchEngine(self)

        self.q_episodic = nn.Linear(dim, dim, bias=False)
        self.q_alarm = nn.Linear(dim, dim, bias=False)
        self.q_procedural = nn.Linear(dim, dim, bias=False)

        self.dequantization_corrector = nn.Sequential(nn.Linear(dim, dim), nn.Mish(), nn.Linear(dim, dim))

        self.register_buffer("core", torch.zeros(1024, dim, dtype=torch.float16))

        self.register_buffer("core_source_segmentids", torch.zeros(1024, dtype=torch.long))
        self.register_buffer("core_source_recordindices", torch.zeros(1024, dtype=torch.long))
        self.register_buffer("core_source_qualitybands", torch.zeros(1024, dtype=torch.long))

        self.register_buffer("episodic_bank", torch.zeros(self.hot_capacity, dim, dtype=torch.float16))
        self.register_buffer("td_error_profiles", torch.zeros(self.hot_capacity, dim, dtype=torch.float16))

        self.register_buffer("dones", torch.zeros(self.hot_capacity, dtype=torch.bool))
        self.register_buffer("episode_ids", torch.zeros(self.hot_capacity, dtype=torch.long))
        self.register_buffer("actions", torch.zeros(self.hot_capacity, dtype=torch.long))
        self.register_buffer("old_logprobs", torch.zeros(self.hot_capacity, dtype=torch.float32))
        self.register_buffer("returns", torch.zeros(self.hot_capacity, dtype=torch.float32))
        self.register_buffer("advantages", torch.zeros(self.hot_capacity, dtype=torch.float32))
        self.register_buffer("next_states", torch.zeros(self.hot_capacity, dim, dtype=torch.float16))
        self.register_buffer("costs", torch.zeros(self.hot_capacity, dtype=torch.float32))

        self.recurrent_state_snapshot = {
            "metagru_h": torch.zeros(self.hot_capacity, 256, dtype=torch.float16, device=self.core.device),
            "pondergru_h": torch.zeros(self.hot_capacity, 256, dtype=torch.float16, device=self.core.device),
            "gradientstm_h": torch.zeros(self.hot_capacity, 256, dtype=torch.float16, device=self.core.device),
            "stmptr": torch.zeros(self.hot_capacity, dtype=torch.long, device=self.core.device),
            "stmtensor_k": torch.zeros(self.hot_capacity, 8, 256, dtype=torch.float16, device=self.core.device),
        }

        self.register_buffer("bank_ptr", torch.tensor(0, dtype=torch.long))

        self.ring_buffer_capacity = MEM_CFG.ring_buffer_capacity
        self.ram_ring_buffer_ep = torch.zeros(self.ring_buffer_capacity, dim, dtype=torch.int8, device="cpu")
        self.ram_ring_buffer_td = torch.zeros(self.ring_buffer_capacity, dim, dtype=torch.float16, device="cpu")

        self.ram_ring_buffer_next_states = torch.zeros(
            self.ring_buffer_capacity, dim, dtype=torch.float16, device="cpu"
        )
        self.ram_ring_buffer_actions = torch.zeros(self.ring_buffer_capacity, dtype=torch.long, device="cpu")
        self.ram_ring_buffer_old_logprobs = torch.zeros(self.ring_buffer_capacity, dtype=torch.float32, device="cpu")
        self.ram_ring_buffer_returns = torch.zeros(self.ring_buffer_capacity, dtype=torch.float32, device="cpu")
        self.ram_ring_buffer_advantages = torch.zeros(self.ring_buffer_capacity, dtype=torch.float32, device="cpu")
        self.ram_ring_buffer_costs = torch.zeros(self.ring_buffer_capacity, dtype=torch.float32, device="cpu")
        self.ram_ring_buffer_dones = torch.zeros(self.ring_buffer_capacity, dtype=torch.bool, device="cpu")
        self.ram_ring_buffer_episode_ids = torch.zeros(self.ring_buffer_capacity, dtype=torch.long, device="cpu")

        self.ram_ring_snapshot_metagru = torch.zeros(self.ring_buffer_capacity, 256, dtype=torch.float16, device="cpu")
        self.ram_ring_snapshot_pondergru = torch.zeros(
            self.ring_buffer_capacity, 256, dtype=torch.float16, device="cpu"
        )
        self.ram_ring_snapshot_gradstm = torch.zeros(self.ring_buffer_capacity, 256, dtype=torch.float16, device="cpu")
        self.ram_ring_snapshot_stmptr = torch.zeros(self.ring_buffer_capacity, dtype=torch.long, device="cpu")
        self.ram_ring_snapshot_stmtensor = torch.zeros(
            self.ring_buffer_capacity, 8, 256, dtype=torch.float16, device="cpu"
        )

        self.ring_ptr = 0

        self.lsh_hash_planes_cpu = torch.randn(dim, 32, dtype=torch.float32, device="cpu")
        self.lsh_signatures_ram = torch.zeros(self.ring_buffer_capacity, 32, dtype=torch.bool, device="cpu")
        self.segment_id_ram = torch.zeros(self.ring_buffer_capacity, dtype=torch.long, device="cpu")

        self.agent_uuid = uuid.uuid4().hex[:8]
        self.global_seq_id = 0
        self.step_counter = 0

    def query_lsh_cpu_background(self, query_vector: torch.Tensor, top_k: int = 16):
        """Dispatches an asynchronous LSH lookup utilizing CPU thread pools.

        Isolates Hamming-distance computation from the GPU to prevent
        CUDA synchronization blocks during forward/backward passes.

        Args:
            query_vector: Dense state representation of shape [D].
            top_k: Limit of unique segments to queue for prefetching.
        """
        query_cpu = query_vector.detach().cpu().float()

        def _cpu_hamming_search(q_vec, limit, k):
            query_sig = torch.matmul(q_vec, self.lsh_hash_planes_cpu) > 0.0

            if limit > 0:
                valid_sigs = self.lsh_signatures_ram[:limit]
                hamming_distances = torch.sum(query_sig.unsqueeze(0) ^ valid_sigs, dim=-1)
                k_safe = min(k, valid_sigs.size(0))
                _, best_idx = torch.topk(hamming_distances, k_safe, largest=False)

                for idx in best_idx:
                    seg_id = self.segment_id_ram[idx].item()
                    if not self.runtime_context.lmdb_bank.prefetch_queue.full():
                        self.runtime_context.lmdb_bank.prefetch_queue.put_nowait(seg_id)

            seg_cache = list(self.runtime_context.lmdb_bank.segproto_cache.items())
            if seg_cache:
                seg_ids = [item[0] for item in seg_cache]
                sig_matrix = torch.from_numpy(np.stack([np.frombuffer(item[1], dtype=np.bool_) for item in seg_cache]))

                ssd_dists = torch.sum(query_sig.unsqueeze(0) ^ sig_matrix, dim=-1)
                _, sorted_indices = torch.sort(ssd_dists)

                for idx in sorted_indices[:64]:
                    seg_id = seg_ids[idx.item()]
                    if not self.runtime_context.lmdb_bank.prefetch_queue.full():
                        self.runtime_context.lmdb_bank.prefetch_queue.put_nowait(seg_id)

        self.runtime_context.compute_worker.submit(_cpu_hamming_search, query_cpu, self.ring_ptr, top_k)

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        old_logprob: torch.Tensor,
        ret: torch.Tensor,
        adv: torch.Tensor,
        next_state: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        episode_id: torch.Tensor,
        snapshot: RecurrentStateSnapshot,
        td_error: typing.Optional[torch.Tensor] = None,
        is_external: bool = False,
    ) -> None:

        if state.dim() not in [1, 2]:
            raise ValueError("State tensor bounds violation.")
        if state.shape[-1] != self.dim:
            raise ValueError(f"State dimension mismatch. Expected {self.dim}, got {state.shape[-1]}.")
        if action.dtype != torch.long:
            raise TypeError(f"Discrete action topology requires torch.long, found {action.dtype}.")

        PlannerValidator.assert_no_nan_inf(state, "MemoryStore_State")
        PlannerValidator.assert_no_nan_inf(next_state, "MemoryStore_NextState")

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if td_error is None:
            td_error = torch.zeros_like(state)
        if td_error.dim() == 1:
            td_error = td_error.unsqueeze(0)

        with torch.no_grad():
            ptr = self.bank_ptr.item()
            b_size = state.size(0)

            if ptr + b_size >= self.hot_capacity:
                if hasattr(self, "consolidator"):
                    self.consolidator.execute_consolidation()
                else:
                    self._semantic_consolidation()
                ptr = self.bank_ptr.item()

            if ptr + b_size < self.hot_capacity:
                self.episodic_bank[ptr : ptr + b_size] = state.to(torch.float16)
                self.actions[ptr : ptr + b_size] = action
                self.old_logprobs[ptr : ptr + b_size] = old_logprob
                self.returns[ptr : ptr + b_size] = ret
                self.advantages[ptr : ptr + b_size] = adv
                self.next_states[ptr : ptr + b_size] = next_state.to(torch.float16)
                self.costs[ptr : ptr + b_size] = cost
                self.dones[ptr : ptr + b_size] = done
                self.episode_ids[ptr : ptr + b_size] = episode_id

                self.recurrent_state_snapshot["metagru_h"][ptr : ptr + b_size] = snapshot.metagru_h
                self.recurrent_state_snapshot["pondergru_h"][ptr : ptr + b_size] = snapshot.pondergru_h
                self.recurrent_state_snapshot["gradientstm_h"][ptr : ptr + b_size] = snapshot.gradientstm_h
                self.recurrent_state_snapshot["stmptr"][ptr : ptr + b_size] = snapshot.stmptr
                self.recurrent_state_snapshot["stmtensor_k"][ptr : ptr + b_size] = snapshot.stmtensor_k

                self.td_error_profiles[ptr : ptr + b_size] = td_error.to(torch.float16)
                self.bank_ptr += b_size
            else:
                raise RuntimeError(
                    f"QualityGate Failed: Insufficient hot memory capacity after consolidation. "
                    f"Needed {b_size}, available {self.hot_capacity - ptr}."
                )

    def store(
        self, x: torch.Tensor, td_error: typing.Optional[torch.Tensor] = None, is_external: bool = False
    ) -> None:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Dimension constraint violation on generic store. Expected {self.dim}, got {x.shape[-1]}."
            )
        PlannerValidator.assert_no_nan_inf(x, "MemoryStore_Legacy_X")

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if td_error is None:
            td_error = torch.zeros_like(x)
        if td_error.dim() == 1:
            td_error = td_error.unsqueeze(0)

        with torch.no_grad():
            ptr = self.bank_ptr.item()
            b_size = x.size(0)

            if ptr + b_size >= self.hot_capacity:
                if hasattr(self, "consolidator"):
                    self.consolidator.execute_consolidation()
                else:
                    self._semantic_consolidation()
                ptr = self.bank_ptr.item()

            if ptr + b_size < self.hot_capacity:
                self.episodic_bank[ptr : ptr + b_size] = x.to(torch.float16)
                self.td_error_profiles[ptr : ptr + b_size] = td_error.to(torch.float16)
                self.bank_ptr += b_size

    def retrieve_triple_head(self, current_context: torch.Tensor, top_k: int = 16) -> RetrievedMemoryBatch:
        if current_context.dim() not in [1, 2]:
            raise ValueError("Invalid dimension constraint on retrieval context mapping.")
        if current_context.shape[-1] != self.dim:
            raise ValueError(f"Context mismatch. Expected {self.dim}, got {current_context.shape[-1]}.")
        PlannerValidator.assert_no_nan_inf(current_context, "RetrieveTripleHead_Context")

        if hasattr(self, "prefetcher"):
            self.prefetcher.execute_prefetch(current_context)
        elif hasattr(self, "_sync_lmdb_prefetch"):
            self._sync_lmdb_prefetch(current_context)

        if not hasattr(self, "_semantic_retrieval_interface"):
            self._semantic_retrieval_interface = SemanticRetrievalInterface(
                self.q_episodic, self.q_alarm, self.q_procedural, self.dim
            )

        ptr = self.bank_ptr.item()
        with torch.no_grad():
            if ptr > 0:
                valid_stm = self.episodic_bank[:ptr]
                valid_stm_td = self.td_error_profiles[:ptr].mean(dim=-1, keepdim=True)
                stm_seg_ids = torch.full((ptr,), -1, dtype=torch.long, device=self.episodic_bank.device)
                stm_rec_idx = torch.arange(ptr, dtype=torch.long, device=self.episodic_bank.device)
                stm_q_bands = torch.full(
                    (ptr,), MemoryQualityBand.GREEN, dtype=torch.long, device=self.episodic_bank.device
                )
            else:
                valid_stm = torch.empty(0, self.dim, device=self.episodic_bank.device, dtype=torch.float16)
                valid_stm_td = torch.empty(0, 1, device=self.episodic_bank.device, dtype=torch.float16)
                stm_seg_ids = torch.empty(0, dtype=torch.long, device=self.episodic_bank.device)
                stm_rec_idx = torch.empty(0, dtype=torch.long, device=self.episodic_bank.device)
                stm_q_bands = torch.empty(0, dtype=torch.long, device=self.episodic_bank.device)

            active_mask = torch.abs(self.core).sum(dim=-1) > 0.0
            active_core = self.core[active_mask]
            core_td = torch.zeros(active_core.size(0), 1, device=self.episodic_bank.device, dtype=torch.float16)

            core_seg_ids = self.core_source_segmentids[active_mask]
            core_rec_idx = self.core_source_recordindices[active_mask]
            core_q_bands = self.core_source_qualitybands[active_mask]

            valid_memory = torch.cat([valid_stm, active_core], dim=0)
            valid_td = torch.cat([valid_stm_td, core_td], dim=0)

            combined_seg_ids = torch.cat([stm_seg_ids, core_seg_ids], dim=0)
            combined_rec_idx = torch.cat([stm_rec_idx, core_rec_idx], dim=0)
            combined_q_bands = torch.cat([stm_q_bands, core_q_bands], dim=0)

            return self._semantic_retrieval_interface.execute_tri_path_retrieval(
                current_context, valid_memory, valid_td, combined_seg_ids, combined_rec_idx, combined_q_bands, top_k
            )

    def get_memory(self) -> torch.Tensor:
        if torch.abs(self.core).sum() == 0:
            return torch.zeros(self.dim, device=self.core.device, dtype=torch.float32)

        with torch.no_grad():
            ptr = max(1, self.bank_ptr.item())
            recent_context = self.episodic_bank[:ptr].mean(dim=0, keepdim=True)

            if torch.abs(recent_context).sum() == 0:
                recent_context = torch.ones(1, self.dim, device=self.core.device, dtype=torch.float16)

            scores = F.cosine_similarity(recent_context, self.core, dim=-1)
            attention_weights = F.softmax((scores / 0.1).float(), dim=0).to(scores.dtype).unsqueeze(1)
            retrieved_memory = torch.sum(self.core * attention_weights, dim=0)

        return retrieved_memory.float()

    def store_episode(self, episode: torch.Tensor):
        self.store(episode, is_external=False)

    @dataclass
    class SequenceBuildStats:
        attempts: int
        valid_starts: int
        episode_rejections: int
        done_rejections: int
        returned_batchsize: int

    def samplesequences(self, batch_size: int, chunk_size: int = 16, burnin: int = 4):
        """Samples temporally contiguous trajectories for BPTT.

        Filters out windows crossing terminal states or episode boundaries.

        Returns:
            PolicySequenceBatch of shape [B, T_burn + T_learn, D] or None if insufficient memory.
        """
        ptr = min(self.bank_ptr.item(), self.hot_capacity)
        total_window = chunk_size + burnin
        if ptr < total_window:
            return None

        valid_starts = ptr - total_window + 1
        max_attempts = batch_size * 10

        # Sample base indices and compute sliding windows
        bulk_candidates = torch.randint(0, valid_starts, (max_attempts,), device=self.episodic_bank.device)
        offsets = torch.arange(total_window, device=self.episodic_bank.device)

        gather_idx = bulk_candidates.unsqueeze(1) + offsets.unsqueeze(0)

        # Filter out sequences crossing episode boundaries
        eps_window = self.episode_ids[gather_idx]
        ep_valid = (eps_window == eps_window[:, 0:1]).all(dim=1)

        dones_window = self.dones[bulk_candidates.unsqueeze(1) + offsets[:-1].unsqueeze(0)]
        dones_valid = ~dones_window.any(dim=1)

        combined_valid = ep_valid & dones_valid
        valid_indices = torch.nonzero(combined_valid).squeeze(-1)

        if valid_indices.numel() >= batch_size:
            used_limit = int(valid_indices[batch_size - 1].item()) + 1
            sampled_indices = bulk_candidates[valid_indices[:batch_size]].tolist()
        else:
            used_limit = int(max_attempts)
            sampled_indices = bulk_candidates[valid_indices].tolist()

        attempts = used_limit
        checked_ep_valid = ep_valid[:used_limit]
        checked_dones_valid = dones_valid[:used_limit]

        episode_rejections = (~checked_ep_valid).sum().item()
        done_rejections = (checked_ep_valid & ~checked_dones_valid).sum().item()

        if len(sampled_indices) == 0:
            empty_snap = RecurrentStateSnapshot(
                metagru_h=torch.empty(0, 256, device=self.episodic_bank.device),
                pondergru_h=torch.empty(0, 256, device=self.episodic_bank.device),
                gradientstm_h=torch.empty(0, 256, device=self.episodic_bank.device),
                stmptr=torch.empty(0, dtype=torch.long, device=self.episodic_bank.device),
                stmtensor_k=torch.empty(0, 8, 256, device=self.episodic_bank.device),
            )
            empty_build = self.SequenceBuildStats(
                attempts=attempts,
                valid_starts=valid_starts,
                episode_rejections=episode_rejections,
                done_rejections=done_rejections,
                returned_batchsize=0,
            )
            return PolicySequenceBatch(
                states=torch.empty(0, total_window, self.dim, device=self.episodic_bank.device),
                actions=torch.empty(0, total_window, dtype=torch.long, device=self.episodic_bank.device),
                old_logprobs=torch.empty(0, total_window, dtype=torch.float32, device=self.episodic_bank.device),
                returns=torch.empty(0, total_window, dtype=torch.float32, device=self.episodic_bank.device),
                advantages=torch.empty(0, total_window, dtype=torch.float32, device=self.episodic_bank.device),
                next_states=torch.empty(0, total_window, self.dim, device=self.episodic_bank.device),
                costs=torch.empty(0, total_window, dtype=torch.float32, device=self.episodic_bank.device),
                dones=torch.empty(0, total_window, dtype=torch.bool, device=self.episodic_bank.device),
                episode_ids=torch.empty(0, total_window, dtype=torch.long, device=self.episodic_bank.device),
                recurrent_snapshots=empty_snap,
                valid_mask=torch.empty(0, total_window, dtype=torch.bool, device=self.episodic_bank.device),
                alive=torch.empty(0, total_window, dtype=torch.bool, device=self.episodic_bank.device),
                burnin=burnin,
                learn_length=chunk_size,
                buildstats=empty_build,
            )

        start_tensor = torch.tensor(sampled_indices, device=self.episodic_bank.device)
        chunk_offsets = torch.arange(total_window, device=self.episodic_bank.device)
        gather_indices = start_tensor.unsqueeze(1) + chunk_offsets.unsqueeze(0)

        states = self.episodic_bank[gather_indices].float()
        actions = self.actions[gather_indices]
        old_logprobs = self.old_logprobs[gather_indices]
        returns = self.returns[gather_indices]
        advantages = self.advantages[gather_indices]
        next_states = self.next_states[gather_indices].float()
        costs = self.costs[gather_indices]
        dones = self.dones[gather_indices]
        episode_ids = self.episode_ids[gather_indices]
        valid_mask = torch.ones_like(dones, dtype=torch.bool)

        snapshot_obj = RecurrentStateSnapshot(
            metagru_h=self.recurrent_state_snapshot["metagru_h"][start_tensor].float(),
            pondergru_h=self.recurrent_state_snapshot["pondergru_h"][start_tensor].float(),
            gradientstm_h=self.recurrent_state_snapshot["gradientstm_h"][start_tensor].float(),
            stmptr=self.recurrent_state_snapshot["stmptr"][start_tensor],
            stmtensor_k=self.recurrent_state_snapshot["stmtensor_k"][start_tensor].float(),
        )

        assert states.shape == next_states.shape, "State and Next State tensors must possess identical dimensions."
        assert (
            actions.shape[1] == old_logprobs.shape[1] == returns.shape[1] == advantages.shape[1] == costs.shape[1]
        ), "Sequence length mismatch detected across transition tensors."
        assert not dones[:, :-1].any(), "Terminal states intercepted within the non-terminal segment boundaries."

        build_stats = self.SequenceBuildStats(
            attempts=attempts,
            valid_starts=valid_starts,
            episode_rejections=episode_rejections,
            done_rejections=done_rejections,
            returned_batchsize=len(sampled_indices),
        )

        return PolicySequenceBatch(
            states=states,
            actions=actions,
            old_logprobs=old_logprobs,
            returns=returns,
            advantages=advantages,
            next_states=next_states,
            costs=costs,
            dones=dones,
            episode_ids=episode_ids,
            recurrent_snapshots=snapshot_obj,
            valid_mask=valid_mask,
            alive=valid_mask,
            burnin=burnin,
            learn_length=chunk_size,
            buildstats=build_stats,
        )

    def decode_action_distributions(self, latent_vector, actor_module=None):
        """Decodes closest actions to a given latent state using cosine similarity."""
        with torch.no_grad():
            valid_memories = self.episodic_bank[: max(1, self.bank_ptr.item())]
            if valid_memories.size(0) < 2:
                return "Insufficient memories for VSA decoding."

            latent_norm = F.normalize(latent_vector.unsqueeze(0).float(), p=2, dim=-1)
            memory_norm = F.normalize(valid_memories.float(), p=2, dim=-1)

            similarities = F.cosine_similarity(latent_norm, memory_norm, dim=-1)
            top_scores, top_idx = torch.topk(similarities, min(3, valid_memories.size(0)))

            output_str = f"[LATENT VECTOR] L2 Norm: {latent_vector.norm().item():.3f}\n"
            top_scores_list = top_scores.tolist()
            top_idx_list = top_idx.tolist()

            top_actions = None
            if actor_module is not None:
                ac_outputs = actor_module(valid_memories[top_idx])
                top_actions = torch.argmax(ac_outputs.policy_logits[:, :16], dim=-1).tolist()

                if not hasattr(self, "_action_labels_cache"):
                    self._action_labels_cache = {
                        0: "QUERY_LMDB_HISTORY",
                        1: "ALLOCATE_SUBGOAL",
                        2: "RUN_LATENT_MCTS",
                        3: "PRUNE_COMPONENTS",
                        4: "BIND_CONTEXT",
                        5: "UNBIND_CONTEXT",
                        6: "EVALUATE_ERROR",
                        7: "UPDATE_NETWORK_WEIGHTS",
                        8: "REQUEST_DATA_CHUNK",
                        9: "UPDATE_KNOWLEDGE_GRAPH",
                        10: "EVALUATE_GATING",
                        11: "ATTEND_TO_WORKSPACE",
                        12: "ADJUST_EXPLORATION_RATE",
                    }

            for i in range(len(top_idx_list)):
                score = top_scores_list[i]
                idx = top_idx_list[i]

                intent_str = "UNKNOWN_OP"
                if top_actions is not None:
                    top_action = top_actions[i]
                    intent_str = self._action_labels_cache.get(top_action, f"UNKNOWN_OP {top_action}")

                output_str += (
                    f" -> MEMORY {i+1}: Index {idx} [Action: {intent_str}] (Cosine Similarity: {score:.3f})\n"
                )

            return output_str

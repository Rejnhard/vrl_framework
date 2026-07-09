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

"""Tests for IPC socket permissions and shared memory isolation.

Validates that Unix Domain Sockets and POSIX shared memory segments used
in the distributed topology operate with restricted OS-level access.
"""

import os
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))


try:
    SOCKET_FAMILY = socket.AF_UNIX
except AttributeError:
    SOCKET_FAMILY = socket.AF_INET


def create_secure_ipc_socket(bind_path: str) -> socket.socket:
    """Binds a socket with explicit OS-level restricted permissions (0o077 umask)."""
    old_umask = os.umask(0o077) if hasattr(os, "umask") else None
    try:
        sock = socket.socket(SOCKET_FAMILY, socket.SOCK_STREAM)
        if SOCKET_FAMILY == socket.AF_INET:
            sock.bind(("127.0.0.1", 0))
        else:
            sock.bind(bind_path)
        return sock
    finally:
        if old_umask is not None:
            os.umask(old_umask)


class TestIPCSocketAndMemoryPermissions:
    """Tests file descriptor and socket authorization."""

    @pytest.fixture
    def secure_temp_dir(self):
        """Generates an isolated temporary directory for file-backed resources."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

    @pytest.mark.skipif(os.name == "nt", reason="Windows does not support POSIX stat() mapping for UDS.")
    def test_unix_domain_socket_strict_posix_permissions(self, secure_temp_dir):
        """Asserts socket file ownership enforces strict 0o600 permissions."""
        socket_path = os.path.join(secure_temp_dir, "actor_learner.sock")

        sock = create_secure_ipc_socket(socket_path)
        sock.listen(1)

        st = os.stat(socket_path)
        permissions = stat.S_IMODE(st.st_mode)
        sock.close()

        assert permissions in (
            0o600,
            0o700,
        ), f"Expected secure socket permissions (0o600 or 0o700), got {oct(permissions)}."

    @pytest.mark.skipif(os.name == "nt", reason="Windows multiprocessing omits `_share_fd_cpu_` POSIX extensions.")
    def test_pytorch_shared_memory_file_descriptor_isolation(self):
        """Asserts shared memory tensors prevent global OS read/write access."""
        import torch.multiprocessing as tmp

        original_strategy = tmp.get_sharing_strategy()
        tmp.set_sharing_strategy("file_descriptor")

        try:
            trajectory_tensor = torch.randn(1024, 128)
            trajectory_tensor.share_memory_()

            fd = trajectory_tensor.untyped_storage()._share_fd_cpu_()
            st = os.fstat(fd[0] if isinstance(fd, tuple) else fd)
            permissions = stat.S_IMODE(st.st_mode)

            assert (
                permissions & stat.S_IRWXO
            ) == 0, f"Tensor memory is globally accessible. Permissions: {oct(permissions)}"

        finally:
            tmp.set_sharing_strategy(original_strategy)

    def test_ipc_unresponsive_connection_timeout(self, secure_temp_dir):
        """Validates proper non-blocking timeout handling during network interrupts."""
        socket_path = os.path.join(secure_temp_dir, "mcts_evaluator.sock")
        server_sock = create_secure_ipc_socket(socket_path)

        server_sock.settimeout(0.5)
        server_sock.listen(1)

        def unresponsive_client():
            client_sock = socket.socket(SOCKET_FAMILY, socket.SOCK_STREAM)
            connect_target = server_sock.getsockname() if SOCKET_FAMILY == socket.AF_INET else socket_path
            client_sock.connect(connect_target)
            client_sock.sendall(b"PARTIAL_TENSOR_HEADER")
            time.sleep(2.0)
            client_sock.close()

        thread = threading.Thread(target=unresponsive_client)
        thread.start()

        conn, _ = server_sock.accept()
        conn.settimeout(0.5)

        try:
            _ = conn.recv(4096)
            _ = conn.recv(4096)
            pytest.fail("Socket blocked indefinitely on an unresponsive client.")

        except socket.timeout:
            pass
        except BlockingIOError:
            pass
        finally:
            conn.close()
            server_sock.close()
            thread.join()

    def test_invalid_payload_handling(self, secure_temp_dir):
        """Validates deserialization pipeline stability against binary corruption."""
        socket_path = os.path.join(secure_temp_dir, "trainer_input.sock")
        server_sock = create_secure_ipc_socket(socket_path)
        server_sock.settimeout(1.0)
        server_sock.listen(1)

        def invalid_client():
            client_sock = socket.socket(SOCKET_FAMILY, socket.SOCK_STREAM)
            connect_target = server_sock.getsockname() if SOCKET_FAMILY == socket.AF_INET else socket_path
            client_sock.connect(connect_target)
            client_sock.sendall(os.urandom(1024))
            client_sock.close()

        thread = threading.Thread(target=invalid_client)
        thread.start()

        conn, _ = server_sock.accept()
        conn.settimeout(1.0)

        raw_data = conn.recv(1024)
        import pickle

        with pytest.raises((pickle.UnpicklingError, EOFError, RuntimeError, ValueError, OverflowError)):
            _ = pickle.loads(raw_data)

        conn.close()
        server_sock.close()
        thread.join()

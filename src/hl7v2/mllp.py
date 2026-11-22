"""MLLP (Minimal Lower Layer Protocol): how HL7 v2 messages travel over TCP.

Each message is wrapped as   <VT> message <FS><CR>   i.e. 0x0B ... 0x1C 0x0D
"""
import socket
import socketserver
import threading

START = b"\x0b"
END = b"\x1c\x0d"


def frame(message: str) -> bytes:
    return START + message.encode("utf-8") + END


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        buffer = b""
        self.request.settimeout(30)
        while True:
            try:
                chunk = self.request.recv(4096)
            except (socket.timeout, ConnectionResetError):
                return
            if not chunk:
                return
            buffer += chunk
            while END in buffer:
                raw, buffer = buffer.split(END, 1)
                start = raw.find(START)
                message = raw[start + 1:].decode("utf-8", errors="replace") if start >= 0 else raw.decode()
                ack = self.server.on_message(message)
                self.request.sendall(frame(ack))


class MLLPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, on_message):
        self.on_message = on_message
        super().__init__((host, port), _Handler)


def start_in_thread(host: str, port: int, on_message) -> MLLPServer:
    server = MLLPServer(host, port, on_message)
    threading.Thread(target=server.serve_forever, name="mllp-listener", daemon=True).start()
    return server


def send(host: str, port: int, message: str, timeout: float = 10) -> str:
    """Send one message and return the ACK text."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(frame(message))
        buffer = b""
        while END not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk
    return buffer.split(END, 1)[0].lstrip(START).decode("utf-8", errors="replace")


def is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False

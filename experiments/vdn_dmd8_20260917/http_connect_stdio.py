import socket
import sys
import threading
import os


def recv_headers(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("proxy closed before CONNECT response")
        data.extend(chunk)
        if len(data) > 65536:
            raise RuntimeError("oversized CONNECT response")
    return bytes(data)


def pump_stdin(sock: socket.socket) -> None:
    try:
        while True:
            payload = os.read(sys.stdin.fileno(), 65536)
            if not payload:
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            sock.sendall(payload)
    except OSError:
        return


def main() -> None:
    target_host = sys.argv[1]
    target_port = int(sys.argv[2])
    proxy_host = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
    proxy_port = int(sys.argv[4]) if len(sys.argv) > 4 else 10808

    with socket.create_connection((proxy_host, proxy_port), timeout=20) as upstream:
        request = (
            f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
            f"Host: {target_host}:{target_port}\r\n"
            "Proxy-Connection: Keep-Alive\r\n\r\n"
        ).encode("ascii")
        upstream.sendall(request)
        status_line = recv_headers(upstream).split(b"\r\n", 1)[0]
        if b" 200 " not in status_line:
            raise RuntimeError(f"CONNECT failed: {status_line!r}")
        upstream.settimeout(None)

        writer = threading.Thread(target=pump_stdin, args=(upstream,), daemon=True)
        writer.start()
        while True:
            payload = upstream.recv(65536)
            if not payload:
                return
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()

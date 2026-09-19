import select
import socket
import sys


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


def bridge(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    while True:
        readable, _, _ = select.select(sockets, [], [], 30)
        if not readable:
            continue
        for src in readable:
            payload = src.recv(65536)
            if not payload:
                return
            dst = right if src is left else left
            dst.sendall(payload)


def main() -> None:
    listen_port = int(sys.argv[1])
    target_host = sys.argv[2]
    target_port = int(sys.argv[3])
    proxy_host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
    proxy_port = int(sys.argv[5]) if len(sys.argv) > 5 else 10808

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", listen_port))
        listener.listen(1)
        client, _ = listener.accept()
        with client, socket.create_connection((proxy_host, proxy_port), timeout=20) as upstream:
            request = (
                f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
                f"Host: {target_host}:{target_port}\r\n"
                "Proxy-Connection: Keep-Alive\r\n\r\n"
            ).encode("ascii")
            upstream.sendall(request)
            response = recv_headers(upstream)
            status_line = response.split(b"\r\n", 1)[0]
            if b" 200 " not in status_line:
                raise RuntimeError(f"CONNECT failed: {status_line!r}")
            bridge(client, upstream)


if __name__ == "__main__":
    main()

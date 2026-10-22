import http.server
import threading
from pathlib import Path

import pytest

from videox.engines.downloads import download_file
from videox.errors import VideoXError


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"x" * 4000
        self.send_response(200)
        # The full file announces 4000 bytes; the cut one closes after 1000.
        self.send_header("Content-Length", "4000")
        self.end_headers()
        self.wfile.write(body if self.path == "/full" else body[:1000])

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_download_keeps_a_complete_file(server, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    dest = tmp_path / "weights" / "model.pth"
    download_file(f"{server}/full", dest)
    assert dest.read_bytes() == b"x" * 4000
    assert list(dest.parent.iterdir()) == [dest]


def test_download_rejects_a_cut_off_file(server, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    dest = tmp_path / "model.pth"
    with pytest.raises(VideoXError, match="下载"):
        download_file(f"{server}/cut", dest)
    # Nothing is left behind, so the next run downloads again.
    assert list(tmp_path.iterdir()) == []

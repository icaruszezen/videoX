import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from videox.engines.downloads import download_file, missing_files
from videox.errors import VideoXError


class _Truncating(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "1000")
        self.end_headers()
        # The server announces 1000 bytes, sends 300 and closes cleanly.
        self.wfile.write(b"x" * 300)

    def log_message(self, *args: object) -> None:
        return


def test_truncated_download_is_not_kept_as_the_weight_file(tmp_path: Path, monkeypatch):
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "*")
    server = HTTPServer(("127.0.0.1", 0), _Truncating)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    dest = tmp_path / "model.safetensors"
    try:
        with pytest.raises(VideoXError, match="下载不完整"):
            download_file(f"http://127.0.0.1:{server.server_port}/model", dest)
    finally:
        thread.join(5)
        server.server_close()
    assert not dest.exists()
    assert not (tmp_path / "model.safetensors.part").exists()
    assert missing_files(tmp_path, ("model.safetensors",)) == ["model.safetensors"]

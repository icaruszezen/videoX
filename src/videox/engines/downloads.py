from __future__ import annotations

import http.client
import os
import urllib.error
import urllib.request
from pathlib import Path

from videox.engines.progress import emit_progress
from videox.errors import VideoXError


def download_file(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # One temporary file per process, so two jobs fetching the same weight never share it.
    partial = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "videox"})
    print(f"正在下载 {dest.name}", flush=True)
    got = 0
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            total = int(response.headers.get("Content-Length") or 0)
            next_report = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    got += len(chunk)
                    if total and got >= next_report:
                        emit_progress("download", got, total)
                        next_report = got + max(total // 20, 1)
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        partial.unlink(missing_ok=True)
        raise VideoXError(f"下载失败 {dest.name}: {exc}") from exc
    if got <= 0:
        partial.unlink(missing_ok=True)
        raise VideoXError(f"下载结果是空的: {dest.name}")
    # A connection closed early ends the read without an error, so compare with the announced size.
    if total and got != total:
        partial.unlink(missing_ok=True)
        raise VideoXError(f"下载不完整 {dest.name}: 收到 {got} / {total} 字节，请重试")
    try:
        partial.replace(dest)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        # Another job may have finished the same file first; Windows refuses to replace it while open.
        if not (dest.is_file() and dest.stat().st_size == got):
            raise VideoXError(f"保存下载文件失败 {dest.name}: {exc}") from exc


def missing_files(directory: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (directory / name).is_file() or (directory / name).stat().st_size <= 0]


def present_files(directory: Path, names: tuple[str, ...]) -> list[str]:
    missing = set(missing_files(directory, names))
    return [name for name in names if name not in missing]

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from videox.engines.progress import emit_progress
from videox.errors import VideoXError


def download_file(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
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
    except (OSError, urllib.error.URLError) as exc:
        partial.unlink(missing_ok=True)
        raise VideoXError(f"下载失败 {dest.name}: {exc}") from exc
    if got <= 0:
        partial.unlink(missing_ok=True)
        raise VideoXError(f"下载结果是空的: {dest.name}")
    if total and got != total:
        # A connection closed early still ends the read loop normally.
        partial.unlink(missing_ok=True)
        raise VideoXError(f"下载不完整 {dest.name}: 收到 {got}/{total} 字节，请重试")
    partial.replace(dest)


def missing_files(directory: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (directory / name).is_file() or (directory / name).stat().st_size <= 0]


def present_files(directory: Path, names: tuple[str, ...]) -> list[str]:
    missing = set(missing_files(directory, names))
    return [name for name in names if name not in missing]

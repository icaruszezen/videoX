import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# A parent that starts one SeedVR2 worker the way upstream does and then waits
# for its result forever, as upstream's _gpu_processing does with no timeout.
_PARENT = """
import multiprocessing as mp
import time
from videox.engines.seedvr2 import progress_entry as entry

if __name__ == "__main__":
    entry._install_worker_watchdog()
    worker = mp.get_context("spawn").Process(target=entry._wrapped_worker, args=(1, 0, None, {}, None, None))
    worker.start()
    print("tracked", len(entry._workers), flush=True)
    time.sleep(60)
    print("still waiting", flush=True)
"""


def test_watchdog_ends_the_job_when_a_worker_dies(tmp_path: Path):
    script = tmp_path / "parent.py"
    script.write_text(_PARENT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    # Without the SeedVR2 checkout the worker fails at once, standing in for an OOM.
    env.pop("VIDEOX_SEEDVR_ROOT", None)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, encoding="utf-8", timeout=50
    )
    assert "tracked 1" in completed.stdout
    assert "多卡子进程异常退出" in completed.stdout
    assert "still waiting" not in completed.stdout
    assert completed.returncode == 1
    assert time.monotonic() - started < 30

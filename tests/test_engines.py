from pathlib import Path

import pytest

from videox.cli import default_restore_output, restore_params
from videox.engines.base import EngineContext
from videox.engines.flashvsr.engine import FlashVSREngine
from videox.engines.progress import parse_engine_line
from videox.engines.realbasicvsr.engine import RealBasicVSREngine
from videox.engines.registry import default_registry
from videox.engines.swiftvr.engine import SwiftVREngine
from videox.errors import VideoXError
from videox.hardware.gpus import GpuInfo
from videox.jobs.models import JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore

PROBE = Path(__file__).resolve().parents[1] / ".work" / "realbasicvsr_probe.pth"


def _context(tmp_path: Path) -> EngineContext:
    return EngineContext(
        project_root=tmp_path,
        python_executable="python",
        model_dir=tmp_path / "models" / "SEEDVR2",
        gpus=[
            GpuInfo(0, "RTX 4080", 16376),
            GpuInfo(1, "RTX 3080", 20480),
        ],
    )


def _video_job(tmp_path: Path, engine_id: str, **params: object) -> JobSpec:
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    return JobSpec(
        engine_id=engine_id,
        input_path=str(source),
        output_path=str(tmp_path / "out.mp4"),
        devices=[1, 0],
        params=params,
    )


def test_registry_lists_four_engines():
    manifests = {item.id: item for item in default_registry().manifests()}
    assert list(manifests) == ["seedvr2", "flashvsr", "swiftvr", "realbasicvsr"]
    assert manifests["seedvr2"].device_policy == "multi"
    assert manifests["seedvr2"].media == ["video", "image"]
    for engine_id in ("flashvsr", "swiftvr", "realbasicvsr"):
        assert manifests[engine_id].device_policy == "single"
        assert manifests[engine_id].media == ["video"]
        for name, prop in manifests[engine_id].params_schema["properties"].items():
            assert prop.get("description"), name
            assert "default" in prop


def test_switching_schema_changes_defaults():
    manifests = {item.id: item for item in default_registry().manifests()}
    flash = manifests["flashvsr"].params_schema["properties"]
    swift = manifests["swiftvr"].params_schema["properties"]
    basic = manifests["realbasicvsr"].params_schema["properties"]
    seed = manifests["seedvr2"].params_schema["properties"]
    assert "resolution" in seed and "mode" not in seed
    assert flash["mode"]["default"] == "tiny-long"
    assert flash["scale"]["default"] == "4"
    assert flash["sparse_ratio"]["default"] == 2.0
    assert flash["local_range"]["default"] == "11"
    assert flash["color_fix"]["default"] is True
    assert swift["upscale"]["default"] == 4
    assert swift["clip_len"]["default"] == 24
    assert swift["dtype"]["default"] == "bfloat16"
    assert basic["max_seq_len"]["default"] == 16
    assert "resolution" not in flash
    assert "mode" not in swift
    assert "upscale" not in basic


def test_single_gpu_engines_keep_the_first_device(tmp_path: Path):
    context = _context(tmp_path)
    flash = FlashVSREngine().build_command(_video_job(tmp_path, "flashvsr"), context)
    assert flash[flash.index("--device") + 1] == "1"
    assert "--cuda_device" not in flash
    assert flash[flash.index("--mode") + 1] == "tiny-long"
    assert flash[flash.index("--scale") + 1] == "4"
    assert "--color-fix" in flash
    swift = SwiftVREngine().build_command(_video_job(tmp_path, "swiftvr"), context)
    assert swift[swift.index("--device") + 1] == "1"
    assert swift[swift.index("--upscale") + 1] == "4"
    assert swift[swift.index("--clip-len") + 1] == "24"
    with pytest.raises(VideoXError, match="4 的倍数"):
        SwiftVREngine().build_command(_video_job(tmp_path, "swiftvr", clip_len=6), context)
    basic = RealBasicVSREngine().build_command(_video_job(tmp_path, "realbasicvsr"), context)
    assert basic[basic.index("--device") + 1] == "1"
    assert basic[basic.index("--max-seq-len") + 1] == "16"


def test_optional_engines_need_their_own_setup(tmp_path: Path):
    context = _context(tmp_path)
    with pytest.raises(VideoXError, match="videox setup"):
        FlashVSREngine().resolve_python(context)
    with pytest.raises(VideoXError, match="videox setup"):
        SwiftVREngine().resolve_python(context)
    assert RealBasicVSREngine().resolve_python(context) == "python"


def test_new_engines_reject_images(tmp_path: Path):
    source = tmp_path / "still.png"
    source.write_bytes(b"png")
    runner = JobRunner(default_registry(), JobStore(), tmp_path)
    for engine_id in ("flashvsr", "swiftvr", "realbasicvsr"):
        with pytest.raises(VideoXError, match="不支持图片"):
            runner.submit(
                JobSpec(
                    engine_id=engine_id,
                    input_path=str(source),
                    output_path=str(tmp_path / "out.png"),
                    media="image",
                ),
                background=False,
            )


def test_restore_flags_stay_on_seedvr2(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    assert default_restore_output(tmp_path, source, "flashvsr", "video").name == "clip.flashvsr.mp4"
    assert default_restore_output(tmp_path, source, "seedvr2", "image").name == "clip.seedvr2.png"
    assert restore_params("flashvsr", 720, 5) == {}
    assert restore_params("seedvr2", 720, 5) == {"resolution": 720, "batch_size": 5}


def test_progress_lines_parse():
    insight = parse_engine_line('{"videox_progress": {"phase": "restore", "current": 1, "total": 4}}')
    assert insight is not None
    assert insight.stage == "restore"
    assert insight.progress == 0.25
    preview = parse_engine_line('{"videox_preview": {"frame": 3}}')
    assert preview is not None
    assert preview.preview_frame == 3
    assert parse_engine_line('{"videox_preview": {"frame": 0}}') is None


@pytest.mark.skipif(not PROBE.is_file(), reason="RealBasicVSR checkpoint was not downloaded")
def test_realbasicvsr_checkpoint_matches_network():
    torch = pytest.importorskip("torch")
    from videox.engines.realbasicvsr.arch import load_realbasicvsr

    load_realbasicvsr(PROBE, torch.device("cpu"))


def test_realbasicvsr_forward_shape():
    torch = pytest.importorskip("torch")
    from videox.engines.realbasicvsr.arch import RealBasicVSRNet

    model = RealBasicVSRNet().eval()
    video = torch.rand(1, 2, 3, 64, 64)
    with torch.inference_mode():
        output = model(video)
    assert tuple(output.shape) == (1, 2, 3, 256, 256)
    single = torch.rand(1, 1, 3, 64, 64)
    with torch.inference_mode():
        one = model(single)
    assert tuple(one.shape) == (1, 1, 3, 256, 256)

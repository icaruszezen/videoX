from pathlib import Path

import pytest

from videox.cli import default_restore_output, engine_params, restore_params
from videox.engines.base import EngineContext
from videox.engines.flashvsr.engine import FlashVSREngine
from videox.engines.progress import parse_engine_line
from videox.engines.realbasicvsr.engine import RealBasicVSREngine
from videox.engines.registry import default_registry
from videox.engines.seedvr2.engine import SeedVR2Engine
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


IMAGE_ENGINES = ("realesrgan", "swinir", "scunet", "gfpgan")


def test_registry_lists_video_and_image_engines():
    manifests = {item.id: item for item in default_registry().manifests()}
    assert list(manifests) == ["seedvr2", "flashvsr", "swiftvr", "realbasicvsr", *IMAGE_ENGINES]
    assert manifests["seedvr2"].device_policy == "multi"
    assert manifests["seedvr2"].media == ["video", "image"]
    for engine_id in ("flashvsr", "swiftvr", "realbasicvsr"):
        assert manifests[engine_id].device_policy == "single"
        assert manifests[engine_id].media == ["video"]
        for name, prop in manifests[engine_id].params_schema["properties"].items():
            assert prop.get("description"), name
            assert "default" in prop
    for engine_id in IMAGE_ENGINES:
        assert manifests[engine_id].device_policy == "single"
        assert manifests[engine_id].media == ["image"]


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


def test_param_flags_are_typed_by_the_engine_schema():
    assert engine_params("gfpgan", []) == {}
    typed = engine_params(
        "gfpgan", ["upscale=4", "strength=0.5", "tile=256", "fp16=false", "background=none"]
    )
    assert typed == {"upscale": "4", "strength": 0.5, "tile": 256, "fp16": False, "background": "none"}
    with pytest.raises(VideoXError, match="没有参数 nope"):
        engine_params("swinir", ["nope=1"])
    with pytest.raises(VideoXError, match="tile 的值无法识别"):
        engine_params("swinir", ["tile=big"])
    with pytest.raises(VideoXError, match="fp16 的值无法识别"):
        engine_params("realesrgan", ["fp16=yes"])


def test_progress_lines_parse():
    insight = parse_engine_line('{"videox_progress": {"phase": "restore", "current": 1, "total": 4}}')
    assert insight is not None
    assert insight.stage == "restore"
    assert insight.progress == 0.10 + 0.90 * 0.25
    loaded = parse_engine_line('{"videox_progress": {"phase": "load", "current": 1, "total": 1}}')
    assert loaded is not None and loaded.progress == 0.10
    downloaded = parse_engine_line('{"videox_progress": {"phase": "download", "current": 1, "total": 1}}')
    assert downloaded is not None and downloaded.progress == 0.05
    preview = parse_engine_line('{"videox_preview": {"frame": 3}}')
    assert preview is not None
    assert preview.preview_frame == 3
    assert parse_engine_line('{"videox_preview": {"frame": 0}}') is None


def test_gfpgan_progress_moves_forward_from_faces_to_background():
    from videox.engines.gfpgan.engine import GFPGANEngine

    parse = GFPGANEngine().new_log_parser()
    lines = [
        ("load", 1, 1),
        ("face", 0, 2),
        ("face", 2, 2),
        ("restore", 0, 1),
        ("restore", 1, 1),
    ]
    values = []
    for phase, current, total in lines:
        insight = parse(f'{{"videox_progress": {{"phase": "{phase}", "current": {current}, "total": {total}}}}}')
        assert insight is not None
        values.append(insight.progress)
    assert values == sorted(values)
    assert values[-1] == 1.0


def test_flashvsr_counts_only_default_mode_weights_as_missing(tmp_path: Path):
    from videox.engines.flashvsr.presets import DEFAULT_MODE, weights_for
    from videox.engines.flashvsr.weights import weight_lists

    for name in weights_for(DEFAULT_MODE):
        (tmp_path / name).write_bytes(b"x")
    present, missing = weight_lists(tmp_path)
    assert missing == []
    assert sorted(present) == sorted(weights_for(DEFAULT_MODE))


def test_flashvsr_venv_from_before_the_import_check_needs_setup(tmp_path: Path):
    from videox.paths import deps_marker, flashvsr_dir, venv_python

    root = flashvsr_dir(tmp_path)
    python = venv_python(root)
    python.parent.mkdir(parents=True)
    python.write_text("")
    with pytest.raises(VideoXError, match="videox setup"):
        FlashVSREngine().resolve_python(_context(tmp_path))
    deps_marker(root).write_text("ready", encoding="utf-8")
    assert FlashVSREngine().resolve_python(_context(tmp_path)) == str(python)


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


def test_flashvsr_and_swiftvr_check_their_new_options(tmp_path: Path):
    context = _context(tmp_path)
    flash = FlashVSREngine().build_command(_video_job(tmp_path, "flashvsr", kv_ratio=4), context)
    assert flash[flash.index("--kv-ratio") + 1] == "4"
    with pytest.raises(VideoXError, match="KV"):
        FlashVSREngine().build_command(_video_job(tmp_path, "flashvsr", kv_ratio=1), context)
    # Tiny now runs the same streaming loop as Tiny Long; old requests still work.
    tiny = FlashVSREngine().build_command(_video_job(tmp_path, "flashvsr", mode="tiny"), context)
    assert tiny[tiny.index("--mode") + 1] == "tiny-long"
    swift = SwiftVREngine().build_command(_video_job(tmp_path, "swiftvr", short_side=1081), context)
    assert swift[swift.index("--short-side") + 1] == "1080"
    plain = SwiftVREngine().build_command(_video_job(tmp_path, "swiftvr"), context)
    assert "--short-side" not in plain
    with pytest.raises(VideoXError, match="输出短边"):
        SwiftVREngine().build_command(_video_job(tmp_path, "swiftvr", short_side=32), context)


def test_frame_and_size_helpers():
    from videox.engines.flashvsr.infer import vram_estimate_gb
    from videox.engines.flashvsr.stream import target_size
    from videox.engines.swiftvr.infer import _output_size

    # 1080p output pads to 1920x1152: about 3 GB of history per unit of KV ratio,
    # and Full keeps another 10 GB of decoder cache.
    tiny = vram_estimate_gb(1920, 1152, 2, full=False)
    assert round(vram_estimate_gb(1920, 1152, 3, full=False) - tiny, 2) == 2.97
    assert 14 < tiny < 16
    assert vram_estimate_gb(1920, 1152, 2, full=True) - tiny > 10

    # 400x240 ×4 = 1600x960, already multiples of 128 on neither side.
    assert target_size(400, 240, 4) == (1600, 960, 1664, 1024)
    assert target_size(256, 128, 2) == (512, 256, 512, 256)
    with pytest.raises(VideoXError):
        target_size(0, 240, 4)
    assert _output_size(1920, 1080, 0) is None
    assert _output_size(640, 360, 1080) == (1920, 1080)
    assert _output_size(360, 640, 1080) == (1080, 1920)
    assert _output_size(720, 480, 1080) == (1620, 1080)


def test_step_fields_reach_the_schema():
    manifests = {item.id: item for item in default_registry().manifests()}
    seed = manifests["seedvr2"].params_schema["properties"]
    swift = manifests["swiftvr"].params_schema["properties"]
    assert (seed["batch_size"]["minimum"], seed["batch_size"]["step"]) == (1, 4)
    assert (swift["clip_len"]["minimum"], swift["clip_len"]["step"]) == (4, 4)


def test_seedvr2_long_clips_stay_on_one_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videox.engines.seedvr2 import engine as seedvr2_engine
    from videox.engines.seedvr2.progress_entry import pop_note
    from videox.media.frames import VideoProbe

    frames = {"count": 108_000}
    monkeypatch.setattr(
        seedvr2_engine,
        "probe_video",
        lambda path: VideoProbe(640, 360, 30.0, "30/1", None, frames["count"]),
    )
    monkeypatch.setattr(seedvr2_engine, "physical_memory_bytes", lambda: 64 * 1024**3)
    context = _context(tmp_path)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    auto = JobSpec(engine_id="seedvr2", input_path=str(source), output_path=str(tmp_path / "out.mp4"))
    # An hour at 1080p would need terabytes of RAM across two GPUs; use one and say why.
    command = SeedVR2Engine().build_command(auto, context)
    assert command[command.index("--cuda_device") + 1] == "0"
    note = pop_note(command)
    assert note is not None and "只用第一张显卡" in note
    assert "--videox_note" not in command
    with pytest.raises(VideoXError, match="请只选一张显卡"):
        SeedVR2Engine().build_command(_video_job(tmp_path, "seedvr2"), context)
    # A few seconds still splits across both cards.
    frames["count"] = 150
    short = SeedVR2Engine().build_command(auto, context)
    assert short[short.index("--cuda_device") + 1] == "0,1"
    assert "--videox_note" not in short

from pathlib import Path

import pytest

from videox.engines.base import EngineContext
from videox.engines.realesrgan import RealESRGANEngine
from videox.engines.realesrgan.presets import (
    DEFAULT_MODEL,
    DENOISE_MODEL,
    MODELS,
    required_weights,
)
from videox.engines.realesrgan.weights import ensure_model, known_weights, weight_lists
from videox.engines.registry import EngineRegistry
from videox.errors import VideoXError
from videox.hardware.gpus import GpuInfo
from videox.jobs.models import JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore

WEIGHTS = Path(__file__).resolve().parents[1] / ".work" / "weights" / "RealESRGAN"


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


def _image_job(tmp_path: Path, **params: object) -> JobSpec:
    source = tmp_path / "in.png"
    source.write_bytes(b"x")
    return JobSpec(
        engine_id="realesrgan",
        input_path=str(source),
        output_path=str(tmp_path / "out.png"),
        media="image",
        devices=[1, 0],
        params=params,
    )


def _command(tmp_path: Path, **params: object) -> list[str]:
    return RealESRGANEngine().build_command(_image_job(tmp_path, **params), _context(tmp_path))


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _allowed(prop: dict, value: object) -> bool:
    if "enum" in prop:
        return value in prop["enum"]
    if prop["type"] == "boolean":
        return isinstance(value, bool)
    return False


def test_manifest_describes_an_image_engine():
    manifest = RealESRGANEngine().manifest()
    assert manifest.id == "realesrgan"
    assert manifest.name == "Real-ESRGAN"
    assert manifest.media == ["image"]
    assert manifest.device_policy == "single"
    assert manifest.capabilities == ["upscale"]
    assert manifest.description
    props = manifest.params_schema["properties"]
    assert list(props) == ["model", "outscale", "denoise_strength", "tile", "tile_pad", "fp16"]
    groups = {group["id"] for group in manifest.params_schema["groups"]}
    seen: list[str] = []
    for name, prop in props.items():
        assert prop.get("description"), name
        assert "default" in prop, name
        assert prop["media"] == ["image"], name
        assert prop["group"] in groups, name
        conditions = [prop.get("visible_when") or {}]
        conditions += [override["when"] for override in prop.get("overrides") or []]
        for condition in conditions:
            for key, values in condition.items():
                assert key in seen, f"{name} depends on later field {key}"
                assert values and all(_allowed(props[key], value) for value in values), (name, key)
        seen.append(name)
    model = props["model"]
    assert model["enum"] == list(MODELS)
    assert model["default"] == DEFAULT_MODEL
    assert set(model["enum_labels"]) == set(MODELS)
    assert set(model["enum_descriptions"]) == set(MODELS)
    assert all(model["enum_descriptions"].values())
    assert props["denoise_strength"]["visible_when"] == {"model": [DENOISE_MODEL]}
    assert props["tile"]["title"] == "分块大小"
    assert props["tile_pad"]["title"] == "分块重叠"
    assert props["fp16"]["title"] == "半精度"


def test_build_command_uses_defaults(tmp_path: Path):
    command = _command(tmp_path)
    assert command[:4] == ["python", "-u", "-m", "videox.engines.realesrgan.progress_entry"]
    assert command[4] == str((tmp_path / "in.png").resolve())
    assert _arg(command, "--output") == str((tmp_path / "out.png").resolve())
    assert _arg(command, "--model-dir") == str(tmp_path / "models" / "RealESRGAN")
    assert _arg(command, "--model") == "RealESRGAN_x4plus"
    assert _arg(command, "--outscale") == "4"
    assert _arg(command, "--tile") == "512"
    assert _arg(command, "--tile-pad") == "10"
    assert "--fp16" in command and "--no-fp16" not in command
    assert "--denoise-strength" not in command
    assert command[-2:] == ["--device", "1"]


def test_build_command_follows_the_model(tmp_path: Path):
    x2 = _command(tmp_path, model="RealESRGAN_x2plus")
    assert _arg(x2, "--outscale") == "2"
    assert _arg(_command(tmp_path, model="RealESRGAN_x2plus", outscale=3), "--outscale") == "3"
    assert _arg(_command(tmp_path, model="realesr-animevideov3", outscale=2.5), "--outscale") == "2.5"
    general = _command(tmp_path, model=DENOISE_MODEL)
    assert _arg(general, "--denoise-strength") == "0.5"
    assert _arg(_command(tmp_path, model=DENOISE_MODEL, denoise_strength=1), "--denoise-strength") == "1"
    assert _arg(_command(tmp_path, model=DENOISE_MODEL, denoise_strength=0.25), "--denoise-strength") == "0.25"
    # Hidden for this model: ignored, even when out of range.
    hidden = _command(tmp_path, model="RealESRGAN_x4plus_anime_6B", denoise_strength=7)
    assert "--denoise-strength" not in hidden
    tuned = _command(tmp_path, tile=0, tile_pad=32, fp16=False)
    assert _arg(tuned, "--tile") == "0"
    assert _arg(tuned, "--tile-pad") == "32"
    assert "--no-fp16" in tuned and "--fp16" not in tuned


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"model": "RealESRGAN_x8"}, "无法识别"),
        ({"outscale": 9}, "不能大于 8"),
        ({"outscale": 0.5}, "不能小于 1"),
        ({"model": "RealESRGAN_x2plus", "outscale": "2"}, "需要是数字"),
        ({"model": DENOISE_MODEL, "denoise_strength": 1.5}, "不能大于 1"),
        ({"model": DENOISE_MODEL, "denoise_strength": -0.1}, "不能小于 0"),
        ({"tile": -1}, "不能小于 0"),
        ({"tile": 5000}, "不能大于 4096"),
        ({"tile": 64.5}, "需要是整数"),
        ({"tile_pad": 129}, "不能大于 128"),
        ({"fp16": 1}, "需要是布尔值"),
    ],
)
def test_build_command_rejects_bad_values(tmp_path: Path, params: dict, message: str):
    with pytest.raises(VideoXError, match=message):
        _command(tmp_path, **params)


def test_runner_rejects_video(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"mp4")
    registry = EngineRegistry()
    registry.register(RealESRGANEngine())
    runner = JobRunner(registry, JobStore(), tmp_path)
    with pytest.raises(VideoXError, match="不支持视频"):
        runner.submit(
            JobSpec(
                engine_id="realesrgan",
                input_path=str(source),
                output_path=str(tmp_path / "out.mp4"),
                media="video",
            ),
            background=False,
        )


def test_weight_lists_only_require_the_default_model(tmp_path: Path):
    assert weight_lists(tmp_path) == ([], ["RealESRGAN_x4plus.pth"])
    (tmp_path / "realesr-animevideov3.pth").write_bytes(b"x")
    (tmp_path / "RealESRGAN_x2plus.pth").write_bytes(b"")
    assert weight_lists(tmp_path) == (["realesr-animevideov3.pth"], ["RealESRGAN_x4plus.pth"])
    (tmp_path / "RealESRGAN_x4plus.pth").write_bytes(b"x")
    present, missing = weight_lists(tmp_path)
    assert present == ["RealESRGAN_x4plus.pth", "realesr-animevideov3.pth"]
    assert missing == []
    assert len(known_weights()) == 7
    assert set(known_weights()) == {weight.name for spec in MODELS.values() for weight in spec.files}


def test_denoise_weights_are_only_needed_below_full_strength(tmp_path: Path):
    assert [w.name for w in required_weights(DENOISE_MODEL, 1.0)] == ["realesr-general-x4v3.pth"]
    assert [w.name for w in required_weights(DENOISE_MODEL, 0.5)] == [
        "realesr-general-x4v3.pth",
        "realesr-general-wdn-x4v3.pth",
    ]
    assert [w.name for w in required_weights("RealESRGAN_x2plus", 0.5)] == ["RealESRGAN_x2plus.pth"]
    for spec in MODELS.values():
        for weight in spec.files:
            assert weight.url.startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/")
            assert weight.url.endswith("/" + weight.name)
    # Files already on disk are not downloaded again.
    for name in ("realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth"):
        (tmp_path / name).write_bytes(b"x")
    assert ensure_model(tmp_path, DENOISE_MODEL, denoise_strength=0.2) == [
        tmp_path / "realesr-general-x4v3.pth",
        tmp_path / "realesr-general-wdn-x4v3.pth",
    ]
    assert ensure_model(tmp_path, DENOISE_MODEL, denoise_strength=1) == [tmp_path / "realesr-general-x4v3.pth"]
    with pytest.raises(VideoXError, match="未知的 Real-ESRGAN 模型"):
        ensure_model(tmp_path, "nope")


@pytest.mark.parametrize("model_id", list(MODELS))
def test_networks_have_the_right_output_shape(model_id: str):
    torch = pytest.importorskip("torch")
    from videox.engines.realesrgan.arch import build_network

    spec = MODELS[model_id]
    torch.manual_seed(0)
    network = build_network(spec.arch, **spec.arch_kwargs).eval()
    with torch.inference_mode():
        output = network(torch.rand(1, 3, 6, 10))
    assert output.shape == (1, 3, 6 * spec.scale, 10 * spec.scale)
    assert torch.isfinite(output).all()
    assert spec.multiple_of == (2 if spec.scale == 2 else 1)


def test_pixel_unshuffle_matches_torch():
    torch = pytest.importorskip("torch")
    from videox.engines.realesrgan.arch import RRDBNet, pixel_unshuffle

    x = torch.rand(2, 3, 8, 12)
    assert torch.equal(pixel_unshuffle(x, 2), torch.nn.functional.pixel_unshuffle(x, 2))
    assert torch.equal(pixel_unshuffle(x[:, :, 2:6, 4:8], 2), torch.nn.functional.pixel_unshuffle(x[:, :, 2:6, 4:8], 2))
    with torch.inference_mode():
        assert RRDBNet(scale=1, num_block=1).eval()(torch.rand(1, 3, 8, 8)).shape == (1, 3, 8, 8)


def test_dni_blends_every_parameter():
    torch = pytest.importorskip("torch")
    from videox.engines.realesrgan.infer import dni

    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([4.0])}
    b = {"w": torch.tensor([3.0, 6.0]), "b": torch.tensor([0.0])}
    mixed = dni(a, b, 0.25)
    assert torch.allclose(mixed["w"], torch.tensor([2.5, 5.0]))
    assert torch.allclose(mixed["b"], torch.tensor([1.0]))
    assert torch.equal(dni(a, b, 1.0)["w"], a["w"])
    assert torch.equal(dni(a, b, 0.0)["w"], b["w"])
    with pytest.raises(VideoXError, match="结构不一致"):
        dni(a, {"w": b["w"]}, 0.5)


def test_pipeline_runs_with_local_weights(tmp_path: Path, capsys):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    from videox.engines.realesrgan.arch import build_network
    from videox.engines.realesrgan.infer import (
        RealESRGANRequest,
        load_model,
        run_realesrgan,
        upscale_rgb,
    )

    # A random-init checkpoint stands in for the release file, so nothing downloads.
    spec = MODELS["realesr-animevideov3"]
    torch.manual_seed(0)
    network = build_network(spec.arch, **spec.arch_kwargs)
    torch.save({"params": network.state_dict()}, tmp_path / spec.weights.name)

    rgb = np.random.default_rng(0).integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    rgb.flags.writeable = False
    # Half precision only applies on CUDA, so the CPU model stays in float32.
    model = load_model("realesr-animevideov3", [tmp_path / spec.weights.name], device=torch.device("cpu"), half=True)
    assert next(model.parameters()).dtype == torch.float32
    whole = upscale_rgb(model, rgb, model_id="realesr-animevideov3", tile=0, tile_pad=10, outscale=4)
    assert whole.shape == (36, 52, 3) and whole.dtype == np.uint8
    tiled = upscale_rgb(model, rgb, model_id="realesr-animevideov3", tile=4, tile_pad=2, outscale=2.5)
    assert tiled.shape == (22, 32, 3)

    rgba = np.zeros((9, 13, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.linspace(0, 255, 13, dtype=np.uint8)
    source = tmp_path / "in.png"
    Image.fromarray(rgba, "RGBA").save(source)
    out = tmp_path / "out" / "result.png"
    run_realesrgan(
        RealESRGANRequest(
            input_path=source,
            output_path=out,
            model_dir=tmp_path,
            model_id="realesr-animevideov3",
            outscale=3,
            tile=8,
            tile_pad=2,
        )
    )
    with Image.open(out) as written:
        assert written.mode == "RGBA" and written.size == (39, 27)
        alpha = np.asarray(written)[..., 3]
        assert alpha[:, 0].max() < 40 and alpha[:, -1].min() > 215
    lines = capsys.readouterr().out
    for phase in ("download", "load", "restore"):
        assert f'"phase": "{phase}"' in lines
    with pytest.raises(VideoXError, match="不支持的图片格式"):
        run_realesrgan(RealESRGANRequest(input_path=source, output_path=tmp_path / "x.gif", model_dir=tmp_path))
    # The size limit is checked before any download or inference.
    wide = tmp_path / "wide.png"
    Image.fromarray(np.zeros((2, 8200, 3), np.uint8)).save(wide)
    with pytest.raises(VideoXError, match="JPG 图片每边最多 65500 像素"):
        run_realesrgan(
            RealESRGANRequest(input_path=wide, output_path=tmp_path / "x.jpg", model_dir=tmp_path / "none", outscale=8)
        )


@pytest.mark.parametrize("model_id", list(MODELS))
def test_released_weights_load_and_run(model_id: str):
    spec = MODELS[model_id]
    if not all((WEIGHTS / weight.name).is_file() for weight in spec.files):
        pytest.skip("Real-ESRGAN weights were not downloaded to .work/weights/RealESRGAN")
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from videox.engines.realesrgan.infer import load_model, upscale_rgb
    from videox.engines.realesrgan.presets import required_weights

    # load_model loads every file with strict=True and blends the denoise pair.
    paths = [WEIGHTS / weight.name for weight in required_weights(model_id, 0.3)]
    model = load_model(model_id, paths, device=torch.device("cpu"), half=False, denoise_strength=0.3)
    rgb = np.random.default_rng(1).integers(0, 256, size=(11, 15, 3), dtype=np.uint8)
    output = upscale_rgb(model, rgb, model_id=model_id, tile=8, tile_pad=2, outscale=spec.scale)
    assert output.shape == (11 * spec.scale, 15 * spec.scale, 3)

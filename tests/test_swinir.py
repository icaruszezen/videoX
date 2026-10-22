import dataclasses
from pathlib import Path

import pytest

from videox.engines.base import EngineContext
from videox.engines.registry import EngineRegistry
from videox.engines.swinir import SwinIREngine
from videox.engines.swinir.presets import (
    DEFAULT_MODEL,
    MODELS,
    RELEASE_URL,
    SR_MODELS,
    model_spec,
)
from videox.engines.swinir.weights import ensure_model, known_files, weight_lists
from videox.errors import VideoXError
from videox.hardware.gpus import GpuInfo
from videox.jobs.models import JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore

WEIGHTS = Path(__file__).resolve().parents[1] / ".work" / "weights" / "SwinIR"

# Layout of each release checkpoint: tensor count and the stored shifted-window mask.
RELEASE_LAYOUT = {
    "real_x4_m": (552, (64, 64, 64)),
    "real_x4_l": (859, (64, 64, 64)),
    "real_x2_m": (550, (64, 64, 64)),
    "denoise_15": (544, (256, 64, 64)),
    "denoise_25": (544, (256, 64, 64)),
    "denoise_50": (544, (256, 64, 64)),
    "jpeg_40": (544, (324, 49, 49)),
    "jpeg_30": (544, (324, 49, 49)),
    "jpeg_20": (544, (324, 49, 49)),
    "jpeg_10": (544, (324, 49, 49)),
}


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
        engine_id="swinir",
        input_path=str(source),
        output_path=str(tmp_path / "out.png"),
        media="image",
        devices=[1, 0],
        params=params,
    )


def _command(tmp_path: Path, **params: object) -> list[str]:
    return SwinIREngine().build_command(_image_job(tmp_path, **params), _context(tmp_path))


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _allowed(prop: dict, value: object) -> bool:
    if "enum" in prop:
        return value in prop["enum"]
    if prop["type"] == "boolean":
        return isinstance(value, bool)
    return False


def _tiny(model_id: str, **changes: object):
    """A release config shrunk to a few blocks so it runs in a blink."""
    small = {"depths": (2, 2), "embed_dim": 16, "num_heads": (2, 2), "img_size": 16, "file": "tiny.pth"}
    if MODELS[model_id].window_size == 7:
        small["img_size"] = 14
    return dataclasses.replace(MODELS[model_id], **{**small, **changes})


def test_manifest_describes_an_image_engine():
    manifest = SwinIREngine().manifest()
    assert manifest.id == "swinir"
    assert manifest.name == "SwinIR"
    assert manifest.media == ["image"]
    assert manifest.device_policy == "single"
    assert manifest.capabilities == ["upscale", "denoise"]
    assert manifest.description
    props = manifest.params_schema["properties"]
    assert list(props) == ["model", "outscale", "tile", "tile_pad"]
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
    assert model["default"] == DEFAULT_MODEL == "real_x4_m"
    assert set(model["enum_labels"]) == set(MODELS)
    assert set(model["enum_descriptions"]) == set(MODELS)
    assert all(model["enum_descriptions"].values())
    assert props["outscale"]["visible_when"] == {"model": ["real_x4_m", "real_x4_l", "real_x2_m"]}
    assert props["outscale"]["overrides"] == [{"when": {"model": ["real_x2_m"]}, "default": 2}]
    assert props["tile"]["title"] == "分块大小"
    assert props["tile_pad"]["title"] == "分块重叠"
    assert "fp16" not in props


def test_build_command_uses_defaults(tmp_path: Path):
    command = _command(tmp_path)
    assert command[:4] == ["python", "-u", "-m", "videox.engines.swinir.progress_entry"]
    assert command[4] == str((tmp_path / "in.png").resolve())
    assert _arg(command, "--output") == str((tmp_path / "out.png").resolve())
    assert _arg(command, "--model-dir") == str(tmp_path / "models" / "SwinIR")
    assert _arg(command, "--model") == "real_x4_m"
    assert _arg(command, "--outscale") == "4"
    assert _arg(command, "--tile") == "384"
    assert _arg(command, "--tile-pad") == "32"
    assert command[-2:] == ["--device", "1"]


def test_build_command_follows_the_model(tmp_path: Path):
    assert _arg(_command(tmp_path, model="real_x2_m"), "--outscale") == "2"
    assert _arg(_command(tmp_path, model="real_x2_m", outscale=3), "--outscale") == "3"
    assert _arg(_command(tmp_path, model="real_x4_l", outscale=2.5), "--outscale") == "2.5"
    assert _arg(_command(tmp_path, model="real_x4_l"), "--outscale") == "4"
    for model_id in MODELS:
        if model_id in SR_MODELS:
            continue
        # Denoise and JPEG models keep the input size: outscale is hidden and ignored, even out of range.
        command = _command(tmp_path, model=model_id, outscale=99)
        assert _arg(command, "--model") == model_id
        assert "--outscale" not in command
    tuned = _command(tmp_path, model="denoise_50", tile=0, tile_pad=0)
    assert _arg(tuned, "--tile") == "0"
    assert _arg(tuned, "--tile-pad") == "0"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"model": "classical_x4"}, "无法识别"),
        ({"outscale": 9}, "不能大于 8"),
        ({"outscale": 0.5}, "不能小于 1"),
        ({"model": "real_x2_m", "outscale": "2"}, "需要是数字"),
        ({"tile": -1}, "不能小于 0"),
        ({"tile": 4096}, "不能大于 2048"),
        ({"tile": 64.5}, "需要是整数"),
        ({"tile_pad": 257}, "不能大于 256"),
        ({"tile_pad": True}, "需要是整数"),
    ],
)
def test_build_command_rejects_bad_values(tmp_path: Path, params: dict, message: str):
    with pytest.raises(VideoXError, match=message):
        _command(tmp_path, **params)


def test_runner_rejects_video(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"mp4")
    registry = EngineRegistry()
    registry.register(SwinIREngine())
    runner = JobRunner(registry, JobStore(), tmp_path)
    with pytest.raises(VideoXError, match="不支持视频"):
        runner.submit(
            JobSpec(
                engine_id="swinir",
                input_path=str(source),
                output_path=str(tmp_path / "out.mp4"),
                media="video",
            ),
            background=False,
        )


def test_model_table_matches_the_release():
    assert list(MODELS) == list(RELEASE_LAYOUT)
    assert SR_MODELS == ["real_x4_m", "real_x4_l", "real_x2_m"]
    # (upsampler, img_size, window_size, img_range, param key) per family, as in upstream define_model.
    families = {
        "real": ("nearest+conv", 64, 8, 1.0, "params_ema"),
        "denoise": ("", 128, 8, 1.0, "params"),
        "jpeg": ("", 126, 7, 255.0, "params"),
    }
    for model_id, spec in MODELS.items():
        assert spec.id == model_id and spec.label
        assert spec.urls == (f"{RELEASE_URL}/{spec.file}",)
        kwargs = spec.arch_kwargs()
        family = families[model_id.split("_")[0]]
        assert (
            kwargs["upsampler"],
            kwargs["img_size"],
            kwargs["window_size"],
            kwargs["img_range"],
            spec.param_key,
        ) == family
        assert kwargs["mlp_ratio"] == 2.0
        if model_id != "real_x4_l":
            assert (kwargs["depths"], kwargs["embed_dim"], kwargs["num_heads"], kwargs["resi_connection"]) == (
                [6] * 6,
                180,
                [6] * 6,
                "1conv",
            )
    large = MODELS["real_x4_l"].arch_kwargs()
    assert (large["depths"], large["embed_dim"], large["num_heads"], large["resi_connection"]) == (
        [6] * 9,
        240,
        [8] * 9,
        "3conv",
    )
    assert [spec.scale for spec in MODELS.values()] == [4, 4, 2] + [1] * 7
    with pytest.raises(VideoXError, match="未知的 SwinIR 模型"):
        model_spec("nope")


def test_weight_lists_only_require_the_default_model(tmp_path: Path):
    default = MODELS[DEFAULT_MODEL].file
    assert weight_lists(tmp_path) == ([], [default])
    (tmp_path / MODELS["jpeg_40"].file).write_bytes(b"x")
    (tmp_path / MODELS["real_x4_l"].file).write_bytes(b"")
    assert weight_lists(tmp_path) == ([MODELS["jpeg_40"].file], [default])
    (tmp_path / default).write_bytes(b"x")
    assert weight_lists(tmp_path) == ([default, MODELS["jpeg_40"].file], [])
    assert len(known_files()) == len(MODELS) == 10


def test_ensure_model_downloads_only_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetched: list[tuple[str, Path]] = []

    def fake_download(url: str, dest: Path) -> None:
        if dest.is_file() and dest.stat().st_size > 0:
            return
        fetched.append((url, dest))
        dest.write_bytes(b"x")

    monkeypatch.setattr("videox.engines.swinir.weights.download_file", fake_download)
    spec = MODELS["denoise_25"]
    assert ensure_model(tmp_path, "denoise_25") == [tmp_path / spec.file]
    assert ensure_model(tmp_path, "denoise_25") == [tmp_path / spec.file]
    assert fetched == [(f"{RELEASE_URL}/{spec.file}", tmp_path / spec.file)]
    with pytest.raises(VideoXError, match="未知的 SwinIR 模型"):
        ensure_model(tmp_path, "nope")


@pytest.mark.parametrize("model_id", list(MODELS))
def test_networks_match_the_release_layout(model_id: str):
    torch = pytest.importorskip("torch")
    from videox.engines.swinir.infer import build_model

    with torch.device("meta"):
        network = build_model(MODELS[model_id])
    state = network.state_dict()
    count, mask = RELEASE_LAYOUT[model_id]
    assert len(state) == count
    assert tuple(state["layers.0.residual_group.blocks.1.attn_mask"].shape) == mask
    assert "layers.0.residual_group.blocks.0.attn_mask" not in state
    assert "mean" not in state


@pytest.mark.parametrize(
    ("model_id", "size"),
    [("real_x4_m", 16), ("real_x4_l", 16), ("real_x2_m", 16), ("denoise_25", 16), ("jpeg_40", 14)],
)
def test_networks_have_the_right_output_shape(model_id: str, size: int):
    torch = pytest.importorskip("torch")
    from videox.engines.swinir.arch import SwinIR

    spec = _tiny(model_id)
    torch.manual_seed(0)
    network = SwinIR(**spec.arch_kwargs()).eval()
    with torch.inference_mode():
        # Odd sizes go through reflect padding and a per-forward mask; the training size uses the stored one.
        for height, width in ((11, 13), (size, size), (spec.window_size * 3, 5 + spec.window_size)):
            output = network(torch.rand(1, 3, height, width))
            assert output.shape == (1, 3, height * spec.scale, width * spec.scale)
            assert torch.isfinite(output).all()


def test_arch_rejects_unsupported_heads():
    pytest.importorskip("torch")
    from videox.engines.swinir.arch import SwinIR

    with pytest.raises(ValueError):
        SwinIR(upscale=3, upsampler="nearest+conv", depths=(2,), num_heads=(2,), embed_dim=8, img_size=16)
    with pytest.raises(ValueError):
        SwinIR(upscale=2, upsampler="", depths=(2,), num_heads=(2,), embed_dim=8, img_size=16)
    with pytest.raises(ValueError):
        SwinIR(upscale=2, upsampler="pixelshuffle", depths=(2,), num_heads=(2,), embed_dim=8, img_size=16)


def test_mirror_pad_matches_upstream_test_script():
    torch = pytest.importorskip("torch")
    from videox.engines.swinir.arch import SwinIR
    from videox.engines.swinir.infer import mirror_pad, mirror_padded

    image = torch.rand(1, 3, 13, 21)
    window = 8
    # main_test_swinir.py: always pad to the next multiple above the size by mirroring.
    h_pad = (13 // window + 1) * window - 13
    w_pad = (21 // window + 1) * window - 21
    expected = torch.cat([image, torch.flip(image, [2])], 2)[:, :, : 13 + h_pad, :]
    expected = torch.cat([expected, torch.flip(expected, [3])], 3)[:, :, :, : 21 + w_pad]
    assert torch.equal(mirror_pad(image, window), expected)
    assert mirror_pad(torch.rand(1, 3, 16, 16), window).shape == (1, 3, 24, 24)
    tiny = mirror_pad(torch.rand(1, 3, 2, 3), 7)
    assert tiny.shape == (1, 3, 7, 7)

    torch.manual_seed(0)
    network = SwinIR(**_tiny("real_x2_m").arch_kwargs()).eval()
    with torch.inference_mode():
        assert mirror_padded(network, 8)(torch.rand(1, 3, 3, 5)).shape == (1, 3, 6, 10)


def test_pipeline_runs_with_local_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    from videox.engines.swinir.infer import SwinIRRequest, build_model, run_swinir

    # A shrunken random-init checkpoint stands in for the release file, so nothing downloads.
    specs = {model_id: _tiny(model_id) for model_id in ("real_x2_m", "jpeg_40")}
    for model_id, spec in specs.items():
        spec = dataclasses.replace(spec, file=f"{model_id}.pth")
        specs[model_id] = spec
        torch.manual_seed(0)
        state = build_model(spec).state_dict()
        torch.save({spec.param_key: state} if model_id == "jpeg_40" else state, tmp_path / spec.file)

    def tiny_spec(model_id: str):
        if model_id not in specs:
            raise VideoXError(f"未知的 SwinIR 模型: {model_id}")
        return specs[model_id]

    monkeypatch.setattr("videox.engines.swinir.infer.model_spec", tiny_spec)
    monkeypatch.setattr("videox.engines.swinir.weights.model_spec", tiny_spec)

    rgba = np.zeros((9, 13, 4), dtype=np.uint8)
    rgba[..., :3] = np.random.default_rng(0).integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    rgba[..., 3] = np.linspace(0, 255, 13, dtype=np.uint8)
    source = tmp_path / "in.png"
    Image.fromarray(rgba, "RGBA").save(source)

    def request(**changes: object) -> SwinIRRequest:
        values = {
            "input_path": source,
            "output_path": tmp_path / "out" / "result.png",
            "model_dir": tmp_path,
            "model": "real_x2_m",
            "outscale": 3.0,
            "tile": 8,
            "tile_pad": 2,
        }
        values.update(changes)
        return SwinIRRequest(**values)

    run_swinir(request())
    with Image.open(tmp_path / "out" / "result.png") as written:
        assert written.mode == "RGBA" and written.size == (39, 27)
        alpha = np.asarray(written)[..., 3]
        assert alpha[:, 0].max() < 40 and alpha[:, -1].min() > 215
    lines = capsys.readouterr().out
    for phase in ("download", "load", "restore"):
        assert f'"phase": "{phase}"' in lines
    assert '"total": 4' in lines  # 13x9 in 8-pixel tiles

    run_swinir(request(model="jpeg_40", outscale=None, tile=0, output_path=tmp_path / "clean.jpg"))
    with Image.open(tmp_path / "clean.jpg") as written:
        assert written.mode == "RGB" and written.size == (13, 9)

    with pytest.raises(VideoXError, match="不支持的图片格式"):
        run_swinir(request(output_path=tmp_path / "x.gif"))
    with pytest.raises(VideoXError, match="未知的 SwinIR 模型"):
        run_swinir(request(model="nope"))


@pytest.mark.parametrize("model_id", list(MODELS))
def test_released_weights_load_and_run(model_id: str):
    spec = MODELS[model_id]
    if not (WEIGHTS / spec.file).is_file():
        pytest.skip("SwinIR weights were not downloaded to .work/weights/SwinIR")
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from videox.engines.image_io import run_tiled
    from videox.engines.swinir.infer import load_model, mirror_padded

    # load_model reads the upstream param key and loads with strict=True.
    model = load_model(WEIGHTS / spec.file, spec, torch.device("cpu"))
    torch.manual_seed(1)
    image = torch.rand(1, 3, 11, 15)
    output = run_tiled(mirror_padded(model, spec.window_size), image, scale=spec.scale, tile=8, tile_pad=2)
    assert output.shape == (11 * spec.scale, 15 * spec.scale, 3)

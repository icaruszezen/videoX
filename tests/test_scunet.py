import dataclasses
from pathlib import Path

import pytest

from videox.engines.base import EngineContext
from videox.engines.registry import EngineRegistry
from videox.engines.scunet import SCUNetEngine
from videox.engines.scunet.presets import DEFAULT_MODEL, MODELS, RELEASE_URL, model_spec
from videox.engines.scunet.weights import ensure_model, known_files, weight_lists
from videox.errors import VideoXError
from videox.hardware.gpus import GpuInfo
from videox.jobs.models import JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore

WEIGHTS = Path(__file__).resolve().parents[1] / ".work" / "weights" / "SCUNet"

# Tensors in each release checkpoint (plain state dicts, no buffers).
RELEASE_TENSORS = 540


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
        engine_id="scunet",
        input_path=str(source),
        output_path=str(tmp_path / "out.png"),
        media="image",
        devices=[1, 0],
        params=params,
    )


def _command(tmp_path: Path, **params: object) -> list[str]:
    return SCUNetEngine().build_command(_image_job(tmp_path, **params), _context(tmp_path))


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _allowed(prop: dict, value: object) -> bool:
    if "enum" in prop:
        return value in prop["enum"]
    if prop["type"] == "boolean":
        return isinstance(value, bool)
    return False


def _tiny(model_id: str, **changes: object):
    """A release config cut to one block per stage so it runs in a blink."""
    return dataclasses.replace(MODELS[model_id], **{"config": (1,) * 7, "file": "tiny.pth", **changes})


def test_manifest_describes_an_image_engine():
    manifest = SCUNetEngine().manifest()
    assert manifest.id == "scunet"
    assert manifest.name == "SCUNet"
    assert manifest.media == ["image"]
    assert manifest.device_policy == "single"
    assert manifest.capabilities == ["denoise"]
    assert manifest.description
    props = manifest.params_schema["properties"]
    assert list(props) == ["model", "strength", "tile", "tile_pad"]
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
    assert model["enum"] == list(MODELS) == ["real_psnr", "real_gan"]
    assert model["default"] == DEFAULT_MODEL == "real_psnr"
    assert model["enum_labels"] == {"real_psnr": "去噪（保真）", "real_gan": "去噪（更锐）"}
    assert set(model["enum_descriptions"]) == set(MODELS)
    assert all(model["enum_descriptions"].values())
    strength = props["strength"]
    assert (strength["type"], strength["title"], strength["default"]) == ("number", "去噪强度", 1.0)
    assert (strength["minimum"], strength["maximum"]) == (0, 1)
    assert (props["tile"]["title"], props["tile"]["default"], props["tile"]["maximum"]) == ("分块大小", 512, 4096)
    assert (props["tile_pad"]["title"], props["tile_pad"]["default"], props["tile_pad"]["maximum"]) == ("分块重叠", 32, 256)
    assert {"fp16", "outscale"}.isdisjoint(props)


def test_build_command_uses_defaults(tmp_path: Path):
    command = _command(tmp_path)
    assert command[:4] == ["python", "-u", "-m", "videox.engines.scunet.progress_entry"]
    assert command[4] == str((tmp_path / "in.png").resolve())
    assert _arg(command, "--output") == str((tmp_path / "out.png").resolve())
    assert _arg(command, "--model-dir") == str(tmp_path / "models" / "SCUNet")
    assert _arg(command, "--model") == "real_psnr"
    assert _arg(command, "--strength") == "1"
    assert _arg(command, "--tile") == "512"
    assert _arg(command, "--tile-pad") == "32"
    assert command[-2:] == ["--device", "1"]


def test_build_command_passes_choices_and_ignores_unknown_params(tmp_path: Path):
    command = _command(tmp_path, model="real_gan", strength=0.65, tile=0, tile_pad=0)
    assert _arg(command, "--model") == "real_gan"
    assert _arg(command, "--strength") == "0.65"
    assert _arg(command, "--tile") == "0"
    assert _arg(command, "--tile-pad") == "0"
    assert _arg(_command(tmp_path, strength=0), "--strength") == "0"
    # SCUNet never resizes: an output scale or fp16 switch from the caller is dropped, even out of range.
    ignored = _command(tmp_path, outscale=99, fp16="yes")
    assert "--outscale" not in ignored and "--fp16" not in ignored
    assert ignored == _command(tmp_path)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"model": "color_50"}, "无法识别"),
        ({"strength": 1.5}, "不能大于 1"),
        ({"strength": -0.1}, "不能小于 0"),
        ({"strength": "1"}, "需要是数字"),
        ({"strength": True}, "需要是数字"),
        ({"tile": -1}, "不能小于 0"),
        ({"tile": 4097}, "不能大于 4096"),
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
    registry.register(SCUNetEngine())
    runner = JobRunner(registry, JobStore(), tmp_path)
    with pytest.raises(VideoXError, match="不支持视频"):
        runner.submit(
            JobSpec(
                engine_id="scunet",
                input_path=str(source),
                output_path=str(tmp_path / "out.mp4"),
                media="video",
            ),
            background=False,
        )


def test_model_table_matches_the_release():
    files = {"real_psnr": "scunet_color_real_psnr.pth", "real_gan": "scunet_color_real_gan.pth"}
    for model_id, spec in MODELS.items():
        assert spec.id == model_id and spec.label and spec.note
        assert spec.file == files[model_id]
        assert spec.urls == (f"{RELEASE_URL}/{spec.file}",)
        assert spec.scale == 1
        # main_test_scunet_real_application.py: SCUNet(in_nc=3, config=[4,4,4,4,4,4,4], dim=64).
        assert spec.arch_kwargs() == {"in_nc": 3, "config": [4] * 7, "dim": 64}
    assert RELEASE_URL == "https://github.com/cszn/KAIR/releases/download/v1.0"
    with pytest.raises(VideoXError, match="未知的 SCUNet 模型"):
        model_spec("nope")


def test_weight_lists_only_require_the_default_model(tmp_path: Path):
    default = MODELS[DEFAULT_MODEL].file
    gan = MODELS["real_gan"].file
    assert weight_lists(tmp_path) == ([], [default])
    (tmp_path / gan).write_bytes(b"x")
    assert weight_lists(tmp_path) == ([gan], [default])
    (tmp_path / default).write_bytes(b"")
    assert weight_lists(tmp_path) == ([gan], [default])
    (tmp_path / default).write_bytes(b"x")
    assert weight_lists(tmp_path) == ([default, gan], [])
    assert known_files() == (default, gan)


def test_ensure_model_downloads_only_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetched: list[tuple[str, Path]] = []

    def fake_download(url: str, dest: Path) -> None:
        if dest.is_file() and dest.stat().st_size > 0:
            return
        fetched.append((url, dest))
        dest.write_bytes(b"x")

    monkeypatch.setattr("videox.engines.scunet.weights.download_file", fake_download)
    spec = MODELS["real_gan"]
    assert ensure_model(tmp_path, "real_gan") == [tmp_path / spec.file]
    assert ensure_model(tmp_path, "real_gan") == [tmp_path / spec.file]
    assert fetched == [(f"{RELEASE_URL}/{spec.file}", tmp_path / spec.file)]
    with pytest.raises(VideoXError, match="未知的 SCUNet 模型"):
        ensure_model(tmp_path, "nope")


@pytest.mark.parametrize("model_id", list(MODELS))
def test_network_matches_the_release_layout(model_id: str):
    torch = pytest.importorskip("torch")
    from videox.engines.scunet.infer import build_model

    with torch.device("meta"):
        network = build_model(MODELS[model_id])
    state = network.state_dict()
    assert len(state) == RELEASE_TENSORS
    assert tuple(state["m_down1.0.trans_block.msa.relative_position_params"].shape) == (1, 15, 15)
    assert tuple(state["m_body.0.trans_block.msa.relative_position_params"].shape) == (8, 15, 15)
    assert tuple(state["m_down3.4.weight"].shape) == (512, 256, 2, 2)
    assert tuple(state["m_tail.0.weight"].shape) == (3, 64, 3, 3)
    assert not any("relative_rows" in key or "relative_cols" in key for key in state)
    # Blocks alternate plain and shifted windows in every stage.
    assert [block.type for block in network.m_body] == ["W", "SW", "W", "SW"]


def test_networks_have_the_right_output_shape():
    torch = pytest.importorskip("torch")
    from videox.engines.scunet.arch import SCUNet

    torch.manual_seed(0)
    network = SCUNet(in_nc=3, config=[2] * 7, dim=64).eval()
    with torch.inference_mode():
        # Any size works: the network pads to a multiple of 64 and crops back.
        for height, width in ((1, 1), (11, 13), (64, 64), (70, 33)):
            output = network(torch.rand(1, 3, height, width))
            assert output.shape == (1, 3, height, width)
            assert torch.isfinite(output).all()
        gray = SCUNet(in_nc=1, config=[1] * 7, dim=64).eval()
        assert gray(torch.rand(2, 1, 20, 9)).shape == (2, 1, 20, 9)
    # A small training size turns the deep stages into plain windows, as upstream does.
    small = SCUNet(config=[2] * 7, dim=64, input_resolution=16)
    assert [block.type for block in small.m_down1[:2]] == ["W", "SW"]
    assert [block.type for block in small.m_body] == ["W", "W"]
    with pytest.raises(ValueError):
        SCUNet(config=[2] * 6)


def test_shifted_window_mask_blocks_pairs_from_opposite_edges():
    torch = pytest.importorskip("torch")
    from videox.engines.scunet.arch import WMSA

    p, shift = 8, 4
    attention = WMSA(32, 32, 32, p, "SW")
    rows, cols = 3, 2
    mask = attention.generate_mask(rows, cols, shift, torch.device("cpu"))
    assert mask.shape == (1, 1, rows * cols, p * p, p * p)
    # After rolling by -shift, the last ``shift`` rows and columns wrapped round from the far edge.
    wrapped_row = torch.arange(rows * p) >= rows * p - shift
    wrapped_col = torch.arange(cols * p) >= cols * p - shift
    region = wrapped_row[:, None].long() * 2 + wrapped_col[None, :].long()
    windows = region.view(rows, p, cols, p).permute(0, 2, 1, 3).reshape(rows * cols, p * p)
    expected = windows[:, :, None] != windows[:, None, :]
    assert torch.equal(mask[0, 0], expected)
    plain = WMSA(32, 32, 32, p, "W").generate_mask(rows, cols, shift, torch.device("cpu"))
    assert not plain.any()
    embedding = attention.relative_embedding()
    assert embedding.shape == (1, p * p, p * p)
    # Same offset, same bias: (0,0)->(1,1) and (2,3)->(3,4) share one entry.
    assert embedding[0, 0, 9] == embedding[0, 2 * p + 3, 3 * p + 4]


def test_strength_blends_with_the_input():
    torch = pytest.importorskip("torch")
    from videox.engines.scunet.infer import blended

    def fake(image):
        return torch.full_like(image, 1.4)  # clamped to 1 before mixing

    image = torch.rand(1, 3, 4, 5)
    assert blended(fake, 1.0) is fake
    assert torch.allclose(blended(fake, 0.25)(image), 0.25 + 0.75 * image)
    assert torch.equal(blended(fake, 0.0)(image), image)


def test_pipeline_runs_with_local_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    from videox.engines.scunet.infer import SCUNetRequest, build_model, run_scunet

    # A shrunken random-init checkpoint stands in for the release file, so nothing downloads.
    specs = {model_id: _tiny(model_id, file=f"{model_id}.pth") for model_id in MODELS}
    for spec in specs.values():
        torch.manual_seed(0)
        torch.save(build_model(spec).state_dict(), tmp_path / spec.file)

    def tiny_spec(model_id: str):
        if model_id not in specs:
            raise VideoXError(f"未知的 SCUNet 模型: {model_id}")
        return specs[model_id]

    monkeypatch.setattr("videox.engines.scunet.infer.model_spec", tiny_spec)
    monkeypatch.setattr("videox.engines.scunet.weights.model_spec", tiny_spec)

    rgba = np.zeros((9, 13, 4), dtype=np.uint8)
    rgba[..., :3] = np.random.default_rng(0).integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    rgba[..., 3] = np.linspace(0, 255, 13, dtype=np.uint8)
    source = tmp_path / "in.png"
    Image.fromarray(rgba, "RGBA").save(source)

    def request(**changes: object) -> SCUNetRequest:
        values = {
            "input_path": source,
            "output_path": tmp_path / "out" / "result.png",
            "model_dir": tmp_path,
            "model": "real_psnr",
            "strength": 1.0,
            "tile": 8,
            "tile_pad": 2,
        }
        values.update(changes)
        return SCUNetRequest(**values)

    run_scunet(request())
    with Image.open(tmp_path / "out" / "result.png") as written:
        assert written.mode == "RGBA" and written.size == (13, 9)
        assert np.array_equal(np.asarray(written)[..., 3], rgba[..., 3])
    lines = capsys.readouterr().out
    for phase in ("download", "load", "restore"):
        assert f'"phase": "{phase}"' in lines
    assert '"total": 4' in lines  # 13x9 in 8-pixel tiles

    run_scunet(request(model="real_gan", tile=0, strength=0.0, output_path=tmp_path / "clean.png"))
    with Image.open(tmp_path / "clean.png") as written:
        # Strength 0 gives the input back.
        assert written.mode == "RGBA" and np.array_equal(np.asarray(written), rgba)
    run_scunet(request(tile=0, output_path=tmp_path / "clean.jpg"))
    with Image.open(tmp_path / "clean.jpg") as written:
        assert written.mode == "RGB" and written.size == (13, 9)

    with pytest.raises(VideoXError, match="不支持的图片格式"):
        run_scunet(request(output_path=tmp_path / "x.gif"))
    with pytest.raises(VideoXError, match="未知的 SCUNet 模型"):
        run_scunet(request(model="nope"))
    with pytest.raises(VideoXError, match="去噪强度"):
        run_scunet(request(strength=1.5))
    with pytest.raises(VideoXError, match="不能是负数"):
        run_scunet(request(tile_pad=-1))
    with pytest.raises(VideoXError, match="输入文件不存在"):
        run_scunet(request(input_path=tmp_path / "missing.png"))


@pytest.mark.parametrize("model_id", list(MODELS))
def test_released_weights_load_and_denoise(model_id: str):
    spec = MODELS[model_id]
    if not (WEIGHTS / spec.file).is_file():
        pytest.skip("SCUNet weights were not downloaded to .work/weights/SCUNet")
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from videox.engines.image_io import run_tiled
    from videox.engines.scunet.infer import load_model

    # load_model loads the plain state dict with strict=True.
    model = load_model(WEIGHTS / spec.file, spec, torch.device("cpu"))
    torch.manual_seed(1)
    clean = torch.linspace(0.2, 0.8, 72).repeat(3, 56, 1).unsqueeze(0)
    noisy = (clean + 0.08 * torch.randn_like(clean)).clamp(0, 1)
    output = run_tiled(model, noisy, scale=1, tile=48, tile_pad=8)
    assert output.shape == (56, 72, 3)
    restored = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float() / 255
    before = (noisy - clean).abs().mean()
    after = (restored - clean).abs().mean()
    assert after < before / 2

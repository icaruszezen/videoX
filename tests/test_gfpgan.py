import os
import sys
import types
from pathlib import Path

import pytest

from videox.engines.base import EngineContext
from videox.engines.gfpgan import GFPGANEngine
from videox.engines.gfpgan.presets import (
    BACKGROUND_UPSCALERS,
    BACKGROUNDS,
    DEFAULT_VERSION,
    DETECTOR,
    MODELS,
    NO_BACKGROUND,
    model_spec,
)
from videox.engines.gfpgan.weights import ensure_model, known_weights, weight_lists
from videox.engines.realesrgan.presets import MODELS as REALESRGAN_MODELS
from videox.engines.registry import EngineRegistry
from videox.errors import VideoXError
from videox.hardware.gpus import GpuInfo
from videox.jobs.models import JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore

WEIGHTS = Path(__file__).resolve().parents[1] / ".work" / "weights" / "GFPGAN"


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
        engine_id="gfpgan",
        input_path=str(source),
        output_path=str(tmp_path / "out.png"),
        media="image",
        devices=[1, 0],
        params=params,
    )


def _command(tmp_path: Path, **params: object) -> list[str]:
    return GFPGANEngine().build_command(_image_job(tmp_path, **params), _context(tmp_path))


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _allowed(prop: dict, value: object) -> bool:
    if "enum" in prop:
        return value in prop["enum"]
    if prop["type"] == "boolean":
        return isinstance(value, bool)
    return False


def test_manifest_describes_an_image_engine():
    manifest = GFPGANEngine().manifest()
    assert manifest.id == "gfpgan"
    assert manifest.name == "GFPGAN"
    assert manifest.media == ["image"]
    assert manifest.device_policy == "single"
    assert manifest.capabilities == ["face", "upscale"]
    assert manifest.description
    props = manifest.params_schema["properties"]
    assert list(props) == [
        "version",
        "upscale",
        "background",
        "strength",
        "only_center_face",
        "tile",
        "tile_pad",
        "fp16",
    ]
    groups = {group["id"] for group in manifest.params_schema["groups"]}
    seen: list[str] = []
    for name, prop in props.items():
        assert prop.get("description"), name
        assert "default" in prop, name
        assert prop["media"] == ["image"], name
        assert prop["group"] in groups, name
        if "enum" in prop:
            assert prop["default"] in prop["enum"], name
            assert set(prop["enum_labels"]) == set(prop["enum"]), name
        conditions = [prop.get("visible_when") or {}]
        conditions += [override["when"] for override in prop.get("overrides") or []]
        for condition in conditions:
            for key, values in condition.items():
                assert key in seen, f"{name} depends on later field {key}"
                assert values and all(_allowed(props[key], value) for value in values), (name, key)
        for override in prop.get("overrides") or []:
            if "default" in override and "enum" in prop:
                assert override["default"] in prop["enum"], name
        seen.append(name)
    version = props["version"]
    assert version["enum"] == list(MODELS)
    assert version["default"] == DEFAULT_VERSION == "v1.4"
    assert set(version["enum_descriptions"]) == set(MODELS)
    assert all(version["enum_descriptions"].values())
    assert props["upscale"]["enum"] == ["1", "2", "4"]
    assert props["upscale"]["default"] == "2"
    background = props["background"]
    assert background["enum"] == list(BACKGROUNDS)
    assert background["default"] == "RealESRGAN_x2plus"
    assert set(background["enum_descriptions"]) == set(BACKGROUNDS)
    assert all(background["enum_descriptions"].values())
    # Every background upscaler is a model the Real-ESRGAN package knows.
    assert BACKGROUND_UPSCALERS and all(key in REALESRGAN_MODELS for key in BACKGROUND_UPSCALERS)
    for name in ("tile", "tile_pad", "fp16"):
        assert props[name]["visible_when"] == {"background": BACKGROUND_UPSCALERS}, name
    assert props["strength"]["minimum"] == 0 and props["strength"]["maximum"] == 1
    assert props["strength"]["default"] == 1.0
    assert props["only_center_face"]["default"] is False


def test_build_command_uses_defaults(tmp_path: Path):
    command = _command(tmp_path)
    assert command[:4] == ["python", "-u", "-m", "videox.engines.gfpgan.progress_entry"]
    assert command[4] == str((tmp_path / "in.png").resolve())
    assert _arg(command, "--output") == str((tmp_path / "out.png").resolve())
    assert _arg(command, "--model-dir") == str(tmp_path / "models" / "GFPGAN")
    assert _arg(command, "--version") == "v1.4"
    assert _arg(command, "--upscale") == "2"
    assert _arg(command, "--background") == "RealESRGAN_x2plus"
    assert _arg(command, "--strength") == "1"
    assert "--no-only-center-face" in command and "--only-center-face" not in command
    assert _arg(command, "--tile") == "512"
    assert _arg(command, "--tile-pad") == "10"
    assert "--fp16" in command and "--no-fp16" not in command
    assert command[-2:] == ["--device", "1"]


def test_build_command_follows_the_choices(tmp_path: Path):
    # A 4x output defaults to the 4x background network; an explicit choice wins.
    assert _arg(_command(tmp_path, upscale="4"), "--background") == "RealESRGAN_x4plus"
    assert _arg(_command(tmp_path, upscale="4", background="RealESRGAN_x2plus"), "--background") == "RealESRGAN_x2plus"
    assert _arg(_command(tmp_path, upscale="1"), "--background") == "RealESRGAN_x2plus"
    v13 = _command(tmp_path, version="v1.3", strength=0.25, only_center_face=True)
    assert _arg(v13, "--version") == "v1.3"
    assert _arg(v13, "--strength") == "0.25"
    assert "--only-center-face" in v13 and "--no-only-center-face" not in v13
    # Faces only: the background settings are hidden and ignored, even when out of range.
    faces_only = _command(tmp_path, background=NO_BACKGROUND, tile=99999, tile_pad=-3, fp16="yes")
    assert _arg(faces_only, "--background") == "none"
    for flag in ("--tile", "--tile-pad", "--fp16", "--no-fp16"):
        assert flag not in faces_only
    tuned = _command(tmp_path, background="RealESRGAN_x4plus", tile=0, tile_pad=32, fp16=False)
    assert _arg(tuned, "--tile") == "0"
    assert _arg(tuned, "--tile-pad") == "32"
    assert "--no-fp16" in tuned and "--fp16" not in tuned


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"version": "v1.2"}, "无法识别"),
        ({"upscale": "3"}, "无法识别"),
        ({"upscale": 2}, "需要是字符串"),
        ({"background": "RealESRGAN_x8"}, "无法识别"),
        ({"strength": 1.5}, "不能大于 1"),
        ({"strength": -0.1}, "不能小于 0"),
        ({"strength": "1"}, "需要是数字"),
        ({"only_center_face": 1}, "需要是布尔值"),
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


def test_progress_entry_reads_the_built_command(tmp_path: Path, monkeypatch):
    from videox.engines.gfpgan import progress_entry

    captured: list[object] = []

    class Request:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    # Stand in for the torch-side module so this runs without torch.
    fake = types.ModuleType("videox.engines.gfpgan.infer")
    fake.GFPGANRequest = Request
    fake.run_gfpgan = captured.append
    monkeypatch.setitem(sys.modules, "videox.engines.gfpgan.infer", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    command = _command(tmp_path, upscale="4", strength=0.5, only_center_face=True, fp16=False, tile=256)
    assert progress_entry.main(command[4:]) == 0
    request = captured[-1].__dict__
    assert request["input_path"] == Path(command[4])
    assert request["model_dir"] == tmp_path / "models" / "GFPGAN"
    assert request["version"] == "v1.4"
    assert request["upscale"] == 4
    assert request["background"] == "RealESRGAN_x4plus"
    assert request["strength"] == 0.5
    assert request["only_center_face"] is True
    assert request["tile"] == 256 and request["tile_pad"] == 10
    assert request["half"] is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    faces_only = _command(tmp_path, background=NO_BACKGROUND)
    assert progress_entry.main(faces_only[4:]) == 0
    request = captured[-1].__dict__
    assert request["background"] == "none"
    assert request["upscale"] == 2 and request["only_center_face"] is False


def test_runner_rejects_video(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"mp4")
    registry = EngineRegistry()
    registry.register(GFPGANEngine())
    runner = JobRunner(registry, JobStore(), tmp_path)
    with pytest.raises(VideoXError, match="不支持视频"):
        runner.submit(
            JobSpec(
                engine_id="gfpgan",
                input_path=str(source),
                output_path=str(tmp_path / "out.mp4"),
                media="video",
            ),
            background=False,
        )


def test_weight_lists_only_require_the_default_version(tmp_path: Path):
    assert weight_lists(tmp_path) == ([], ["GFPGANv1.4.pth", "detection_Resnet50_Final.pth"])
    (tmp_path / "GFPGANv1.3.pth").write_bytes(b"x")
    (tmp_path / "detection_Resnet50_Final.pth").write_bytes(b"")
    assert weight_lists(tmp_path) == (["GFPGANv1.3.pth"], ["GFPGANv1.4.pth", "detection_Resnet50_Final.pth"])
    (tmp_path / "GFPGANv1.4.pth").write_bytes(b"x")
    (tmp_path / "detection_Resnet50_Final.pth").write_bytes(b"x")
    present, missing = weight_lists(tmp_path)
    assert present == ["GFPGANv1.4.pth", "detection_Resnet50_Final.pth", "GFPGANv1.3.pth"]
    assert missing == []
    assert set(known_weights()) == {"GFPGANv1.4.pth", "GFPGANv1.3.pth", "detection_Resnet50_Final.pth"}
    for spec in MODELS.values():
        assert spec.weights.url == f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/{spec.weights.name}"
        assert spec.files[-1] == DETECTOR
    assert DETECTOR.url == "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    # Files already on disk are not downloaded again.
    assert ensure_model(tmp_path, "v1.3") == [tmp_path / "GFPGANv1.3.pth", tmp_path / "detection_Resnet50_Final.pth"]
    with pytest.raises(VideoXError, match="未知的 GFPGAN 版本"):
        ensure_model(tmp_path, "v9")
    with pytest.raises(VideoXError, match="未知的 GFPGAN 版本"):
        model_spec("nope")


def _tiny_gfpgan_kwargs(**changes: object) -> dict:
    # 512 output like the released models, with very narrow channels.
    kwargs = {
        "out_size": 512,
        "num_style_feat": 32,
        "channel_multiplier": 2,
        "num_mlp": 1,
        "input_is_latent": True,
        "different_w": True,
        "narrow": 0.125,
        "sft_half": True,
    }
    kwargs.update(changes)
    return kwargs


@pytest.mark.parametrize(
    "changes",
    [
        {"out_size": 32, "channel_multiplier": 1},
        {"out_size": 32, "channel_multiplier": 1, "different_w": False, "sft_half": False, "input_is_latent": False},
        {},
    ],
)
def test_gfpgan_network_shapes(changes: dict):
    torch = pytest.importorskip("torch")
    from videox.engines.gfpgan.arch import GFPGANv1Clean

    kwargs = _tiny_gfpgan_kwargs(**changes)
    size = kwargs["out_size"]
    torch.manual_seed(0)
    network = GFPGANv1Clean(**kwargs).eval()
    x = torch.rand(1, 3, size, size) * 2 - 1
    with torch.inference_mode():
        fixed_a = network(x, randomize_noise=False)
        fixed_b = network(x, randomize_noise=False)
        torch.manual_seed(3)
        noisy_a = network(x)
        torch.manual_seed(3)
        noisy_b = network(x)
    assert fixed_a.shape == (1, 3, size, size)
    assert torch.isfinite(fixed_a).all()
    assert torch.equal(fixed_a, fixed_b)
    assert torch.equal(noisy_a, noisy_b)


def test_released_gfpgan_config_has_checkpoint_layout():
    pytest.importorskip("torch")
    from videox.engines.gfpgan.arch import GFPGANv1Clean

    state = GFPGANv1Clean(**MODELS["v1.4"].arch_kwargs).state_dict()
    assert len(state) == 285
    assert "stylegan_decoder.noises.noise14" in state
    assert "toRGB.6.weight" in state
    assert tuple(state["final_linear.weight"].shape) == (16 * 512, 256 * 16)


def test_retinaface_shapes_and_priors():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from videox.engines.gfpgan.arch import RetinaFace, prior_box

    torch.manual_seed(0)
    network = RetinaFace().eval()
    height, width = 50, 70
    count = sum(-(-height // step) * -(-width // step) * 2 for step in (8, 16, 32))
    with torch.inference_mode():
        loc, conf, landmarks = network(torch.rand(1, 3, height, width) * 255 - 110)
    assert loc.shape == (1, count, 4)
    assert conf.shape == (1, count, 2)
    assert landmarks.shape == (1, count, 10)
    assert torch.allclose(conf.sum(-1), torch.ones(1, count))
    image = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    rows = network.detect_faces(image, conf_threshold=0.0, nms_threshold=0.4)
    assert rows.ndim == 2 and rows.shape[1] == 15 and rows.dtype == np.float32
    assert np.all(np.diff(rows[:, 4]) <= 0)

    # The vectorised priors follow facexlib's PriorBox loop order exactly.
    expected = []
    for step, sizes in zip((8, 16, 32), ((16, 32), (64, 128), (256, 512))):
        for i in range(-(-height // step)):
            for j in range(-(-width // step)):
                for size in sizes:
                    expected += [(j + 0.5) * step / width, (i + 0.5) * step / height, size / width, size / height]
    assert torch.equal(prior_box(height, width), torch.Tensor(expected).view(-1, 4))


def _synthetic_landmarks(scale: float, left: float, top: float):
    from videox.engines.gfpgan.faces import FACE_TEMPLATE

    return (FACE_TEMPLATE * scale + [left, top]).astype("float32")


def test_align_and_paste_back_round_trip():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from videox.engines.gfpgan.faces import align_face, inverse_affine, paste_faces

    yy, xx = np.mgrid[0:200, 0:220]
    image = np.dstack([xx * 255 / 220, yy * 255 / 200, (xx + yy) * 255 / 420]).astype(np.uint8)
    matrix, cropped = align_face(image, _synthetic_landmarks(0.25, 40, 30))
    assert cropped.shape == (512, 512, 3)
    assert np.allclose(matrix[:, :2], [[4, 0], [0, 4]], atol=1e-3)
    # Pasting the unchanged crop back leaves the face area as it was.
    same = paste_faces(image, [cropped], [inverse_affine(matrix, 1)], input_size=(200, 220), upscale=1)
    assert same.shape == image.shape and same.dtype == np.uint8
    centre = (slice(60, 130), slice(70, 140))
    assert np.abs(same[centre].astype(int) - image[centre].astype(int)).max() <= 2
    # A bright restored face lands only inside the face box at 2x.
    bright = np.full_like(cropped, 255)
    up = paste_faces(
        np.zeros((400, 440, 3), np.uint8), [bright], [inverse_affine(matrix, 2)], input_size=(200, 220), upscale=2
    )
    assert up.shape == (400, 440, 3)
    assert up[200, 230].min() >= 254
    assert up[10, 10].max() == 0 and up[390, 430].max() == 0
    assert paste_faces(image, [], [], input_size=(200, 220), upscale=1).tolist() == image.tolist()


def test_detect_faces_filters_and_rescales():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from videox.engines.gfpgan.faces import detect_faces

    class FakeDetector:
        def __init__(self, rows):
            self.rows = np.asarray(rows, dtype=np.float32)
            self.seen: list[tuple[int, int]] = []

        def detect_faces(self, image, conf_threshold):
            self.seen.append(image.shape[:2])
            return self.rows.copy()

    rows = [
        # box, score, eyes 20 px apart
        [10, 10, 50, 50, 0.99, 20, 25, 40, 25, 30, 35, 22, 42, 38, 42],
        # near the centre
        [70, 40, 110, 80, 0.98, 80, 55, 100, 55, 90, 65, 82, 72, 98, 72],
        # eyes 3 px apart: skipped
        [150, 10, 160, 20, 0.99, 154, 14, 157, 14, 155, 16, 154, 18, 157, 18],
    ]
    image = np.zeros((120, 180, 3), np.uint8)
    detector = FakeDetector(rows)
    faces = detect_faces(detector, image, only_center_face=False, eye_dist_threshold=5)
    assert len(faces) == 2 and detector.seen == [(120, 180)]
    assert faces[0].landmarks.shape == (5, 2) and faces[0].landmarks.dtype == np.float32
    centre = detect_faces(detector, image, only_center_face=True, eye_dist_threshold=5)
    assert len(centre) == 1 and centre[0].box[0] == 70
    # Above the detection limit the image is shrunk and coordinates scaled back.
    shrunk = detect_faces(detector, image, only_center_face=False, eye_dist_threshold=5, max_edge=90)
    assert detector.seen[-1] == (60, 90)
    assert np.allclose(shrunk[0].box[:4], [20, 20, 100, 100])
    assert np.isclose(shrunk[0].box[4], 0.99)
    assert np.allclose(shrunk[0].landmarks[0], [40, 50])


def _full_frame_paste(background, faces, inverses, *, input_size, upscale):
    """facexlib's square-mask paste-back, warping every face over the whole frame."""
    import cv2
    import numpy as np

    height, width = input_size
    h_up, w_up = int(height * upscale), int(width * upscale)
    image = cv2.resize(background, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
    for face, matrix in zip(faces, inverses, strict=True):
        matrix = matrix.copy()
        if upscale > 1:
            matrix[:, 2] += 0.5 * upscale
        inv_restored = cv2.warpAffine(face, matrix, (w_up, h_up))
        inv_mask = cv2.warpAffine(np.ones((512, 512), np.float32), matrix, (w_up, h_up))
        erosion = cv2.erode(inv_mask, np.ones((int(2 * upscale), int(2 * upscale)), np.uint8))
        pasted = erosion[:, :, None] * inv_restored
        w_edge = int(np.sum(erosion) ** 0.5) // 20
        center = cv2.erode(erosion, np.ones((w_edge * 2, w_edge * 2), np.uint8))
        soft = cv2.GaussianBlur(center, (w_edge * 2 + 1, w_edge * 2 + 1), 0)[:, :, None]
        image = soft * pasted + (1 - soft) * image
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("upscale", [1, 2, 4])
def test_paste_faces_matches_full_frame_paste(upscale: int):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from videox.engines.gfpgan.faces import align_face, inverse_affine, paste_faces

    rng = np.random.default_rng(upscale)
    image = rng.integers(0, 256, size=(150, 170, 3), dtype=np.uint8)
    faces, inverses = [], []
    # One face inside, one cut by the top-left corner, one overlapping the first.
    for scale, left, top in ((0.2, 50, 40), (0.15, -20, -15), (0.18, 70, 55)):
        matrix, _cropped = align_face(image, _synthetic_landmarks(scale, left, top))
        faces.append(rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8))
        inverses.append(inverse_affine(matrix, upscale))
    expected = _full_frame_paste(image, faces, inverses, input_size=(150, 170), upscale=upscale)
    pasted = paste_faces(image, faces, inverses, input_size=(150, 170), upscale=upscale)
    assert pasted.shape == expected.shape and pasted.dtype == np.uint8
    # Only rounding differs: each face is truncated to 8 bits before the next one blends.
    assert np.abs(pasted.astype(int) - expected.astype(int)).max() <= 2


def test_paste_faces_strength_scales_the_mask():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from videox.engines.gfpgan.faces import align_face, inverse_affine, paste_faces

    background = np.full((200, 220, 3), 40, np.uint8)
    matrix, _cropped = align_face(background, _synthetic_landmarks(0.25, 40, 30))
    face = np.full((512, 512, 3), 240, np.uint8)
    inverses = [inverse_affine(matrix, 2)]
    kept = paste_faces(background, [face], inverses, input_size=(200, 220), upscale=2, strength=0)
    assert kept.tolist() == np.full((400, 440, 3), 40, np.uint8).tolist()
    half = paste_faces(background, [face], inverses, input_size=(200, 220), upscale=2, strength=0.5)
    assert abs(int(half[200, 230, 0]) - 140) <= 1
    full = paste_faces(background, [face], inverses, input_size=(200, 220), upscale=2)
    assert full[200, 230].min() >= 239


def test_pipeline_runs_with_tiny_networks(tmp_path: Path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from PIL import Image

    from videox.engines.gfpgan import infer
    from videox.engines.gfpgan.arch import GFPGANv1Clean, RetinaFace
    from videox.engines.gfpgan.faces import DetectedFace
    from videox.engines.gfpgan.presets import GFPGANModel, WeightFile

    # A narrow random network stands in for the release checkpoint, so nothing downloads.
    spec = GFPGANModel("测试", "测试", WeightFile("GFPGANv1.4.pth", "https://invalid/"), _tiny_gfpgan_kwargs())
    monkeypatch.setattr(infer, "model_spec", lambda version: spec)
    torch.manual_seed(0)
    torch.save({"params_ema": GFPGANv1Clean(**spec.arch_kwargs).state_dict()}, tmp_path / "GFPGANv1.4.pth")
    (tmp_path / DETECTOR.name).write_bytes(b"x")
    monkeypatch.setattr(infer, "load_detector", lambda path, device: RetinaFace().eval())

    faces = [DetectedFace(np.array([30, 20, 90, 80, 0.99], np.float32), _synthetic_landmarks(0.125, 25, 12))]
    calls: list[dict] = []

    def fake_detect(detector, bgr, **kwargs):
        calls.append(kwargs)
        return faces if not kwargs["only_center_face"] else []

    monkeypatch.setattr(infer, "detect_faces", fake_detect)

    rgb = np.random.default_rng(0).integers(0, 256, size=(90, 120, 3), dtype=np.uint8)
    rgba = np.dstack([rgb, np.tile(np.linspace(0, 255, 120, dtype=np.uint8), (90, 1))])
    source = tmp_path / "in.png"
    Image.fromarray(rgba, "RGBA").save(source)
    out = tmp_path / "out" / "result.png"
    infer.run_gfpgan(
        infer.GFPGANRequest(input_path=source, output_path=out, model_dir=tmp_path, background=NO_BACKGROUND, upscale=2)
    )
    with Image.open(out) as written:
        assert written.mode == "RGBA" and written.size == (240, 180)
        alpha = np.asarray(written)[..., 3]
        assert alpha[:, 0].max() < 20 and alpha[:, -1].min() > 235
    lines = capsys.readouterr().out
    for phase in ("download", "load", "face", "restore"):
        assert f'"phase": "{phase}"' in lines
    assert '"phase": "face", "current": 1, "total": 1' in lines
    assert calls[-1] == {"only_center_face": False, "eye_dist_threshold": 5}

    # No face: the whole image is still upscaled and written.
    plain = tmp_path / "plain.jpg"
    infer.run_gfpgan(
        infer.GFPGANRequest(
            input_path=source, output_path=plain, model_dir=tmp_path, background=NO_BACKGROUND,
            upscale=1, only_center_face=True, strength=0.5,
        )
    )
    with Image.open(plain) as written:
        assert written.mode == "RGB" and written.size == (120, 90)
    assert "没有修复任何人脸" in capsys.readouterr().out

    # Strength 0 keeps the background, so no face network runs.
    calls.clear()
    background_only = tmp_path / "background.png"
    infer.run_gfpgan(
        infer.GFPGANRequest(
            input_path=source, output_path=background_only, model_dir=tmp_path, background=NO_BACKGROUND,
            upscale=1, strength=0,
        )
    )
    assert calls == [] and '"phase": "face"' not in capsys.readouterr().out
    with Image.open(background_only) as written:
        assert np.asarray(written)[..., :3].tolist() == rgb.tolist()

    with pytest.raises(VideoXError, match="不支持的图片格式"):
        infer.run_gfpgan(infer.GFPGANRequest(input_path=source, output_path=tmp_path / "x.gif", model_dir=tmp_path))
    wide = tmp_path / "wide.png"
    Image.fromarray(np.zeros((2, 4100, 3), np.uint8)).save(wide)
    with pytest.raises(VideoXError, match="WEBP 图片每边最多 16383 像素"):
        infer.run_gfpgan(infer.GFPGANRequest(input_path=wide, output_path=tmp_path / "x.webp", model_dir=tmp_path, upscale=4))
    with pytest.raises(VideoXError, match="未知的 Real-ESRGAN 模型"):
        infer.run_gfpgan(infer.GFPGANRequest(input_path=source, output_path=out, model_dir=tmp_path, background="x"))
    assert infer.background_dir(tmp_path / "models" / "GFPGAN") == tmp_path / "models" / "RealESRGAN"


def _released(*names: str) -> bool:
    return all((WEIGHTS / name).is_file() for name in names)


@pytest.mark.parametrize("version", list(MODELS))
def test_released_weights_restore_a_face(version: str, tmp_path: Path, capsys):
    spec = MODELS[version]
    if not _released(*(weight.name for weight in spec.files)):
        pytest.skip("GFPGAN weights were not downloaded to .work/weights/GFPGAN")
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from PIL import Image

    from videox.engines.gfpgan import infer
    from videox.engines.gfpgan.faces import detect_faces

    device = torch.device("cpu")
    restorer = infer.load_restorer(WEIGHTS / spec.weights.name, spec, device)
    detector = infer.load_detector(WEIGHTS / DETECTOR.name, device)
    # From a flat gray crop the StyleGAN prior paints a frontal face; use it as the test photo.
    generated = infer.restore_face(restorer, np.full((512, 512, 3), 100, np.uint8), device, randomize_noise=False)
    assert generated is not None and generated.shape == (512, 512, 3)
    face = np.asarray(Image.fromarray(generated).resize((192, 192), Image.Resampling.LANCZOS))
    photo = np.full((232, 252, 3), 100, np.uint8)
    photo[20:212, 30:222] = face
    bgr = np.ascontiguousarray(photo[..., ::-1])
    with torch.inference_mode():
        found = detect_faces(detector, bgr, only_center_face=False, eye_dist_threshold=5)
        shrunk = detect_faces(detector, bgr, only_center_face=False, eye_dist_threshold=5, max_edge=200)
    assert len(found) == 1 and found[0].box[4] > 0.97
    # Landmarks sit where the template puts them inside the pasted 192 px face.
    from videox.engines.gfpgan.faces import FACE_TEMPLATE

    expected = FACE_TEMPLATE * (192 / 512) + [30, 20]
    assert np.abs(found[0].landmarks - expected).max() < 8
    assert len(shrunk) == 1 and np.abs(shrunk[0].landmarks - found[0].landmarks).max() < 4

    source = tmp_path / "face.png"
    Image.fromarray(np.dstack([photo, np.full(photo.shape[:2], 200, np.uint8)]), "RGBA").save(source)
    out = tmp_path / "restored.png"
    infer.run_gfpgan(
        infer.GFPGANRequest(
            input_path=source, output_path=out, model_dir=WEIGHTS, version=version, background=NO_BACKGROUND
        )
    )
    assert '"phase": "face", "current": 1, "total": 1' in capsys.readouterr().out
    with Image.open(out) as written:
        assert written.size == (504, 464) and written.mode == "RGBA"
        assert np.asarray(written)[..., 3].min() == 200


def test_released_weights_with_realesrgan_background(tmp_path: Path):
    background = WEIGHTS.parent / "RealESRGAN" / "RealESRGAN_x2plus.pth"
    if not _released(*(weight.name for weight in MODELS[DEFAULT_VERSION].files)) or not background.is_file():
        pytest.skip("GFPGAN and Real-ESRGAN weights were not downloaded to .work/weights")
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from PIL import Image

    from videox.engines.gfpgan import infer

    yy, xx = np.mgrid[0:40, 0:56]
    image = np.dstack([xx * 4, yy * 6, (xx + yy) * 2]).astype(np.uint8)
    source = tmp_path / "in.jpg"
    Image.fromarray(image).save(source)
    out = tmp_path / "out.png"
    infer.run_gfpgan(
        infer.GFPGANRequest(input_path=source, output_path=out, model_dir=WEIGHTS, upscale=4, tile=24, tile_pad=4,
                            background="RealESRGAN_x2plus", half=False)
    )
    with Image.open(out) as written:
        assert written.size == (224, 160) and written.mode == "RGB"

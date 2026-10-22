from pathlib import Path

import pytest

from videox.engines.fields import (
    IMAGE,
    choice_notes,
    conditional,
    integer_field,
    number_field,
    resolve_params,
    schema,
    string_field,
)
from videox.errors import VideoXError


def _schema() -> dict:
    return schema(
        [("primary", "模型")],
        {
            "model": choice_notes(
                string_field(
                    "模型",
                    "x4",
                    group="primary",
                    media=IMAGE,
                    choices=[("x4", "4 倍"), ("x2", "2 倍"), ("denoise", "去噪")],
                    description="选择模型。",
                ),
                {"x4": "四倍", "x2": "两倍", "denoise": "只去噪"},
            ),
            "outscale": conditional(
                number_field(
                    "输出倍数",
                    4,
                    group="primary",
                    media=IMAGE,
                    minimum=1,
                    maximum=8,
                    description="输出倍数。",
                ),
                visible_when={"model": ["x4", "x2"]},
                overrides=[({"model": ["x2"]}, {"default": 2, "maximum": 4})],
            ),
            "tile": integer_field("分块", 512, group="primary", media=IMAGE, description="分块。"),
        },
    )


def test_conditional_fields_follow_the_model():
    spec = _schema()
    assert resolve_params(spec, {}, "image") == {"model": "x4", "outscale": 4, "tile": 512}
    assert resolve_params(spec, {"model": "x2"}, "image")["outscale"] == 2
    assert resolve_params(spec, {"model": "x2", "outscale": 3}, "image")["outscale"] == 3.0
    with pytest.raises(VideoXError, match="不能大于 4"):
        resolve_params(spec, {"model": "x2", "outscale": 6}, "image")
    hidden = resolve_params(spec, {"model": "denoise", "outscale": 99}, "image")
    assert "outscale" not in hidden
    assert resolve_params(spec, {}, "video") == {}
    props = spec["properties"]
    assert props["model"]["enum_descriptions"]["x2"] == "两倍"
    assert props["outscale"]["visible_when"] == {"model": ["x4", "x2"]}
    assert props["outscale"]["overrides"] == [{"when": {"model": ["x2"]}, "default": 2, "maximum": 4}]


def test_resolve_params_rejects_bad_values():
    spec = _schema()
    with pytest.raises(VideoXError, match="无法识别"):
        resolve_params(spec, {"model": "x8"}, "image")
    with pytest.raises(VideoXError, match="需要是整数"):
        resolve_params(spec, {"tile": 1.5}, "image")
    with pytest.raises(VideoXError, match="需要是整数"):
        resolve_params(spec, {"tile": True}, "image")
    with pytest.raises(VideoXError, match="不能小于 0"):
        resolve_params(spec, {"tile": -1}, "image")


def test_boolean_conditions_do_not_match_numbers():
    spec = schema(
        [("primary", "模型")],
        {
            "half": {"type": "boolean", "default": True, "media": IMAGE, "title": "半精度"},
            "extra": conditional(
                integer_field("附加", 1, group="primary", media=IMAGE, description="附加。"),
                visible_when={"half": [True]},
            ),
        },
    )
    assert "extra" in resolve_params(spec, {}, "image")
    assert "extra" not in resolve_params(spec, {"half": False}, "image")


def test_tiled_inference_matches_whole_image():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from videox.engines.image_io import run_tiled

    torch.manual_seed(0)
    blur = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)

    def model(x):
        return torch.nn.functional.interpolate(blur(x), scale_factor=2, mode="nearest")

    image = torch.rand(1, 3, 37, 53)
    whole = run_tiled(model, image, scale=2, tile=0, tile_pad=0)
    tiled = run_tiled(model, image, scale=2, tile=16, tile_pad=2)
    assert whole.shape == (74, 106, 3)
    assert (whole == tiled).all()
    padded = run_tiled(lambda x: torch.nn.functional.interpolate(x, scale_factor=4), image, scale=4, tile=16, tile_pad=4, multiple_of=8)
    assert padded.shape == (148, 212, 3)


def test_images_round_trip_with_alpha(tmp_path: Path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image

    from videox.engines.image_io import StillImage, read_image, resize_plane, write_image

    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 128
    source = tmp_path / "in.png"
    Image.fromarray(rgba).save(source)
    image = read_image(source)
    assert image.alpha is not None and image.alpha.shape == (6, 8)
    assert image.rgb.shape == (6, 8, 3)

    opaque = tmp_path / "opaque.png"
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(opaque)
    assert read_image(opaque).alpha is None

    out = tmp_path / "nested" / "out.png"
    write_image(out, StillImage(resize_plane(image.rgb, 16, 12), resize_plane(image.alpha, 16, 12)))
    with Image.open(out) as written:
        assert written.mode == "RGBA" and written.size == (16, 12)
    jpeg = tmp_path / "out.jpg"
    write_image(jpeg, image)
    with Image.open(jpeg) as written:
        assert written.mode == "RGB"

    deep = tmp_path / "deep.png"
    Image.fromarray(np.full((4, 4), 65535, dtype=np.uint16)).save(deep)
    assert int(read_image(deep).rgb.min()) == 255
    with pytest.raises(VideoXError, match="不支持"):
        write_image(tmp_path / "out.gif", image)

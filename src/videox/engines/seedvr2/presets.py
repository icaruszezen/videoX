from __future__ import annotations

DEFAULT_DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"
DEFAULT_RESOLUTION = 1080
DEFAULT_TEMPORAL_OVERLAP = 3
DEFAULT_PREPEND_FRAMES = 4
DEFAULT_COLOR_CORRECTION = "lab"
DEFAULT_SEED = 42
DEFAULT_ATTENTION_MODE = "sdpa"
DEFAULT_BATCH_SIZE_IMAGE = 1

REQUIRED_MODELS = (DEFAULT_DIT_MODEL, DEFAULT_VAE_MODEL)

BOTH = ["video", "image"]
VIDEO = ["video"]

DIT_MODEL_CHOICES = [
    ("seedvr2_ema_3b_fp8_e4m3fn.safetensors", "3B FP8"),
    ("seedvr2_ema_3b_fp16.safetensors", "3B FP16"),
    ("seedvr2_ema_3b-Q8_0.gguf", "3B GGUF Q8"),
    ("seedvr2_ema_3b-Q4_K_M.gguf", "3B GGUF Q4"),
    ("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors", "7B FP8"),
    ("seedvr2_ema_7b_fp16.safetensors", "7B FP16"),
    ("seedvr2_ema_7b-Q4_K_M.gguf", "7B GGUF Q4"),
    ("seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "7B Sharp FP8"),
    ("seedvr2_ema_7b_sharp_fp16.safetensors", "7B Sharp FP16"),
    ("seedvr2_ema_7b_sharp-Q4_K_M.gguf", "7B Sharp GGUF Q4"),
]
COLOR_CORRECTION_CHOICES = [
    ("lab", "LAB"),
    ("wavelet", "小波"),
    ("wavelet_adaptive", "自适应小波"),
    ("hsv", "HSV"),
    ("adain", "AdaIN"),
    ("none", "关闭"),
]
ATTENTION_CHOICES = [
    ("sdpa", "PyTorch SDPA"),
    ("flash_attn_2", "Flash Attention 2"),
    ("flash_attn_3", "Flash Attention 3"),
    ("sageattn_2", "SageAttention 2"),
    ("sageattn_3", "SageAttention 3"),
]
COMPILE_BACKEND_CHOICES = [
    ("inductor", "Inductor"),
    ("cudagraphs", "CUDA Graphs"),
]
COMPILE_MODE_CHOICES = [
    ("default", "default"),
    ("reduce-overhead", "reduce-overhead"),
    ("max-autotune", "max-autotune"),
    ("max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs"),
]
OFFLOAD_CHOICES = [
    ("none", "不卸载"),
    ("cpu", "CPU"),
]
OUTPUT_FORMAT_CHOICES = [
    ("mp4", "MP4"),
    ("png", "PNG 序列"),
]


def max_blocks_for_model(name: str) -> int:
    if "7b" in name.lower():
        return 36
    return 32


def _labels(choices: list[tuple[str, str]]) -> dict[str, str]:
    return {value: label for value, label in choices}


def _values(choices: list[tuple[str, str]]) -> list[str]:
    return [value for value, _ in choices]


def _string(
    title: str,
    default: str,
    *,
    group: str,
    media: list[str],
    description: str = "",
    choices: list[tuple[str, str]] | None = None,
) -> dict:
    field: dict = {
        "type": "string",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
    }
    if description:
        field["description"] = description
    if choices is not None:
        field["enum"] = _values(choices)
        field["enum_labels"] = _labels(choices)
    return field


def _integer(
    title: str,
    *,
    group: str,
    media: list[str],
    default: int | None = None,
    minimum: int = 0,
    maximum: int | None = None,
    description: str = "",
) -> dict:
    field: dict = {
        "type": "integer",
        "title": title,
        "group": group,
        "media": media,
        "minimum": minimum,
    }
    if default is not None:
        field["default"] = default
    if maximum is not None:
        field["maximum"] = maximum
    if description:
        field["description"] = description
    return field


def _number(
    title: str,
    default: float,
    *,
    group: str,
    media: list[str],
    minimum: float = 0,
    maximum: float = 1,
    description: str = "",
) -> dict:
    field: dict = {
        "type": "number",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
        "minimum": minimum,
        "maximum": maximum,
    }
    if description:
        field["description"] = description
    return field


def _boolean(
    title: str,
    default: bool,
    *,
    group: str,
    media: list[str],
    description: str = "",
) -> dict:
    field: dict = {
        "type": "boolean",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
    }
    if description:
        field["description"] = description
    return field


def params_schema() -> dict:
    return {
        "type": "object",
        "groups": [
            {"id": "primary", "title": "模型与分辨率"},
            {"id": "quality", "title": "画质"},
            {"id": "memory", "title": "显存"},
            {"id": "performance", "title": "性能"},
            {"id": "output", "title": "输出"},
        ],
        "properties": {
            "dit_model": _string(
                "扩散模型",
                DEFAULT_DIT_MODEL,
                group="primary",
                media=BOTH,
                choices=DIT_MODEL_CHOICES,
                description="决定修复用的扩散网络。3B 更省显存、速度更快，细节略少；7B 细节更多，更吃显存也更慢。Sharp 会让边缘更锐，过锐时换回普通 7B。FP8 和 GGUF 更省显存，FP16 质量更稳但更占显存。首次选用会下载对应权重。",
            ),
            "resolution": _integer(
                "短边分辨率",
                group="primary",
                media=BOTH,
                default=DEFAULT_RESOLUTION,
                minimum=1,
                description="输出画面短边的像素，宽高比保持不变。调高更清晰，同时更慢、更吃显存；调低更快，细节更少。1080 适合大多数成片。",
            ),
            "max_resolution": _integer(
                "长边上限",
                group="primary",
                media=BOTH,
                default=0,
                minimum=0,
                description="限制长边不超过该像素。调低会先缩小再处理，用来避免超大画面爆显存；调高允许更大的画面。0 表示不限制。",
            ),
            "batch_size": _integer(
                "批大小",
                group="primary",
                media=VIDEO,
                minimum=1,
                description="一次送进模型的帧数，必须是 4 的倍数加 1，例如 1、5、9、13。调大更快，但显存成倍上升，不够会失败；调小更稳、更慢。框里显示的是按显存算出的推荐值：约 12GB 用 5，约 16GB 及以上用 9。没改过时仍按显存自动选择，清空则恢复自动。",
            ),
            "color_correction": _string(
                "色彩校正",
                DEFAULT_COLOR_CORRECTION,
                group="quality",
                media=BOTH,
                choices=COLOR_CORRECTION_CHOICES,
                description="把修复后的颜色拉回原片。LAB 最接近原色，优先用它。小波和自适应小波保留更多局部反差；HSV、AdaIN 改变更大。选关闭会让模型自己决定颜色，可能偏色。",
            ),
            "input_noise_scale": _number(
                "输入噪声",
                0,
                group="quality",
                media=BOTH,
                description="在输入上加少量噪声，范围 0 到 1。保持 0 最干净。高分辨率出现麻点或块状噪点时略微提高，例如 0.05 到 0.1，可以压噪；提太高会糊掉细节。",
            ),
            "latent_noise_scale": _number(
                "潜空间噪声",
                0,
                group="quality",
                media=BOTH,
                description="在潜空间加噪声，范围 0 到 1。保持 0 时细节最硬。纹理发死或有塑料感时略微提高会更柔和；提太高会发虚，失去锐度。",
            ),
            "seed": _integer(
                "随机种子",
                group="quality",
                media=BOTH,
                default=DEFAULT_SEED,
                description="控制随机性。相同种子和相同输入会得到相同结果，方便复现。换一个数字只会改变纹理细节，画质档次不变。",
            ),
            "vae_encode_tiled": _boolean(
                "VAE 编码分块",
                True,
                group="memory",
                media=BOTH,
                description="编码时把画面切块，降低显存。打开后更不容易爆显存，速度略慢，块太小可能出现接缝。显存充足、想略快一点时可以关掉。",
            ),
            "vae_encode_tile_size": _integer(
                "编码块大小",
                group="memory",
                media=BOTH,
                default=1024,
                minimum=1,
                description="编码每一块的边长，只在编码分块打开时生效。调大更快、接缝更少，但更吃显存；调小更省显存，接缝风险更高。默认 1024。",
            ),
            "vae_encode_tile_overlap": _integer(
                "编码块重叠",
                group="memory",
                media=BOTH,
                default=128,
                minimum=0,
                description="相邻编码块重叠的像素，用来减轻接缝。调大接缝更轻，略慢、略占显存；调小更快，接缝更明显。默认 128。",
            ),
            "vae_decode_tiled": _boolean(
                "VAE 解码分块",
                True,
                group="memory",
                media=BOTH,
                description="解码时把画面切块，降低显存。打开后更不容易爆显存，速度略慢。显存充足、想略快一点时可以关掉。",
            ),
            "vae_decode_tile_size": _integer(
                "解码块大小",
                group="memory",
                media=BOTH,
                default=1024,
                minimum=1,
                description="解码每一块的边长，只在解码分块打开时生效。调大更快、接缝更少，但更吃显存；调小更省显存，接缝风险更高。默认 1024。",
            ),
            "vae_decode_tile_overlap": _integer(
                "解码块重叠",
                group="memory",
                media=BOTH,
                default=128,
                minimum=0,
                description="相邻解码块重叠的像素，用来减轻接缝。调大接缝更轻，略慢、略占显存；调小更快，接缝更明显。默认 128。",
            ),
            "blocks_to_swap": _integer(
                "换出块数",
                group="memory",
                media=BOTH,
                default=0,
                minimum=0,
                description="把若干 Transformer 块放到内存。调大更省显存，速度明显变慢；0 表示整网留在显卡上，最快。3B 最多 32，7B 最多 36。大于 0 时会自动改用 CPU 卸载。",
            ),
            "swap_io_components": _boolean(
                "交换输入输出组件",
                False,
                group="memory",
                media=BOTH,
                description="连同输入输出组件一起换出。打开后再省一点显存，速度更慢。显存已经够用时保持关闭。",
            ),
            "dit_offload_device": _string(
                "DiT 卸载设备",
                "none",
                group="memory",
                media=BOTH,
                choices=OFFLOAD_CHOICES,
                description="DiT 空闲时放在哪。不卸载最快，模型一直占着显存。选 CPU 更省显存，切换时更慢。换出块数大于 0 时，即使这里是不卸载也会改用 CPU。",
            ),
            "vae_offload_device": _string(
                "VAE 卸载设备",
                "none",
                group="memory",
                media=BOTH,
                choices=OFFLOAD_CHOICES,
                description="VAE 空闲时放在哪。不卸载更快，但一直占着显存。选 CPU 会在编码和解码的间隙腾出显存，切换时慢一截。显存紧张再改到 CPU。",
            ),
            "tensor_offload_device": _string(
                "张量卸载设备",
                "cpu",
                group="memory",
                media=BOTH,
                choices=OFFLOAD_CHOICES,
                description="中间张量在计算间隙放到哪。CPU 更省显存，是稳妥默认；不卸载更快，但更容易占满显存。",
            ),
            "allow_vram_overflow": _boolean(
                "允许显存溢出",
                False,
                group="memory",
                media=BOTH,
                description="显存不够时借用系统内存继续跑，避免直接失败。打开后速度会明显变慢。只在反复爆显存时打开。",
            ),
            "attention_mode": _string(
                "注意力模式",
                DEFAULT_ATTENTION_MODE,
                group="performance",
                media=BOTH,
                choices=ATTENTION_CHOICES,
                description="注意力的计算实现。SDPA 兼容性最好，始终可用。Flash Attention 和 SageAttention 在支持的显卡上更快、更省显存；机器没有对应依赖时会失败，改回 SDPA。",
            ),
            "compile_dit": _boolean(
                "编译 DiT",
                False,
                group="performance",
                media=BOTH,
                description="用 torch.compile 编译扩散网络。打开后稳定阶段更快，第一次编译要多等很久，换分辨率时可能重新编译。日常建议关闭。",
            ),
            "compile_vae": _boolean(
                "编译 VAE",
                False,
                group="performance",
                media=BOTH,
                description="用 torch.compile 编译 VAE。打开后编码和解码更快，第一次编译要多等很久。和编译 DiT 互不影响。",
            ),
            "compile_backend": _string(
                "编译后端",
                "inductor",
                group="performance",
                media=BOTH,
                choices=COMPILE_BACKEND_CHOICES,
                description="编译使用的后端，只在打开编译后生效。Inductor 更通用。CUDA Graphs 跑起来开销更低，但输入形状一变就更挑剔。",
            ),
            "compile_mode": _string(
                "编译模式",
                "default",
                group="performance",
                media=BOTH,
                choices=COMPILE_MODE_CHOICES,
                description="编译的优化程度，只在打开编译后生效。default 最稳。reduce-overhead 降低调度开销。max-autotune 最激进，编译最久，跑起来可能更快。",
            ),
            "compile_fullgraph": _boolean(
                "整图编译",
                False,
                group="performance",
                media=BOTH,
                description="把整个模型编成一张图。打开后更快，但输入形状一变就更容易编译失败。尺寸固定时可以试。",
            ),
            "compile_dynamic": _boolean(
                "动态形状",
                False,
                group="performance",
                media=BOTH,
                description="允许动态形状。输入分辨率经常变化时打开，可以复用已编译的图，少重新编译。尺寸固定时保持关闭更稳。",
            ),
            "temporal_overlap": _integer(
                "时间重叠",
                group="output",
                media=VIDEO,
                default=DEFAULT_TEMPORAL_OVERLAP,
                minimum=0,
                description="相邻批次以及双卡交界重叠的帧数，用来衔接画面。调高衔接更顺、闪烁更少，但重复计算更多、更慢；调低更快，接缝或闪烁更明显。默认 3。",
            ),
            "prepend_frames": _integer(
                "前置帧",
                group="output",
                media=VIDEO,
                default=DEFAULT_PREPEND_FRAMES,
                minimum=0,
                description="在开头补若干反向帧，处理完再丢掉，用来减轻第一帧附近的闪烁。这些帧不会出现在成片里。调高开头更稳，但多算几帧；0 最快，开头更容易闪。默认 4。",
            ),
            "uniform_batch_size": _boolean(
                "统一批大小",
                True,
                group="output",
                media=VIDEO,
                description="最后一批帧数不足时补齐到相同批大小。打开后时序更稳，末尾会多算一点；关掉可能稍快，但最后一段更容易闪或不一致。",
            ),
            "chunk_size": _integer(
                "分块大小",
                group="output",
                media=VIDEO,
                default=0,
                minimum=0,
                description="长视频按多少帧一段读入。0 表示整段读入，最简单。调大则分段更少、占用更多内存；调小更省内存，段与段之间多一次衔接。很长的视频可以改成几百帧。",
            ),
            "skip_first_frames": _integer(
                "跳过开头",
                group="output",
                media=VIDEO,
                default=0,
                minimum=0,
                description="从第几帧之后开始处理，用来跳过片头。调大会丢掉更多开头画面；0 表示从第一帧开始。",
            ),
            "load_cap": _integer(
                "最多处理帧数",
                group="output",
                media=VIDEO,
                default=0,
                minimum=0,
                description="最多处理多少帧，用来只出一段试效果。调小更快，成片更短；0 表示处理跳过之后剩下的全部帧。",
            ),
            "output_format": _string(
                "输出格式",
                "mp4",
                group="output",
                media=VIDEO,
                choices=OUTPUT_FORMAT_CHOICES,
                description="MP4 会封装成视频，并尽量保留原片音轨。PNG 序列把每一帧写成图片，没有音轨，方便后期，体积也更大。",
            ),
            "use_10bit": _boolean(
                "10-bit 输出",
                False,
                group="output",
                media=VIDEO,
                description="用 x265 写出 10-bit MP4，减轻渐变上的色带。打开后编码更慢，文件对播放器的要求更高。只对 MP4 有效。",
            ),
        },
    }

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
                description="3B 更省显存，7B 细节更多。Sharp 会加强锐度。首次使用时下载对应权重。",
            ),
            "resolution": _integer(
                "短边分辨率",
                group="primary",
                media=BOTH,
                default=DEFAULT_RESOLUTION,
                minimum=1,
                description="按短边放大，并保持原来的宽高比。",
            ),
            "max_resolution": _integer(
                "长边上限",
                group="primary",
                media=BOTH,
                default=0,
                minimum=0,
                description="任一长边超过该值时缩小。0 表示不限制。",
            ),
            "batch_size": _integer(
                "批大小",
                group="primary",
                media=VIDEO,
                minimum=1,
                description="必须是 4n+1。留空时按显存选择：约 12GB 用 5，约 16GB 及以上用 9。",
            ),
            "color_correction": _string(
                "色彩校正",
                DEFAULT_COLOR_CORRECTION,
                group="quality",
                media=BOTH,
                choices=COLOR_CORRECTION_CHOICES,
                description="把修复后的颜色拉回原图。LAB 最接近原片。",
            ),
            "input_noise_scale": _number(
                "输入噪声",
                0,
                group="quality",
                media=BOTH,
                description="高分辨率出现噪点时可以略微提高。",
            ),
            "latent_noise_scale": _number(
                "潜空间噪声",
                0,
                group="quality",
                media=BOTH,
                description="细节过硬时可以提高，用来让纹理柔和一些。",
            ),
            "seed": _integer(
                "随机种子",
                group="quality",
                media=BOTH,
                default=DEFAULT_SEED,
                description="相同种子和相同输入会得到相同结果。",
            ),
            "vae_encode_tiled": _boolean(
                "VAE 编码分块",
                True,
                group="memory",
                media=BOTH,
                description="编码时分块，降低显存占用。",
            ),
            "vae_encode_tile_size": _integer(
                "编码块大小",
                group="memory",
                media=BOTH,
                default=1024,
                minimum=1,
                description="编码分块打开时生效。",
            ),
            "vae_encode_tile_overlap": _integer(
                "编码块重叠",
                group="memory",
                media=BOTH,
                default=128,
                minimum=0,
                description="相邻编码块重叠的像素，用来减轻接缝。",
            ),
            "vae_decode_tiled": _boolean(
                "VAE 解码分块",
                True,
                group="memory",
                media=BOTH,
                description="解码时分块，降低显存占用。",
            ),
            "vae_decode_tile_size": _integer(
                "解码块大小",
                group="memory",
                media=BOTH,
                default=1024,
                minimum=1,
                description="解码分块打开时生效。",
            ),
            "vae_decode_tile_overlap": _integer(
                "解码块重叠",
                group="memory",
                media=BOTH,
                default=128,
                minimum=0,
                description="相邻解码块重叠的像素，用来减轻接缝。",
            ),
            "blocks_to_swap": _integer(
                "换出块数",
                group="memory",
                media=BOTH,
                default=0,
                minimum=0,
                description="把若干 Transformer 块放到 CPU。3B 最多 32，7B 最多 36。大于 0 时默认卸载到 CPU。",
            ),
            "swap_io_components": _boolean(
                "交换输入输出组件",
                False,
                group="memory",
                media=BOTH,
                description="连同输入输出组件一起换出，进一步省显存。",
            ),
            "dit_offload_device": _string(
                "DiT 卸载设备",
                "none",
                group="memory",
                media=BOTH,
                choices=OFFLOAD_CHOICES,
                description="不卸载时模型留在显卡上。换出块数大于 0 且这里仍是不卸载时，会改用 CPU。",
            ),
            "vae_offload_device": _string(
                "VAE 卸载设备",
                "none",
                group="memory",
                media=BOTH,
                choices=OFFLOAD_CHOICES,
            ),
            "tensor_offload_device": _string(
                "张量卸载设备",
                "cpu",
                group="memory",
                media=BOTH,
                choices=OFFLOAD_CHOICES,
                description="中间张量在计算间隙放到这里。",
            ),
            "allow_vram_overflow": _boolean(
                "允许显存溢出",
                False,
                group="memory",
                media=BOTH,
                description="显存不够时使用系统内存，避免直接失败，速度会明显变慢。",
            ),
            "attention_mode": _string(
                "注意力模式",
                DEFAULT_ATTENTION_MODE,
                group="performance",
                media=BOTH,
                choices=ATTENTION_CHOICES,
                description="SDPA 始终可用。其余后端需要对应的显卡和依赖。",
            ),
            "compile_dit": _boolean(
                "编译 DiT",
                False,
                group="performance",
                media=BOTH,
                description="用 torch.compile 加速 DiT。首次编译会更慢。",
            ),
            "compile_vae": _boolean(
                "编译 VAE",
                False,
                group="performance",
                media=BOTH,
                description="用 torch.compile 加速 VAE。首次编译会更慢。",
            ),
            "compile_backend": _string(
                "编译后端",
                "inductor",
                group="performance",
                media=BOTH,
                choices=COMPILE_BACKEND_CHOICES,
                description="打开编译后生效。",
            ),
            "compile_mode": _string(
                "编译模式",
                "default",
                group="performance",
                media=BOTH,
                choices=COMPILE_MODE_CHOICES,
                description="打开编译后生效。",
            ),
            "compile_fullgraph": _boolean(
                "整图编译",
                False,
                group="performance",
                media=BOTH,
                description="把整个模型编成一张图。更快，输入形状变化时更严格。",
            ),
            "compile_dynamic": _boolean(
                "动态形状",
                False,
                group="performance",
                media=BOTH,
                description="输入尺寸变化时尽量复用已编译的图。",
            ),
            "temporal_overlap": _integer(
                "时间重叠",
                group="output",
                media=VIDEO,
                default=DEFAULT_TEMPORAL_OVERLAP,
                minimum=0,
                description="相邻批次和多卡交界重叠的帧数，用来衔接画面。",
            ),
            "prepend_frames": _integer(
                "前置帧",
                group="output",
                media=VIDEO,
                default=DEFAULT_PREPEND_FRAMES,
                minimum=0,
                description="在开头补若干反向帧，减轻起始处的闪烁，结果里会去掉。",
            ),
            "uniform_batch_size": _boolean(
                "统一批大小",
                True,
                group="output",
                media=VIDEO,
                description="最后一批不足时补齐到相同批大小。",
            ),
            "chunk_size": _integer(
                "分块大小",
                group="output",
                media=VIDEO,
                default=0,
                minimum=0,
                description="0 表示整段读入。大于 0 时按帧分段处理长视频。",
            ),
            "skip_first_frames": _integer(
                "跳过开头",
                group="output",
                media=VIDEO,
                default=0,
                minimum=0,
                description="从第几帧之后开始处理。",
            ),
            "load_cap": _integer(
                "最多处理帧数",
                group="output",
                media=VIDEO,
                default=0,
                minimum=0,
                description="0 表示处理跳过之后剩下的全部帧。",
            ),
            "output_format": _string(
                "输出格式",
                "mp4",
                group="output",
                media=VIDEO,
                choices=OUTPUT_FORMAT_CHOICES,
                description="PNG 序列会写到一个目录，不再封装音轨。",
            ),
            "use_10bit": _boolean(
                "10-bit 输出",
                False,
                group="output",
                media=VIDEO,
                description="用 x265 写出 10-bit MP4，减轻渐变色带。",
            ),
        },
    }

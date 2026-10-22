from __future__ import annotations

from videox.engines.fields import integer_field, schema

DEFAULT_MAX_SEQ_LEN = 16
CHECKPOINT_NAME = "RealBasicVSR_x4.pth"
CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmediting/restorers/real_basicvsr/"
    "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth"
)


def params_schema() -> dict:
    return schema(
        [("primary", "模型")],
        {
            "max_seq_len": integer_field(
                "一次处理帧数",
                DEFAULT_MAX_SEQ_LEN,
                group="primary",
                minimum=1,
                maximum=64,
                description="长视频按这个帧数分段送进网络。调小更省显存，段与段之间的衔接会弱一点；调大更连贯，也更吃显存。固定 4 倍放大。默认 16。",
            ),
        },
    )

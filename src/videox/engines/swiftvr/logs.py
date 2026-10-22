from videox.engines.base import LogInsight
from videox.engines.progress import parse_engine_line


def parse_swiftvr_line(line: str) -> LogInsight | None:
    return parse_engine_line(line)

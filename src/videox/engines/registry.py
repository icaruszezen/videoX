from __future__ import annotations

from videox.engines.base import RestorationEngine
from videox.errors import VideoXError
from videox.jobs.models import EngineManifest


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, RestorationEngine] = {}

    def register(self, engine: RestorationEngine) -> None:
        self._engines[engine.id] = engine

    def get(self, engine_id: str) -> RestorationEngine:
        try:
            return self._engines[engine_id]
        except KeyError:
            raise VideoXError(f"未知引擎: {engine_id}") from None

    def manifests(self) -> list[EngineManifest]:
        return [engine.manifest() for engine in self._engines.values()]


def default_registry() -> EngineRegistry:
    from videox.engines.flashvsr.engine import FlashVSREngine
    from videox.engines.gfpgan.engine import GFPGANEngine
    from videox.engines.realbasicvsr.engine import RealBasicVSREngine
    from videox.engines.realesrgan.engine import RealESRGANEngine
    from videox.engines.scunet.engine import SCUNetEngine
    from videox.engines.seedvr2.engine import SeedVR2Engine
    from videox.engines.swiftvr.engine import SwiftVREngine
    from videox.engines.swinir.engine import SwinIREngine

    registry = EngineRegistry()
    registry.register(SeedVR2Engine())
    registry.register(FlashVSREngine())
    registry.register(SwiftVREngine())
    registry.register(RealBasicVSREngine())
    registry.register(RealESRGANEngine())
    registry.register(SwinIREngine())
    registry.register(SCUNetEngine())
    registry.register(GFPGANEngine())
    return registry

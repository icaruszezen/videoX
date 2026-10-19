from __future__ import annotations

import asyncio
import queue

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from videox import __version__
from videox.doctor import collect_report
from videox.engines.registry import default_registry
from videox.errors import VideoXError
from videox.jobs.models import EventType, JobRecord, JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore
from videox.paths import ensure_tool_path, project_root


class CreateJobBody(BaseModel):
    engine_id: str = "seedvr2"
    input_path: str
    output_path: str
    devices: list[int] | None = None
    params: dict = Field(default_factory=dict)

    def to_spec(self) -> JobSpec:
        return JobSpec(
            engine_id=self.engine_id,
            input_path=self.input_path,
            output_path=self.output_path,
            devices=self.devices,
            params=self.params,
        )


def create_app(runner: JobRunner | None = None) -> FastAPI:
    ensure_tool_path()
    if runner is None:
        root = project_root()
        runner = JobRunner(default_registry(), JobStore(), root)
    app = FastAPI(title="VideoX", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runner = runner

    @app.get("/engines")
    def engines() -> list[dict]:
        return [manifest.model_dump() for manifest in runner.registry.manifests()]

    @app.get("/system")
    def system() -> dict:
        return collect_report(runner.project_root).model_dump()

    @app.post("/jobs", status_code=202, response_model=JobRecord)
    def create_job(body: CreateJobBody) -> JobRecord:
        try:
            return runner.submit(body.to_spec(), background=True)
        except VideoXError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        record = runner.store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return record

    @app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        try:
            return runner.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        if runner.store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        events = runner.store.subscribe(job_id)

        async def stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(events.get, True, 15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    payload = event.model_dump_json()
                    yield f"event: {event.type.value}\ndata: {payload}\n\n"
                    if event.type in {
                        EventType.completed,
                        EventType.failed,
                        EventType.cancelled,
                    }:
                        break
            finally:
                runner.store.unsubscribe(job_id, events)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)

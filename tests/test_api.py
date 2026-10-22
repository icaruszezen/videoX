from fastapi.testclient import TestClient

from videox.engines.registry import default_registry
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore
from videox.server import create_app


def test_engines_and_missing_input(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setenv("VIDEOX_ROOT", str(root))
    runner = JobRunner(default_registry(), JobStore(), root)
    client = TestClient(create_app(runner))

    engines = client.get("/engines")
    assert engines.status_code == 200
    body = engines.json()
    assert [item["id"] for item in body] == ["seedvr2", "flashvsr", "swiftvr", "realbasicvsr"]
    assert "upscale" in body[0]["capabilities"]
    assert body[0]["media"] == ["video", "image"]
    assert body[0]["device_policy"] == "multi"
    assert "dit_model" in body[0]["params_schema"]["properties"]
    assert body[0]["params_schema"]["properties"]["batch_size"]["media"] == ["video"]
    flash = body[1]
    assert flash["media"] == ["video"]
    assert flash["device_policy"] == "single"
    assert flash["params_schema"]["properties"]["mode"]["default"] == "tiny-long"
    assert "dit_model" not in flash["params_schema"]["properties"]

    created = client.post(
        "/jobs",
        json={"input_path": str(tmp_path / "missing.mp4"), "output_path": str(tmp_path / "out.mp4")},
    )
    assert created.status_code == 400

    unknown = client.post(
        "/jobs",
        json={
            "engine_id": "missing",
            "input_path": str(tmp_path / "missing.mp4"),
            "output_path": str(tmp_path / "out.mp4"),
        },
    )
    assert unknown.status_code == 400

    system = client.get("/system")
    assert system.status_code == 200
    payload = system.json()
    assert payload["project_root"] == str(root)
    assert [item["id"] for item in payload["engines"]] == [
        "seedvr2",
        "flashvsr",
        "swiftvr",
        "realbasicvsr",
    ]

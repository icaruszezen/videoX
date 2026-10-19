from videox.jobs.models import JobSpec


def test_job_spec_round_trip():
    spec = JobSpec(
        input_path="a.mp4",
        output_path="b.mp4",
        devices=[0, 1],
        params={"resolution": 720},
    )
    restored = JobSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert restored.engine_id == "seedvr2"

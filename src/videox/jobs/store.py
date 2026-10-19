from __future__ import annotations

import queue
import threading
import uuid
from datetime import datetime, timezone

from videox.jobs.models import (
    TERMINAL_STATUSES,
    EventType,
    JobRecord,
    JobSpec,
    JobStatus,
    ProgressEvent,
)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._events: dict[str, list[ProgressEvent]] = {}
        self._subscribers: dict[str, list[queue.Queue[ProgressEvent]]] = {}

    def create(self, spec: JobSpec) -> JobRecord:
        record = JobRecord(
            id=uuid.uuid4().hex[:12],
            spec=spec,
            status=JobStatus.queued,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._jobs[record.id] = record
            self._events[record.id] = []
            self._subscribers[record.id] = []
        return record.model_copy(deep=True)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.model_copy(deep=True) if record else None

    def update(self, job_id: str, **changes: object) -> bool:
        with self._lock:
            record = self._jobs[job_id]
            if record.status in TERMINAL_STATUSES:
                return False
            for key, value in changes.items():
                setattr(record, key, value)
            return True

    def finish(self, job_id: str, status: JobStatus, message: str) -> bool:
        with self._lock:
            record = self._jobs[job_id]
            if record.status in TERMINAL_STATUSES:
                return False
            record.status = status
            record.error = message if status == JobStatus.failed else None
            if status == JobStatus.completed:
                record.progress = 1.0
                record.stage = "done"
        event_type = EventType(status.value)
        self.publish(ProgressEvent(job_id=job_id, type=event_type, message=message, progress=1.0 if status == JobStatus.completed else None))
        return True

    def publish(self, event: ProgressEvent) -> None:
        with self._lock:
            record = self._jobs.get(event.job_id)
            if record is not None and record.status not in TERMINAL_STATUSES:
                if event.progress is not None:
                    record.progress = event.progress
                if event.stage:
                    record.stage = event.stage
            history = self._events.setdefault(event.job_id, [])
            history.append(event)
            if len(history) > 2000:
                del history[: len(history) - 2000]
            subscribers = list(self._subscribers.get(event.job_id, []))
        for subscriber in subscribers:
            subscriber.put(event)

    def subscribe(self, job_id: str) -> queue.Queue[ProgressEvent]:
        events: queue.Queue[ProgressEvent] = queue.Queue()
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            for event in self._events.get(job_id, []):
                events.put(event)
            status = self._jobs[job_id].status
            if status not in TERMINAL_STATUSES:
                self._subscribers[job_id].append(events)
        return events

    def unsubscribe(self, job_id: str, events: queue.Queue[ProgressEvent]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job_id, [])
            if events in subscribers:
                subscribers.remove(events)

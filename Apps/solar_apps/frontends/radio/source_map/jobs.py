"""Cancellable subprocess jobs and in-memory artifact registration."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.processes import (
    miniforge_subprocess_environment,
    python_module_command,
)

from .artifacts import validate_source_map_artifact
from .service import PathPolicy


class ArtifactRegistry:
    def __init__(self, policy: PathPolicy) -> None:
        self.policy = policy
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        image_path: str | Path,
        sidecar_path: str | Path,
        *,
        roi_set: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image = self.policy.resolve(image_path, must_exist=True, kind="file")
        sidecar = self.policy.resolve(sidecar_path, must_exist=True, kind="file")
        metadata = validate_source_map_artifact(image, sidecar)
        artifact_id = uuid.uuid4().hex
        record = {
            "id": artifact_id,
            "image_path": image,
            "sidecar_path": sidecar,
            "metadata": metadata,
            "roi_set": roi_set,
        }
        with self._lock:
            self._artifacts[artifact_id] = record
        return record

    def get(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._artifacts.get(str(artifact_id))
        if record is None:
            raise KeyError("Artifact not found or expired")
        return record


class JobRegistry:
    def __init__(self, *, policy: PathPolicy, artifacts: ArtifactRegistry) -> None:
        self.policy = policy
        self.artifacts = artifacts
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(
        self, config: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        return self._start_job(
            payload={"config": config, "candidate": candidate},
            kind="single",
            candidate_ids=[str(candidate["id"])],
        )

    def start_sequence(
        self, config: dict[str, Any], candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("A sequence render requires at least one candidate")
        frozen: list[dict[str, Any]] = []
        for fallback_sequence, candidate in enumerate(candidates, start=1):
            item = dict(candidate)
            item["sequence"] = int(item.get("sequence") or fallback_sequence)
            frozen.append(item)
        return self._start_job(
            payload={"config": config, "candidates": frozen},
            kind="sequence",
            candidate_ids=[str(candidate["id"]) for candidate in frozen],
        )

    def _start_job(
        self,
        *,
        payload: dict[str, Any],
        kind: str,
        candidate_ids: list[str],
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        workspace = tempfile.TemporaryDirectory(prefix="source-map-job-")
        root = Path(workspace.name)
        job_file = root / "job.json"
        result_file = root / "result.json"
        progress_file = root / "progress.json"
        job_file.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        command = python_module_command(
            "solar_apps.frontends.radio.source_map.worker",
            [
                "--job-file",
                str(job_file),
                "--result-file",
                str(result_file),
                "--progress-file",
                str(progress_file),
            ],
        )
        process = subprocess.Popen(
            command,
            cwd=RuntimeLayout.discover().repo_root,
            env=_worker_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        record = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "candidate_id": candidate_ids[0],
            "candidate_ids": candidate_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "process": process,
            "workspace": workspace,
            "result_file": result_file,
            "progress_file": progress_file,
            "command": command,
            "stdout": "",
            "stderr": "",
            "artifact_id": None,
            "artifact_ids": [],
            "total": len(candidate_ids),
            "completed": 0,
            "current_frame": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = record
        threading.Thread(target=self._wait, args=(job_id,), daemon=True).start()
        return self.public(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._record(job_id)
        process: subprocess.Popen = record["process"]
        with self._lock:
            if record["status"] not in {"running", "canceling"}:
                return self._public_record(record)
            record["status"] = "canceling"
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        with self._lock:
            if record["status"] == "canceling":
                record["status"] = "canceled"
        return self._public_record(record)

    def public(self, job_id: str) -> dict[str, Any]:
        return self._public_record(self._record(job_id))

    def stop_all(self) -> None:
        with self._lock:
            ids = [
                job_id
                for job_id, item in self._jobs.items()
                if item["status"] == "running"
            ]
        for job_id in ids:
            self.cancel(job_id)

    def _wait(self, job_id: str) -> None:
        record = self._record(job_id)
        process: subprocess.Popen = record["process"]
        stdout, stderr = process.communicate()
        with self._lock:
            record["stdout"] = stdout[-12000:]
            record["stderr"] = stderr[-12000:]
            if record["status"] in {"canceling", "canceled"}:
                record["status"] = "canceled"
                return
        try:
            if not record["result_file"].is_file():
                raise RuntimeError(
                    stderr.strip()
                    or f"Render worker exited with code {process.returncode}"
                )
            result = json.loads(record["result_file"].read_text(encoding="utf-8"))
            if process.returncode != 0 or not result.get("ok"):
                raise RuntimeError(
                    result.get("error") or stderr.strip() or "Render failed"
                )
            rendered = result.get("artifacts")
            if rendered is None:
                rendered = [result]
            artifact_ids: list[str] = []
            for item in rendered:
                artifact = self.artifacts.register(
                    item["image_path"], item["sidecar_path"]
                )
                artifact["source_index"] = int(
                    item.get("sequence") or len(artifact_ids) + 1
                )
                artifact["candidate_id"] = str(item.get("candidate_id") or "")
                artifact_ids.append(artifact["id"])
            with self._lock:
                record["artifact_ids"] = artifact_ids
                record["artifact_id"] = artifact_ids[0] if artifact_ids else None
                record["completed"] = len(artifact_ids)
                record["current_frame"] = len(artifact_ids) or None
                record["status"] = "completed"
        except Exception as exc:
            with self._lock:
                record["status"] = "failed"
                record["error"] = str(exc)

    def _record(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(str(job_id))
        if record is None:
            raise KeyError("Render job not found or expired")
        return record

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        progress: dict[str, Any] = {}
        progress_file = record.get("progress_file")
        if record["status"] in {"running", "canceling"} and progress_file:
            try:
                progress = json.loads(Path(progress_file).read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                progress = {}
        current_frame = progress.get("current_frame", record.get("current_frame"))
        artifact_ids = list(record.get("artifact_ids", []))
        return {
            "id": record["id"],
            "kind": record.get("kind", "single"),
            "status": record["status"],
            "candidate_id": record["candidate_id"],
            "candidate_ids": list(record.get("candidate_ids", [])),
            "created_at": record["created_at"],
            "artifact_id": record["artifact_id"],
            "artifact_ids": artifact_ids,
            "artifacts": artifact_ids,
            "total": int(progress.get("total", record.get("total", 1))),
            "completed": int(progress.get("completed", record.get("completed", 0))),
            "current_frame": current_frame,
            "current_index": current_frame,
            "error": record["error"],
            "warnings": [],
            "stdout": record["stdout"],
            "stderr": record["stderr"],
        }


def _worker_environment() -> dict[str, str]:
    return miniforge_subprocess_environment()

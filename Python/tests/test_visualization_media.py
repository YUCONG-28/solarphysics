"""Regression tests for shared visualization media output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solar_toolkit.visualization import media


def _fake_pyav(events: list[str], *, fail_on_pts: int | None = None):
    class FakeVideoFrame:
        def __init__(self, array):
            self.array = array
            self.pts = None
            self.time_base = None

        @classmethod
        def from_ndarray(cls, array, *, format):
            assert format == "rgb24"
            events.append("from-ndarray")
            return cls(array)

        def reformat(self, *, width, height, format):
            events.append(f"reformat-{width}x{height}-{format}")
            return self

    class FakeCodecContext:
        def open(self):
            events.append("codec-open")

    class FakeStream:
        def __init__(self):
            self.codec_context = FakeCodecContext()
            self.options = {}

        def encode(self, frame=None):
            if frame is None:
                events.append("flush")
                return ["flush-packet"]
            events.append(f"encode-{frame.pts}")
            if frame.pts == fail_on_pts:
                raise RuntimeError("synthetic encoder failure")
            return [f"packet-{frame.pts}"]

    class FakeContainer:
        def __init__(self, path):
            self.path = Path(path)
            self.stream = FakeStream()
            self.closed = False

        def add_stream(self, codec_name, *, rate):
            events.append(f"add-stream-{codec_name}-{rate}")
            return self.stream

        def mux(self, packet):
            events.append(f"mux-{packet}")

        def close(self):
            if not self.closed:
                events.append("close")
                self.path.write_bytes(b"synthetic-pyav-video")
                self.closed = True

    def fake_open(path, **kwargs):
        events.append("open")
        return FakeContainer(path)

    return SimpleNamespace(open=fake_open, VideoFrame=FakeVideoFrame)


def test_pyav_stream_initializes_before_lazy_frames_and_flushes(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    output_path = tmp_path / "sequence.mp4"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    def frames():
        for index in range(2):
            events.append(f"yield-{index}")
            yield frame, (6, 4)

    monkeypatch.setattr(media, "resolve_ffprobe", lambda: None)

    assert media.write_frames_pyav_stream(
        frames(),
        output_path,
        fps=10.0,
        width=6,
        height=4,
        output_format="mp4",
        quality="high",
        av_module=_fake_pyav(events),
    )

    assert events.index("codec-open") < events.index("yield-0")
    assert events.count("from-ndarray") == 2
    assert "flush" in events
    assert "mux-flush-packet" in events
    assert output_path.read_bytes() == b"synthetic-pyav-video"
    assert not list(tmp_path.glob(".media-*.mp4"))


def test_pyav_runtime_error_cleans_partial_output(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    monkeypatch.setattr(media, "resolve_ffprobe", lambda: None)

    with pytest.raises(media.MediaProcessingError, match="PyAV failed while encoding"):
        media.write_frames_pyav_stream(
            [(frame, (6, 4)), (frame, (6, 4))],
            tmp_path / "sequence.mp4",
            fps=10.0,
            width=6,
            height=4,
            output_format="mp4",
            quality="high",
            av_module=_fake_pyav(events, fail_on_pts=1),
        )

    assert not (tmp_path / "sequence.mp4").exists()
    assert not list(tmp_path.glob(".media-*.mp4"))


def test_missing_pyav_falls_back_to_existing_ffmpeg_path(
    tmp_path: Path, monkeypatch
) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    calls = 0

    def fake_ffmpeg(frame_iter, output_path, **kwargs):
        nonlocal calls
        calls += 1
        assert len(list(frame_iter)) == 2
        Path(output_path).write_bytes(b"ffmpeg-fallback")
        return True

    monkeypatch.setattr(media, "_load_pyav", lambda: None)
    monkeypatch.setattr(media, "write_frames_ffmpeg_stream", fake_ffmpeg)

    assert media.write_media_from_frames(
        [(frame, (6, 4)), (frame, (6, 4))],
        tmp_path / "sequence.mp4",
        10.0,
        frame_size=(6, 4),
        output_format="mp4",
    )
    assert calls == 1


def test_real_pyav_mp4_is_probe_valid(tmp_path: Path) -> None:
    pytest.importorskip("av")
    if media.resolve_ffprobe() is None:
        pytest.skip("FFprobe is required for the real PyAV smoke")
    frames = [
        (np.full((6, 8, 3), value, dtype=np.uint8), (8, 6))
        for value in (0, 64, 128, 255)
    ]
    output_path = tmp_path / "pyav.mp4"

    assert media.write_frames_pyav_stream(
        frames,
        output_path,
        fps=2.0,
        width=8,
        height=6,
        output_format="mp4",
        quality="high",
    )
    probe = media.probe_video(
        output_path,
        expected_size=(8, 6),
        expected_frame_count=4,
    )
    assert probe["codec"] == "h264"
    assert probe["frame_rate"] == pytest.approx(2.0)


def test_ffmpeg_publication_retries_transient_windows_file_lock(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "sequence.mp4"
    temporary_path = tmp_path / ".sequence.partial-test.mp4"
    real_replace = os.replace
    replace_attempts = 0

    def fake_ffmpeg(command, frame_iter, *, width, height):
        list(frame_iter)
        Path(command[-1]).write_bytes(b"synthetic-mp4")
        return True

    def flaky_replace(source, destination):
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts < 3:
            error = PermissionError(13, "file is being used by another process")
            error.winerror = 32
            raise error
        real_replace(source, destination)

    monkeypatch.setattr(media, "resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media, "resolve_ffprobe", lambda: None)
    monkeypatch.setattr(media, "_run_ffmpeg_stream", fake_ffmpeg)
    monkeypatch.setattr(media, "_partial_output_path", lambda _path: temporary_path)
    monkeypatch.setattr(media.os, "replace", flaky_replace)
    monkeypatch.setattr(media.time, "sleep", lambda _seconds: None)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert media.write_frames_ffmpeg_stream(
        [(frame, (2, 2))],
        output_path,
        fps=10.0,
        width=2,
        height=2,
        output_format="mp4",
        quality="high",
    )

    assert replace_attempts == 3
    assert output_path.read_bytes() == b"synthetic-mp4"
    assert not temporary_path.exists()


def test_atomic_publication_hardlinks_when_delete_sharing_stays_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / ".media-locked.mp4"
    destination = tmp_path / "sequence.mp4"
    source.write_bytes(b"validated-video")
    sharing_error = PermissionError(13, "file is being used by another process")
    sharing_error.winerror = 32

    def blocked_replace(_operation):
        raise sharing_error

    monkeypatch.setattr(media, "_retry_windows_file_operation", blocked_replace)

    media._atomic_replace(source, destination)

    assert source.is_file()
    assert destination.read_bytes() == b"validated-video"
    assert source.samefile(destination)


def test_frame_source_error_is_not_downgraded_to_encoder_false(monkeypatch) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def broken_frames():
        yield frame, (2, 2)
        raise OSError("source frame became unavailable")

    class FakeStdin:
        def write(self, _payload):
            return None

        def close(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        media.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )

    with pytest.raises(media.MediaProcessingError, match="composite frame failed"):
        media._run_ffmpeg_stream(
            ["ffmpeg"],
            broken_frames(),
            width=2,
            height=2,
        )


def test_probe_video_reuses_identity_cache_and_invalidates_after_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sequence.mp4"
    path.write_bytes(b"first-video")
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 8,
                            "height": 6,
                            "nb_read_packets": "4",
                            "duration": "2.0",
                            "avg_frame_rate": "2/1",
                            "r_frame_rate": "2/1",
                        }
                    ],
                    "format": {"duration": "2.0"},
                }
            ),
        )

    with media._PROBE_CACHE_LOCK:
        media._PROBE_CACHE.clear()
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    first = media.probe_video(path, ffprobe="ffprobe")
    second = media.probe_video(path, ffprobe="ffprobe")
    path.write_bytes(b"second-video-with-new-identity")
    third = media.probe_video(path, ffprobe="ffprobe")

    assert first == second == third
    assert calls == 2


def test_failed_probe_is_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sequence.mp4"
    path.write_bytes(b"video")
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stderr="invalid", stdout="")

    with media._PROBE_CACHE_LOCK:
        media._PROBE_CACHE.clear()
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    for _ in range(2):
        with pytest.raises(media.MediaProcessingError, match="no decodable"):
            media.probe_video(path, ffprobe="ffprobe")

    assert calls == 2

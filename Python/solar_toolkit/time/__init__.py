"""Time helpers for local solar-observation workflows.

English: Lightweight timestamp parsing, nearest-time matching, and range
filtering shared by AIA, HMI, radio, X-ray, and CME tools.

中文：面向本地太阳观测流程的轻量时间工具，用于时间戳解析、最近时间匹配和时间范围筛选。
"""

from .parsing import (
    extract_time_from_filename,
    extract_timestamp_from_filename,
    parse_time,
)
from .precision import TimeStamp, elapsed_seconds, normalize_time, seconds_to_ns
from .selection import filter_by_time_range, nearest_by_time
from .sequence import (
    OneToOneMatch,
    SequenceMatchPlan,
    SequenceMatchRow,
    TimeFrame,
    TimeSelection,
    TimeStream,
    build_sequence_match_plan,
    filter_time_frames,
    monotone_one_to_one,
)

__all__ = [
    "extract_time_from_filename",
    "extract_timestamp_from_filename",
    "filter_by_time_range",
    "nearest_by_time",
    "parse_time",
    "TimeStamp",
    "elapsed_seconds",
    "normalize_time",
    "seconds_to_ns",
    "TimeFrame",
    "TimeStream",
    "TimeSelection",
    "SequenceMatchRow",
    "SequenceMatchPlan",
    "OneToOneMatch",
    "build_sequence_match_plan",
    "filter_time_frames",
    "monotone_one_to_one",
]

"""Native-frame loading and time labels, with no viewpoint mapping dependency."""

from astropy.time import Time
from copy import deepcopy
import math

from solar_toolkit.map.jet_annotations import JetDocument


def matching_sample_index(manifest, documents):
    """Resolve a sample only from the ordered original-image identities.

    Names, timestamps and the previous combo-box index cannot identify restored
    originals. Ambiguous duplicates and incomplete pairs remain unselected.
    """
    if not manifest or len(documents) != 2 or not all(documents):
        return -1
    identities = tuple(document.sha256 for document in documents)
    matches = [
        index
        for index, sample in enumerate(manifest[1].get("pairs", []))
        if tuple(view.get("image_sha256") for view in sample.get("views", []))
        == identities
    ]
    return matches[0] if len(matches) == 1 else -1


def sample_pairing(sample, documents):
    """Retain structured, image-bound evidence; legacy method strings prove no cadence.

    Loading a pair must not infer a cadence from the six sparse pilot samples.
    Invalid evidence is rejected before it can attach to different originals.
    """
    evidence = sample.get("pairing")
    if not isinstance(evidence, dict):
        return None
    if len(documents) != 2 or not all(documents):
        return None
    for name, document in zip(("AIA", "EUVI"), documents, strict=True):
        record = evidence.get(name)
        if (
            not isinstance(record, dict)
            or record.get("image_sha256") != document.sha256
        ):
            raise ValueError("样本时间配对证据与原图身份不一致")
        if (
            record.get("midpoint_utc")
            and abs(
                (Time(record["midpoint_utc"]) - Time(document.info["midpoint_utc"])).sec
            )
            > 0.001
        ):
            raise ValueError("样本时间配对证据与原图曝光中点不一致")
    if not isinstance(evidence.get("status"), str):
        raise ValueError("样本时间配对证据缺少状态")
    if evidence["status"] == "matched":
        tolerance = evidence.get("tolerance_s")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("样本时间配对证据缺少有效采样容差")
        if all(d.info.get("dsun_m") for d in documents):
            emission = [
                Time(d.info["midpoint_utc"]).unix - d.info["dsun_m"] / 299792458.0
                for d in documents
            ]
            delta = emission[0] - emission[1]
            if abs(delta) > tolerance + 1e-6:
                raise ValueError("样本实际时间差超过所记录的配对容差")
            if evidence.get("delta_emission_s") is not None and not math.isclose(
                evidence["delta_emission_s"], delta, abs_tol=0.001
            ):
                raise ValueError("样本光行时修正差与原图不一致")
    return deepcopy(evidence)


def load_native_pair(pair, known, snapshots, thaw, *, lazy_segmentation=False):
    """Load both actual frames independently; a missing counterpart stays missing."""
    documents = []
    for instrument in ("AIA", "EUVI"):
        record = pair[instrument]
        if record is None:
            documents.append(None)
            continue
        digest = record["image_sha256"]
        document = known.get(digest)
        if document is None and digest in snapshots:
            document = thaw(snapshots[digest])
        if document is None:
            document = JetDocument(record["_path"], lazy_segmentation=lazy_segmentation)
            if record.get("_difference"):
                document.load_difference(record["_difference"])
        if document.sha256 != digest:
            raise ValueError("时序原图校验不一致")
        if record.get("_difference") and (
            document.difference_sha256 != record["difference_sha256"]
        ):
            raise ValueError("差分校验不一致")
        documents.append(document)
    return documents


def native_pair_status(documents, pair=None):
    """Describe actual UTC and centre-light-time pairing, never imply simultaneity."""
    stamps = [
        f"{name}: {document.info['midpoint_utc']}" if document else f"{name}: 缺配"
        for name, document in zip(("AIA", "EUVI"), documents, strict=True)
    ]
    matches = pair and all(
        (document.sha256 if document else None)
        == (pair[name]["image_sha256"] if pair[name] else None)
        for name, document in zip(("AIA", "EUVI"), documents, strict=True)
    )
    if matches:
        delta = pair.get("delta_emission_s")
        states = {
            "matched": "时间配对通过",
            "no_other_frame": "缺少另一视角",
            "outside_tolerance": "时间差超限，缺配",
            "ambiguous_nearest": "多个最近帧，配对不唯一",
            "cadence_unavailable": "采样间隔不足，未自动配对",
        }
        status = states.get(pair["status"], pair["status"])
    else:
        delta = None
        status = "手动配对；未验证采样容差"
        if all(documents) and all(d.info.get("dsun_m") for d in documents):
            emission = [
                Time(d.info["midpoint_utc"]).unix - d.info["dsun_m"] / 299792458.0
                for d in documents
            ]
            delta = emission[0] - emission[1]
    text = " | ".join(stamps) + " | " + status
    text += " | 日心光行时修正差 " + (f"{delta:.3f} s" if delta is not None else "未定")
    if matches and pair.get("repeated_other"):
        text += " | 重用 EUVI/AIA 配对帧，不是新增独立观测"
    return text + " | 保持原始双视角"


__all__ = [
    "load_native_pair",
    "native_pair_status",
    "sample_pairing",
    "matching_sample_index",
]

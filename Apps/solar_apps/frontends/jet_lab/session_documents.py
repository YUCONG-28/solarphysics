"""Image cache shared by native sessions and legacy display adapters."""

from copy import deepcopy
import hashlib
import json

from solar_toolkit.map.jet_annotations import JetDocument, json_safe


def science_signature(owner=None):
    """Fingerprint persisted scientific results/options, independent of display.

    The default state is clean even before the native fitting controls exist.
    The selected model and robust comparison live inside the result payload.
    """
    options = {"include_curve": False, **getattr(owner, "fit_options", {})}
    payload = dict(
        result=getattr(owner, "reconstruction_result", None),
        signature=getattr(owner, "reconstruction_signature", None),
        fit_options=options,
    )
    return hashlib.sha256(
        json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


_EMPTY_SCIENCE_SIGNATURE = science_signature()


class DocumentStore:
    def remember(self):
        for pane in self.owner.panes:
            if pane.doc is not None:
                self.documents[pane.doc.sha256] = pane.doc
        active = {p.doc.sha256 for p in self.owner.panes if p.doc} | {
            self.reference_hash
        }

        # Evict image arrays, retaining annotations/undo as lightweight revisions.
        def size(d):
            return (
                d.raw.nbytes
                + d.map.data.nbytes
                + d.labels.nbytes
                + (d.difference.nbytes if d.difference is not None else 0)
            )

        total = sum(size(d) for d in self.documents.values())
        for key in list(self.documents):
            if total <= 256 * 1024**2:
                break
            if key in active:
                continue
            d = self.documents.pop(key)
            total -= size(d)
            self.document_snapshots[key] = {
                "path": str(d.path),
                "difference": str(d.difference_path) if d.difference_path else None,
                "difference_sha256": d.difference_sha256,
                "state": deepcopy(d.state),
                "history": deepcopy(d.history),
                "undo": deepcopy(d._undo),
                "dirty": d.dirty,
            }

    def thaw(self, snapshot):
        lazy = getattr(self.owner, "native_only", False)
        d = JetDocument(snapshot["path"], lazy_segmentation=lazy)
        if snapshot["difference"]:
            d.load_difference(snapshot["difference"])
        d.state = deepcopy(snapshot["state"])
        d.history = deepcopy(snapshot["history"])
        d._undo = deepcopy(snapshot["undo"])
        d.dirty = snapshot["dirty"]
        if not lazy or d.state.get("axis") or d.state.get("seed") is not None:
            d.resegment(restore=True)
        return d

    def has_unsaved(self):
        return (
            any(d.dirty for d in self.documents.values())
            or any(s["dirty"] for s in self.document_snapshots.values())
            or self.has_unsaved_science()
        )

    def has_unsaved_science(self):
        if not getattr(self.owner, "native_only", False):
            return False
        baseline = (
            getattr(self.owner, "saved_science_signature", None)
            or _EMPTY_SCIENCE_SIGNATURE
        )
        return science_signature(self.owner) != baseline

    def mark_science_saved(self):
        self.owner.saved_science_signature = science_signature(self.owner)

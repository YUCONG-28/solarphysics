"""Joint native-pixel fits of straight or single-bend stereo jet candidates.

The observations remain the original two images. Neither an image warp nor an
epipolar snap enters the objective. A fitted curve is a conditional shape model,
not evidence that the identified structures or their exposure times agree.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.time import Time
from scipy.integrate import quad
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import softmax
from sunpy.coordinates import frames
from sunpy.map import Map

from .euvi_preprocessing import geometry_issues
from .jet_annotations import json_safe
from .jet_viewpoint import observer

__all__ = ["GEOMETRY_FIT_VERSION", "freeze_jet_documents", "fit_jet_geometry"]

GEOMETRY_FIT_VERSION = "native-joint-fit-v1"
_RADIUS_M = 695700000.0
_LIGHT_SPEED_M_S = 299792458.0
_POSITION_EPS = 1e-10


@dataclass(frozen=True)
class _FrozenDocument:
    path: str
    sha256: str
    map: object
    info: dict
    state: dict


def freeze_jet_documents(documents):
    """Copy scientific inputs for an isolated worker; no Qt objects are retained.

    Array copies are read-only. Annotation dictionaries and map metadata belong
    to the snapshot, so an edit or a frame change cannot alter a running fit.
    The caller must still reject results whose input signature became stale.
    """
    result = []
    for doc in documents:
        if doc is None:
            result.append(None)
            continue
        data = np.array(doc.map.data, copy=True)
        data.flags.writeable = False
        mask = None if doc.map.mask is None else np.array(doc.map.mask, copy=True)
        if mask is not None:
            mask.flags.writeable = False
        result.append(
            _FrozenDocument(
                str(doc.path),
                str(doc.sha256),
                Map(data, deepcopy(doc.map.meta), mask=mask),
                deepcopy(doc.info),
                {"tiepoints": deepcopy(doc.state.get("tiepoints", []))},
            )
        )
    return tuple(result)


def _input_signature(documents, pairing):
    inputs = [
        (
            None
            if doc is None
            else dict(
                sha256=doc.sha256,
                native_wcs=doc.map.wcs.to_header_string(),
                native_metadata=dict(doc.map.meta),
                observation=doc.info,
                tiepoints=doc.state.get("tiepoints", []),
            )
        )
        for doc in documents
    ]
    payload = json.dumps(
        json_safe(dict(version=GEOMETRY_FIT_VERSION, views=inputs, pairing=pairing)),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class _NativeProjectionContext:
    """Frozen per-pair affine frame transforms and vectorized native WCS calls."""

    def __init__(self, documents):
        if len(documents) != 2 or any(d is None for d in documents):
            raise ValueError("missing_view")
        problems = [reason for d in documents for reason in geometry_issues(d.map)]
        if problems:
            raise ValueError(";".join(problems))
        self.documents = tuple(documents)
        self.common = frames.HeliographicStonyhurst(
            obstime=Time(documents[0].info["midpoint_utc"])
        )
        self.native = [
            frames.HeliographicStonyhurst(obstime=d.map.coordinate_frame.obstime)
            for d in documents
        ]
        self.native_origins = [
            observer(d.map, f) for d, f in zip(documents, self.native, strict=True)
        ]
        self.rotations, self.offsets, self.bases, self.origins = [], [], [], []
        self.midpoints = [Time(d.info["midpoint_utc"]).unix for d in documents]
        basis_points = np.vstack((np.zeros(3), np.eye(3)))
        coordinate = SkyCoord(
            CartesianRepresentation(basis_points.T * _RADIUS_M * u.m), frame=self.common
        )
        for native, origin in zip(self.native, self.native_origins, strict=True):
            transformed = (
                coordinate.transform_to(native).cartesian.xyz.to_value(u.m).T
                / _RADIUS_M
            )
            offset = transformed[0]
            rotation = (transformed[1:] - offset).T
            self.rotations.append(rotation)
            self.offsets.append(offset)
            self.origins.append(np.linalg.solve(rotation, origin - offset))
            z = origin / np.linalg.norm(origin)
            x = np.cross([0.0, 0.0, 1.0], z)
            if np.linalg.norm(x) <= _POSITION_EPS:
                raise ValueError("observer_polar_basis_undefined")
            x /= np.linalg.norm(x)
            self.bases.append((x, np.cross(z, x), z))
        self.unit_from_rad = [
            np.array([u.rad.to(u.Unit(unit)) for unit in d.map.wcs.world_axis_units])
            for d in documents
        ]

    def native_positions(self, xyz):
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        return [
            xyz @ rotation.T + offset
            for rotation, offset in zip(self.rotations, self.offsets, strict=True)
        ]

    def project(self, xyz):
        """Return (point, view, x/y) native pixels and HPC arcseconds."""
        pixels, hpc = [], []
        for i, positions in enumerate(self.native_positions(xyz)):
            direction = positions - self.native_origins[i]
            norms = np.linalg.norm(direction, axis=1)
            if np.any(norms <= _POSITION_EPS):
                raise ValueError("position_at_observer")
            direction /= norms[:, None]
            x, y, z = self.bases[i]
            angles = np.c_[
                np.arctan2(direction @ x, -(direction @ z)),
                np.arcsin(np.clip(direction @ y, -1, 1)),
            ]
            world = angles * self.unit_from_rad[i]
            px, py = self.documents[i].map.wcs.world_to_pixel_values(
                world[:, 0], world[:, 1]
            )
            pixels.append(np.c_[px, py])
            hpc.append(angles * u.rad.to(u.arcsec))
        return np.stack(pixels, axis=1), np.stack(hpc, axis=1)

    def rays(self, view, pixels):
        """Batch native pixels to exact finite-distance common-frame rays."""
        xy = np.asarray(pixels, float).reshape(-1, 2)
        wx, wy = self.documents[view].map.wcs.pixel_to_world_values(xy[:, 0], xy[:, 1])
        angles = np.c_[wx, wy] / self.unit_from_rad[view]
        angles[:, 0] = (angles[:, 0] + np.pi) % (2 * np.pi) - np.pi
        tx, ty = angles.T
        x, y, z = self.bases[view]
        native_ray = (
            np.cos(ty)[:, None] * np.sin(tx)[:, None] * x
            + np.sin(ty)[:, None] * y
            - np.cos(ty)[:, None] * np.cos(tx)[:, None] * z
        )
        rays = np.linalg.solve(self.rotations[view], native_ray.T).T
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        return rays, angles * u.rad.to(u.arcsec)

    def physical(self, xyz):
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        visible, distances = [], []
        for positions, origin in zip(
            self.native_positions(xyz), self.native_origins, strict=True
        ):
            direction = positions - origin
            den = np.sum(direction**2, axis=1)
            t = np.clip(-direction @ origin / np.maximum(den, _POSITION_EPS**2), 0, 1)
            closest = origin + t[:, None] * direction
            visible.append(np.linalg.norm(closest, axis=1) >= 1 - _POSITION_EPS)
            distances.append(np.sqrt(den))
        return dict(
            height_Rsun=np.linalg.norm(xyz, axis=1) - 1,
            visible=np.stack(visible, axis=1),
            source_delta_emission_s=(
                self.midpoints[0]
                - np.asarray(distances[0]) * _RADIUS_M / _LIGHT_SPEED_M_S
            )
            - (
                self.midpoints[1]
                - np.asarray(distances[1]) * _RADIUS_M / _LIGHT_SPEED_M_S
            ),
        )


def _unit(value):
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= _POSITION_EPS:
        raise ValueError("coincident_fitted_endpoints")
    return np.asarray(value) / norm


def _angle(first, second):
    return float(np.rad2deg(np.arccos(np.clip(_unit(first) @ _unit(second), -1, 1))))


class _Shape:
    """Endpoint gauge and two transverse bend components, no depth bounds."""

    def __init__(self, xyz, model):
        self.model = model
        self.centre = np.asarray(xyz).mean(axis=0)
        self.scale = float(np.linalg.norm(xyz[-1] - xyz[0]))
        if self.scale <= _POSITION_EPS:
            raise ValueError("coincident_endpoints")
        initial_direction = _unit(xyz[-1] - xyz[0])
        # The fixed chart avoids a discontinuous least-aligned-axis choice at
        # every objective call. Leaving this chart is a diagnosed failed fit.
        self.reference = np.eye(3)[np.argmin(abs(initial_direction))]

    def initial(self, xyz):
        endpoints = ((np.asarray(xyz)[[0, -1]] - self.centre) / self.scale).ravel()
        distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if np.any(distances <= _POSITION_EPS):
            raise ValueError("duplicate_initial_positions")
        logits = np.log(distances[:-1] / distances[-1])
        return np.r_[endpoints, [0.0, 0.0] if self.model == "curve" else [], logits]

    def unpack(self, parameters):
        endpoints = parameters[:6].reshape(2, 3) * self.scale + self.centre
        direction = _unit(endpoints[1] - endpoints[0])
        bend = np.zeros(3)
        offset = 6
        if self.model == "curve":
            first = self.reference - direction * (self.reference @ direction)
            first = _unit(first)
            second = np.cross(direction, first)
            bend = self.scale * (parameters[6] * first + parameters[7] * second)
            offset = 8
        # Fix the last logit to zero: the softmax then has no additive gauge.
        increments = softmax(np.r_[parameters[offset:], 0.0])
        t = np.r_[0.0, np.cumsum(increments)]
        t[-1] = 1.0
        return endpoints, bend, t

    @staticmethod
    def evaluate(endpoints, bend, t):
        t = np.asarray(t, float).reshape(-1)
        return (
            endpoints[0]
            + t[:, None] * (endpoints[1] - endpoints[0])
            + 4 * (t * (1 - t))[:, None] * bend
        )


def _blank_model(model, issues=()):
    return dict(
        model=model,
        valid=False,
        converged=False,
        status="unavailable",
        issues=list(issues),
        geometry_valid=False,
        parameters=None,
        fitted_points=[],
        pixel_rmse=None,
        per_view_rmse_px=None,
        length_Rsun=None,
        length_Mm=None,
        direction_xyz=None,
        endpoint_tangents_xyz=None,
        radial_angle_deg=None,
        directed=False,
        unsigned_radial_angle_deg=None,
        ordering_source=None,
        length_interpretation=None,
        optimizer=None,
        validation=None,
        sampled_curve_xyz_Rsun=[],
    )


def _cancel_if_requested(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("jet_geometry_fit_cancelled")


def _path_height_range(endpoints, bend):
    """Exact radial extrema of the quadratic path, including interior extrema."""
    coefficients = np.stack(
        (endpoints[0], endpoints[1] - endpoints[0] + 4 * bend, -4 * bend)
    )
    squared_radius = sum(
        np.convolve(coefficients[:, axis], coefficients[:, axis]) for axis in range(3)
    )
    roots = np.polynomial.polynomial.polyroots(
        np.polynomial.polynomial.polyder(squared_radius)
    )
    parameters = np.array(
        [0.0, 1.0]
        + [
            float(root.real)
            for root in roots
            if abs(root.imag) < 1e-9 and 0 < root.real < 1
        ]
    )
    positions = _Shape.evaluate(endpoints, bend, parameters)
    heights = np.linalg.norm(positions, axis=1) - 1
    return dict(
        min_height_Rsun=float(heights.min()),
        max_height_Rsun=float(heights.max()),
        extremum_parameters=parameters,
        method="polynomial_radial_extrema_including_endpoints",
    )


def _uncertainty_weights(rows, context):
    """All-or-none supplied angular sigmas, otherwise uniform native pixels.

    Full angular sigmas are transformed with each WCS local Jacobian. They are
    conditional independent localization scales, not a complete error budget.
    Partial/invalid sigmas never cause a hidden mixture of units or weights.
    """

    def numeric_sigma(correspondence):
        try:
            return (
                float(correspondence.get("sigma_arcsec")) if correspondence else np.nan
            )
        except (TypeError, ValueError):
            return np.nan

    sigma = np.array(
        [
            [numeric_sigma(c) for c in r.get("input_correspondences", [None, None])]
            for r in rows
        ],
        float,
    )
    if (
        sigma.shape != (len(rows), 2)
        or not np.isfinite(sigma).all()
        or np.any(sigma <= 0)
    ):
        supplied = int(np.count_nonzero(np.isfinite(sigma) & (sigma > 0)))
        return None, dict(
            mode="equal_native_pixels",
            complete_sigma=False,
            supplied_sigma_count=supplied,
            partial_sigma_diagnostic_only=True,
            interpretation="engineering_residual_not_statistical_confidence",
        )
    whiteners = []
    for row, sigmas in zip(rows, sigma, strict=True):
        per_view = []
        for view, (pixel, s) in enumerate(zip(row["pixel_xy"], sigmas, strict=True)):
            offsets = np.array(
                [pixel, np.asarray(pixel) + [0.01, 0], np.asarray(pixel) + [0, 0.01]]
            )
            _, hpc = context.rays(view, offsets)
            jac = (hpc[1:] - hpc[0]).T / 0.01
            # For scalar isotropic HPC sigma, J/sigma maps pixel offsets to
            # normalized angular residuals without discarding PC/rotation.
            if not np.isfinite(jac).all() or abs(np.linalg.det(jac)) <= 1e-12:
                raise ValueError("sigma_wcs_jacobian_singular")
            per_view.append(jac / s)
        whiteners.append(per_view)
    return np.array(whiteners), dict(
        mode="supplied_angular_sigma",
        complete_sigma=True,
        supplied_sigma_count=int(sigma.size),
        transformation="native_pixel_residual_whitened_by_local_HPC_Jacobian",
        interpretation="conditional_independent_localization_weights; evolution_and_common_pointing_not_included",
    )


def _solve(rows, context, model, loss, max_nfev, whiteners=None, cancelled=None):
    _cancel_if_requested(cancelled)
    result = _blank_model(model)
    xyz = np.asarray([r["xyz_Rsun"] for r in rows], float)
    observed = np.asarray([r["pixel_xy"] for r in rows], float)
    try:
        shape = _Shape(xyz, model)
        initial = shape.initial(xyz)
    except ValueError as error:
        result["issues"].append(str(error))
        return result

    def residual(parameters):
        _cancel_if_requested(cancelled)
        endpoints, bend, t = shape.unpack(parameters)
        predicted, _ = context.project(shape.evaluate(endpoints, bend, t))
        differences = predicted - observed
        if whiteners is not None:
            differences = np.einsum("nvij,nvj->nvi", whiteners, differences)
        if not np.isfinite(differences).all():
            raise ValueError("nonfinite_native_projection")
        return differences.ravel()

    try:
        solution = least_squares(
            residual,
            initial,
            method="trf",
            loss=loss,
            f_scale=1.0,
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        endpoints, bend, t = shape.unpack(solution.x)
        fitted = shape.evaluate(endpoints, bend, t)
        predicted, predicted_hpc = context.project(fitted)
        differences = predicted - observed
        objective_differences = (
            differences
            if whiteners is None
            else np.einsum("nvij,nvj->nvi", whiteners, differences)
        )
        score_weights = (
            np.ones_like(objective_differences)
            if loss == "linear"
            else 1 / np.sqrt(1 + objective_differences**2)
        )
        singular = np.linalg.svd(solution.jac, compute_uv=False)
        threshold = singular[0] * 1e-8 if len(singular) else 0.0
        rank = int(np.count_nonzero(singular > threshold))
        condition = (
            float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > 0
            else None
        )
        physical = context.physical(fitted)
        inner_tangent = _unit(endpoints[1] - endpoints[0] + 4 * bend)
        outer_tangent = _unit(endpoints[1] - endpoints[0] - 4 * bend)
        centre = shape.evaluate(endpoints, bend, [0.5])[0]
        direction = _unit(endpoints[1] - endpoints[0])
        length, quadrature_error = quad(
            lambda value: float(
                np.linalg.norm(endpoints[1] - endpoints[0] + 4 * (1 - 2 * value) * bend)
            ),
            0,
            1,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        sampled_curve = shape.evaluate(endpoints, bend, np.linspace(0, 1, 101))
        path_height = _path_height_range(endpoints, bend)
        path_visibility = context.physical(sampled_curve)["visible"]
        result.update(
            converged=bool(solution.success),
            status="fitted" if solution.success else "not_converged",
            parameters=dict(
                endpoint_xyz_Rsun=endpoints,
                bend_xyz_Rsun=bend,
                t=t,
                formula="P0+t*(P1-P0)+4*t*(1-t)*b",
                bend_orthogonal_to_chord=True,
            ),
            pixel_rmse=float(np.sqrt(np.mean(differences**2))),
            per_view_rmse_px=np.sqrt(np.mean(differences**2, axis=(0, 2))),
            length_Rsun=float(length),
            length_Mm=float(length * _RADIUS_M / 1e6),
            direction_xyz=direction,
            endpoint_tangents_xyz=[inner_tangent, outer_tangent],
            radial_angle_deg=_angle(direction, centre),
            optimizer=dict(
                success=bool(solution.success),
                status=int(solution.status),
                message=str(solution.message),
                nfev=int(solution.nfev),
                cost=float(solution.cost),
                jacobian_rank=rank,
                parameter_count=len(solution.x),
                jacobian_condition=condition,
                rank_relative_threshold=1e-8,
                quadrature_error_Rsun=float(quadrature_error),
            ),
            sampled_curve_xyz_Rsun=sampled_curve,
            path_height_range=path_height,
            path_visibility=dict(
                sample_count=101,
                all_samples_visible=bool(path_visibility.all()),
                interpretation="sampled_visibility_check_not_continuous_visibility_proof",
            ),
        )
        if not solution.success:
            result["issues"].append("optimizer_not_converged")
        if rank < len(solution.x):
            result["issues"].append("rank_deficient_fit")
        if np.any(np.diff(t) <= 1e-10):
            result["issues"].append("fitted_nodes_not_resolvably_ordered")
        if path_height["min_height_Rsun"] < -1e-10:
            result["issues"].append("fitted_path_below_photosphere")
        if not path_visibility.all():
            result["issues"].append("fitted_path_occulted_at_sample")
        for index, row in enumerate(rows):
            issues = []
            if physical["height_Rsun"][index] < -1e-10:
                issues.append("below_photosphere")
            if not np.all(physical["visible"][index]):
                issues.append("solar_occultation")
            tolerance = getattr(context, "tolerance_s", None)
            if (
                tolerance is not None
                and abs(physical["source_delta_emission_s"][index]) > tolerance + 1e-6
            ):
                issues.append("fitted_source_time_outside_tolerance")
            result["fitted_points"].append(
                dict(
                    number=row["number"],
                    xyz_Rsun=fitted[index],
                    predicted_pixel_xy=predicted[index],
                    observed_pixel_xy=observed[index],
                    residual_pixel_xy=differences[index],
                    residual_px=np.linalg.norm(differences[index], axis=1),
                    objective_residual=objective_differences[index],
                    robust_score_weights=score_weights[index],
                    robust_weight_interpretation="loss_score_weights_not_probabilities_or_rejection_flags",
                    predicted_hpc_arcsec=predicted_hpc[index],
                    height_Rsun=physical["height_Rsun"][index],
                    visible=physical["visible"][index],
                    source_delta_emission_s=physical["source_delta_emission_s"][index],
                    admissible_candidate=not issues,
                    issues=issues,
                    geometry_valid=False,
                )
            )
        if any(p["issues"] for p in result["fitted_points"]):
            result["issues"].append("fitted_physical_conditions_failed")
        result["valid"] = bool(solution.success and not result["issues"])
        if solution.success and result["issues"]:
            result["status"] = "diagnostic_candidate"
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result.update(status="failed", issues=[str(error)])
    return result


def _cross_validate(rows, context, full, loss, max_nfev, whiteners, cancelled=None):
    folds = []
    for omitted in range(1, len(rows) - 1):
        _cancel_if_requested(cancelled)
        keep = np.arange(len(rows)) != omitted
        training = [row for index, row in enumerate(rows) if keep[index]]
        fitted = _solve(
            training,
            context,
            full["model"],
            loss,
            max_nfev,
            None if whiteners is None else whiteners[keep],
            cancelled,
        )
        fold = dict(
            omitted_number=rows[omitted]["number"],
            valid=False,
            issues=fitted["issues"],
            pixel_rmse=None,
            predicted_pixel_xy=None,
            endpoint_direction_change_deg=None,
            relative_length_change=None,
        )
        if fitted["valid"]:
            parameters = fitted["parameters"]
            endpoints, bend = np.asarray(parameters["endpoint_xyz_Rsun"]), np.asarray(
                parameters["bend_xyz_Rsun"]
            )
            observed = np.asarray(rows[omitted]["pixel_xy"])
            # The held-out feature parameter is nuisance distance on the fitted
            # locus, constrained between its unchanged ordered neighbours.
            lower, upper = parameters["t"][omitted - 1 : omitted + 1]

            def objective(t, endpoints=endpoints, bend=bend, observed=observed):
                _cancel_if_requested(cancelled)
                projected, _ = context.project(_Shape.evaluate(endpoints, bend, [t]))
                delta = projected[0] - observed
                return float(np.sum(delta**2))

            prediction = minimize_scalar(
                objective,
                bounds=(float(lower), float(upper)),
                method="bounded",
                options={"xatol": 1e-10},
            )
            predicted, _ = context.project(
                _Shape.evaluate(endpoints, bend, [prediction.x])
            )
            fold.update(
                valid=bool(prediction.success),
                pixel_rmse=float(np.sqrt(prediction.fun / 4)),
                predicted_pixel_xy=predicted[0],
                endpoint_direction_change_deg=[
                    _angle(a, b)
                    for a, b in zip(
                        full["endpoint_tangents_xyz"],
                        fitted["endpoint_tangents_xyz"],
                        strict=True,
                    )
                ],
                relative_length_change=float(
                    (fitted["length_Rsun"] - full["length_Rsun"]) / full["length_Rsun"]
                ),
            )
        folds.append(fold)
    valid = [fold for fold in folds if fold["valid"]]
    return dict(
        method="leave_one_interior_feature_out; endpoints_retained",
        interpretation="predictive_locus_distance_and_sensitivity_not_confidence",
        held_out_parameter_metric="equal_native_pixel_residual",
        folds=folds,
        valid_fold_count=len(valid),
        expected_fold_count=len(rows) - 2,
        pixel_rmse=(
            float(np.sqrt(np.mean([fold["pixel_rmse"] ** 2 for fold in valid])))
            if valid
            else None
        ),
        complete=len(valid) == len(folds),
        max_endpoint_direction_change_deg=max(
            (max(fold["endpoint_direction_change_deg"]) for fold in valid), default=None
        ),
        max_abs_relative_length_change=max(
            (abs(fold["relative_length_change"]) for fold in valid), default=None
        ),
    )


def fit_jet_geometry(
    documents,
    *,
    reconstruction=None,
    pairing=None,
    include_curve=False,
    loss="linear",
    cross_validate=True,
    max_nfev=400,
    cancelled=None
):
    """Compare conditional joint fits in the two original image coordinate systems.

    The line is always the default, never silently replaced by a curve. With no
    reliable endpoint/order metadata its internal ordering is only a nuisance
    parameterization and it returns an unsigned axis and fitted-sample span.
    A curve requires at least six complete, ordered jet features and unique endpoints.
    ``soft_l1`` uses scale 1 pixel without complete sigmas (engineering only), or
    1 normalized residual when every localization sigma is supplied. Both fit
    types retain the same input sample. No result overwrites original clicks.
    A true ``cancelled()`` callback raises ``InterruptedError`` inside objectives
    and between leave-one-out fits; cancellation never returns a partial success.
    """
    _cancel_if_requested(cancelled)
    if loss not in {"linear", "soft_l1"}:
        raise ValueError("loss must be linear or soft_l1")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    result = dict(
        schema_version=1,
        method_version=GEOMETRY_FIT_VERSION,
        status="unavailable",
        issues=[],
        input_point_numbers=[],
        excluded_points=[],
        weighting=None,
        loss=loss,
        selected_model="line",
        line=_blank_model("line"),
        curve=_blank_model("curve", ["curve_not_requested"]),
        comparison=None,
        geometry_valid=False,
        display_geometry_used=False,
        context={},
        settings=dict(
            include_curve=bool(include_curve),
            cross_validate=bool(cross_validate),
            max_nfev=max_nfev,
            robust_scale=1.0,
        ),
    )
    if len(documents) != 2 or any(d is None for d in documents):
        result["issues"].append("missing_view")
        return result
    if reconstruction is None:
        from .jet_reconstruction import reconstruct_jet

        reconstruction = reconstruct_jet(documents, pairing=pairing)
    if pairing is None:
        pairing = reconstruction.get("provenance", {}).get("pairing")
    frame_records = reconstruction.get("provenance", {}).get("frames", [])
    if [r.get("sha256") for r in frame_records] != [d.sha256 for d in documents]:
        result["issues"].append("reconstruction_input_identity_mismatch")
        return result
    current_signature = _input_signature(documents, pairing)
    saved_signature = reconstruction.get("provenance", {}).get("input_signature")
    if saved_signature is not None and saved_signature != current_signature:
        result["issues"].append("reconstruction_input_signature_stale")
        return result
    rows = reconstruction.get("points", [])
    # State/coordinate comparison prevents an old reconstruction on the same
    # image pair from supplying outdated initial points or feature roles.
    for row in rows:
        current = [
            next(
                (
                    p
                    for p in d.state.get("tiepoints", [])
                    if p.get("number") == row["number"]
                ),
                None,
            )
            for d in documents
        ]
        if json_safe(current) != row.get("input_correspondences"):
            result["issues"].append("reconstruction_annotations_stale")
            return result
    if {r["number"] for r in rows} != {
        p.get("number") for d in documents for p in d.state.get("tiepoints", [])
    }:
        result["issues"].append("reconstruction_annotations_stale")
        return result
    selected = []
    for row in rows:
        eligible = row.get("eligibility", {}).get(
            "axis_candidate", row.get("admissible_candidate", False)
        )
        if row.get("role") == "jet" and eligible:
            selected.append(row)
        else:
            result["excluded_points"].append(
                dict(
                    number=row["number"],
                    reasons=(
                        ["not_explicit_jet_point"]
                        if row.get("role") != "jet"
                        else row.get("issues", ["candidate_unavailable"])
                    ),
                )
            )
    result["input_point_numbers"] = [r["number"] for r in selected]
    all_jets = [
        r
        for r in rows
        if r.get("role") == "jet"
        or (
            r.get("role") == "unknown"
            and any(
                c and c.get("role") == "jet" for c in r.get("input_correspondences", [])
            )
        )
    ]
    curve_issues = []
    if len(selected) != len(all_jets):
        result["issues"].append(
            "unusable_jet_features_excluded; line_span_is_not_complete_jet_length"
        )
        curve_issues.append("complete_jet_features_required_for_curve")
    if len(selected) < 3:
        result["issues"].append("at_least_three_jet_features_required")
        result["line"]["issues"] = result["issues"].copy()
        return json_safe(result)
    orders = [row.get("order") for row in selected]
    ordered = all(
        isinstance(value, int) and not isinstance(value, bool) for value in orders
    ) and len(set(orders)) == len(orders)
    ordering_source = (
        "annotated_jet_order"
        if ordered
        else "PCA_projection_for_nuisance_parameters_only"
    )
    if ordered:
        selected.sort(key=lambda r: r["order"])
    else:
        curve_issues.append("unique_annotated_jet_order_required_for_curve")
        positions = np.asarray([row["xyz_Rsun"] for row in selected])
        centered = positions - positions.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        if (
            values[-1] <= _POSITION_EPS**2
            or values[-1] - values[-2] <= 1e-10 * values[-1]
        ):
            result["issues"].append("initial_eigenline_not_unique")
            return json_safe(result)
        direction = vectors[:, -1]
        if direction[np.argmax(abs(direction))] < 0:
            direction = -direction
        selected = [
            selected[index] for index in np.argsort(centered @ direction, kind="stable")
        ]
    inner = [r for r in all_jets if r.get("endpoint") == "inner"]
    outer = [r for r in all_jets if r.get("endpoint") == "outer"]
    unique_endpoints = len(inner) == 1 and len(outer) == 1
    if (
        not ordered
        and unique_endpoints
        and inner[0]["number"] == selected[-1]["number"]
        and outer[0]["number"] == selected[0]["number"]
    ):
        selected.reverse()
    directed = bool(
        unique_endpoints
        and inner[0]["number"] == selected[0]["number"]
        and outer[0]["number"] == selected[-1]["number"]
    )
    if not directed:
        curve_issues.append("explicit_unique_extreme_endpoints_required_for_curve")
    if len(selected) < 6:
        curve_issues.append("at_least_six_ordered_features_required")
    try:
        context = _NativeProjectionContext(documents)
        tolerance = (pairing or {}).get("tolerance_s")
        context.tolerance_s = (
            float(tolerance)
            if isinstance(tolerance, (int, float))
            and not isinstance(tolerance, bool)
            and np.isfinite(tolerance)
            and tolerance > 0
            else None
        )
        whiteners, weighting = _uncertainty_weights(selected, context)
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        result["issues"].append(str(error))
        return json_safe(result)
    result.update(
        input_point_numbers=[r["number"] for r in selected],
        weighting=weighting,
        context=dict(
            input_signature=current_signature,
            reference_frame="HeliographicStonyhurst",
            reference_time_utc=context.common.obstime.utc.isot,
            image_sha256=[d.sha256 for d in documents],
            physical_radius_m=_RADIUS_M,
            projection="finite_distance_native_WCS; no_image_reprojection",
            bounds="feature_parameter_0_to_1_only; no_height_or_curvature_bounds",
            identity_status="not_upgraded_by_fit",
            reconstruction_issues=reconstruction.get("issues", []),
        ),
    )
    result["settings"]["robust_scale_units"] = (
        "normalized_localization_residual"
        if weighting["complete_sigma"]
        else "native_pixel"
    )
    result["line"] = _solve(
        selected, context, "line", loss, max_nfev, whiteners, cancelled
    )
    if include_curve:
        result["curve"] = (
            _solve(selected, context, "curve", loss, max_nfev, whiteners, cancelled)
            if not curve_issues
            else _blank_model("curve", curve_issues)
        )
    if cross_validate:
        for model in ("line", "curve"):
            if result[model]["valid"]:
                result[model]["validation"] = _cross_validate(
                    selected,
                    context,
                    result[model],
                    loss,
                    max_nfev,
                    whiteners,
                    cancelled,
                )
    for name in ("line", "curve"):
        model = result[name]
        model.update(
            directed=directed,
            ordering_source=ordering_source,
            length_interpretation=(
                "fitted_span_between_explicit_inner_outer_features"
                if directed and len(selected) == len(all_jets)
                else "fitted_sample_span_not_confirmed_complete_jet_length"
            ),
        )
        if model["radial_angle_deg"] is not None:
            model["unsigned_radial_angle_deg"] = min(
                model["radial_angle_deg"], 180 - model["radial_angle_deg"]
            )
            if not directed:
                model["radial_angle_deg"] = None
    result["status"] = "fitted" if result["line"]["valid"] else "diagnostic_candidate"
    result["comparison"] = dict(
        same_input_point_numbers=result["input_point_numbers"],
        automatic_model_selection=False,
        line_pixel_rmse=result["line"]["pixel_rmse"],
        curve_pixel_rmse=result["curve"]["pixel_rmse"],
        interpretation="curve_is_a_conditional_single_bend_model; lower_training_residual_is_not_validation",
    )
    return json_safe(result)

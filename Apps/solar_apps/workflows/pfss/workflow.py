"""A reproducible global field, forward-radio comparison and portable bundle."""

from dataclasses import asdict
from pathlib import Path
import json
import platform
import time


def run(config):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import numpy as np
    import pandas as pd
    import sunpy.map
    from sunpy.coordinates import frames, get_earth, sun
    from solar_toolkit.modeling.pfss import (
        PFSSConfig,
        prepare_boundary,
        solve_pfss,
        trace_fieldlines,
        uniform_seeds,
        project_fieldlines,
        save_bundle,
    )
    from solar_toolkit.modeling.pfss.bundle import file_hash
    from solar_toolkit.modeling.pfss.observations import observation_midpoint
    from solar_toolkit.modeling.pfss.projection import visible_from_observer
    from solar_toolkit.radio.fieldline_association import (
        projected_distances,
        associate_sources,
        model_shell_candidates,
    )
    from solar_toolkit.radio.io import truthy

    output = Path(config["output_dir"])
    if output.exists() and (output / "manifest.json").exists():
        raise FileExistsError(
            "Use a fresh run directory; completed results are immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    physics = PFSSConfig(**config.get("pfss", {}))
    if physics.max_seeds > 512:
        raise ValueError("This workflow is limited to 512 seeds per run")
    step = float(config.get("trace_step_size", 0.25))
    steps = int(config.get("trace_max_steps", 4096))
    if not np.isfinite(step) or not 0 < step <= 1 or not 1 <= steps <= 16384:
        raise ValueError("Tracer requires 0 < step_size <= 1 and max_steps <= 16384")
    event = Time(config["event_utc"], scale="utc")
    print("Preparing global radial boundary", flush=True)
    boundary, audit = prepare_boundary(
        config["boundary_fits"], nphi=physics.nphi, ns=physics.ns, component="Br"
    )
    correction = config.get("net_flux_correction", "none")
    if correction not in {"none", "subtract_global_mean"}:
        raise ValueError("Unknown explicit net flux correction")
    if correction == "subtract_global_mean":
        mean = float(boundary.data.mean())
        boundary = sunpy.map.Map(boundary.data - mean, boundary.meta.copy())
        audit.update(net_flux_correction=correction, removed_mean_G=mean)
    audit["final_input_signed_flux_G_sr"] = float(boundary.data.mean() * 4 * np.pi)
    audit["final_input_unsigned_flux_G_sr"] = float(
        np.abs(boundary.data).mean() * 4 * np.pi
    )
    audit["solver_monopole_policy"] = (
        "Excluded by the backend independently of input mean subtraction"
    )
    boundary.save(output / "prepared_boundary.fits", overwrite=False)
    (output / "boundary_audit.json").write_text(json.dumps(audit, indent=2))
    from solar_toolkit.modeling.pfss.cache import cache_keys
    from solar_toolkit.modeling.pfss.bundle import load_bundle
    import shutil

    keys = cache_keys(config)
    observer = get_earth(event)
    aia = sunpy.map.Map(config["aia_fits"])
    aia_mid = observation_midpoint(aia)
    cached = None
    if config.get("reuse_run"):
        previous, previous_lines = load_bundle(config["reuse_run"])
        if not {"source_surface_br.fits", "field_samples.npz", "seeds.csv"}.issubset(
            previous["_verified_files"]
        ):
            raise ValueError(
                "Reusable magnetic artifacts must be listed in the verified manifest"
            )
        if previous.get("cache_keys", {}).get("trace") == keys["trace"]:
            cached = (previous, previous_lines)
    if cached is not None:
        previous, lines = cached
        for name in ["source_surface_br.fits", "field_samples.npz", "seeds.csv"]:
            shutil.copyfile(Path(config["reuse_run"]) / name, output / name)
        ss = sunpy.map.Map(output / "source_surface_br.fits")
        receipt = dict(previous["receipt"])
        receipt.update(cache_reused_from=config["reuse_run"], elapsed_s=0)
        print("Reusing checksum-verified magnetic field and traces", flush=True)
    else:
        print("Solving PFSS", flush=True)
        field, receipt = solve_pfss(boundary, physics)
        ss = field.source_surface_br
        ss.save(output / "source_surface_br.fits", overwrite=False)
        # Portable public field samples; no pickle or private solver internals.
        nlon, nlat, nrad = 36, 18, 12
        lon, slat, rad = np.meshgrid(
            (np.arange(nlon) + 0.5) * 360 / nlon,
            -1 + (np.arange(nlat) + 0.5) * 2 / nlat,
            np.geomspace(1.001, physics.rss - 0.001, nrad),
            indexing="ij",
        )
        sample = SkyCoord(
            lon.ravel() * u.deg,
            np.arcsin(slat.ravel()) * u.rad,
            rad.ravel() * u.R_sun,
            frame=field.coordinate_frame,
        )
        bvec = field.get_bvec(sample, out_type="spherical")
        np.savez_compressed(
            output / "field_samples.npz",
            longitude_deg=lon,
            latitude_rad=np.arcsin(slat),
            radius_rsun=rad,
            b_spherical_G=bvec.to_value(u.G).reshape(lon.shape + (3,)),
        )
        seeds = uniform_seeds(field.coordinate_frame, nlon=16, nlat=16)
        if config.get("seed_hpc_box"):
            xmin, xmax, ymin, ymax = config["seed_hpc_box"]
            nlocal = int(config.get("local_seed_side", 8))
            if not 1 <= nlocal <= 16 or 256 + nlocal * nlocal > physics.max_seeds:
                raise ValueError("Local seed grid exceeds the configured seed budget")
            x, y = np.meshgrid(
                np.linspace(xmin, xmax, nlocal), np.linspace(ymin, ymax, nlocal)
            )
            foot = SkyCoord(
                x.ravel() * u.arcsec,
                y.ravel() * u.arcsec,
                frame=frames.Helioprojective(observer=observer, obstime=event),
            ).make_3d()
            foot = foot.transform_to(
                frames.HeliographicCarrington(observer="earth", obstime=event)
            )
            keep = np.isfinite(foot.radius)
            seeds = SkyCoord(
                np.r_[seeds.lon.to_value(u.deg), foot.lon[keep].to_value(u.deg)]
                * u.deg,
                np.r_[seeds.lat.to_value(u.deg), foot.lat[keep].to_value(u.deg)]
                * u.deg,
                1.001 * u.R_sun,
                frame=field.coordinate_frame,
            )
        pd.DataFrame(
            {
                "seed_index": np.arange(len(seeds)),
                "lon_deg": seeds.lon.to_value(u.deg),
                "lat_deg": seeds.lat.to_value(u.deg),
                "radius_rsun": seeds.radius.to_value(u.R_sun),
            }
        ).to_csv(output / "seeds.csv", index=False)
        print(f"Tracing {len(seeds)} explicit candidate seeds", flush=True)
        lines = trace_fieldlines(
            field,
            seeds,
            rss=physics.rss,
            max_seeds=physics.max_seeds,
            step_size=float(config.get("trace_step_size", 0.25)),
            max_steps=int(config.get("trace_max_steps", 4096)),
        )
    projections = project_fieldlines(
        lines, observer=aia.observer_coordinate, obstime=aia_mid
    )
    # Portable, WCS-preserving AIA cutout for display on hosts without raw data.
    corner = SkyCoord(
        [650, 1450] * u.arcsec, [-600, 0] * u.arcsec, frame=aia.coordinate_frame
    )
    aia.submap(corner[0], top_right=corner[1]).save(
        output / "aia_display.fits", overwrite=False
    )
    # Choose candidate subset on EUV alone, before opening the radio source table.
    candidates = []
    ranking = []
    axis = None
    if config.get("jet_axis_csv"):
        axis_df = pd.read_csv(config["jet_axis_csv"])
        xcol = next(
            c for c in axis_df if c in {"x_arcsec", "tx_arcsec", "axis_x_arcsec"}
        )
        ycol = next(
            c for c in axis_df if c in {"y_arcsec", "ty_arcsec", "axis_y_arcsec"}
        )
        axis = axis_df[[xcol, ycol]].to_numpy(float)
    for line, projected in zip(lines, projections, strict=True):
        score = np.inf
        if line["trace_valid"] and axis is not None:
            dist, _ = projected_distances(axis, projected["hpc_arcsec"])
            if dist.shape[1]:
                score = float(np.median(np.min(dist, axis=1)))
        ranking.append(
            dict(
                fieldline_id=line["fieldline_id"],
                euv_median_residual_arcsec=score,
                trace_valid=line["trace_valid"],
                classification=line["classification"],
            )
        )
    rank = pd.DataFrame(ranking).sort_values(
        "euv_median_residual_arcsec", kind="stable"
    )
    ids = set(
        rank[rank.trace_valid & np.isfinite(rank.euv_median_residual_arcsec)]
        .head(int(config.get("candidate_limit", 20)))
        .fieldline_id
    )
    if axis is None:
        ids = set(row["fieldline_id"] for row in lines if row["trace_valid"])
    candidates = [line for line in lines if line["fieldline_id"] in ids]
    rank["selected_for_radio_comparison"] = rank.fieldline_id.isin(ids)
    rank.to_csv(output / "euv_candidate_ranking.csv", index=False)
    frozen_ids = sorted(ids)
    (output / "candidate_selection.json").write_text(
        json.dumps(
            {
                "fieldline_ids": frozen_ids,
                "criterion": "EUV-only median projected distance; exploratory ranking, no identification threshold",
                "newkirk_used": False,
            },
            indent=2,
        )
    )
    selection_hash = file_hash(output / "candidate_selection.json")
    associations = []
    model_rows = []
    sources = pd.DataFrame()
    if config.get("sources_csv"):
        sources = pd.read_csv(config["sources_csv"])
        sources["time_utc"] = pd.to_datetime(
            sources.time_utc, utc=True, format="ISO8601"
        )
        sources_all = sources.copy()
        if sources.source_id.duplicated().any():
            raise ValueError("Source identifiers must be normalized and unique")
        sources = sources[sources.source_valid.map(truthy)].copy()
        sources = sources[
            (sources.time_utc >= pd.Timestamp(config["window_start_utc"]))
            & (sources.time_utc <= pd.Timestamp(config["window_end_utc"]))
        ]
        sources.to_csv(output / "sources.csv", index=False)
        shells = model_shell_candidates(
            candidates, sources.frequency_mhz, rss=physics.rss
        )
        usable = [row for row in shells if row["model_valid"]]
        shell_points = np.array([row["xyz_carrington_rsun"] for row in usable])
        print(
            f"Comparing {len(sources)} sources against {len(candidates)} fixed lines",
            flush=True,
        )
        for stamp, group in sources.groupby("time_utc"):
            instant = Time(stamp.to_pydatetime())
            obs = get_earth(instant)
            frame = frames.HeliographicCarrington(observer="earth", obstime=instant)
            proj = project_fieldlines(candidates, observer=obs, obstime=instant)
            chunk = associate_sources(
                group,
                candidates,
                proj,
                observer_carrington_rsun=obs.transform_to(frame).cartesian.xyz.to_value(
                    u.R_sun
                ),
            )
            if not chunk.empty:
                associations.append(chunk)
            if len(shell_points):
                coords = SkyCoord(
                    x=shell_points[:, 0] * u.R_sun,
                    y=shell_points[:, 1] * u.R_sun,
                    z=shell_points[:, 2] * u.R_sun,
                    representation_type="cartesian",
                    frame=frame,
                )
                hp = coords.transform_to(
                    frames.Helioprojective(observer=obs, obstime=instant)
                )
                predicted = np.column_stack(
                    [hp.Tx.to_value(u.arcsec), hp.Ty.to_value(u.arcsec)]
                )
                visibility = visible_from_observer(
                    shell_points,
                    obs.transform_to(frame).cartesian.xyz.to_value(u.R_sun),
                )
                for _, source in group.iterrows():
                    for i, candidate in enumerate(usable):
                        if candidate["frequency_mhz"] != source.frequency_mhz:
                            continue
                        model_rows.append(
                            {
                                k: v
                                for k, v in candidate.items()
                                if k != "xyz_carrington_rsun"
                            }
                            | dict(
                                source_id=source.source_id,
                                time_utc=str(stamp),
                                predicted_x_arcsec=float(predicted[i, 0]),
                                predicted_y_arcsec=float(predicted[i, 1]),
                                visible=bool(visibility[i]),
                                projected_residual_arcsec=(
                                    float(
                                        np.linalg.norm(
                                            predicted[i]
                                            - [
                                                source.center_x_arcsec,
                                                source.center_y_arcsec,
                                            ]
                                        )
                                    )
                                    if visibility[i]
                                    else np.nan
                                ),
                            )
                        )
        pd.DataFrame(
            [
                {k: v for k, v in row.items() if k != "xyz_carrington_rsun"}
                for row in shells
            ]
        ).to_csv(output / "model_domain_diagnostics.csv", index=False)
    association = (
        pd.concat(associations, ignore_index=True) if associations else pd.DataFrame()
    )
    models = pd.DataFrame(model_rows)
    association.to_csv(output / "source_fieldline_association.csv", index=False)
    if len(association):
        from solar_toolkit.radio.newkirk import newkirk_height_from_frequency_mhz

        projected_height = (
            np.hypot(sources.center_x_arcsec, sources.center_y_arcsec)
            / float(aia.rsun_obs.to_value(u.arcsec))
            - 1
        )
        legacy = sources[["source_id"]].assign(
            projected_height_rsun=projected_height.to_numpy()
        )
        conditional = []
        for m in (1, 2, 4):
            for h in (1, 2):
                a = association.merge(legacy, on="source_id", validate="many_to_one")
                a["multiplier"] = m
                a["harmonic"] = h
                a["newkirk_height_rsun"] = newkirk_height_from_frequency_mhz(
                    a.frequency_mhz.to_numpy(), m, h
                )
                a["conditional_height_residual_rsun"] = (
                    a.conditional_candidate_height_rsun - a.newkirk_height_rsun
                )
                a["height_change_from_projection_rsun"] = (
                    a.conditional_candidate_height_rsun - a.projected_height_rsun
                )
                conditional.append(a)
        pd.concat(conditional, ignore_index=True).to_csv(
            output / "conditional_height_comparison.csv", index=False
        )
    models.to_csv(output / "newkirk_forward_candidates.csv", index=False)
    ranks = pd.DataFrame()
    burst_ranks = pd.DataFrame()
    if not models.empty:
        # Minimum over intersections on each fixed line; never a different line per frequency.
        per_source = (
            models[models.visible]
            .groupby(
                ["source_id", "fieldline_id", "multiplier", "harmonic"], as_index=False
            )
            .projected_residual_arcsec.min()
        )
        coverage = per_source.groupby(["source_id", "fieldline_id"]).size()
        common = coverage[coverage == 6].index
        per_source = (
            per_source.set_index(["source_id", "fieldline_id"])
            .loc[lambda f: f.index.isin(common)]
            .reset_index()
        )
        ranks = (
            per_source.groupby(["fieldline_id", "multiplier", "harmonic"])
            .projected_residual_arcsec.agg(["median", "count", "max"])
            .reset_index()
            .sort_values("median")
        )
        if config.get("burst_matches_csv"):
            matches = pd.read_csv(config["burst_matches_csv"])
            matches.to_csv(output / "burst_matches_input.csv", index=False)
            matches = matches[matches.match_valid.map(truthy)]
            matched = per_source.merge(
                matches[["source_id", "burst_id"]], on="source_id", how="inner"
            )
            matched.to_csv(output / "burst_model_rows.csv", index=False)
            burst_ranks = (
                matched.groupby(["burst_id", "fieldline_id", "multiplier", "harmonic"])
                .projected_residual_arcsec.agg(["median", "count"])
                .reset_index()
            )
            burst_ranks["trajectory_supported"] = False
            burst_ranks["minimum_frequency_count_met"] = burst_ranks["count"] >= 3
        if config.get("drifts_csv"):
            from solar_toolkit.radio.source_geometry import match_burst_crossings

            drift = pd.read_csv(config["drifts_csv"])
            all_matches = []
            clock_ranks = []
            for offset in [-0.45, 0, 0.45]:
                matching = match_burst_crossings(
                    sources_all, drift, clock_offset_s=offset
                )
                all_matches.append(matching)
                valid = matching[
                    matching.match_valid & matching.source_id.isin(sources.source_id)
                ]
                joined = per_source.merge(
                    valid[["source_id", "burst_id"]], on="source_id"
                )
                if not joined.empty:
                    result = (
                        joined.groupby(
                            ["burst_id", "fieldline_id", "multiplier", "harmonic"]
                        )
                        .projected_residual_arcsec.agg(["median", "count"])
                        .reset_index()
                    )
                    result["clock_offset_s"] = offset
                    result["clock_offset_verified"] = False
                    result["trajectory_supported"] = False
                    clock_ranks.append(result)
            pd.concat(all_matches, ignore_index=True).to_csv(
                output / "clock_sensitivity_matches.csv", index=False
            )
            if clock_ranks:
                pd.concat(clock_ranks, ignore_index=True).to_csv(
                    output / "clock_sensitivity_ranking.csv", index=False
                )
    ranks.to_csv(output / "newkirk_forward_ranking.csv", index=False)
    burst_ranks.to_csv(output / "burst_forward_ranking.csv", index=False)
    assert selection_hash == file_hash(output / "candidate_selection.json")
    from .render import render

    render(
        output,
        boundary,
        ss,
        aia,
        lines,
        projections,
        sources,
        axis,
        rank,
        ranks,
        association,
        models,
        event,
    )
    inputs = [
        {"role": key, "path": config[key], "sha256": file_hash(config[key])}
        for key in [
            "boundary_fits",
            "aia_fits",
            "sources_csv",
            "burst_matches_csv",
            "jet_axis_csv",
            "drifts_csv",
        ]
        if config.get(key)
    ]
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    import importlib.metadata
    import resource

    receipt.update(
        host=platform.node(),
        python=platform.python_version(),
        elapsed_total_s=time.monotonic() - start,
        peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        dependencies={
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "scipy",
                "sunpy",
                "astropy",
                "matplotlib",
                "pandas",
                "sunkit-magex",
                "streamtracer",
            ]
        },
    )
    code_roots = [
        Path(__file__).parent,
        Path(
            __import__("solar_toolkit.modeling.pfss", fromlist=["__file__"]).__file__
        ).parent,
    ]
    receipt["code_sha256"] = {
        str(p.relative_to(root.parent)): file_hash(p)
        for root in code_roots
        for p in sorted(root.glob("*.py"))
    }
    metadata = dict(
        obstime=event.utc.isot,
        aia_midpoint_utc=aia_mid.utc.isot,
        aia_projection_time_difference_s=float((aia_mid - event).to_value(u.s)),
        carrington_rotation=float(sun.carrington_rotation_number(event)),
        config=asdict(physics),
        receipt=receipt,
        inputs=inputs,
        boundary_audit=audit,
        cache_keys=keys,
        candidate_selection_sha256=selection_hash,
        selected_fieldline_ids=frozen_ids,
        projection_obstime=aia_mid.utc.isot,
        projection_observer="AIA FITS observer",
        source_count=len(sources),
        independent_source_height_count=0,
        limitations=[
            "PFSS geometry and Newkirk shells are conditional models, not independent 3D positions",
            "EUV rank is not a verified jet association",
            "Source covariance and global clock offset remain unconfirmed",
            "Frozen Carrington synoptic pattern; the boundary is not simultaneous with the event",
            "Portable field samples are not a lossless restart of the backend solver; changing seeds across processes currently rebuilds the solution",
        ],
    )
    save_bundle(output, lines, projections, metadata)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "fieldlines": len(lines),
                "valid_lines": sum(row["trace_valid"] for row in lines),
                "sources": len(sources),
                "elapsed_s": receipt["elapsed_total_s"],
            }
        ),
        flush=True,
    )
    return metadata

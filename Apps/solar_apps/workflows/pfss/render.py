"""Static scientific figures rendered on the compute host."""


def render(
    directory,
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
):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm, AsinhNorm
    from astropy import units as u

    colors = {
        "open_positive": "#d94842",
        "open_negative": "#2877b5",
        "closed": "#6a9b45",
        "failed": "#999999",
    }

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m, title in zip(
        axes, [boundary, ss], ["Prepared global Br", "Source surface Br"]
    ):
        vmax = np.percentile(abs(m.data), 99)
        left = float(m.meta["crval1"]) + (0.5 - float(m.meta["crpix1"])) * float(
            m.scale.axis1.to_value(u.deg / u.pix)
        )
        left = left % 360
        extent = [left, left + 360, -1, 1]
        im = ax.imshow(
            m.data,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            norm=SymLogNorm(max(vmax / 100, 0.001), vmin=-vmax, vmax=vmax),
        )
        ax.set(
            xlabel="Carrington longitude (deg; grid orientation)",
            ylabel="sin(latitude)",
            title=title,
        )
        # Every zero-contour component is retained.
        ax.contour(
            m.data,
            levels=[0],
            origin="lower",
            extent=extent,
            colors="black",
            linewidths=0.3,
        )
        fig.colorbar(im, ax=ax, label="G")
    fig.suptitle(
        "Global PFSS boundary; synoptic assembly is not an event-time snapshot"
    )
    save(fig, "global_field")
    for include_radio in [False, True]:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(projection=aia)
        normalized = aia.data / float(aia.meta["exptime"])
        ax.imshow(
            normalized,
            origin="lower",
            cmap="sdoaia171",
            norm=AsinhNorm(
                linear_width=30, vmin=0, vmax=float(np.nanpercentile(normalized, 99.7))
            ),
        )
        world = ax.get_transform("world")
        chosen = set(rank[rank.selected_for_radio_comparison].fieldline_id)
        for line, projected in zip(lines, projections, strict=True):
            if line["fieldline_id"] not in chosen:
                continue
            xy = projected["hpc_arcsec"] / 3600
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                transform=world,
                color=colors[line["classification"]],
                alpha=0.5,
                lw=0.8,
            )
        if axis is not None:
            ax.plot(
                axis[:, 0] / 3600,
                axis[:, 1] / 3600,
                transform=world,
                color="#ff5296",
                lw=3,
                label="Observed EUV corridor (not a footpoint)",
            )
        if include_radio and len(sources):
            sc = ax.scatter(
                sources.center_x_arcsec / 3600,
                sources.center_y_arcsec / 3600,
                transform=world,
                c=sources.frequency_mhz,
                cmap="viridis",
                s=12,
                edgecolor="white",
                linewidth=0.2,
            )
            fig.colorbar(sc, ax=ax, label="Radio frequency (MHz)")
        # Use WCS conversion for the plot window; no linear display-grid shortcut.
        from astropy.coordinates import SkyCoord

        a, b = aia.world_to_pixel(
            SkyCoord(
                [700, 1370] * u.arcsec,
                [-500, -60] * u.arcsec,
                frame=aia.coordinate_frame,
            )
        )
        ax.set_xlim(a.value)
        ax.set_ylim(b.value)
        ax.set_title(
            f"AIA + conditional PFSS{' + radio' if include_radio else ''}\nEvent reference {event.utc.isot} UTC"
        )
        if axis is not None:
            ax.legend(loc="lower left", fontsize=8)
        save(fig, "aia_pfss_radio" if include_radio else "aia_pfss")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    for line in lines:
        xyz = line["xyz_carrington_rsun"]
        if line["trace_valid"]:
            ax.plot(*xyz.T, color=colors[line["classification"]], lw=0.5, alpha=0.5)
    ax.set(
        xlabel="Carrington X / R_sun",
        ylabel="Y / R_sun",
        zlabel="Z / R_sun",
        title="Conditional global PFSS field lines",
    )
    ax.set_box_aspect([1, 1, 1])
    save(fig, "fieldlines_3d")
    if len(ranks):
        fig, ax = plt.subplots(figsize=(10, 5))
        selected = rank[rank.selected_for_radio_comparison].fieldline_id.head(5)
        labels = ["1xH1", "1xH2", "2xH1", "2xH2", "4xH1", "4xH2"]
        for line_id in selected:
            g = ranks[ranks.fieldline_id == line_id].set_index(
                ["multiplier", "harmonic"]
            )
            ax.plot(
                labels,
                [
                    g.loc[(m, s), "median"] if (m, s) in g.index else np.nan
                    for m in (1, 2, 4)
                    for s in (1, 2)
                ],
                "o-",
                label=line_id,
            )
        ax.set(
            ylabel="Median forward residual (arcsec)",
            title="Fixed EUV-selected lines + Newkirk shells: conditional comparison",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, "newkirk_residuals")
    if len(association):
        fig, ax = plt.subplots(figsize=(10, 5))
        for line_id in rank[rank.selected_for_radio_comparison].fieldline_id.head(5):
            a = (
                association[association.fieldline_id == line_id]
                .sort_values("projected_residual_arcsec")
                .drop_duplicates("source_id")
            )
            stats = a.groupby(
                "frequency_mhz"
            ).conditional_candidate_height_rsun.median()
            ax.plot(stats.index, stats.values, "o-", label=line_id)
        from solar_toolkit.radio.newkirk import newkirk_height_from_frequency_mhz

        f = np.linspace(sources.frequency_mhz.min(), sources.frequency_mhz.max(), 200)
        for m, s in [(1, 1), (2, 1), (4, 1), (2, 2), (4, 2)]:
            ax.plot(
                f,
                newkirk_height_from_frequency_mhz(f, m, s),
                "--",
                lw=1,
                label=f"{m}xH{s}" + (" = 1xH2" if (m, s) == (4, 1) else ""),
            )
        ax.invert_xaxis()
        ax.set(
            xlabel="Frequency (MHz)",
            ylabel="Height above photosphere / R_sun",
            title="Nearest field-line candidate heights; source association UNVERIFIED",
        )
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
        save(fig, "height_frequency")

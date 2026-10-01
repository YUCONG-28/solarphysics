"""Application adapters for radio fitting and coordinate reprojection.

Functions retain the historical core dependency hooks; scientific operations
and product ordering are unchanged.
"""

# ruff: noqa: F401, I001

from ._overlay_context import (
    Config,
    GAUSSIAN_DIAGNOSTIC_FIELDS,
    GaussianFitResult,
    GaussianReprojectResult,
    IntArray,
    RawRadioReprojectResult,
    RegularGridInterpolator,
    SkyCoord,
    _canonical_data_coord_to_pixel,
    _canonical_elliptical_gaussian_2d,
    _canonical_estimate_background_noise,
    _canonical_gaussian_fit_diag_defaults,
    _canonical_gaussian_quality_config,
    _canonical_limit_fit_pixels,
    _canonical_pixel_to_data_coord,
    _canonical_roi_slices_from_mask,
    _canonical_safe_rms_map,
    _canonical_select_peak_connected_mask,
    _canonical_true_indices,
    _canonical_unravel_2d_index,
    _gaussian_diagnostics_row,
    _normalise_band_key,
    _radec_file_cache,
    binary_dilation,
    csv,
    curve_fit,
    datetime,
    fits,
    gaussian_filter,
    glob,
    griddata,
    math,
    normalize_roi_bounds_arcsec,
    np,
    os,
    re,
    sunpy,
    u,
    warnings,
)


def _workflow():
    """Resolve historical core replacement hooks only when a function runs."""

    from . import _overlay_workflow_core

    return _overlay_workflow_core


def _combine_polarization_data(
    rr_data: np.ndarray, ll_data: np.ndarray, cfg: Config
) -> np.ndarray:
    """组合RR和LL数据（加权平均或简单相加）"""
    if cfg.weighted_average:
        combined_data = rr_data * cfg.rr_weight + ll_data * cfg.ll_weight
    else:
        combined_data = rr_data + ll_data
    return combined_data


def get_solar_position(obs_time: datetime) -> tuple[float, float]:
    """计算观测时刻太阳中心在 ICRS 坐标系中的位置（RA, Dec，单位度）"""
    try:
        from astropy.coordinates import get_sun
        from astropy.time import Time

        sun_coord = get_sun(Time(obs_time, format="datetime", scale="utc"))
        return sun_coord.ra.deg, sun_coord.dec.deg
    except Exception as e:
        print(f"    [警告] 使用 astropy 计算太阳位置失败: {e}")
        return 306.413395, -19.231661


def _find_radec_file(search_dirs: list[str], freq_value: str, kind: str) -> str | None:
    """
    查找赤经（kind='ra'）或赤纬（kind='dec'）坐标文件，结果缓存避免重复搜索。
    """
    if kind == "ra":
        patterns = [
            f"{freq_value}MHz_RightAscensionDegree.fits",
            f"{freq_value}MHz_RA.fits",
            f"*{freq_value}*RightAscension*.fits",
            f"*{freq_value}*RA*.fits",
        ]
    else:
        patterns = [
            f"{freq_value}MHz_DeclinationDegree.fits",
            f"{freq_value}MHz_Dec.fits",
            f"*{freq_value}*Declination*.fits",
            f"*{freq_value}*Dec*.fits",
        ]

    for sd in search_dirs:
        # 使用绝对路径作为缓存 key，避免相对路径引发的歧义
        abs_sd = os.path.abspath(sd)
        cache_key = (abs_sd, freq_value, kind)

        # ─── 修复核心逻辑 ──────────────────────────────
        if cache_key in _workflow()._radec_file_cache:
            if _workflow()._radec_file_cache[cache_key] is not None:
                return _workflow()._radec_file_cache[cache_key]  # 找到了，直接返回
            else:
                continue  # 缓存显示当前目录没有，继续去下一个目录找！

        if not os.path.isdir(abs_sd):
            _workflow()._radec_file_cache[cache_key] = None
            continue

        found = None
        for pat in patterns:
            matches = glob.glob(os.path.join(abs_sd, pat))
            if matches:
                found = matches[0]
                break

        _workflow()._radec_file_cache[cache_key] = found
        if found:
            return found

    return None


def _load_fits_2d(path: str, use_float32: bool) -> np.ndarray | None:
    """读取 FITS 并压缩到 2D，失败返回 None"""
    try:
        with _workflow().fits.open(path) as hdu:
            data = hdu[0].data
            while data.ndim > 2:
                data = data[0]
            data = np.squeeze(data)
            return data.astype(np.float32 if use_float32 else np.float64)
    except Exception:
        return None


def extract_radio_2d_data(
    fits_path: str,
    use_float32: bool = True,
    cfg: Config | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    fits.Header | None,
    None,
]:
    """
    提取射电强度数据及配套的赤经赤纬坐标图。

    返回
    ----
    (radio_data_2d, ra_map, dec_map, header, None)

    赤经赤纬坐标图（ra_map / dec_map）：
        与 radio_data_2d 像素一一对应的坐标映射图，
        每个像素存储该点天球 ICRS 赤经/赤纬值（度）。
        用于后续精确坐标变换（ICRS → HPC）。
    """
    try:
        with _workflow().fits.open(fits_path) as hdu:
            data = hdu[0].data
            while data.ndim > 2:
                data = data[0]
            data = np.squeeze(data)
            header = hdu[0].header.copy()

        dtype = np.float32 if use_float32 else np.float64
        data = data.astype(dtype)

        ra_map = dec_map = None

        if cfg is not None and cfg.use_radec_maps:
            base_dir = os.path.dirname(fits_path)
            base_name = os.path.basename(fits_path)

            # 提取频率值
            m = re.search(r"(\d+)MHz", base_name, re.IGNORECASE)
            if not m:
                m = re.search(r"(\d+)MHz", os.path.basename(base_dir), re.IGNORECASE)
            freq_value = m.group(1) if m else None

            if freq_value:
                search_dirs = [
                    base_dir,
                    os.path.dirname(base_dir),
                    os.path.join(os.path.dirname(base_dir), ".."),
                ]
                ra_path = _workflow()._find_radec_file(search_dirs, freq_value, "ra")
                dec_path = _workflow()._find_radec_file(search_dirs, freq_value, "dec")

                if ra_path:
                    ra_map = _workflow()._load_fits_2d(ra_path, use_float32)
                    if cfg.debug_mode and ra_map is not None:
                        print(
                            f"    [坐标图] 已加载 RA 文件: {os.path.basename(ra_path)}"
                        )
                if dec_path:
                    dec_map = _workflow()._load_fits_2d(dec_path, use_float32)
                    if cfg.debug_mode and dec_map is not None:
                        print(
                            f"    [坐标图] 已加载 Dec 文件: {os.path.basename(dec_path)}"
                        )

                if cfg.debug_mode:
                    if ra_map is not None and dec_map is not None:
                        print("    [坐标图] 成功加载赤经赤纬坐标图")

                        ra_valid = ra_map[np.isfinite(ra_map)]
                        dec_valid = dec_map[np.isfinite(dec_map)]
                        if len(ra_valid) > 0:
                            print("\n" + "=" * 30)
                            print("【坐标原始数值探针 - 诊断用】")
                            print(
                                f"RA 范围:  {np.nanmin(ra_valid):.4f} 至 {np.nanmax(ra_valid):.4f}"
                            )
                            print(
                                f"Dec 范围: {np.nanmin(dec_valid):.4f} 至 {np.nanmax(dec_valid):.4f}"
                            )
                            print(f"检测到为 0 的异常像素数: {np.sum(ra_map == 0)}")
                            print("=" * 30 + "\n")
                    else:
                        print("    [坐标图] 警告: 未找到完整的坐标图文件")

        return data, ra_map, dec_map, header, None

    except Exception as e:
        print(f"读取 FITS 文件失败 {fits_path}: {e}")
        return None, None, None, None, None


def compute_contour_levels(data: np.ndarray, cfg: Config) -> list[float]:
    """直接基于高斯模型的峰值计算等值线级别"""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return []
    peak = float(np.nanmax(finite))
    # 直接返回峰值的百分比（例如 90%）
    return [f * peak for f in cfg.contour_levels_peak]


def elliptical_gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta):
    """
    二维椭圆高斯函数
    参数：
        A      : 峰值振幅
        x0, y0 : 中心位置（质心）
        sigma_x, sigma_y : 沿长轴和短轴的 rms 宽度
        theta  : 长轴相对于 x 轴的角度（弧度）
    返回：对应坐标的高斯值
    """
    return _workflow()._canonical_elliptical_gaussian_2d(
        xy, A, x0, y0, sigma_x, sigma_y, theta
    )


def _unravel_2d_index(
    flat_index: int | np.integer, shape: tuple[int, ...]
) -> tuple[int, int]:
    return _workflow()._canonical_unravel_2d_index(flat_index, shape)


def _true_indices(mask: np.ndarray) -> IntArray:
    return _workflow()._canonical_true_indices(mask)


def fit_elliptical_gaussian(data, x, y, initial_guess=None):
    """
    拟合二维椭圆高斯到图像数据

    输入：
        data : 2D numpy数组，图像强度
        x    : 1D numpy数组，x坐标（长度 = data.shape[1]）
        y    : 1D numpy数组，y坐标（长度 = data.shape[0]）
        initial_guess : 可选，初始参数 (A, x0, y0, sigma_x, sigma_y, theta)

    输出：
        popt : 拟合参数 [A, x0, y0, sigma_x, sigma_y, theta]
        pcov : 协方差矩阵
    """
    X, Y = np.meshgrid(x, y)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    data_flat = data.ravel()

    # 如果没有提供初始猜测，自动估计
    if initial_guess is None:
        max_y, max_x = _workflow()._unravel_2d_index(int(np.argmax(data)), data.shape)
        init_x0 = x[max_x]
        init_y0 = y[max_y]
        init_A = np.max(data)

        # 粗略估计 sigma（通过半高宽）
        half_max = init_A / 2.0
        # x方向
        row_max = data[max_y, :]
        indices = _workflow()._true_indices(row_max >= half_max)
        if len(indices) > 1:
            init_sigma_x = (x[indices[-1]] - x[indices[0]]) / (2.355)  # FWHM -> sigma
        else:
            init_sigma_x = (x[-1] - x[0]) / 10.0
        # y方向
        col_max = data[:, max_x]
        indices_y = _workflow()._true_indices(col_max >= half_max)
        if len(indices_y) > 1:
            init_sigma_y = (y[indices_y[-1]] - y[indices_y[0]]) / (2.355)
        else:
            init_sigma_y = (y[-1] - y[0]) / 10.0
        init_theta = 0.0
        initial_guess = (
            init_A,
            init_x0,
            init_y0,
            init_sigma_x,
            init_sigma_y,
            init_theta,
        )

    # 参数边界
    bounds = (
        [0, -np.inf, -np.inf, 1e-3, 1e-3, -np.pi / 2],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi / 2],
    )

    popt, pcov = _workflow().curve_fit(
        _workflow().elliptical_gaussian_2d,
        (x_flat, y_flat),
        data_flat,
        p0=initial_guess,
        bounds=bounds,
        maxfev=5000,
    )
    return popt, pcov


def elliptical_gaussian_2d_with_constant_bg(xy, A, x0, y0, sigma_x, sigma_y, theta, b0):
    return (
        _workflow().elliptical_gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta) + b0
    )


def elliptical_gaussian_2d_with_plane_bg(
    xy, A, x0, y0, sigma_x, sigma_y, theta, b0, bx, by
):
    x, y = xy
    return (
        _workflow().elliptical_gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta)
        + b0
        + bx * x
        + by * y
    )


def gaussian_only_from_popt(xy, popt, background_model):
    return _workflow().elliptical_gaussian_2d(xy, *popt[:6])


def estimate_background_noise(data, source_exclusion_mask=None):
    exclusion = (
        None
        if source_exclusion_mask is None
        else np.asarray(source_exclusion_mask, dtype=np.bool_)
    )
    return _workflow()._canonical_estimate_background_noise(data, exclusion)


def _robust_median_mad(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.nanstd(arr))
    return med, rms


def _sigma_clip_values(values, sigma=3.0, iters=3):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    for _ in range(max(int(iters), 0)):
        med, rms = _workflow()._robust_median_mad(arr)
        if not np.isfinite(med) or not np.isfinite(rms) or rms <= 0:
            break
        keep = np.abs(arr - med) <= float(sigma) * rms
        if np.count_nonzero(keep) == arr.size:
            break
        arr = arr[keep]
        if arr.size == 0:
            break
    return arr


def _safe_rms_map(rms_map):
    # Overlay historically enforced an additional positive numerical floor.
    return np.maximum(_workflow()._canonical_safe_rms_map(rms_map), 1e-12)


def _mesh_values_to_map(mesh_y, mesh_x, mesh_values, shape, fill_value):
    ny, nx = shape
    if len(mesh_values) == 0:
        return np.full(shape, fill_value, dtype=np.float64)
    mesh_y = np.asarray(mesh_y, dtype=np.float64)
    mesh_x = np.asarray(mesh_x, dtype=np.float64)
    values = np.asarray(mesh_values, dtype=np.float64)
    valid = np.isfinite(mesh_y) & np.isfinite(mesh_x) & np.isfinite(values)
    mesh_y = mesh_y[valid]
    mesh_x = mesh_x[valid]
    values = values[valid]
    if values.size == 0:
        return np.full(shape, fill_value, dtype=np.float64)
    y_unique = np.unique(mesh_y)
    x_unique = np.unique(mesh_x)
    grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=np.float64)
    y_lookup = {v: i for i, v in enumerate(y_unique)}
    x_lookup = {v: i for i, v in enumerate(x_unique)}
    for yv, xv, val in zip(mesh_y, mesh_x, values, strict=False):
        grid[y_lookup[yv], x_lookup[xv]] = val
    global_fill = float(np.nanmedian(values)) if values.size else fill_value
    if not np.isfinite(global_fill):
        global_fill = fill_value
    grid = np.where(np.isfinite(grid), grid, global_fill)
    x_pixels = np.arange(nx, dtype=np.float64)
    y_pixels = np.arange(ny, dtype=np.float64)
    row_interp = np.empty((grid.shape[0], nx), dtype=np.float64)
    for i in range(grid.shape[0]):
        row_interp[i, :] = np.interp(x_pixels, x_unique, grid[i, :])
    out = np.empty((ny, nx), dtype=np.float64)
    for j in range(nx):
        out[:, j] = np.interp(y_pixels, y_unique, row_interp[:, j])
    return out


def estimate_background_rms_mesh(data, cfg, source_mask=None):
    work = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(work)
    finite_values = work[finite]
    global_bg, global_rms = _workflow()._robust_median_mad(finite_values)
    if not np.isfinite(global_bg):
        global_bg = 0.0
    if not np.isfinite(global_rms) or global_rms <= 0:
        global_rms = 1.0
    mesh_size = max(int(cfg.get("background_mesh_size", 96)), 1)
    mesh_step = max(int(cfg.get("background_mesh_step", mesh_size)), 1)
    min_valid = max(int(cfg.get("background_min_valid_pixels", 20)), 1)
    exclude = np.zeros(work.shape, dtype=np.bool_)
    if source_mask is not None and np.asarray(source_mask).shape == work.shape:
        exclude = np.asarray(source_mask, dtype=np.bool_)
    diagnostics = {
        "background_rms_median": np.nan,
        "background_level_median": np.nan,
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "mesh_count": 0,
        "warning": "",
    }
    if work.ndim != 2 or finite_values.size == 0:
        diagnostics["warning"] = "non_finite_data"
        bg = np.full_like(work, global_bg, dtype=np.float64)
        rms = np.full_like(work, global_rms, dtype=np.float64)
        return bg, _workflow()._safe_rms_map(rms), diagnostics
    ny, nx = work.shape
    mesh_y, mesh_x, bg_values, rms_values = [], [], [], []
    for y0 in range(0, ny, mesh_step):
        y1 = min(y0 + mesh_size, ny)
        for x0 in range(0, nx, mesh_step):
            x1 = min(x0 + mesh_size, nx)
            box = work[y0:y1, x0:x1]
            valid = np.isfinite(box) & ~exclude[y0:y1, x0:x1]
            clipped = _workflow()._sigma_clip_values(
                box[valid],
                sigma=float(cfg.get("background_sigma_clip", 3.0)),
                iters=int(cfg.get("background_sigma_clip_iters", 3)),
            )
            if clipped.size < min_valid:
                continue
            bg, rms = _workflow()._robust_median_mad(clipped)
            if not np.isfinite(bg) or not np.isfinite(rms) or rms <= 0:
                continue
            mesh_y.append(0.5 * (y0 + y1 - 1))
            mesh_x.append(0.5 * (x0 + x1 - 1))
            bg_values.append(bg)
            rms_values.append(rms)
    if not bg_values:
        diagnostics["warning"] = "mesh_insufficient; fallback_global_median_mad"
        background_map = np.full_like(work, global_bg, dtype=np.float64)
        rms_map = np.full_like(work, global_rms, dtype=np.float64)
    else:
        background_map = _workflow()._mesh_values_to_map(
            mesh_y, mesh_x, bg_values, work.shape, global_bg
        )
        rms_map = _workflow()._mesh_values_to_map(
            mesh_y, mesh_x, rms_values, work.shape, global_rms
        )
    rms_map = np.maximum(
        _workflow()._safe_rms_map(rms_map),
        float(cfg.get("background_rms_floor", 1e-12)),
    )
    diagnostics.update(
        {
            "background_rms_median": float(np.nanmedian(rms_map[np.isfinite(rms_map)])),
            "background_level_median": float(
                np.nanmedian(background_map[np.isfinite(background_map)])
            ),
            "mesh_count": int(len(bg_values)),
        }
    )
    return background_map, rms_map, diagnostics


def _select_peak_connected_mask(
    source_mask_bool,
    grow_mask,
    peak_y,
    peak_x,
    use_snr=False,
    snr_map=None,
    work=None,
):
    return _workflow()._canonical_select_peak_connected_mask(
        source_mask_bool,
        grow_mask,
        peak_y,
        peak_x,
        use_snr=use_snr,
        snr_map=snr_map,
        work=work,
    )


def create_source_mask(data, cfg, background_map=None, rms_map=None):
    """Build the overlay-calibrated source mask.

    This remains overlay-owned intentionally: its default peak/growth fractions,
    target mask sizes, dilation, and minimum pixel count are tuned differently
    from :mod:`solar_toolkit.radio.gaussian_masks`. Direct delegation would alter
    which radio source pixels enter the scientific fit.
    """

    work = np.asarray(data, dtype=np.float64)
    finite_data = work[np.isfinite(work)]
    diagnostics = {
        "quality_flag": "ok",
        "background_level": np.nan,
        "noise_sigma": np.nan,
        "threshold": np.nan,
        "peak": np.nan,
        "mask_pixel_count": 0,
        "source_snr_peak": np.nan,
        "source_snr_mean": np.nan,
        "background_rms_median": np.nan,
        "background_level_median": np.nan,
        "mask_method": "raw_threshold",
        "fit_peak_fraction_threshold_used": np.nan,
        "fit_peak_fraction_candidate_counts": "",
    }
    if finite_data.size == 0:
        diagnostics["quality_flag"] = "non_finite_data"
        return None, diagnostics
    peak = float(np.max(finite_data))
    if not np.isfinite(peak):
        diagnostics["quality_flag"] = "non_finite_data"
        return None, diagnostics
    use_snr = False
    snr_map = None
    if (
        cfg.get("background_use_for_mask", True)
        and background_map is not None
        and rms_map is not None
    ):
        bg = np.asarray(background_map, dtype=np.float64)
        rms = _workflow()._safe_rms_map(rms_map)
        use_snr = bg.shape == work.shape and rms.shape == work.shape
    else:
        bg = rms = None
    finite_peak_work = np.where(np.isfinite(work), work, -np.inf)
    peak_y, peak_x = _workflow()._unravel_2d_index(
        int(np.argmax(finite_peak_work)), work.shape
    )
    base_peak_fraction = float(cfg.get("fit_peak_fraction_threshold", 0.40))
    grow_peak_fraction = float(cfg.get("fit_grow_peak_fraction_threshold", 0.22))
    min_peak_fraction = float(
        cfg.get("fit_peak_fraction_threshold_min", base_peak_fraction)
    )
    max_peak_fraction = float(
        cfg.get("fit_peak_fraction_threshold_max", base_peak_fraction)
    )
    step_peak_fraction = abs(float(cfg.get("fit_peak_fraction_threshold_step", 0.03)))
    min_peak_fraction, max_peak_fraction = sorted(
        (min_peak_fraction, max_peak_fraction)
    )
    base_peak_fraction = min(
        max(base_peak_fraction, min_peak_fraction), max_peak_fraction
    )
    grow_peak_fraction = min(max(grow_peak_fraction, 0.0), base_peak_fraction)
    target_min_pixels = int(cfg.get("fit_mask_target_min_pixels", 18))
    target_max_pixels = int(cfg.get("fit_mask_target_max_pixels", 260))
    if target_max_pixels < target_min_pixels:
        target_min_pixels, target_max_pixels = target_max_pixels, target_min_pixels
    if use_snr:
        snr_map = (work - bg) / rms
        fit_threshold = float(cfg.get("fit_snr_threshold", 5.0))
        grow_threshold = float(cfg.get("fit_grow_snr_threshold", 3.0))
        diagnostics.update(
            {
                "background_level": float(np.nanmedian(bg[np.isfinite(bg)])),
                "noise_sigma": float(np.nanmedian(rms[np.isfinite(rms) & (rms > 0)])),
                "threshold": fit_threshold,
                "background_rms_median": float(
                    np.nanmedian(rms[np.isfinite(rms) & (rms > 0)])
                ),
                "background_level_median": float(np.nanmedian(bg[np.isfinite(bg)])),
                "mask_method": "snr_mesh",
            }
        )
    else:
        background_level, noise_sigma = _workflow().estimate_background_noise(work)
        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            noise_sigma = max(float(np.std(finite_data)), 1e-12)
        diagnostics.update(
            {
                "background_level": background_level,
                "noise_sigma": noise_sigma,
                "background_rms_median": noise_sigma,
                "background_level_median": background_level,
                "mask_method": "raw_threshold",
            }
        )

    def build_mask_for_peak_fraction(peak_fraction):
        intensity_threshold = float(peak_fraction * peak)
        grow_intensity_threshold = float(grow_peak_fraction * peak)
        if use_snr:
            core_mask = (
                np.isfinite(snr_map)
                & (snr_map >= fit_threshold)
                & np.isfinite(work)
                & (work > intensity_threshold)
            )
            grow_mask_local = (
                np.isfinite(snr_map)
                & (snr_map >= grow_threshold)
                & np.isfinite(work)
                & (work > grow_intensity_threshold)
            )
            threshold_used = fit_threshold
        else:
            noise_sigma = diagnostics["noise_sigma"]
            threshold_used = max(
                float(cfg.get("fit_snr_threshold", 5.0)) * noise_sigma,
                intensity_threshold,
            )
            core_mask = np.isfinite(work) & (work > threshold_used)
            grow_threshold_used = max(
                float(cfg.get("fit_grow_snr_threshold", 3.0)) * noise_sigma,
                grow_intensity_threshold,
            )
            grow_mask_local = np.isfinite(work) & (work > grow_threshold_used)
        if not np.any(core_mask):
            return None, 0, threshold_used
        candidate_mask = _select_peak_connected_mask(
            core_mask,
            grow_mask_local,
            peak_y,
            peak_x,
            use_snr=use_snr,
            snr_map=snr_map,
            work=work,
        )
        dilation_pixels = int(cfg.get("fit_mask_dilation_pixels", 1))
        if dilation_pixels > 0:
            candidate_mask = binary_dilation(candidate_mask, iterations=dilation_pixels)
        return (
            np.asarray(candidate_mask, dtype=np.bool_),
            int(np.count_nonzero(candidate_mask)),
            threshold_used,
        )

    if step_peak_fraction <= 0:
        candidate_fractions = [base_peak_fraction]
    else:
        candidate_fractions = []
        frac = min_peak_fraction
        while frac <= max_peak_fraction + 1e-12:
            candidate_fractions.append(round(frac, 10))
            frac += step_peak_fraction
        if candidate_fractions[-1] < max_peak_fraction - 1e-12:
            candidate_fractions.append(max_peak_fraction)
    candidates = []
    for candidate_fraction in sorted(set(candidate_fractions)):
        candidate_mask, candidate_count, candidate_threshold = (
            build_mask_for_peak_fraction(candidate_fraction)
        )
        candidates.append(
            {
                "fraction": float(candidate_fraction),
                "mask": candidate_mask,
                "count": int(candidate_count),
                "threshold": float(candidate_threshold),
            }
        )
    target_mid_pixels = 0.5 * (target_min_pixels + target_max_pixels)
    in_target_candidates = [
        item
        for item in candidates
        if item["mask"] is not None
        and target_min_pixels <= item["count"] <= target_max_pixels
    ]
    usable_candidates = [
        item
        for item in candidates
        if item["mask"] is not None
        and item["count"] >= int(cfg.get("fit_min_mask_pixels", 12))
    ]
    nonempty_candidates = [
        item for item in candidates if item["mask"] is not None and item["count"] > 0
    ]
    if in_target_candidates:
        selected = min(
            in_target_candidates,
            key=lambda item: (
                abs(item["count"] - target_mid_pixels),
                abs(item["fraction"] - base_peak_fraction),
            ),
        )
    elif usable_candidates:
        selected = min(
            usable_candidates,
            key=lambda item: (
                abs(item["count"] - target_min_pixels),
                abs(item["fraction"] - base_peak_fraction),
            ),
        )
    elif nonempty_candidates:
        selected = max(nonempty_candidates, key=lambda item: item["count"])
    else:
        selected = {
            "fraction": base_peak_fraction,
            "mask": None,
            "count": 0,
            "threshold": np.nan,
        }
    diagnostics.update(
        {
            "fit_peak_fraction_threshold_used": float(selected["fraction"]),
            "fit_peak_fraction_candidate_counts": ";".join(
                f"{item['fraction']:.3f}:{item['count']}" for item in candidates
            ),
            "threshold": selected["threshold"],
            "peak": peak,
            "mask_pixel_count": int(selected["count"]),
        }
    )
    main_mask = selected["mask"]
    if main_mask is None or not np.any(main_mask):
        diagnostics["quality_flag"] = "mask_too_small"
        return None, diagnostics
    if use_snr and snr_map is not None:
        source_snr = snr_map[main_mask & np.isfinite(snr_map)]
        diagnostics["source_snr_peak"] = (
            float(np.nanmax(source_snr)) if source_snr.size else np.nan
        )
        diagnostics["source_snr_mean"] = (
            float(np.nanmean(source_snr)) if source_snr.size else np.nan
        )
    if int(selected["count"]) < int(cfg.get("fit_min_mask_pixels", 12)):
        diagnostics["quality_flag"] = "mask_too_small"
    return np.asarray(main_mask, dtype=np.bool_), diagnostics


def _gaussian_fit_diag_defaults(cfg):
    return _workflow()._canonical_gaussian_fit_diag_defaults(cfg)


def _roi_slices_from_mask(mask, shape, padding):
    return _workflow()._canonical_roi_slices_from_mask(mask, shape, padding)


def _weighted_moment_initial_guess(x, y, z, bg, nx, ny, peak_x, peak_y, cfg):
    weights = np.asarray(z, dtype=np.float64) - bg
    finite = np.isfinite(weights) & np.isfinite(x) & np.isfinite(y)
    weights = np.where(finite & (weights > 0), weights, 0.0)
    total = float(np.sum(weights))
    sigma_min = 1.0
    sigma_max = max(2.0, float(cfg.get("max_sigma_fraction", 0.18)) * max(nx, ny))
    if total > 0:
        cx = float(np.sum(x * weights) / total)
        cy = float(np.sum(y * weights) / total)
        var_x = float(np.sum(((x - cx) ** 2) * weights) / total)
        var_y = float(np.sum(((y - cy) ** 2) * weights) / total)
        sigma_x = math.sqrt(max(var_x, sigma_min**2))
        sigma_y = math.sqrt(max(var_y, sigma_min**2))
    else:
        cx, cy = float(peak_x), float(peak_y)
        sigma_x = max(min(nx, ny) / 12.0, sigma_min)
        sigma_y = sigma_x
    sigma_x = float(np.clip(sigma_x, sigma_min, sigma_max))
    sigma_y = float(np.clip(sigma_y, sigma_min, sigma_max))
    if not np.isfinite(cx) or not np.isfinite(cy):
        cx, cy = float(peak_x), float(peak_y)
    return cx, cy, sigma_x, sigma_y


def _limit_fit_pixels(x, y, z, peak_x, peak_y, max_pixels):
    return _workflow()._canonical_limit_fit_pixels(x, y, z, peak_x, peak_y, max_pixels)


def _gaussian_quality_config(cfg):
    return _workflow()._canonical_gaussian_quality_config(cfg)


def _set_gaussian_failure_diag(
    cfg: dict, source_file, reason: str, mask_diag: dict | None = None, **extra
) -> None:
    mask_diag = mask_diag or {}
    cfg["_last_gaussian_failure_diag"] = {
        "source_file": source_file or "",
        "reason": reason,
        "quality_flag": reason,
        "quality_flag_detail": extra.get("quality_flag_detail", ""),
        "finite_pixel_count": extra.get("finite_pixel_count", ""),
        "mask_pixel_count": mask_diag.get(
            "mask_pixel_count", extra.get("mask_pixel_count", 0)
        ),
        "background_rms_median": mask_diag.get("background_rms_median", np.nan),
        "background_level_median": mask_diag.get("background_level_median", np.nan),
        "fit_peak_fraction_threshold_used": mask_diag.get(
            "fit_peak_fraction_threshold_used",
            extra.get("fit_peak_fraction_threshold_used", np.nan),
        ),
        "fit_peak_fraction_candidate_counts": mask_diag.get(
            "fit_peak_fraction_candidate_counts",
            extra.get("fit_peak_fraction_candidate_counts", ""),
        ),
        "gaussian_fit_method": extra.get("gaussian_fit_method", "skipped"),
        "roi_used": extra.get("roi_used", False),
        "roi_shape": extra.get("roi_shape", ""),
    }


def _fit_failure_warning(source_file, quality_flag, detail=""):
    name = os.path.basename(source_file) if source_file else "radio image"
    suffix = f" / {detail}" if detail else ""
    if detail or str(quality_flag) not in {"mask_too_small", "low_snr"}:
        warnings.warn(
            f"Gaussian fit skipped for {name}: reason={quality_flag}{suffix}",
            stacklevel=2,
        )


def pixel_to_data_coord(
    x_pix, y_pix, extent, shape, origin="upper"
) -> tuple[float, float]:
    return _workflow()._canonical_pixel_to_data_coord(
        x_pix, y_pix, extent, shape, origin
    )


def data_coord_to_pixel(
    x_arcsec, y_arcsec, extent, shape, origin="upper"
) -> tuple[float, float]:
    return _workflow()._canonical_data_coord_to_pixel(
        x_arcsec, y_arcsec, extent, shape, origin
    )


def coordinate_roundtrip_error_pixel(
    x_pix, y_pix, extent, shape, origin="upper"
) -> float:
    x_arcsec, y_arcsec = _workflow().pixel_to_data_coord(
        x_pix, y_pix, extent, shape, origin
    )
    x_back, y_back = _workflow().data_coord_to_pixel(
        x_arcsec, y_arcsec, extent, shape, origin
    )
    return float(math.hypot(float(x_pix) - x_back, float(y_pix) - y_back))


def _attach_gaussian_fit_metadata(result, cfg, mask_diag, fit_input_type, fit_meta):
    result.fit_input_type = fit_input_type
    result.background_rms_median = mask_diag.get("background_rms_median", np.nan)
    result.background_level_median = mask_diag.get("background_level_median", np.nan)
    result.fit_peak_fraction_threshold_used = mask_diag.get(
        "fit_peak_fraction_threshold_used", np.nan
    )
    result.fit_peak_fraction_candidate_counts = mask_diag.get(
        "fit_peak_fraction_candidate_counts", ""
    )
    result.source_snr_peak = mask_diag.get("source_snr_peak", np.nan)
    result.source_snr_mean = mask_diag.get("source_snr_mean", np.nan)
    for key, value in fit_meta.items():
        setattr(result, key, value)
    return result


def _update_gaussian_quality(fit_result, extent, img_shape, cfg):
    ny, nx = img_shape
    dx = abs((extent[1] - extent[0]) / max(nx, 1))
    dy = abs((extent[3] - extent[2]) / max(ny, 1))
    width = 2.355 * fit_result.sigma_pixel[0] * dx
    height = 2.355 * fit_result.sigma_pixel[1] * dy
    fit_result.fwhm_major_arcsec = float(max(width, height))
    fit_result.fwhm_minor_arcsec = float(min(width, height))
    raw_center = getattr(fit_result, "raw_center_arcsec", None)
    fit_result.center_peak_distance_arcsec = np.nan
    if raw_center is not None:
        cx, cy = fit_result.center_arcsec
        rx, ry = raw_center
        fit_result.center_peak_distance_arcsec = float(math.hypot(cx - rx, cy - ry))
    quality_cfg = _workflow()._gaussian_quality_config(cfg)
    is_moment_fallback = fit_result.quality_flag == "moment_fallback"
    detail = getattr(fit_result, "quality_flag_detail", "")
    valid = True
    if fit_result.fwhm_major_arcsec > float(quality_cfg.get("max_fwhm_arcsec", 1800.0)):
        fit_result.quality_flag = "unphysical_size"
        detail = "skipped_large_fwhm"
        valid = False
    max_dist = float(quality_cfg.get("max_center_peak_distance_arcsec", 300.0))
    if np.isfinite(fit_result.center_peak_distance_arcsec):
        max_dist = min(
            max_dist,
            float(cfg.get("gaussian_max_center_peak_distance_fraction_of_fwhm", 0.5))
            * fit_result.fwhm_minor_arcsec,
        )
        if fit_result.center_peak_distance_arcsec > max_dist:
            fit_result.quality_flag = "center_far_from_peak"
            detail = "center_far_from_peak"
            valid = False
    min_snr = float(quality_cfg.get("min_snr", cfg.get("fit_snr_threshold", 5.0)))
    if (
        fit_result.snr is not None
        and np.isfinite(fit_result.snr)
        and fit_result.snr < min_snr
    ):
        if not is_moment_fallback:
            fit_result.quality_flag = "low_snr"
        detail = "low_snr"
        valid = False
    max_resid = float(quality_cfg.get("max_residual_rms_fraction", 0.8))
    if (
        fit_result.residual_rms is not None
        and np.isfinite(fit_result.residual_rms)
        and np.isfinite(fit_result.amplitude)
        and abs(fit_result.amplitude) > 0
        and fit_result.residual_rms / abs(fit_result.amplitude) > max_resid
    ):
        fit_result.quality_flag = "high_residual"
        detail = "high_residual"
        valid = False
    if is_moment_fallback and quality_cfg.get("require_quality_ok", True):
        valid = False
        detail = detail or "moment_fallback"
    if quality_cfg.get("require_quality_ok", True) and fit_result.quality_flag != "ok":
        valid = False
    fit_result.quality_flag_detail = detail
    fit_result.overlay_valid = (
        bool(valid) if cfg.get("gaussian_valid_only_for_overlay", True) else True
    )
    fit_result.trajectory_valid = (
        bool(valid) if cfg.get("gaussian_valid_only_for_trajectory", True) else True
    )
    if is_moment_fallback and not cfg.get(
        "gaussian_allow_moment_fallback_for_trajectory", False
    ):
        fit_result.trajectory_valid = False
    return bool(fit_result.overlay_valid)


def fit_elliptical_gaussian_on_radio_image(
    data,
    extent,
    cfg,
    source_file=None,
    background_map=None,
    rms_map=None,
    fit_input_type="raw",
    image_origin=None,
):
    """Fit with the overlay-specific mask, ROI, and quality contract.

    The canonical fitter additionally supports source-mask overrides and uses
    different default mask sizes, ROI padding, fit-pixel limits, failure
    metadata, and validity rules. Keeping this body local avoids changing
    established overlay products while pure helpers delegate above.
    """

    work = np.asarray(data, dtype=np.float64)
    image_origin = image_origin or cfg.get("_current_radio_image_origin", "upper")
    cfg.pop("_last_gaussian_failure_diag", None)
    fit_meta = _workflow()._gaussian_fit_diag_defaults(cfg)
    if work.ndim != 2 or not np.any(np.isfinite(work)):
        _workflow()._set_gaussian_failure_diag(
            cfg, source_file, "non_finite_data", **fit_meta
        )
        _workflow()._fit_failure_warning(source_file, "non_finite_data")
        return None
    finite_work = work[np.isfinite(work)]
    ny, nx = work.shape
    X, Y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    source_mask, mask_diag = _workflow().create_source_mask(
        work, cfg, background_map=background_map, rms_map=rms_map
    )
    if cfg.get("fit_use_source_mask", True) and source_mask is None:
        reason = mask_diag.get("quality_flag", "mask_too_small")
        _workflow()._set_gaussian_failure_diag(
            cfg,
            source_file,
            reason,
            mask_diag,
            finite_pixel_count=finite_work.size,
            **fit_meta,
        )
        _workflow()._fit_failure_warning(source_file, reason)
        return None
    fit_mask = (
        np.asarray(source_mask, dtype=np.bool_) & np.isfinite(work)
        if cfg.get("fit_use_source_mask", True)
        else np.isfinite(work)
    )
    if cfg.get("gaussian_fit_use_roi", True):
        y_slice, x_slice, roi_used = _workflow()._roi_slices_from_mask(
            source_mask if source_mask is not None else fit_mask,
            work.shape,
            cfg.get("gaussian_fit_roi_padding_pixels", 4),
        )
    else:
        y_slice, x_slice, roi_used = slice(0, ny), slice(0, nx), False
    x_offset = int(x_slice.start or 0)
    y_offset = int(y_slice.start or 0)
    roi_work = work[y_slice, x_slice]
    roi_fit_mask = fit_mask[y_slice, x_slice]
    roi_ny, roi_nx = roi_work.shape
    fit_meta["roi_used"] = bool(roi_used)
    fit_meta["roi_shape"] = f"{roi_ny}x{roi_nx}"
    X_roi, Y_roi = np.meshgrid(
        np.arange(roi_nx, dtype=np.float64),
        np.arange(roi_ny, dtype=np.float64),
    )
    xy_fit = (X_roi[roi_fit_mask].ravel(), Y_roi[roi_fit_mask].ravel())
    z_fit = roi_work[roi_fit_mask].ravel()
    finite = np.isfinite(z_fit) & np.isfinite(xy_fit[0]) & np.isfinite(xy_fit[1])
    xy_fit = (xy_fit[0][finite], xy_fit[1][finite])
    z_fit = z_fit[finite]
    if z_fit.size < int(cfg.get("fit_min_mask_pixels", 12)):
        _workflow()._set_gaussian_failure_diag(
            cfg, source_file, "mask_too_small", mask_diag, **fit_meta
        )
        _workflow()._fit_failure_warning(source_file, "mask_too_small")
        return None
    local_bg = float(mask_diag.get("background_level", np.nan))
    if not np.isfinite(local_bg):
        local_bg = float(np.nanmedian(z_fit))
    local_peak = float(np.nanmax(z_fit))
    local_peak_idx = int(np.nanargmax(z_fit))
    peak_x = float(xy_fit[0][local_peak_idx])
    peak_y = float(xy_fit[1][local_peak_idx])
    xy_x, xy_y, z_fit, before_limit, after_limit = _workflow()._limit_fit_pixels(
        xy_fit[0],
        xy_fit[1],
        z_fit,
        peak_x,
        peak_y,
        cfg.get("gaussian_fit_max_pixels", 400),
    )
    xy_fit = (xy_x, xy_y)
    fit_meta["fit_pixel_count_before_limit"] = int(before_limit)
    fit_meta["fit_pixel_count_after_limit"] = int(after_limit)
    mask_diag["mask_pixel_count"] = int(after_limit)
    if z_fit.size < int(cfg.get("fit_min_mask_pixels", 12)):
        _workflow()._set_gaussian_failure_diag(
            cfg, source_file, "mask_too_small", mask_diag, **fit_meta
        )
        _workflow()._fit_failure_warning(source_file, "mask_too_small")
        return None
    A0 = max(local_peak - local_bg, 1e-12)
    cx0, cy0, sigma_x0, sigma_y0 = _workflow()._weighted_moment_initial_guess(
        xy_fit[0], xy_fit[1], z_fit, local_bg, roi_nx, roi_ny, peak_x, peak_y, cfg
    )
    fit_meta["initial_center_pixel"] = f"{cx0 + x_offset:.3f},{cy0 + y_offset:.3f}"
    fit_meta["initial_sigma_x_pixel"] = float(sigma_x0)
    fit_meta["initial_sigma_y_pixel"] = float(sigma_y0)
    if cfg.get("gaussian_fit_normalize_data", True):
        centered = z_fit - local_bg
        finite_centered = centered[np.isfinite(centered)]
        norm_scale = (
            float(np.nanpercentile(np.abs(finite_centered), 99))
            if finite_centered.size
            else np.nan
        )
        if not np.isfinite(norm_scale) or norm_scale <= 0:
            norm_scale = max(abs(A0), 1.0)
        z_curve = centered / norm_scale
        A0_curve = max(A0 / norm_scale, 1e-6)
        bg0_curve = 0.0
        bg_abs_limit = 10.0
    else:
        norm_scale = 1.0
        z_curve = z_fit
        A0_curve = A0
        bg0_curve = local_bg
        bg_abs_limit = max(abs(local_peak) * 10.0, abs(local_bg) * 10.0, 1.0)
    fit_meta["normalization_scale"] = float(norm_scale)
    background_model = cfg.get("fit_background_model", "constant")
    model_map = {
        "none": _workflow().elliptical_gaussian_2d,
        "constant": _workflow().elliptical_gaussian_2d_with_constant_bg,
        "plane": _workflow().elliptical_gaussian_2d_with_plane_bg,
    }
    if background_model not in model_map:
        background_model = "constant"
    model_func = model_map[background_model]
    sigma_upper = max(
        2.0, float(cfg.get("max_sigma_fraction", 0.18)) * max(roi_nx, roi_ny)
    )
    amp_upper = max(abs(A0_curve) * 10.0, 2.0)
    slope_limit = bg_abs_limit / max(roi_nx, roi_ny, 1)
    p0 = [A0_curve, float(cx0), float(cy0), sigma_x0, sigma_y0, 0.0]
    lower = [0.0, 0.0, 0.0, 0.5, 0.5, -np.pi / 2]
    upper = [amp_upper, roi_nx - 1, roi_ny - 1, sigma_upper, sigma_upper, np.pi / 2]
    if background_model == "constant":
        p0 += [bg0_curve]
        lower += [-bg_abs_limit]
        upper += [bg_abs_limit]
    elif background_model == "plane":
        p0 += [bg0_curve, 0.0, 0.0]
        lower += [-bg_abs_limit, -slope_limit, -slope_limit]
        upper += [bg_abs_limit, slope_limit, slope_limit]
    p0_arr = np.minimum(
        np.maximum(np.asarray(p0, dtype=float), np.asarray(lower, dtype=float)),
        np.asarray(upper, dtype=float),
    )
    maxfev = int(cfg.get("gaussian_fit_maxfev", 8000))
    fit_meta["maxfev"] = maxfev
    fit_exception = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, pcov = _workflow().curve_fit(
                model_func,
                xy_fit,
                z_curve,
                p0=p0_arr,
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                maxfev=maxfev,
            )
        fit_meta["gaussian_fit_method"] = "curve_fit"
    except Exception as exc:
        fit_exception = exc
        if not cfg.get("gaussian_fit_fallback_to_moment", True):
            _workflow()._set_gaussian_failure_diag(
                cfg,
                source_file,
                "fit_failed_no_fallback",
                mask_diag,
                quality_flag_detail=str(exc),
                **fit_meta,
            )
            _workflow()._fit_failure_warning(
                source_file, "fit_failed_no_fallback", str(exc)
            )
            return None
        popt = p0_arr.copy()
        pcov = None
        fit_meta["gaussian_fit_method"] = "moment_fallback"
    X_local_full = X - x_offset
    Y_local_full = Y - y_offset
    model_curve = model_func((X_local_full, Y_local_full), *popt).reshape(work.shape)
    model_total = (
        model_curve * norm_scale + local_bg
        if cfg.get("gaussian_fit_normalize_data", True)
        else model_curve
    )
    gaussian_curve = (
        _workflow()
        .gaussian_only_from_popt((X_local_full, Y_local_full), popt, background_model)
        .reshape(work.shape)
    )
    gaussian_only_model = (
        gaussian_curve * norm_scale
        if cfg.get("gaussian_fit_normalize_data", True)
        else gaussian_curve
    )
    if not np.all(np.isfinite(model_total[fit_mask])):
        _workflow()._set_gaussian_failure_diag(
            cfg, source_file, "non_finite_fit_result", mask_diag, **fit_meta
        )
        _workflow()._fit_failure_warning(source_file, "non_finite_fit_result")
        return None
    x0_fit, y0_fit = float(popt[1]) + x_offset, float(popt[2]) + y_offset
    sigma_x, sigma_y = abs(float(popt[3])), abs(float(popt[4]))
    center_arcsec = _workflow().pixel_to_data_coord(
        x0_fit, y0_fit, extent, work.shape, origin=image_origin
    )
    raw_peak_arcsec = _workflow().pixel_to_data_coord(
        peak_x + x_offset, peak_y + y_offset, extent, work.shape, origin=image_origin
    )
    residual = work[fit_mask] - model_total[fit_mask]
    residual_rms = float(np.sqrt(np.nanmean(residual**2))) if residual.size else np.nan
    if (
        cfg.get("background_use_for_mask", True)
        and rms_map is not None
        and np.asarray(rms_map).shape == work.shape
    ):
        safe_rms = _workflow()._safe_rms_map(rms_map)
        noise_values = safe_rms[fit_mask & np.isfinite(safe_rms)]
        noise_sigma = float(np.nanmedian(noise_values)) if noise_values.size else np.nan
    else:
        _, noise_sigma = _workflow().estimate_background_noise(work, fit_mask)
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        noise_sigma = mask_diag.get("noise_sigma", np.nan)
    amplitude_curve = float(popt[0])
    amplitude_original = (
        amplitude_curve * norm_scale
        if cfg.get("gaussian_fit_normalize_data", True)
        else amplitude_curve
    )
    snr = (
        float(amplitude_original / noise_sigma)
        if np.isfinite(noise_sigma) and noise_sigma > 0
        else np.nan
    )
    quality_flag = (
        "moment_fallback"
        if fit_meta["gaussian_fit_method"] == "moment_fallback"
        else "ok"
    )
    result = GaussianFitResult(
        model=model_total,
        gaussian_only_model=gaussian_only_model,
        center_pixel=(x0_fit, y0_fit),
        center_arcsec=center_arcsec,
        sigma_pixel=(sigma_x, sigma_y),
        theta_rad=float(popt[5]),
        amplitude=amplitude_original,
        background_level=(
            float(local_bg + popt[6] * norm_scale)
            if len(popt) >= 7 and cfg.get("gaussian_fit_normalize_data", True)
            else (float(popt[6]) if len(popt) >= 7 else None)
        ),
        noise_sigma=float(noise_sigma) if np.isfinite(noise_sigma) else None,
        snr=snr if np.isfinite(snr) else None,
        residual_rms=residual_rms if np.isfinite(residual_rms) else None,
        quality_flag=quality_flag,
        covariance=pcov,
        mask_pixel_count=int(after_limit),
        source_file=source_file,
    )
    result.source_mask = source_mask
    result.image_origin = image_origin
    result.image_extent = extent
    result.raw_center_arcsec = raw_peak_arcsec
    result.coordinate_roundtrip_error_pixel = (
        _workflow().coordinate_roundtrip_error_pixel(
            x0_fit, y0_fit, extent, work.shape, origin=image_origin
        )
    )
    if fit_meta["gaussian_fit_method"] == "moment_fallback":
        result.reason = "fit_failed_moment_fallback"
        result.quality_flag_detail = str(fit_exception or "curve_fit_failed")
    result = _workflow()._attach_gaussian_fit_metadata(
        result, cfg, mask_diag, fit_input_type, fit_meta
    )
    _workflow()._update_gaussian_quality(result, extent, work.shape, cfg)
    return result


def config_for_gaussian_band(cfg: Config, band_label=None) -> dict:
    base = dict(vars(cfg))
    per_band = getattr(cfg, "gaussian_per_band_params", {}) or {}
    candidates = []
    key = _workflow()._normalise_band_key(band_label)
    if key:
        candidates.append(key)
        candidates.append(key.replace(".0MHz", "MHz"))
        candidates.append(key.replace("MHz", ""))
    for candidate in candidates:
        if candidate in per_band and isinstance(per_band[candidate], dict):
            base.update(per_band[candidate])
            break
    base["_gaussian_band_label"] = band_label
    return base


def save_gaussian_diagnostics_row(row, output_dir, cfg):
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(
            output_dir,
            getattr(
                cfg,
                "gaussian_diagnostics_csv",
                "aia_radio_gaussian_fit_diagnostics.csv",
            ),
        )
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=GAUSSIAN_DIAGNOSTIC_FIELDS,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {name: row.get(name, "") for name in GAUSSIAN_DIAGNOSTIC_FIELDS}
            )
    except Exception as exc:
        if getattr(cfg, "debug_mode", False):
            print(f"    [高斯诊断] CSV 写入失败: {exc}")


def _radio_overlay_mode(cfg: Config) -> str:
    mode = str(getattr(cfg, "radio_overlay_mode", "gaussian") or "gaussian").lower()
    if mode in {"raw", "direct", "direct_overlay", "raw_overlay"}:
        return "raw"
    return "gaussian"


def _build_radio_pixel_to_aia_pixel_mapper(
    radio_shape: tuple[int, int],
    ra_map: np.ndarray | None,
    dec_map: np.ndarray | None,
    aia_cutout_map: sunpy.map.GenericMap,
    cfg: Config,
    radio_header: fits.Header | None = None,
):
    ny_r, nx_r = radio_shape
    x_pix = np.arange(nx_r, dtype=float)
    y_pix = np.arange(ny_r, dtype=float)
    use_radec = cfg.use_radec_maps and (ra_map is not None) and (dec_map is not None)

    if use_radec:
        ra_abs = ra_map.copy().astype(np.float64)
        dec_abs = dec_map.copy().astype(np.float64)
        invalid_mask = (ra_abs == 0.0) & (dec_abs == 0.0)
        ra_abs[invalid_mask] = np.nan
        dec_abs[invalid_mask] = np.nan

        interp_ra = _workflow().RegularGridInterpolator(
            (y_pix, x_pix), ra_abs, bounds_error=False, fill_value=np.nan
        )
        interp_dec = _workflow().RegularGridInterpolator(
            (y_pix, x_pix), dec_abs, bounds_error=False, fill_value=np.nan
        )
    elif radio_header is not None:
        crpix1 = radio_header.get("CRPIX1", 0)
        crpix2 = radio_header.get("CRPIX2", 0)
        crval1 = radio_header.get("CRVAL1", 0)
        crval2 = radio_header.get("CRVAL2", 0)
        cdelt1 = radio_header.get("CDELT1", 1)
        cdelt2 = radio_header.get("CDELT2", 1)
    else:
        return None

    def radio_pix_to_aia_pix(xp, yp):
        if use_radec:
            ra_val = float(interp_ra((yp, xp)))
            dec_val = float(interp_dec((yp, xp)))
            if np.isnan(ra_val) or np.isnan(dec_val):
                iy = np.clip(int(round(yp)), 0, ny_r - 1)
                ix = np.clip(int(round(xp)), 0, nx_r - 1)
                ra_val = ra_abs[iy, ix]
                dec_val = dec_abs[iy, ix]
                if np.isnan(ra_val) or np.isnan(dec_val):
                    return np.nan, np.nan
            tx_arcsec = ra_val * 3600.0
            ty_arcsec = dec_val * 3600.0
        else:
            tx_arcsec = crval1 + (xp + 1 - crpix1) * cdelt1
            ty_arcsec = crval2 + (yp + 1 - crpix2) * cdelt2

        coord_target = SkyCoord(
            Tx=tx_arcsec * u.arcsec,
            Ty=ty_arcsec * u.arcsec,
            frame=aia_cutout_map.coordinate_frame,
        )
        px, py = aia_cutout_map.wcs.world_to_pixel(coord_target)
        return float(px), float(py)

    return radio_pix_to_aia_pix


def _aia_pixel_to_arcsec(
    aia_cutout_map: sunpy.map.GenericMap, pixel_x: float, pixel_y: float
) -> tuple[float, float]:
    try:
        center_world = aia_cutout_map.pixel_to_world(
            pixel_x * u.pixel, pixel_y * u.pixel
        )
        return (
            float(center_world.Tx.to_value(u.arcsec)),
            float(center_world.Ty.to_value(u.arcsec)),
        )
    except Exception:
        return float(pixel_x), float(pixel_y)


def reproject_raw_radio_to_aia(
    radio_data: np.ndarray,
    ra_map: np.ndarray | None,
    dec_map: np.ndarray | None,
    aia_cutout_map: sunpy.map.GenericMap,
    cfg: Config,
    radio_header: fits.Header | None = None,
    source_file: str | None = None,
    band_label: str | None = None,
    polarization: str | None = None,
    radio_time: datetime | None = None,
) -> RawRadioReprojectResult | None:
    del band_label, polarization, radio_time
    if radio_data is None or np.ndim(radio_data) != 2:
        return None
    ny_a, nx_a = aia_cutout_map.data.shape
    radio_values = np.asarray(radio_data, dtype=np.float64)
    finite_mask = np.isfinite(radio_values)
    if not finite_mask.any():
        return None

    radio_pix_to_aia_pix = _workflow()._build_radio_pixel_to_aia_pixel_mapper(
        radio_values.shape, ra_map, dec_map, aia_cutout_map, cfg, radio_header
    )
    if radio_pix_to_aia_pix is None:
        return None

    y_idx, x_idx = np.indices(radio_values.shape, dtype=np.float64)
    source_x = x_idx[finite_mask]
    source_y = y_idx[finite_mask]
    values = radio_values[finite_mask]
    points = []
    mapped_values = []
    for xp, yp, value in zip(source_x, source_y, values, strict=False):
        aia_x, aia_y = radio_pix_to_aia_pix(float(xp), float(yp))
        if not (np.isfinite(aia_x) and np.isfinite(aia_y)):
            continue
        points.append((aia_x, aia_y))
        mapped_values.append(float(value))
    if not points:
        return None

    points_arr = np.asarray(points, dtype=np.float64)
    values_arr = np.asarray(mapped_values, dtype=np.float64)
    grid_y, grid_x = np.mgrid[0:ny_a, 0:nx_a]
    method = str(
        getattr(cfg, "raw_reproject_interpolation_method", "linear") or "linear"
    ).lower()
    if method not in {"linear", "nearest", "cubic"}:
        method = "linear"
    if method in {"linear", "cubic"} and len(points_arr) < 3:
        method = "nearest"
    try:
        model = _workflow().griddata(
            points_arr,
            values_arr,
            (grid_x, grid_y),
            method=method,
            fill_value=np.nan,
        )
    except Exception:
        model = _workflow().griddata(
            points_arr,
            values_arr,
            (grid_x, grid_y),
            method="nearest",
            fill_value=np.nan,
        )
    if np.all(~np.isfinite(model)):
        model = _workflow().griddata(
            points_arr,
            values_arr,
            (grid_x, grid_y),
            method="nearest",
            fill_value=np.nan,
        )
    if np.all(~np.isfinite(model)):
        return None

    peak_y, peak_x = np.unravel_index(np.nanargmax(model), model.shape)
    peak_arcsec = _workflow()._aia_pixel_to_arcsec(
        aia_cutout_map, float(peak_x), float(peak_y)
    )
    return RawRadioReprojectResult(
        model=np.asarray(model, dtype=np.float32),
        peak_pixel=(float(peak_x), float(peak_y)),
        peak_arcsec=peak_arcsec,
        amplitude=float(np.nanmax(model)),
        source_file=source_file,
    )


def reproject_radio_for_overlay(
    radio_data: np.ndarray,
    ra_map: np.ndarray | None,
    dec_map: np.ndarray | None,
    aia_cutout_map: sunpy.map.GenericMap,
    cfg: Config,
    radio_header: fits.Header | None = None,
    source_file: str | None = None,
    band_label: str | None = None,
    polarization: str | None = None,
    radio_time: datetime | None = None,
) -> GaussianReprojectResult | RawRadioReprojectResult | None:
    if _workflow()._radio_overlay_mode(cfg) == "raw":
        return _workflow().reproject_raw_radio_to_aia(
            radio_data,
            ra_map,
            dec_map,
            aia_cutout_map,
            cfg,
            radio_header,
            source_file=source_file,
            band_label=band_label,
            polarization=polarization,
            radio_time=radio_time,
        )
    return _workflow().reproject_radio_via_gaussian_fit(
        radio_data,
        ra_map,
        dec_map,
        aia_cutout_map,
        cfg,
        radio_header,
        source_file=source_file,
        band_label=band_label,
        polarization=polarization,
        radio_time=radio_time,
    )


def reproject_radio_via_gaussian_fit(
    radio_data: np.ndarray,
    ra_map: np.ndarray | None,
    dec_map: np.ndarray | None,
    aia_cutout_map: sunpy.map.GenericMap,
    cfg: Config,
    radio_header: fits.Header | None = None,
    source_file: str | None = None,
    band_label: str | None = None,
    polarization: str | None = None,
    radio_time: datetime | None = None,
) -> GaussianReprojectResult | None:
    ny_a, nx_a = aia_cutout_map.data.shape
    ny_r, nx_r = radio_data.shape
    x_pix = np.arange(nx_r, dtype=float)
    y_pix = np.arange(ny_r, dtype=float)

    gaussian_cfg = _workflow().config_for_gaussian_band(cfg, band_label)
    background_map, rms_map, bg_diag = _workflow().estimate_background_rms_mesh(
        radio_data, gaussian_cfg
    )
    radio_fit = _workflow().fit_elliptical_gaussian_on_radio_image(
        radio_data,
        extent=[0, nx_r - 1, 0, ny_r - 1],
        cfg=gaussian_cfg,
        source_file=source_file,
        background_map=background_map,
        rms_map=rms_map,
        fit_input_type="raw",
        image_origin="lower",
    )
    if radio_fit is None:
        if getattr(cfg, "save_gaussian_diagnostics", True):
            _workflow().save_gaussian_diagnostics_row(
                _workflow()._gaussian_diagnostics_row(
                    None,
                    gaussian_cfg,
                    source_file=source_file,
                    band=band_label,
                    polarization=polarization,
                    radio_time=radio_time,
                    bg_diag=bg_diag,
                ),
                cfg.output_dir,
                cfg,
            )
        if cfg.debug_mode:
            print(f"    [高斯拟合] 失败或质量不足: {source_file}")
        return None
    if (
        not getattr(radio_fit, "overlay_valid", True)
        and not cfg.draw_low_quality_gaussian_contours
    ):
        if getattr(cfg, "save_gaussian_diagnostics", True):
            _workflow().save_gaussian_diagnostics_row(
                _workflow()._gaussian_diagnostics_row(
                    radio_fit,
                    gaussian_cfg,
                    source_file=source_file,
                    band=band_label,
                    polarization=polarization,
                    radio_time=radio_time,
                    bg_diag=bg_diag,
                ),
                cfg.output_dir,
                cfg,
            )
        if cfg.debug_mode:
            print(
                f"    [高斯拟合] 跳过低质量结果: "
                f"quality={radio_fit.quality_flag}, "
                f"detail={getattr(radio_fit, 'quality_flag_detail', '')}"
            )
        return None

    # ---------- 1. 在射电域进行二维椭圆高斯拟合 ----------
    try:
        popt = (
            radio_fit.amplitude,
            radio_fit.center_pixel[0],
            radio_fit.center_pixel[1],
            radio_fit.sigma_pixel[0],
            radio_fit.sigma_pixel[1],
            radio_fit.theta_rad,
        )
        pcov = radio_fit.covariance
    except Exception as e:
        if cfg.debug_mode:
            print(f"    [高斯拟合] 失败: {e}")
        return None

    A_fit, x0_pix, y0_pix, sigma_x_pix, sigma_y_pix, theta_pix = popt
    use_radec = cfg.use_radec_maps and (ra_map is not None) and (dec_map is not None)

    if use_radec:
        # 【应用 AIA_RS_HMI.py 的预处理逻辑】
        ra_abs = ra_map.copy().astype(np.float64)
        dec_abs = dec_map.copy().astype(np.float64)

        # 精准过滤背景：将精确为 0.0 的无效区域置为 NaN，防止坐标拉扯
        invalid_mask = (ra_abs == 0.0) & (dec_abs == 0.0)
        ra_abs[invalid_mask] = np.nan
        dec_abs[invalid_mask] = np.nan

        interp_ra = _workflow().RegularGridInterpolator(
            (y_pix, x_pix), ra_abs, bounds_error=False, fill_value=np.nan
        )
        interp_dec = _workflow().RegularGridInterpolator(
            (y_pix, x_pix), dec_abs, bounds_error=False, fill_value=np.nan
        )

    elif radio_header is not None:
        crpix1 = radio_header.get("CRPIX1", 0)
        crpix2 = radio_header.get("CRPIX2", 0)
        crval1 = radio_header.get("CRVAL1", 0)
        crval2 = radio_header.get("CRVAL2", 0)
        cdelt1 = radio_header.get("CDELT1", 1)
        cdelt2 = radio_header.get("CDELT2", 1)
    else:
        return None

    # ---------- 2. 映射转换函数（度 -> HPC 角秒） ----------
    def radio_pix_to_aia_pix(xp, yp):
        if use_radec:
            ra_val = float(interp_ra((yp, xp)))
            dec_val = float(interp_dec((yp, xp)))

            # 处理越界或 NaN 值
            if np.isnan(ra_val) or np.isnan(dec_val):
                iy = np.clip(int(round(yp)), 0, ny_r - 1)
                ix = np.clip(int(round(xp)), 0, nx_r - 1)
                ra_val = ra_abs[iy, ix]
                dec_val = dec_abs[iy, ix]
                if np.isnan(ra_val) or np.isnan(dec_val):
                    return np.nan, np.nan

            # 【核心修正】：参照 AIA_RS_HMI.py，真实单位为度，转为角秒必须乘以 3600
            tx_arcsec = ra_val * 3600.0
            ty_arcsec = dec_val * 3600.0

            # 构建 AIA 原生的日面投影坐标系 (HPC)
            coord_target = SkyCoord(
                Tx=tx_arcsec * u.arcsec,
                Ty=ty_arcsec * u.arcsec,
                frame=aia_cutout_map.coordinate_frame,
            )
        else:
            x_angle = crval1 + (xp + 1 - crpix1) * cdelt1
            y_angle = crval2 + (yp + 1 - crpix2) * cdelt2
            coord_target = SkyCoord(
                Tx=x_angle * u.arcsec,
                Ty=y_angle * u.arcsec,
                frame=aia_cutout_map.coordinate_frame,
            )

        px, py = aia_cutout_map.wcs.world_to_pixel(coord_target)
        return float(px), float(py)

    # ---------- 3. 核心：三点映射 ----------
    try:
        c_aia_x, c_aia_y = radio_pix_to_aia_pix(x0_pix, y0_pix)
        if np.isnan(c_aia_x) or np.isnan(c_aia_y):
            return None

        p_maj_x = x0_pix + sigma_x_pix * np.cos(theta_pix)
        p_maj_y = y0_pix + sigma_x_pix * np.sin(theta_pix)
        maj_aia_x, maj_aia_y = radio_pix_to_aia_pix(p_maj_x, p_maj_y)

        p_min_x = x0_pix - sigma_y_pix * np.sin(theta_pix)
        p_min_y = y0_pix + sigma_y_pix * np.cos(theta_pix)
        min_aia_x, min_aia_y = radio_pix_to_aia_pix(p_min_x, p_min_y)
    except Exception as e:
        if cfg.debug_mode:
            print(f"    [坐标转换] 映射失败: {e}")
        return None

    dx_maj = maj_aia_x - c_aia_x
    dy_maj = maj_aia_y - c_aia_y
    sigma_aia_x = np.sqrt(dx_maj**2 + dy_maj**2)
    theta_aia = np.arctan2(dy_maj, dx_maj)

    dx_min = min_aia_x - c_aia_x
    dy_min = min_aia_y - c_aia_y
    sigma_aia_y = np.sqrt(dx_min**2 + dy_min**2)

    if not (
        np.isfinite(sigma_aia_x)
        and np.isfinite(sigma_aia_y)
        and sigma_aia_x > 0
        and sigma_aia_y > 0
    ):
        return None

    # ---------- 4. 在 AIA 视场生成最终模型 ----------
    Y_aia, X_aia = np.mgrid[0:ny_a, 0:nx_a]
    model = _workflow().elliptical_gaussian_2d(
        (X_aia, Y_aia), A_fit, c_aia_x, c_aia_y, sigma_aia_x, sigma_aia_y, theta_aia
    )

    model = np.maximum(model, 0)
    center_world = aia_cutout_map.pixel_to_world(c_aia_x * u.pixel, c_aia_y * u.pixel)

    if getattr(cfg, "save_gaussian_diagnostics", True):
        _workflow().save_gaussian_diagnostics_row(
            _workflow()._gaussian_diagnostics_row(
                radio_fit,
                gaussian_cfg,
                source_file=source_file,
                band=band_label,
                polarization=polarization,
                radio_time=radio_time,
                bg_diag=bg_diag,
            ),
            cfg.output_dir,
            cfg,
        )

    return GaussianReprojectResult(
        model=model.astype(np.float32),
        center_pixel=(float(c_aia_x), float(c_aia_y)),
        center_arcsec=(
            float(center_world.Tx.to_value(u.arcsec)),
            float(center_world.Ty.to_value(u.arcsec)),
        ),
        sigma_pixel=(float(sigma_aia_x), float(sigma_aia_y)),
        theta_rad=float(theta_aia),
        amplitude=float(A_fit),
        covariance=None if pcov is None else np.asarray(pcov),
        quality_flag=getattr(radio_fit, "quality_flag", "ok"),
        quality_flag_detail=getattr(radio_fit, "quality_flag_detail", ""),
        overlay_valid=getattr(radio_fit, "overlay_valid", True),
        trajectory_valid=getattr(radio_fit, "trajectory_valid", True),
        snr=getattr(radio_fit, "snr", None),
        residual_rms=getattr(radio_fit, "residual_rms", None),
        mask_pixel_count=getattr(radio_fit, "mask_pixel_count", 0),
        fwhm_major_arcsec=getattr(radio_fit, "fwhm_major_arcsec", None),
        fwhm_minor_arcsec=getattr(radio_fit, "fwhm_minor_arcsec", None),
        center_peak_distance_arcsec=getattr(
            radio_fit, "center_peak_distance_arcsec", None
        ),
        source_file=source_file,
        radio_fit_result=radio_fit,
    )


def smooth_for_contour(data: np.ndarray, sigma: float) -> np.ndarray:
    """对等值线数据做加权高斯平滑（保留 NaN 边界）"""
    if sigma <= 0:
        return data
    nan_mask = np.isnan(data)
    filled = np.where(nan_mask, 0.0, data)
    weights = (~nan_mask).astype(np.float64)
    sm_d = _workflow().gaussian_filter(filled, sigma=sigma)
    sm_w = _workflow().gaussian_filter(weights, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sm_w > 1e-6, sm_d / sm_w, np.nan)


def _get_padded_aia_map(
    aia_map: sunpy.map.GenericMap, cfg: Config
) -> sunpy.map.GenericMap:
    """
    精确截取或扩充画布到用户指定的 ROI 范围，
    彻底解决 sunpy.map.submap 在图像边缘的自动截断问题，
    允许在 AIA 没有数据的外太空区域继续绘制射电和 HMI。
    """
    roi = _workflow().normalize_roi_bounds_arcsec(cfg)
    bl = SkyCoord(
        roi["left"] * u.arcsec,
        roi["bottom"] * u.arcsec,
        frame=aia_map.coordinate_frame,
    )
    tr = SkyCoord(
        roi["right"] * u.arcsec,
        roi["top"] * u.arcsec,
        frame=aia_map.coordinate_frame,
    )

    px_bl = aia_map.wcs.world_to_pixel(bl)
    px_tr = aia_map.wcs.world_to_pixel(tr)

    # 转换为整数像素边界
    x0, y0 = int(np.floor(float(px_bl[0]))), int(np.floor(float(px_bl[1])))
    x1, y1 = int(np.ceil(float(px_tr[0]))), int(np.ceil(float(px_tr[1])))

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    new_nx = x1 - x0
    new_ny = y1 - y0

    # 创建全 NaN 的全新画布（充当深空背景）
    new_data = np.full((new_ny, new_nx), np.nan, dtype=aia_map.data.dtype)

    orig_ny, orig_nx = aia_map.data.shape

    # 计算原图和新画布的重合像素区域
    src_x0 = max(0, x0)
    src_x1 = min(orig_nx, x1)
    src_y0 = max(0, y0)
    src_y1 = min(orig_ny, y1)

    # 如果有重合，则将 AIA 原图的有效部分精准贴入新画布
    if src_x0 < src_x1 and src_y0 < src_y1:
        dst_x0 = src_x0 - x0
        dst_x1 = src_x1 - x0
        dst_y0 = src_y0 - y0
        dst_y1 = src_y1 - y0
        new_data[dst_y0:dst_y1, dst_x0:dst_x1] = aia_map.data[
            src_y0:src_y1, src_x0:src_x1
        ]

    # 更新 WCS 头文件，平移参考坐标原点
    new_meta = aia_map.meta.copy()
    new_meta["CRPIX1"] -= x0
    new_meta["CRPIX2"] -= y0
    new_meta["NAXIS1"] = new_nx
    new_meta["NAXIS2"] = new_ny

    # 返回完全基于用户范围定制的新 Map
    return _workflow().sunpy.map.Map(new_data, new_meta)


def _legacy_check_gaussian_fit_synthetic_source():
    cfg = Config()
    cfg.fit_snr_threshold = 1.0
    cfg.gaussian_quality_requirements["require_quality_ok"] = False
    shape = (96, 128)
    y, x = np.indices(shape, dtype=np.float64)
    x0, y0 = 70.25, 34.75
    data = 2.0 + 80.0 * np.exp(-0.5 * (((x - x0) / 7.0) ** 2 + ((y - y0) / 5.0) ** 2))
    result = _workflow().fit_elliptical_gaussian_on_radio_image(
        data,
        extent=[0, shape[1] - 1, 0, shape[0] - 1],
        cfg=dict(vars(cfg)),
        source_file="synthetic.fits",
        image_origin="lower",
    )
    assert result is not None
    assert math.hypot(result.center_pixel[0] - x0, result.center_pixel[1] - y0) < 1.0
    return True

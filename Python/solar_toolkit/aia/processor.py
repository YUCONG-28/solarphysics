"""Runtime dispatcher for the AIA EUV processor.

The heavy SunPy/Astropy implementation is loaded lazily so importing the
public AIA package remains safe in lightweight test and documentation paths.

中文：本模块延迟加载较重的 SunPy/Astropy 实现，使测试、文档和轻量 import
路径可以安全使用公共 AIA 包。
"""

from __future__ import annotations

from .config import AIAConfig

__all__ = ["process_aia_fits"]


def _actual_mode(cfg: AIAConfig) -> str:
    """Normalize compatibility flags into the runtime mode used by the CLI."""
    if cfg.use_test_mode or cfg.mode == "test":
        return "test"
    if cfg.mode == "mosaic" or cfg.multi_band_composite:
        return "mosaic"
    return "single"


def _configure_matplotlib_backend(mode: str) -> None:
    """Select the non-interactive backend before loading the heavy executor."""
    if mode in ("single", "mosaic"):
        import matplotlib

        matplotlib.use("Agg", force=True)


def _load_impl():
    """Import the heavy implementation only when processing is requested."""
    from . import _euv_processor_impl

    return _euv_processor_impl


def process_aia_fits(cfg: AIAConfig) -> None:
    """Run AIA processing without changing the legacy mode semantics."""
    actual_mode = _actual_mode(cfg)
    _configure_matplotlib_backend(actual_mode)

    if not cfg.draw_original and not cfg.draw_difference:
        raise ValueError(
            "Nothing to draw: at least one of draw_original or "
            "draw_difference must be True."
        )

    impl = _load_impl()
    if actual_mode == "test":
        impl._run_test_mode(cfg)
        if cfg.draw_difference:
            print(
                "Test mode: draw_difference=True detected; "
                "full difference batch skipped."
            )
        return

    if actual_mode == "single":
        if cfg.draw_original:
            impl._run_single_batch(cfg)
        if cfg.draw_difference:
            impl._run_difference_batch(cfg)
        return

    if actual_mode == "mosaic":
        output_mode = cfg.difference_output_mode
        if output_mode == "auto":
            # Documented behavior: "auto" resolves to mosaic difference output.
            output_mode = "mosaic"

        if cfg.draw_difference and output_mode in ("mosaic", "both"):
            cfg.mosaic_difference_inline = True
            impl._run_mosaic_batch(cfg)
        elif not cfg.draw_difference and cfg.draw_original:
            impl._run_mosaic_batch(cfg)

        if cfg.draw_difference and output_mode in ("single", "both"):
            impl._run_difference_batch(cfg)

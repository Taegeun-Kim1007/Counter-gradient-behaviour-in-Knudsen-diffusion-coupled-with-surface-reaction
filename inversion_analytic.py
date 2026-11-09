# -*- coding: utf-8 -*-
"""
Flux-based integral solution for
"Counter-gradient behaviour in Knudsen diffusion coupled with surface reaction"

Taegeun Kim
Department of Mechanical Engineering, Pohang University of Science and Technology,
77 Cheongam-ro, Nam-gu, Pohang, Gyeongbuk 37673, Republic of Korea
ktg1007@postech.ac.kr
"""

import os
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erf
from scipy.fft import next_fast_len
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a GUI/X display (avoids Qt/xcb crash)
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import threading
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation
# ======== Matplotlib global style ========
# Nature figure guideline: ALL text (incl. math) must be sans-serif, preferably Arial.
# Body text -> Arial; math ($n$, $\Gamma$, $x/(2h)$, x10^n) -> custom mathtext in Arial so
# variables/symbols are sans-serif too.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial:italic",
})


# ============================================================
# Shared plot settings
# ============================================================
DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE = "centerline"
DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE = 0.9
DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE = 0.9
AVOGADRO_CONSTANT = 6.02214076e23
DEFAULT_PLOT_WIDTH = 50.0e-9
DEFAULT_CONTOUR_X_TICK_COUNT = 6


def safe_divide(numerator, denominator):
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def default_contour_x_ticks(aspect_ratio, tick_count=DEFAULT_CONTOUR_X_TICK_COUNT):
    depth = DEFAULT_PLOT_WIDTH * float(aspect_ratio)
    return tuple(float(value) for value in np.linspace(0.0, depth, int(tick_count)))

PLOT_CASE_SETTINGS = {
    1: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.3,
            cbar_pad=0.05,
            cbar_aspect=7.5,
            figsize_contour=(15, 6.0),
            font_size=32,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=3,
            mass_fraction_powernorm_gamma=None,
            zoom_factor=3.0,
            zoom_xlim=(2.5e-8, 5.0e-8),
            zoom_font_size=32,
            zoom_cbar_fraction=0.3,
            zoom_cbar_aspect=7.5,
            zoom_x_nticks=2,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(6.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(8.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
    3: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.04,
            cbar_pad=0.05,
            cbar_aspect=6.0,
            figsize_contour=(15, 6.0),
            font_size=32,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=4,
            mass_fraction_powernorm_gamma=None,
            zoom_factor=2.9,
            zoom_xlim=(0.8e-7, 1.5e-7),
            zoom_font_size=32,
            zoom_cbar_fraction=0.04,
            zoom_cbar_aspect=6.0,
            zoom_x_nticks=3,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(6.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(9.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
    5: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.04,
            cbar_pad=0.05,
            cbar_aspect=5.0,
            figsize_contour=(15, 6.0),
            font_size=32,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=6,
            x_ticks=(0.0, 0.5e-7, 1.0e-7, 1.5e-7, 2.0e-7, 2.5e-7),
            mass_fraction_powernorm_gamma=None,
            zoom_factor=3.9,
            zoom_xlim=(1.3e-7, 2.5e-7),
            zoom_font_size=40,
            zoom_cbar_fraction=0.04,
            zoom_cbar_aspect=5.0,
            zoom_x_nticks=4,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(6.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(10.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
    10: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.03,
            cbar_pad=0.05,
            cbar_aspect=4.0,
            figsize_contour=(15, 6.0),
            font_size=32,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=6,
            mass_fraction_powernorm_gamma=None,
            zoom_factor=2.9,
            zoom_xlim=(3.1e-7, 5.0e-7),
            zoom_font_size=40,
            zoom_cbar_fraction=0.06,
            zoom_cbar_aspect=4.0,
            zoom_x_nticks=5,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(6.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(12.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
    20: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.025,
            cbar_pad=0.03,
            cbar_aspect=2.5,
            figsize_contour=(15, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=11,
            mass_fraction_powernorm_gamma=None,
            zoom_factor=2.0,
            zoom_font_size=48,
            zoom_cbar_fraction=0.025,
            zoom_cbar_aspect=2.5,
            zoom_x_nticks=5,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(6.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(18.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
    40: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.020,
            cbar_pad=0.03,
            cbar_aspect=1.5,
            figsize_contour=(20, 10.0),
            font_size=20,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=11,
            mass_fraction_powernorm_gamma=None,
            zoom_factor=2.0,
            zoom_font_size=20,
            zoom_cbar_fraction=0.020,
            zoom_cbar_aspect=1.8,
            zoom_x_nticks=4,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(10.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(40.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
    100: {
        "contour": dict(
            levels=900,
            enable_probe=False,
            dpi=70,
            cbar_shrink=0.8,
            cbar_fraction=0.015,
            cbar_pad=0.03,
            cbar_aspect=1.4,
            figsize_contour=(25, 10.0),
            font_size=20,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            cbar_nticks=3,
            cbar_decimals=2,
            inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
            inversion_product_negative_range_scale=DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
            inversion_product_positive_range_scale=DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
            minline_linewidth=2.0,
            ax_aspect=1.0,
            x_nticks=11,
            mass_fraction_powernorm_gamma=None,
            zoom_factor=2.0,
            zoom_font_size=48,
            zoom_cbar_fraction=0.015,
            zoom_cbar_aspect=1.4,
            zoom_x_nticks=5,
        ),
        "centerline": dict(
            dpi=70,
            figsize_centerline=(12.0, 6.0),
            font_size=24,
            tick_length=3.5,
            tick_width=2.5,
            box_linewidth=3.0,
            centerline_linewidth=3.0,
        ),
        "surface": dict(
            dpi=70,
            figsize=(12.0, 5.0),
            zoom_scale=1.0,
            figsize_contrib=(40.0, 8.0),
            zoom_scale_contrib=1.0,
            font_size=22,
            tick_length=3.5,
            tick_width=3.5,
            box_linewidth=3.0,
            line_width=3.0,
            wall_dashed_width=2.0,
            end_zoom_rate_pad_frac=0.40,
            end_zoom_rate_pad_frac_contrib=0.15,
        ),
    },
}


def get_case_plot_settings(aspect_ratio):
    aspect_ratio = int(aspect_ratio)
    if aspect_ratio in PLOT_CASE_SETTINGS:
        case_plot_cfg = PLOT_CASE_SETTINGS[aspect_ratio]
    else:
        available = sorted(PLOT_CASE_SETTINGS)
        lower_or_equal = [ar for ar in available if ar <= aspect_ratio]
        template_ar = lower_or_equal[-1] if lower_or_equal else available[-1]
        case_plot_cfg = PLOT_CASE_SETTINGS[template_ar]
    contour_cfg = dict(case_plot_cfg["contour"])
    contour_cfg.setdefault(
        "inversion_product_range_source",
        DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
    )
    contour_cfg.setdefault(
        "inversion_product_negative_range_scale",
        DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
    )
    contour_cfg.setdefault(
        "inversion_product_positive_range_scale",
        DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
    )
    if "x_ticks" not in contour_cfg or aspect_ratio not in PLOT_CASE_SETTINGS:
        contour_cfg["x_ticks"] = default_contour_x_ticks(aspect_ratio)
    if contour_cfg.get("x_ticks") is not None:
        contour_cfg["x_nticks"] = len(contour_cfg["x_ticks"])
    return {
        "contour": contour_cfg,
        "centerline": dict(case_plot_cfg["centerline"]),
        "surface": dict(case_plot_cfg["surface"]),
    }


def build_surface_plot_cfg(aspect_ratio, half_width):
    surface_cfg = get_case_plot_settings(aspect_ratio)["surface"]
    zoom_scale = float(surface_cfg.pop("zoom_scale", 1.0))

    figsize = tuple(float(v) for v in np.asarray(surface_cfg["figsize"], dtype=float))
    surface_cfg["figsize"] = figsize
    surface_cfg["zoom_figsize"] = tuple(
        float(v) for v in zoom_scale * np.asarray(figsize, dtype=float)
    )

    zoom_scale_contrib = float(surface_cfg.pop("zoom_scale_contrib", zoom_scale))
    figsize_contrib = tuple(
        float(v) for v in np.asarray(surface_cfg.get("figsize_contrib", figsize), dtype=float)
    )
    surface_cfg["figsize_contrib"] = figsize_contrib
    surface_cfg["zoom_figsize_contrib"] = tuple(
        float(v) for v in zoom_scale_contrib * np.asarray(figsize_contrib, dtype=float)
    )

    surface_cfg["end_zoom_rate_pad_frac_contrib"] = float(
        surface_cfg.get("end_zoom_rate_pad_frac_contrib", surface_cfg.get("end_zoom_rate_pad_frac", 0.08))
    )

    half_width = float(half_width)
    surface_cfg["end_zoom_pad"] = 2.5 * half_width
    return surface_cfg

# ============================================================
# Progress helper
# ============================================================
class Progress:
    def __init__(self, enabled=True, prefix=""):
        self.enabled = enabled
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.prefix = f"{prefix} " if prefix else ""

    def log(self, msg):
        if not self.enabled:
            return
        with self.lock:
            dt = time.perf_counter() - self.t0
            print(f"{self.prefix}[{dt:9.2f}s] {msg}", flush=True)

    def emit(self, msg):
        if not self.enabled:
            return
        with self.lock:
            print(f"{self.prefix}{msg}", flush=True)

# ============================================================
# Jet field
# ============================================================
def jet_fields(x, y, h, n_in, u_in, T_in, R, n_quad):
    """
    Compute n_jet, (nu)_jet, (nv)_jet for a slit of width 2h at x=0.

    IMPORTANT:
    - n_jet is number density.
    - nu_jet and nv_jet are species molecular(number) fluxes, NOT diffusive fluxes.
      nu_jet = n_jet * u_x
      nv_jet = n_jet * u_y

    x, y can be 2D arrays (meshgrid).
    """
        
    beta_in = 1.0 / (2.0 * R * T_in)
    S = np.sqrt(beta_in) * u_in
    S2 = S * S

    # avoid division by zero at x=0
    x_eff = x

    theta1 = np.arctan((y - h) / x_eff) # shape: (ny, nx)
    theta2 = np.arctan((y + h) / x_eff) # shape: (ny, nx)

    dth = theta2 - theta1
    th_mid  = 0.5 * (theta1 + theta2) # map to [-1, 1] for Gauss-Legendre quadrature
    th_half = 0.5 * (theta2 - theta1)

    if u_in == 0:
        n_jet = n_in * (dth / (2.0 * np.pi))
        coef = n_in / (4.0 * np.sqrt(np.pi * beta_in))
        nu_jet = coef * (np.sin(theta2) - np.sin(theta1))
        nv_jet = coef * (np.cos(theta1) - np.cos(theta2))
        return n_jet, nu_jet, nv_jet

    # Gauss-Legendre nodes on [-1,1]
    t, w = leggauss(n_quad)
    
    # accumulated integrals (2D)
    I_n  = np.zeros_like(th_mid)
    I_nu = np.zeros_like(th_mid)

    # reuse temporary 2D arrays to reduce allocations
    th  = np.empty_like(th_mid)
    s   = np.empty_like(th_mid)
    c   = np.empty_like(th_mid)
    a   = np.empty_like(th_mid)
    exp_term = np.empty_like(th_mid)
    a2 = np.empty_like(th_mid)

    for k in range(n_quad):
        # th = th_mid + th_half * t[k]
        np.multiply(th_half, t[k], out=th)
        np.add(th_mid, th, out=th)

        np.sin(th, out=s)
        np.cos(th, out=c)

        # stabilization: exp(-S^2)*exp(a^2) -> exp(-S^2 sin^2 θ)
        np.multiply(s, s, out=exp_term)
        exp_term *= -S2
        np.exp(exp_term, out=exp_term)

        # a = S*cos(th)
        np.multiply(c, S, out=a)
        ea = erf(a)

        # I_n integrand = exp(...) * cos * erf(a)
        I_n += w[k] * (exp_term * c * ea)

        # I_nu integrand = (1 + 2a^2) * cos * exp(...) * (1 + erf(a))
        np.multiply(a, a, out=a2)      # a^2
        a2 *= 2.0
        a2 += 1.0                      # 1 + 2a^2
        I_nu += w[k] * (a2 * c * exp_term * (1.0 + ea))

    # Jacobian of the theta mapping
    I_n  *= th_half
    I_nu *= th_half

    # n_jet
    n_jet = n_in * (
        np.exp(-S2) * dth / (2.0 * np.pi)
        + 0.25 * (erf(S * np.sin(theta2)) - erf(S * np.sin(theta1)))
        + (S / (2.0 * np.sqrt(np.pi))) * I_n
    )

    # molecular(number) flux in x-direction (not diffusive flux)
    nu_jet = (n_in / (4.0 * np.pi * np.sqrt(beta_in))) * (
        np.exp(-S2) * (S * dth + 0.5 * S * (np.sin(2.0 * theta2) - np.sin(2.0 * theta1)))
        + np.sqrt(np.pi) * I_nu
    )

    # molecular(number) flux in y-direction (not diffusive flux)
    nv_jet = (n_in / (4.0 * np.sqrt(np.pi * beta_in))) * (
        np.cos(theta1) * np.exp(-(S2) * (np.sin(theta1)**2)) * (1.0 + erf(S * np.cos(theta1)))
        - np.cos(theta2) * np.exp(-(S2) * (np.sin(theta2)**2)) * (1.0 + erf(S * np.cos(theta2)))
    )

    return n_jet, nu_jet, nv_jet

# ============================================================
# Utilities
# ============================================================
def cell_centers_1d(a, b, n):
    """Split [a,b] into n cells and return cell centers (length n) and edges (length n+1)."""
    edges = np.linspace(a, b, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges

def mid_weights_1d(x):
    """Midpoint-rule weights on uniform cell-center grid."""
    dx = x[1] - x[0]
    return np.full_like(x, dx, dtype=np.float64)

# ============================================================
# Precompute kernels for iterative impacts (wall-to-wall normal flux)
# ============================================================
def precompute_impact_kernels(xw, yw, h, Lm, beta_w):
    """
    Cell-center wall grids:
      xw: centers in (0,Lm)
      yw: centers in (-h,h)

    Subtraction form for:
      - end -> bottom/top (nv at y=±h)
      - bottom/top -> end (nu at x=Lm)

    Returns:
      K_TB, K_BT             : unchanged (top<->bottom)
      K_EB_res, K_ET_res     : residual matrices (end->bottom/top)
      K_BE_res, K_TE_res     : residual matrices (bottom/top->end)
      C_EB, C_ET             : constant coefficient vectors (length nx)
      C_BE, C_TE             : constant coefficient vectors (length ny)

    IMPORTANT:
    - All kernels here are for molecular(number) flux, NOT diffusive flux.

    """
    nx = xw.size
    ny = yw.size

    # midpoint weights for cell-center sampling
    wx = mid_weights_1d(xw)  # (nx,)
    wy = mid_weights_1d(yw)  # (ny,)

    coef = 1.0 / (4.0 * np.sqrt(np.pi * beta_w))

    # ------------------------------------------------------------
    # top <-> bottom (always nonsingular: distance = 2h) molecular(number) flux not diffusive flux
    # ------------------------------------------------------------
    dx = xw[:, None] - xw[None, :]
    dx2 = dx * dx
    denom = (dx2 + (2.0 * h) ** 2) ** 1.5
    K_TB = (-(2.0 * h) ** 2 * coef) * (1.0 / denom) * wx[None, :]
    K_BT = (+(2.0 * h) ** 2 * coef) * (1.0 / denom) * wx[None, :]

    # ------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------
    Lx = (Lm - xw)          # (nx,)  L = Lm - xw  (cell centers => L>0)
    dx_e = (Lm - xw)        # (nx,)  u = Lm - s   (s on bottom/top)

    # end sources: s in yw (cell centers => not ±h)
    dy_b = (-h - yw)        # (ny,)  t = -h - s
    dy_t = (h - yw)         # (ny,)  t =  h - s

    # end evaluation: y in yw (cell centers => not ±h)
    d_be = (yw + h)         # (ny,)  d = y + h
    d_te = (h - yw)         # (ny,)  d = h - y

    # ------------------------------------------------------------
    # end -> bottom/top: 
    #   nv^{E->B} = C_EB(L) * n0 + L * ∫ K(s;L) (n(s)-n0) ds
    #   nv^{E->T} = C_ET(L) * n0 + L * ∫ K(s;L) (n(s)-n0) ds
    # molecular(number) flux not diffusive flux
    # ------------------------------------------------------------
    common_E = (1.0 - (Lx / np.sqrt(Lx * Lx + (2.0 * h) ** 2)))  # (nx,)
    C_EB = -coef * common_E
    C_ET = +coef * common_E

    # ------------------------------------------------------------
    # bottom/top -> end:
    #   nu^{B->E}(y) = C_BE(d) * n0 + d * ∫ K(s;d) (n(s)-n0) ds
    #   nu^{T->E}(y) = C_TE(d) * n0 + d * ∫ K(s;d) (n(s)-n0) ds
    # molecular(number) flux not diffusive flux
    # ------------------------------------------------------------
    C_BE = coef * (1.0 - (d_be / np.sqrt(Lm * Lm + d_be * d_be)))  # (ny,)
    C_TE = coef * (1.0 - (d_te / np.sqrt(Lm * Lm + d_te * d_te)))  # (ny,)

    # ------------------------------------------------------------
    # residual matrices (midpoint quadrature)
    # ------------------------------------------------------------
    # end -> bottom/top (integrate over end-wall y grid)
    L_col   = Lx[:, None]      # (nx,1)
    dyb_row = dy_b[None, :]    # (1,ny)
    dyt_row = dy_t[None, :]    # (1,ny)

    denom_EB = (L_col * L_col + dyb_row * dyb_row) ** 1.5
    denom_ET = (L_col * L_col + dyt_row * dyt_row) ** 1.5

    K_EB_res = (L_col * coef) * (dyb_row / denom_EB) * wy[None, :]
    K_ET_res = (L_col * coef) * (dyt_row / denom_ET) * wy[None, :]

    # bottom/top -> end (integrate over x grid on bottom/top)
    dx_row = dx_e[None, :]     # (1,nx)
    dbe_col = d_be[:, None]    # (ny,1)
    dte_col = d_te[:, None]    # (ny,1)

    denom_BE = (dx_row * dx_row + dbe_col * dbe_col) ** 1.5
    denom_TE = (dx_row * dx_row + dte_col * dte_col) ** 1.5

    K_BE_res = (dbe_col * coef) * (dx_row / denom_BE) * wx[None, :]
    K_TE_res = (dte_col * coef) * (dx_row / denom_TE) * wx[None, :]

    return {
        "K_TB": K_TB,
        "K_BT": K_BT,
        "K_EB_res": K_EB_res,
        "K_ET_res": K_ET_res,
        "K_BE_res": K_BE_res,
        "K_TE_res": K_TE_res,
        "C_EB": C_EB,
        "C_ET": C_ET,
        "C_BE": C_BE,
        "C_TE": C_TE,
    }

# ============================================================
# Iterative solver for n_w
# ============================================================
def solve_reactive_channel(
    nx, ny, h, Lm,
    R_A, R_B, T_in, u_in,
    n_in_A, n_in_B,
    T_w, reaction_probability,
    n_quad_jet, tol, max_iter,
    verbose, max_workers,
    log_prefix="",
):
    progress = Progress(enabled=verbose, prefix=log_prefix)

    beta_w_A = 1.0 / (2.0 * R_A * T_w)
    beta_w_B = 1.0 / (2.0 * R_B * T_w)
    Cnw_A = 2.0 * np.sqrt(np.pi * beta_w_A)
    Cnw_B = 2.0 * np.sqrt(np.pi * beta_w_B)

    # ------------------------------------------------------------
    # (A) cell-center grids (NO corners included)
    # ------------------------------------------------------------
    x, x_edges = cell_centers_1d(0.0, Lm, nx)      # (nx,) in (0,Lm)
    y, y_edges = cell_centers_1d(-h,  h, ny)       # (ny,) in (-h,h)

    X, Y = np.meshgrid(x, y, indexing="xy")        # field grid (cell centers)

    # ------------------------------------------------------------
    # (B) jet fields on cell-center field grid (for later plotting)
    # ------------------------------------------------------------
    progress.log("Jet field calculation (cell centers) ...")
    t_phase = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fA = ex.submit(jet_fields, X, Y, h, n_in_A, u_in, T_in, R_A, n_quad_jet)
        fB = ex.submit(jet_fields, X, Y, h, n_in_B, u_in, T_in, R_B, n_quad_jet)
        nJet_A, nuJet_A, nvJet_A = fA.result()
        nJet_B, nuJet_B, nvJet_B = fB.result()
    progress.log(f"Jet field done ({time.perf_counter() - t_phase:.2f}s)")

    # ------------------------------------------------------------
    # (C) jet normal molecular flux ON walls (sampled at wall-lines, but x/y are cell-centers), not diffusive flux
    #     - bottom/top: x = x_centers, y = ±h
    #     - end:        x = Lm, y = y_centers
    # ------------------------------------------------------------
    Xb = x[None, :]
    Yb = (-h) * np.ones((1, nx), dtype=np.float64)
    Xt = x[None, :]
    Yt = (+h) * np.ones((1, nx), dtype=np.float64)
    Xe = (Lm) * np.ones((ny, 1), dtype=np.float64)
    Ye = y[:, None]

    progress.log("Jet wall calculation ...")
    t_phase = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fAbot = ex.submit(jet_fields, Xb, Yb, h, n_in_A, u_in, T_in, R_A, n_quad_jet)
        fAtop = ex.submit(jet_fields, Xt, Yt, h, n_in_A, u_in, T_in, R_A, n_quad_jet)
        fAend = ex.submit(jet_fields, Xe, Ye, h, n_in_A, u_in, T_in, R_A, n_quad_jet)

        fBbot = ex.submit(jet_fields, Xb, Yb, h, n_in_B, u_in, T_in, R_B, n_quad_jet)
        fBtop = ex.submit(jet_fields, Xt, Yt, h, n_in_B, u_in, T_in, R_B, n_quad_jet)
        fBend = ex.submit(jet_fields, Xe, Ye, h, n_in_B, u_in, T_in, R_B, n_quad_jet)

        _, _, nvJet_bot_A_line = fAbot.result()
        _, _, nvJet_top_A_line = fAtop.result()
        _, nuJet_end_A_col, _  = fAend.result()

        _, _, nvJet_bot_B_line = fBbot.result()
        _, _, nvJet_top_B_line = fBtop.result()
        _, nuJet_end_B_col, _  = fBend.result()

    nvJet_bot_A = nvJet_bot_A_line[0, :].copy()
    nvJet_top_A = nvJet_top_A_line[0, :].copy()
    nuJet_end_A = nuJet_end_A_col[:, 0].copy()
    nvJet_bot_B = nvJet_bot_B_line[0, :].copy()
    nvJet_top_B = nvJet_top_B_line[0, :].copy()
    nuJet_end_B = nuJet_end_B_col[:, 0].copy()
    progress.log(f"Jet wall done ({time.perf_counter() - t_phase:.2f}s)")

    # ------------------------------------------------------------
    # initialize: jet-only impact using incoming normal molecular(number) flux, not diffusive flux!
    # ------------------------------------------------------------
    Gamma_imp_bot_A = -nvJet_bot_A
    Gamma_imp_bot_B = -nvJet_bot_B
    Gamma_imp_top_A =  nvJet_top_A
    Gamma_imp_top_B =  nvJet_top_B
    Gamma_imp_end_A =  nuJet_end_A
    Gamma_imp_end_B =  nuJet_end_B

    nw_bot_A = Cnw_A * ((1.0 - reaction_probability) * Gamma_imp_bot_A)
    nw_bot_B = Cnw_B * (reaction_probability * Gamma_imp_bot_A + Gamma_imp_bot_B)
    nw_top_A = Cnw_A * ((1.0 - reaction_probability) * Gamma_imp_top_A)
    nw_top_B = Cnw_B * (reaction_probability * Gamma_imp_top_A + Gamma_imp_top_B)
    nw_end_A = Cnw_A * ((1.0 - reaction_probability) * Gamma_imp_end_A)
    nw_end_B = Cnw_B * (reaction_probability * Gamma_imp_end_A + Gamma_imp_end_B)

    # ------------------------------------------------------------
    # kernels (cell-center x/y)
    # ------------------------------------------------------------
    progress.log("precompute kernel calculations ...")
    t_phase = time.perf_counter()
    kernels_A = precompute_impact_kernels(x, y, h, Lm, beta_w_A)
    progress.log(f"kernels A done ({time.perf_counter() - t_phase:.2f}s)")

    progress.log("precompute kernel calculations ...")
    t_phase = time.perf_counter()
    kernels_B = precompute_impact_kernels(x, y, h, Lm, beta_w_B)
    progress.log(f"kernels B done ({time.perf_counter() - t_phase:.2f}s)")

    # unpack A
    K_TB_A = kernels_A["K_TB"]; K_BT_A = kernels_A["K_BT"]
    K_EB_A = kernels_A["K_EB_res"]; K_ET_A = kernels_A["K_ET_res"]
    K_BE_A = kernels_A["K_BE_res"]; K_TE_A = kernels_A["K_TE_res"]
    C_EB_A = kernels_A["C_EB"]; C_ET_A = kernels_A["C_ET"]
    C_BE_A = kernels_A["C_BE"]; C_TE_A = kernels_A["C_TE"]

    # unpack B
    K_TB_B = kernels_B["K_TB"]; K_BT_B = kernels_B["K_BT"]
    K_EB_B = kernels_B["K_EB_res"]; K_ET_B = kernels_B["K_ET_res"]
    K_BE_B = kernels_B["K_BE_res"]; K_TE_B = kernels_B["K_TE_res"]
    C_EB_B = kernels_B["C_EB"]; C_ET_B = kernels_B["C_ET"]
    C_BE_B = kernels_B["C_BE"]; C_TE_B = kernels_B["C_TE"]

    # reference indices (cell-center endpoints near corners)
    ix_end = nx - 1     # x ~ Lm - dx/2
    iy_bot = 0          # y ~ -h + dy/2
    iy_top = ny - 1     # y ~ +h - dy/2

    rel = np.inf
    rel_old = np.inf
    converged = False

    for it in range(1, max_iter + 1):
        old = np.concatenate([nw_bot_A, nw_bot_B, nw_top_A, nw_top_B, nw_end_A, nw_end_B])

        # -------------------------
        # A wall->wall reflected normal molecular(number) flux, not diffusive flux!
        # -------------------------
        nv_top_to_bot_A = K_TB_A @ nw_top_A
        nv_bot_to_top_A = K_BT_A @ nw_bot_A

        n0_end_bot_A = nw_end_A[iy_bot]
        n0_end_top_A = nw_end_A[iy_top]
        nv_end_to_bot_A = C_EB_A * n0_end_bot_A + (K_EB_A @ (nw_end_A - n0_end_bot_A))
        nv_end_to_top_A = C_ET_A * n0_end_top_A + (K_ET_A @ (nw_end_A - n0_end_top_A))

        n0_bot_end_A = nw_bot_A[ix_end]
        n0_top_end_A = nw_top_A[ix_end]
        nu_bot_to_end_A = C_BE_A * n0_bot_end_A + (K_BE_A @ (nw_bot_A - n0_bot_end_A))
        nu_top_to_end_A = C_TE_A * n0_top_end_A + (K_TE_A @ (nw_top_A - n0_top_end_A))

        # -------------------------
        # B wall->wall reflected normal molecular(number) flux, not diffusive flux!
        # -------------------------
        nv_top_to_bot_B = K_TB_B @ nw_top_B
        nv_bot_to_top_B = K_BT_B @ nw_bot_B

        n0_end_bot_B = nw_end_B[iy_bot]
        n0_end_top_B = nw_end_B[iy_top]
        nv_end_to_bot_B = C_EB_B * n0_end_bot_B + (K_EB_B @ (nw_end_B - n0_end_bot_B))
        nv_end_to_top_B = C_ET_B * n0_end_top_B + (K_ET_B @ (nw_end_B - n0_end_top_B))

        n0_bot_end_B = nw_bot_B[ix_end]
        n0_top_end_B = nw_top_B[ix_end]
        nu_bot_to_end_B = C_BE_B * n0_bot_end_B + (K_BE_B @ (nw_bot_B - n0_bot_end_B))
        nu_top_to_end_B = C_TE_B * n0_top_end_B + (K_TE_B @ (nw_top_B - n0_top_end_B))

        # -------------------------
        # impacted normal molecular(number) flux on each wall-line (no clamp), not diffusive flux!
        # -------------------------
        nv_imp_bot_A = nvJet_bot_A + nv_top_to_bot_A + nv_end_to_bot_A
        nv_imp_bot_B = nvJet_bot_B + nv_top_to_bot_B + nv_end_to_bot_B

        nv_imp_top_A = nvJet_top_A + nv_bot_to_top_A + nv_end_to_top_A
        nv_imp_top_B = nvJet_top_B + nv_bot_to_top_B + nv_end_to_top_B

        nu_imp_end_A = nuJet_end_A + nu_bot_to_end_A + nu_top_to_end_A
        nu_imp_end_B = nuJet_end_B + nu_bot_to_end_B + nu_top_to_end_B

        # incoming molecular number flux magnitude on each wall
        # bottom/top use wall-normal nv, end uses wall-normal nu
        Gamma_imp_bot_A = -nv_imp_bot_A
        Gamma_imp_bot_B = -nv_imp_bot_B
        Gamma_imp_top_A =  nv_imp_top_A
        Gamma_imp_top_B =  nv_imp_top_B
        Gamma_imp_end_A =  nu_imp_end_A
        Gamma_imp_end_B =  nu_imp_end_B

        # update wall densities
        nw_bot_A = Cnw_A * ((1.0 - reaction_probability) * Gamma_imp_bot_A)
        nw_bot_B = Cnw_B * (reaction_probability * Gamma_imp_bot_A + Gamma_imp_bot_B)
        nw_top_A = Cnw_A * ((1.0 - reaction_probability) * Gamma_imp_top_A)
        nw_top_B = Cnw_B * (reaction_probability * Gamma_imp_top_A + Gamma_imp_top_B)
        nw_end_A = Cnw_A * ((1.0 - reaction_probability) * Gamma_imp_end_A)
        nw_end_B = Cnw_B * (reaction_probability * Gamma_imp_end_A + Gamma_imp_end_B)

        new = np.concatenate([nw_bot_A, nw_bot_B, nw_top_A, nw_top_B, nw_end_A, nw_end_B])
        denom = np.maximum(np.max(np.abs(old)), 1e-20)
        if it == 1:
            rel_old = 0
        if (it % 500 == 0):
            rel_old = rel
   
        rel = np.max(np.abs(new - old)) / denom * 1e10

        if verbose and (it == 1 or it % 500 == 0 or rel < tol):
            progress.emit(f"[iter {it:4d}] rel_change={rel:.3e}")

        if rel < tol or (rel_old == rel and rel / 1e10 < tol):
            converged = True
            break

    if not converged and verbose:
        progress.emit(
            f"WARNING: did not converge within max_iter={max_iter} "
            f"(final_rel_change={rel:.3e})"
        )

    imp = {
        "Gamma_imp": { # incoming normal molecular(number) flux
            "bottom": {"A": Gamma_imp_bot_A.copy(), "B": Gamma_imp_bot_B.copy()},
            "top":    {"A": Gamma_imp_top_A.copy(), "B": Gamma_imp_top_B.copy()},
            "end":    {"A": Gamma_imp_end_A.copy(), "B": Gamma_imp_end_B.copy()},
        },
        "raw_normal_flux": { # raw molecular(number) flux components before wall-normal sign convention
            "nv_imp_bottom": {"A": nv_imp_bot_A.copy(), "B": nv_imp_bot_B.copy()},
            "nv_imp_top":    {"A": nv_imp_top_A.copy(), "B": nv_imp_top_B.copy()},
            "nu_imp_end":    {"A": nu_imp_end_A.copy(), "B": nu_imp_end_B.copy()},
        },
        # Contributions to the incoming normal flux on each wall, stored with the
        # same sign convention as Gamma_imp so they can be used directly for
        # source-decomposed wall flux/reaction postprocessing.
        "Gamma_imp_components": {
            "bottom": {
                "A": {
                    "jet": (-nvJet_bot_A).copy(),
                    "top": (-nv_top_to_bot_A).copy(),
                    "end": (-nv_end_to_bot_A).copy(),
                },
                "B": {
                    "jet": (-nvJet_bot_B).copy(),
                    "top": (-nv_top_to_bot_B).copy(),
                    "end": (-nv_end_to_bot_B).copy(),
                },
            },
            "top": {
                "A": {
                    "jet": nvJet_top_A.copy(),
                    "bottom": nv_bot_to_top_A.copy(),
                    "end": nv_end_to_top_A.copy(),
                },
                "B": {
                    "jet": nvJet_top_B.copy(),
                    "bottom": nv_bot_to_top_B.copy(),
                    "end": nv_end_to_top_B.copy(),
                },
            },
            "end": {
                "A": {
                    "jet": nuJet_end_A.copy(),
                    "bottom": nu_bot_to_end_A.copy(),
                    "top": nu_top_to_end_A.copy(),
                },
                "B": {
                    "jet": nuJet_end_B.copy(),
                    "bottom": nu_bot_to_end_B.copy(),
                    "top": nu_top_to_end_B.copy(),
                },
            },
        },
    }

    return {
        "x": x, "y": y, "X": X, "Y": Y,
        "h": h, "Lm": Lm, "R_A": R_A, "R_B": R_B,
        "beta_w_A": beta_w_A, "beta_w_B": beta_w_B,
        "n_in_A": float(n_in_A),
        "n_in_B": float(n_in_B),
        "n_in": {"A": float(n_in_A), "B": float(n_in_B)},
        "reaction_probability": reaction_probability,
        "jet": {
            "A": {"n": nJet_A, "nu": nuJet_A, "nv": nvJet_A},
            "B": {"n": nJet_B, "nu": nuJet_B, "nv": nvJet_B},
        },
        "nw": {
            "bottom": {"A": nw_bot_A, "B": nw_bot_B},
            "top":    {"A": nw_top_A, "B": nw_top_B},
            "end":    {"A": nw_end_A, "B": nw_end_B},
        },
        "imp": imp,
        "iterations": it,
        "converged": converged,
        "final_rel_change": float(rel),
    }


def format_solver_status(sol, label=None):
    status = "converged" if bool(sol.get("converged", False)) else "NOT converged"
    iterations = sol.get("iterations", "?")
    final_rel_change = sol.get("final_rel_change")
    rel_text = ""
    if final_rel_change is not None and np.isfinite(final_rel_change):
        rel_text = f", final_rel_change={float(final_rel_change):.3e}"
    prefix = f"{label}: " if label else ""
    return f"{prefix}{status} in {iterations} iterations{rel_text}"

# ============================================================
# FFT Toeplitz helpers
# ============================================================
def _toeplitz_embed_from_gshift(g_shift, n, m_fft):
    """
    g_shift stores g[m] for m=-(n-1)..(n-1) in order:
        g_shift[n-1 + m] = g[m]
    We build circulant embedding vector p of length m_fft (>=2n):
        p = [c, 0...0, r_tail]
    where c is first column, r_tail is reversed first row excluding r0.
    """
    if m_fft < 2 * n:
        raise ValueError("m_fft must be >= 2*n")
    c = g_shift[n-1:n-1+n]       # g[0..n-1]
    r = g_shift[n-1::-1][:n]     # [g[0], g[-1], ..., g[-(n-1)]]
    p = np.zeros(m_fft, dtype=np.float64)
    p[:n] = c
    # tail is g[-(n-1)]..g[-1] = r[n-1]..r[1]
    p[m_fft-(n-1):] = r[1:][::-1]
    return p

def _precompute_V(v, n, m_fft):
    vpad = np.zeros(m_fft, dtype=np.float64)
    vpad[:n] = v
    return np.fft.rfft(vpad)

def _apply_toeplitz_fft(P, V, m_fft, n):
    ypad = np.fft.irfft(P * V, n=m_fft)
    return ypad[:n]

# ============================================================
# FFT-accelerated wall contributions (full field reconstruction)
# ============================================================
def wall_contrib_bottom_fft(x, y, h, Lm, beta_w, nw_x):
    """
    Reconstruct bottom-wall contribution to the field.

    Returns
    -------
    n  : number density contribution
    nu : x-direction molecular(number) flux contribution (NOT diffusive flux)
    nv : y-direction molecular(number) flux contribution (NOT diffusive flux)
    """
    nx = x.size
    ny = y.size
    dx = x[1] - x[0]
    wx = mid_weights_1d(x)
    coef = 1.0 / (4.0 * np.sqrt(np.pi * beta_w))

    m_fft = next_fast_len(2 * nx)
    m = (np.arange(-(nx-1), nx, dtype=np.float64) * dx)  # offsets

    vec = nw_x * wx
    V_vec = _precompute_V(vec, nx, m_fft)
    V_wx  = _precompute_V(wx,  nx, m_fft)

    n  = np.zeros((ny, nx), dtype=np.float64)
    nu = np.zeros((ny, nx), dtype=np.float64)
    nv = np.zeros((ny, nx), dtype=np.float64)

    u1 = (Lm - x)
    u0 = (-x)

    for j in range(0, ny):
        yp = y[j] + h  # cell centers => yp>0

        denom = m*m + yp*yp
        g_inv_shift     = 1.0 / denom
        g_inv32_shift   = 1.0 / (denom ** 1.5)
        g_m_inv32_shift = m * g_inv32_shift

        # constant part
        n_const  = (nw_x / (2.0 * np.pi)) * (np.arctan(u1 / yp) - np.arctan(u0 / yp))
        nu_const = (yp * coef) * nw_x * (1.0 / np.sqrt(u1*u1 + yp*yp) - 1.0 / np.sqrt(u0*u0 + yp*yp))
        nv_const = coef * nw_x * (u1 / np.sqrt(u1*u1 + yp*yp) - u0 / np.sqrt(u0*u0 + yp*yp))

        P_inv     = np.fft.rfft(_toeplitz_embed_from_gshift(g_inv_shift, nx, m_fft))
        P_inv32   = np.fft.rfft(_toeplitz_embed_from_gshift(g_inv32_shift, nx, m_fft))
        P_m_inv32 = np.fft.rfft(_toeplitz_embed_from_gshift(g_m_inv32_shift, nx, m_fft))

        I_n_total = _apply_toeplitz_fft(P_inv, V_vec, m_fft, nx)
        I_n_w     = _apply_toeplitz_fft(P_inv, V_wx,  m_fft, nx)
        I_n_res   = I_n_total - nw_x * I_n_w

        I_nv_total = _apply_toeplitz_fft(P_inv32, V_vec, m_fft, nx)
        I_nv_w     = _apply_toeplitz_fft(P_inv32, V_wx,  m_fft, nx)
        I_nv_res   = I_nv_total - nw_x * I_nv_w

        I_nu_total = _apply_toeplitz_fft(P_m_inv32, V_vec, m_fft, nx)
        I_nu_w     = _apply_toeplitz_fft(P_m_inv32, V_wx,  m_fft, nx)
        I_nu_res   = I_nu_total - nw_x * I_nu_w

        n[j, :]  = n_const  + (yp / (2.0 * np.pi)) * I_n_res
        nu[j, :] = nu_const + (yp * coef) * I_nu_res
        nv[j, :] = nv_const + ((yp**2) * coef) * I_nv_res

    return n, nu, nv

def wall_contrib_top_fft(x, y, h, Lm, beta_w, nw_x):
    """
    Reconstruct top-wall contribution to the field.

    Returns
    -------
    n  : number density contribution
    nu : x-direction molecular(number) flux contribution (NOT diffusive flux)
    nv : y-direction molecular(number) flux contribution (NOT diffusive flux)
    """
    nx = x.size
    ny = y.size
    dx = x[1] - x[0]
    wx = mid_weights_1d(x)
    coef = 1.0 / (4.0 * np.sqrt(np.pi * beta_w))

    m_fft = next_fast_len(2 * nx)
    m = (np.arange(-(nx-1), nx, dtype=np.float64) * dx)

    vec = nw_x * wx
    V_vec = _precompute_V(vec, nx, m_fft)
    V_wx  = _precompute_V(wx,  nx, m_fft)

    n  = np.zeros((ny, nx), dtype=np.float64)
    nu = np.zeros((ny, nx), dtype=np.float64)
    nv = np.zeros((ny, nx), dtype=np.float64)

    u1 = (Lm - x)
    u0 = (-x)

    for j in range(0, ny):
        ym = h - y[j]  # cell centers => ym>0

        denom = m*m + ym*ym
        g_inv_shift     = 1.0 / denom
        g_inv32_shift   = 1.0 / (denom ** 1.5)
        g_m_inv32_shift = m * g_inv32_shift

        n_const  = (nw_x / (2.0 * np.pi)) * (np.arctan(u1 / ym) - np.arctan(u0 / ym))
        nu_const = (ym * coef) * nw_x * (1.0 / np.sqrt(u1*u1 + ym*ym) - 1.0 / np.sqrt(u0*u0 + ym*ym))
        nv_const = -coef * nw_x * (u1 / np.sqrt(u1*u1 + ym*ym) - u0 / np.sqrt(u0*u0 + ym*ym))

        P_inv     = np.fft.rfft(_toeplitz_embed_from_gshift(g_inv_shift, nx, m_fft))
        P_inv32   = np.fft.rfft(_toeplitz_embed_from_gshift(g_inv32_shift, nx, m_fft))
        P_m_inv32 = np.fft.rfft(_toeplitz_embed_from_gshift(g_m_inv32_shift, nx, m_fft))

        I_n_total = _apply_toeplitz_fft(P_inv, V_vec, m_fft, nx)
        I_n_w     = _apply_toeplitz_fft(P_inv, V_wx,  m_fft, nx)
        I_n_res   = I_n_total - nw_x * I_n_w

        I_nv_total = _apply_toeplitz_fft(P_inv32, V_vec, m_fft, nx)
        I_nv_w     = _apply_toeplitz_fft(P_inv32, V_wx,  m_fft, nx)
        I_nv_res   = I_nv_total - nw_x * I_nv_w

        I_nu_total = _apply_toeplitz_fft(P_m_inv32, V_vec, m_fft, nx)
        I_nu_w     = _apply_toeplitz_fft(P_m_inv32, V_wx,  m_fft, nx)
        I_nu_res   = I_nu_total - nw_x * I_nu_w

        n[j, :]  = n_const  + (ym / (2.0 * np.pi)) * I_n_res
        nu[j, :] = nu_const + (ym * coef) * I_nu_res
        nv[j, :] = nv_const + (-(ym**2) * coef) * I_nv_res

    return n, nu, nv

def wall_contrib_end_fft(x, y, h, Lm, beta_w, nw_y):
    """
    Reconstruct end-wall contribution to the field.

    Returns
    -------
    n  : number density contribution
    nu : x-direction molecular(number) flux contribution (NOT diffusive flux)
    nv : y-direction molecular(number) flux contribution (NOT diffusive flux)
    """
    nx = x.size
    ny = y.size
    dy = y[1] - y[0]
    wy = mid_weights_1d(y)
    coef = 1.0 / (4.0 * np.sqrt(np.pi * beta_w))

    m_fft = next_fast_len(2 * ny)
    m = (np.arange(-(ny-1), ny, dtype=np.float64) * dy)

    vec = nw_y * wy
    V_vec = _precompute_V(vec, ny, m_fft)
    V_wy  = _precompute_V(wy,  ny, m_fft)

    n  = np.zeros((ny, nx), dtype=np.float64)
    nu = np.zeros((ny, nx), dtype=np.float64)
    nv = np.zeros((ny, nx), dtype=np.float64)

    for i in range(0, nx):
        L = Lm - x[i]  # cell centers => L>0

        denom = L*L + m*m
        g_inv_shift     = 1.0 / denom
        g_inv32_shift   = 1.0 / (denom ** 1.5)
        g_m_inv32_shift = m * g_inv32_shift

        n_const = (nw_y / (2.0 * np.pi)) * (np.arctan((y + h) / L) - np.arctan((y - h) / L))

        Aplus  = (y + h) / np.sqrt(L*L + (y + h)**2)
        Aminus = (y - h) / np.sqrt(L*L + (y - h)**2)
        nu_const = -coef * nw_y * (Aplus - Aminus)

        nv_const = (L * coef) * nw_y * (
            1.0 / np.sqrt(L*L + (y - h)**2) - 1.0 / np.sqrt(L*L + (y + h)**2)
        )

        P_inv     = np.fft.rfft(_toeplitz_embed_from_gshift(g_inv_shift, ny, m_fft))
        P_inv32   = np.fft.rfft(_toeplitz_embed_from_gshift(g_inv32_shift, ny, m_fft))
        P_m_inv32 = np.fft.rfft(_toeplitz_embed_from_gshift(g_m_inv32_shift, ny, m_fft))

        I_n_total = _apply_toeplitz_fft(P_inv, V_vec, m_fft, ny)
        I_n_w     = _apply_toeplitz_fft(P_inv, V_wy,  m_fft, ny)
        I_n_res   = I_n_total - nw_y * I_n_w

        I_nu_total = _apply_toeplitz_fft(P_inv32, V_vec, m_fft, ny)
        I_nu_w     = _apply_toeplitz_fft(P_inv32, V_wy,  m_fft, ny)
        I_nu_res   = I_nu_total - nw_y * I_nu_w

        I_nv_total = _apply_toeplitz_fft(P_m_inv32, V_vec, m_fft, ny)
        I_nv_w     = _apply_toeplitz_fft(P_m_inv32, V_wy,  m_fft, ny)
        I_nv_res   = I_nv_total - nw_y * I_nv_w

        n[:, i]  = n_const  + (L / (2.0 * np.pi)) * I_n_res
        nu[:, i] = nu_const + (-(L**2) * coef) * I_nu_res
        nv[:, i] = nv_const + (L * coef) * I_nv_res

    return n, nu, nv

# ============================================================
# Full-field reconstruction (FFT version) + parallel wrapper
# ============================================================
def compute_full_fields(sol, mass_A, mass_B, max_workers=6, verbose=True, log_prefix=""):
    """
    Using converged wall densities, compute full n, nu, nv for species A and B,
    then compute mass fractions YA, YB.

    This version uses FFT Toeplitz acceleration in wall_contrib_*_fft.

    IMPORTANT:
    - nu and nv returned here are total species molecular(number) fluxes,
      NOT diffusive fluxes.
    - Diffusive flux must be computed afterwards from the total species fields.    
    """
    progress = Progress(enabled=verbose, prefix=log_prefix)

    x = sol["x"]; y = sol["y"]
    h = sol["h"]; Lm = sol["Lm"]
    beta_w_A = sol["beta_w_A"]
    beta_w_B = sol["beta_w_B"]

    # Jet
    nJet_A  = sol["jet"]["A"]["n"];  nuJet_A = sol["jet"]["A"]["nu"];  nvJet_A = sol["jet"]["A"]["nv"]
    nJet_B  = sol["jet"]["B"]["n"];  nuJet_B = sol["jet"]["B"]["nu"];  nvJet_B = sol["jet"]["B"]["nv"]

    # Walls
    nw_bot_A = sol["nw"]["bottom"]["A"]; nw_bot_B = sol["nw"]["bottom"]["B"]
    nw_top_A = sol["nw"]["top"]["A"];    nw_top_B = sol["nw"]["top"]["B"]
    nw_end_A = sol["nw"]["end"]["A"];    nw_end_B = sol["nw"]["end"]["B"]

    progress.log("field calculation (FFT) ...")
    t_phase = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(wall_contrib_bottom_fft, x, y, h, Lm, beta_w_A, nw_bot_A): "bottom A",
            ex.submit(wall_contrib_bottom_fft, x, y, h, Lm, beta_w_B, nw_bot_B): "bottom B",
            ex.submit(wall_contrib_top_fft,    x, y, h, Lm, beta_w_A, nw_top_A): "top A",
            ex.submit(wall_contrib_top_fft,    x, y, h, Lm, beta_w_B, nw_top_B): "top B",
            ex.submit(wall_contrib_end_fft,    x, y, h, Lm, beta_w_A, nw_end_A): "end A",
            ex.submit(wall_contrib_end_fft,    x, y, h, Lm, beta_w_B, nw_end_B): "end B",
        }

        results = {}
        for fut in as_completed(futures):
            name = futures[fut]
            results[name] = fut.result()
            progress.log(f"done: {name}")

    progress.log(f"field calculation done ({time.perf_counter()-t_phase:.2f}s)")

    nB_A, nuB_A, nvB_A = results["bottom A"]
    nB_B, nuB_B, nvB_B = results["bottom B"]
    nT_A, nuT_A, nvT_A = results["top A"]
    nT_B, nuT_B, nvT_B = results["top B"]
    nE_A, nuE_A, nvE_A = results["end A"]
    nE_B, nuE_B, nvE_B = results["end B"]

    # Total molecular number density and molecular number flux per species, not diffusive flux!
    nA  = nJet_A  + nB_A  + nT_A  + nE_A
    nuA = nuJet_A + nuB_A + nuT_A + nuE_A
    nvA = nvJet_A + nvB_A + nvT_A + nvE_A

    nB  = nJet_B  + nB_B  + nT_B  + nE_B
    nuB = nuJet_B + nuB_B + nuT_B + nuE_B
    nvB = nvJet_B + nvB_B + nvT_B + nvE_B

    rho_A = mass_A * nA
    rho_B = mass_B * nB
    rho_total = rho_A + rho_B
    YA = safe_divide(rho_A, rho_total)
    YB = safe_divide(rho_B, rho_total)

    cA = nA / AVOGADRO_CONSTANT
    cB = nB / AVOGADRO_CONSTANT
    c_total = cA + cB
    XA = safe_divide(cA, c_total)
    XB = safe_divide(cB, c_total)

    fields_total = {
        "A": {"n": nA, "c": cA, "nu": nuA, "nv": nvA},
        "B": {"n": nB, "c": cB, "nu": nuB, "nv": nvB},
        "YA": YA, "YB": YB,
        "XA": XA, "XB": XB,
    }
    contrib = {
        "A": {
            "jet":    {"n": nJet_A, "c": nJet_A / AVOGADRO_CONSTANT, "nu": nuJet_A, "nv": nvJet_A},
            "bottom": {"n": nB_A, "c": nB_A / AVOGADRO_CONSTANT, "nu": nuB_A, "nv": nvB_A},
            "top":    {"n": nT_A, "c": nT_A / AVOGADRO_CONSTANT, "nu": nuT_A, "nv": nvT_A},
            "end":    {"n": nE_A, "c": nE_A / AVOGADRO_CONSTANT, "nu": nuE_A, "nv": nvE_A},
        },
        "B": {
            "jet":    {"n": nJet_B, "c": nJet_B / AVOGADRO_CONSTANT, "nu": nuJet_B, "nv": nvJet_B},
            "bottom": {"n": nB_B, "c": nB_B / AVOGADRO_CONSTANT, "nu": nuB_B, "nv": nvB_B},
            "top":    {"n": nT_B, "c": nT_B / AVOGADRO_CONSTANT, "nu": nuT_B, "nv": nvT_B},
            "end":    {"n": nE_B, "c": nE_B / AVOGADRO_CONSTANT, "nu": nuE_B, "nv": nvE_B},
        },
    }
    return fields_total, contrib

def compute_diffusive_mass_flux_fields(fields_total, mass_A, mass_B):
    """
    Compute diffusive mass and molar flux fields from total species fields.

    IMPORTANT:
    - fields_total["A"]["nu"], fields_total["A"]["nv"],
      fields_total["B"]["nu"], fields_total["B"]["nv"]
      are species molecular(number) fluxes, NOT diffusive fluxes.

    - The diffusive mass flux is computed afterwards from the total fields:
          j_i = rho_i * (u_i - u_mix)
              = m_i * Gamma_i - Y_i * (m_A * Gamma_A + m_B * Gamma_B)

    - The molar diffusive flux is computed in the molar-average frame:
          J_i = N_i - X_i * (N_A + N_B)

      where
          Gamma_i = species molecular(number) flux
          N_i     = Gamma_i / N_Avogadro
          rho_i   = m_i * n_i
          Y_i     = rho_i / rho_total
          X_i     = c_i / c_total

    Returns
    -------
    diff_fields : dict
        diff_fields["A"]["ju_diff"], diff_fields["A"]["jv_diff"]
        diff_fields["B"]["ju_diff"], diff_fields["B"]["jv_diff"]
        diff_fields["A"]["Ju_diff"], diff_fields["A"]["Jv_diff"]
        diff_fields["B"]["Ju_diff"], diff_fields["B"]["Jv_diff"]
    """
    nA = fields_total["A"]["n"]
    nB = fields_total["B"]["n"]

    # species molecular(number) flux, not diffusive flux
    Gamma_u_A = fields_total["A"]["nu"]
    Gamma_v_A = fields_total["A"]["nv"]
    Gamma_u_B = fields_total["B"]["nu"]
    Gamma_v_B = fields_total["B"]["nv"]

    rho_A = mass_A * nA
    rho_B = mass_B * nB
    rho_total = rho_A + rho_B

    Y_A = fields_total.get("YA")
    Y_B = fields_total.get("YB")
    if Y_A is None or Y_B is None:
        Y_A = safe_divide(rho_A, rho_total)
        Y_B = safe_divide(rho_B, rho_total)

    c_A = fields_total["A"].get("c", nA / AVOGADRO_CONSTANT)
    c_B = fields_total["B"].get("c", nB / AVOGADRO_CONSTANT)
    c_total = c_A + c_B
    X_A = fields_total.get("XA")
    X_B = fields_total.get("XB")
    if X_A is None or X_B is None:
        X_A = safe_divide(c_A, c_total)
        X_B = safe_divide(c_B, c_total)

    # total mixture mass flux, not diffusive flux
    mass_flux_mix_u = mass_A * Gamma_u_A + mass_B * Gamma_u_B
    mass_flux_mix_v = mass_A * Gamma_v_A + mass_B * Gamma_v_B

    # diffusive mass flux
    ju_diff_A = mass_A * Gamma_u_A - Y_A * mass_flux_mix_u
    jv_diff_A = mass_A * Gamma_v_A - Y_A * mass_flux_mix_v

    ju_diff_B = mass_B * Gamma_u_B - Y_B * mass_flux_mix_u
    jv_diff_B = mass_B * Gamma_v_B - Y_B * mass_flux_mix_v

    molar_flux_A_u = Gamma_u_A / AVOGADRO_CONSTANT
    molar_flux_A_v = Gamma_v_A / AVOGADRO_CONSTANT
    molar_flux_B_u = Gamma_u_B / AVOGADRO_CONSTANT
    molar_flux_B_v = Gamma_v_B / AVOGADRO_CONSTANT
    molar_flux_mix_u = molar_flux_A_u + molar_flux_B_u
    molar_flux_mix_v = molar_flux_A_v + molar_flux_B_v

    Ju_diff_A = molar_flux_A_u - X_A * molar_flux_mix_u
    Jv_diff_A = molar_flux_A_v - X_A * molar_flux_mix_v
    Ju_diff_B = molar_flux_B_u - X_B * molar_flux_mix_u
    Jv_diff_B = molar_flux_B_v - X_B * molar_flux_mix_v

    # ju_diff_A = mass_A * Gamma_u_A - rho_A*(Y_A*Gamma_u_A/nA + Y_B*Gamma_u_B/nB)
    # jv_diff_A = mass_A * Gamma_v_A - rho_A*(Y_A*Gamma_v_A/nA + Y_B*Gamma_v_B/nB)

    # ju_diff_B = mass_B * Gamma_u_B - rho_B*(Y_A*Gamma_u_A/nA + Y_B*Gamma_u_B/nB)
    # jv_diff_B = mass_B * Gamma_v_B - rho_B*(Y_A*Gamma_v_A/nA + Y_B*Gamma_v_B/nB)
    
    return {
        "A": {
            "ju_diff": ju_diff_A,   # diffusive flux in x-direction
            "jv_diff": jv_diff_A,   # diffusive flux in y-direction
            "Ju_diff": Ju_diff_A,   # molar diffusive flux in x-direction
            "Jv_diff": Jv_diff_A,   # molar diffusive flux in y-direction
        },
        "B": {
            "ju_diff": ju_diff_B,   # diffusive flux in x-direction
            "jv_diff": jv_diff_B,   # diffusive flux in y-direction
            "Ju_diff": Ju_diff_B,   # molar diffusive flux in x-direction
            "Jv_diff": Jv_diff_B,   # molar diffusive flux in y-direction
        },
        "mixture": {
            "rho_A": rho_A,
            "rho_B": rho_B,
            "rho_total": rho_total,
            "c_A": c_A,
            "c_B": c_B,
            "c_total": c_total,
            "Y_A": Y_A,
            "Y_B": Y_B,
            "X_A": X_A,
            "X_B": X_B,
            "mass_flux_mix_u": mass_flux_mix_u,   # total mixture mass flux, not diffusive flux
            "mass_flux_mix_v": mass_flux_mix_v,   # total mixture mass flux, not diffusive flux
            "molar_flux_mix_u": molar_flux_mix_u,  # total mixture molar flux, not diffusive flux
            "molar_flux_mix_v": molar_flux_mix_v,  # total mixture molar flux, not diffusive flux
        }
    }

def surface_wall_coordinates(sol):
    return {
        "bottom": np.asarray(sol["x"], dtype=float),
        "top": np.asarray(sol["x"], dtype=float),
        "end": np.asarray(sol["y"], dtype=float),
    }


def surface_wall_weights(sol):
    return {
        "bottom": mid_weights_1d(sol["x"]),
        "top": mid_weights_1d(sol["x"]),
        "end": mid_weights_1d(sol["y"]),
    }


def compute_surface_flux(sol, *, span_depth=1.0):
    """
    Incoming normal molecular(number) flux on each wall.

    This uses the same Gamma_imp sign convention stored in sol["imp"] so the
    returned totals and source contributions can be plotted directly and can
    be reused for surface reaction-rate decomposition.
    """
    if "imp" not in sol or "Gamma_imp" not in sol["imp"]:
        raise KeyError("sol['imp']['Gamma_imp'] not found. Modify solve_reactive_channel to return impacted fluxes.")
    if "Gamma_imp_components" not in sol["imp"]:
        raise KeyError("sol['imp']['Gamma_imp_components'] not found. Modify solve_reactive_channel to return flux components.")

    source_parts = ("jet", "bottom", "top", "end")
    walls = ("bottom", "top", "end")
    wall_coords = surface_wall_coordinates(sol)
    wall_weights = surface_wall_weights(sol)

    local_flux = {sp: {} for sp in ("A", "B", "total")}
    local_flux_components = {sp: {} for sp in ("A", "B", "total")}
    total_number_flux_2D_unit_depth = {sp: {} for sp in ("A", "B", "total")}
    total_number_flux_components_2D_unit_depth = {sp: {} for sp in ("A", "B", "total")}

    for wall in walls:
        coord = wall_coords[wall]
        weight = wall_weights[wall]

        total_components_wall = {
            part: np.zeros_like(coord, dtype=float)
            for part in source_parts
        }

        for sp in ("A", "B"):
            total_arr = np.asarray(sol["imp"]["Gamma_imp"][wall][sp], dtype=float).copy()
            component_arrs = {
                part: np.zeros_like(coord, dtype=float)
                for part in source_parts
            }
            for part, values in sol["imp"]["Gamma_imp_components"][wall][sp].items():
                component_arrs[part] = np.asarray(values, dtype=float).copy()

            local_flux[sp][wall] = total_arr
            local_flux_components[sp][wall] = component_arrs
            total_number_flux_2D_unit_depth[sp][wall] = float(np.sum(total_arr * weight))
            total_number_flux_components_2D_unit_depth[sp][wall] = {
                part: float(np.sum(component_arrs[part] * weight))
                for part in source_parts
            }

            for part in source_parts:
                total_components_wall[part] += component_arrs[part]

        local_flux["total"][wall] = local_flux["A"][wall] + local_flux["B"][wall]
        local_flux_components["total"][wall] = total_components_wall
        total_number_flux_2D_unit_depth["total"][wall] = (
            total_number_flux_2D_unit_depth["A"][wall]
            + total_number_flux_2D_unit_depth["B"][wall]
        )
        total_number_flux_components_2D_unit_depth["total"][wall] = {
            part: (
                total_number_flux_components_2D_unit_depth["A"][wall][part]
                + total_number_flux_components_2D_unit_depth["B"][wall][part]
            )
            for part in source_parts
        }

    for sp in ("A", "B", "total"):
        total_number_flux_2D_unit_depth[sp]["total"] = sum(
            total_number_flux_2D_unit_depth[sp][wall] for wall in walls
        )
        total_number_flux_components_2D_unit_depth[sp]["total"] = {
            part: sum(
                total_number_flux_components_2D_unit_depth[sp][wall][part]
                for wall in walls
            )
            for part in source_parts
        }

    total_number_flux_for_span_depth = {}
    total_number_flux_components_for_span_depth = {}
    for sp in ("A", "B", "total"):
        total_number_flux_for_span_depth[sp] = {
            "span_depth_m": float(span_depth),
            **{
                wall: value * float(span_depth)
                for wall, value in total_number_flux_2D_unit_depth[sp].items()
            },
        }
        total_number_flux_components_for_span_depth[sp] = {
            "span_depth_m": float(span_depth),
            **{
                wall: {
                    part: value * float(span_depth)
                    for part, value in component_dict.items()
                }
                for wall, component_dict in total_number_flux_components_2D_unit_depth[sp].items()
            },
        }

    return {
        "source_parts": source_parts,
        "wall_coord": wall_coords,
        "local_flux": local_flux,
        "local_flux_components": local_flux_components,
        "total_number_flux_2D_unit_depth": total_number_flux_2D_unit_depth,
        "total_number_flux_components_2D_unit_depth": total_number_flux_components_2D_unit_depth,
        "total_number_flux_for_span_depth": total_number_flux_for_span_depth,
        "total_number_flux_components_for_span_depth": total_number_flux_components_for_span_depth,
    }


def compute_surface_reaction_rate(sol, *, span_depth=1.0):
    """
    Surface reaction rate for A -> B on walls, using impacted incoming flux saved from solve_reactive_channel.

    Reaction model:
      - Reaction probability: kappa = reaction_probability
      - Local surface reaction rate along the wall line:
            r_s(s) = kappa * Gamma_imp_A(s)

      - Integrated total reaction rate on each wall:
            Ndot_wall = ∫ r_s dl

    IMPORTANT:
      - Gamma_imp_A(s) is taken from the solver (NO max/clamp):
            bottom: Gamma_imp = -nv_imp_bottom
            top:    Gamma_imp = +nv_imp_top
            end:    Gamma_imp = +nu_imp_end

    Optional:
      - If you want the total for an actual depth/span in z (e.g., slit length in z),
        multiply by span_depth:
            Ndot_3D = Ndot_2D * span_depth
    """
    if "imp" not in sol or "Gamma_imp" not in sol["imp"]:
        raise KeyError("sol['imp']['Gamma_imp'] not found. Modify solve_reactive_channel to return impacted fluxes.")

    kappa = float(sol["reaction_probability"])
    x = sol["x"]
    y = sol["y"]
    source_parts = ("jet", "bottom", "top", "end")
    wall_coords = surface_wall_coordinates(sol)

    # Incoming flux magnitudes on walls (no clamp)
    Gamma_imp_bot_A = sol["imp"]["Gamma_imp"]["bottom"]["A"]
    Gamma_imp_top_A = sol["imp"]["Gamma_imp"]["top"]["A"]
    Gamma_imp_end_A = sol["imp"]["Gamma_imp"]["end"]["A"]

    # Local surface reaction rate along each wall line
    r_bot = kappa * Gamma_imp_bot_A
    r_top = kappa * Gamma_imp_top_A
    r_end = kappa * Gamma_imp_end_A

    # Integrate along wall length to get total reaction rate
    wx = mid_weights_1d(x)  # integrates over x
    wy = mid_weights_1d(y)  # integrates over y

    Ndot_bot = float(np.sum(r_bot * wx))
    Ndot_top = float(np.sum(r_top * wx))
    Ndot_end = float(np.sum(r_end * wy))
    Ndot_tot = Ndot_bot + Ndot_top + Ndot_end

    local_rate_components = None
    total_number_rate_components_2D_unit_depth = None
    total_number_rate_components_for_span_depth = None
    if "Gamma_imp_components" in sol["imp"]:
        local_rate_components = {}
        total_number_rate_components_2D_unit_depth = {}
        wall_weights = {"bottom": wx, "top": wx, "end": wy}

        for wall in ("bottom", "top", "end"):
            coord = wall_coords[wall]
            weight = wall_weights[wall]
            wall_components = {
                part: np.zeros_like(coord, dtype=float)
                for part in source_parts
            }
            for part, values in sol["imp"]["Gamma_imp_components"][wall]["A"].items():
                wall_components[part] = kappa * np.asarray(values, dtype=float)
            local_rate_components[wall] = wall_components
            total_number_rate_components_2D_unit_depth[wall] = {
                part: float(np.sum(wall_components[part] * weight))
                for part in source_parts
            }

        total_number_rate_components_2D_unit_depth["total"] = {
            part: sum(
                total_number_rate_components_2D_unit_depth[wall][part]
                for wall in ("bottom", "top", "end")
            )
            for part in source_parts
        }
        total_number_rate_components_for_span_depth = {
            "span_depth_m": float(span_depth),
            **{
                wall: {
                    part: value * float(span_depth)
                    for part, value in component_dict.items()
                }
                for wall, component_dict in total_number_rate_components_2D_unit_depth.items()
            },
        }

    # If you want "3D total" for a given out-of-plane depth/span
    Ndot_bot_3D = Ndot_bot * float(span_depth)
    Ndot_top_3D = Ndot_top * float(span_depth)
    Ndot_end_3D = Ndot_end * float(span_depth)
    Ndot_tot_3D = Ndot_tot * float(span_depth)

    result = {
        "kappa": kappa,
        "local_rate": {
            "bottom": r_bot,
            "top":    r_top,
            "end":    r_end,
        },
        "total_number_rate_2D_unit_depth": {
            "bottom": Ndot_bot,
            "top":    Ndot_top,
            "end":    Ndot_end,
            "total":  Ndot_tot,
        },
        "total_number_rate_for_span_depth": {  # multiplied by span_depth
            "span_depth_m": float(span_depth),
            "bottom": Ndot_bot_3D,
            "top":    Ndot_top_3D,
            "end":    Ndot_end_3D,
            "total":  Ndot_tot_3D,
        },
        "source_parts": source_parts,
    }
    if local_rate_components is not None:
        result["local_rate_components"] = local_rate_components
        result["total_number_rate_components_2D_unit_depth"] = total_number_rate_components_2D_unit_depth
        result["total_number_rate_components_for_span_depth"] = total_number_rate_components_for_span_depth

    return result

# ============================================================
# Postprocessing (contours + click probe)
# ============================================================
# ======== Custom colormap (rainbow + 8 additional colors) ========
CUSTOM_COLORS = np.array([
    [0.6350, 0.0780, 0.1840],  # red
    [0.8500, 0.3250, 0.0980],  # orange
    [0.9290, 0.6940, 0.1250],  # yellow
    [0.4660, 0.6740, 0.1880],  # green
    [0.0000, 0.4470, 0.7410],  # blue
    [0.4940, 0.1840, 0.5560],  # purple
    [0.7300, 0.8300, 0.9600],  # light blue
], dtype=float)

from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_rainbow", CUSTOM_COLORS, N=256)
INVERSION_CMAP = ListedColormap([CUSTOM_COLORS[4], CUSTOM_COLORS[0]])
INVERSION_PRODUCT_CMAP = LinearSegmentedColormap.from_list(
    "cold_and_hot_product",
    [
        (0.0, (0.0, 1.0, 1.0)),          # min: cyan
        (0.35, (0.0, 0.0, 1.0)),         # cold branch: blue
        (0.499999, (0.0, 0.0, 0.501961)),# just below 0: dark blue
        (0.500001, (1.0, 0.0, 0.0)),     # just above 0: red
        (1.0, (1.0, 1.0, 0.0)),          # max: yellow
    ],
    N=256,
)
INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE = DEFAULT_INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE
INVERSION_PRODUCT_POSITIVE_RANGE_SCALE = DEFAULT_INVERSION_PRODUCT_POSITIVE_RANGE_SCALE

class ScalarFormatterFixed(mticker.ScalarFormatter):
    def __init__(self, decimals=1, **kwargs):
        self._decimals = int(decimals)
        super().__init__(**kwargs)

    def _set_format(self):
        # override the format chosen automatically by ScalarFormatter
        self.format = f"%.{self._decimals}f"

def get_centerline_YA_min(sol, fields_total):
    """
    Find the location of the Y_A minimum on the centerline (y ≈ 0).
    Returns: dict(ix, iy, x, y, YA, percent)
      percent = (Lm - x_min)/Lm*100
    """
    x = sol["x"]
    y = sol["y"]
    Lm = sol["Lm"]

    iy0 = int(np.argmin(np.abs(y - 0.0)))          # index closest to y = 0
    YA_line = fields_total["YA"][iy0, :]
    ix_min = int(np.nanargmin(YA_line))

    x_min = float(x[ix_min])
    percent = (Lm - x_min) / Lm * 100.0 if Lm != 0 else np.nan

    return {
        "ix": ix_min,
        "iy": iy0,
        "x": x_min,
        "y": float(y[iy0]),
        "YA": float(YA_line[ix_min]),
        "percent": float(percent),
    }


def get_centerline_nA_min(sol, fields_total):
    """
    Find the location of the n_A minimum on the centerline (y ≈ 0).
    Returns: dict(ix, iy, x, y, nA, percent)
      percent = (Lm - x_min)/Lm*100
    """
    x = sol["x"]
    y = sol["y"]
    Lm = sol["Lm"]

    iy0 = int(np.argmin(np.abs(y - 0.0)))          # index closest to y = 0
    nA_line = fields_total["A"]["n"][iy0, :]
    ix_min = int(np.nanargmin(nA_line))

    x_min = float(x[ix_min])
    percent = (Lm - x_min) / Lm * 100.0 if Lm != 0 else np.nan

    return {
        "ix": ix_min,
        "iy": iy0,
        "x": x_min,
        "y": float(y[iy0]),
        "nA": float(nA_line[ix_min]),
        "percent": float(percent),
    }


def compare_centerline_YA_nA_min(sol, fields_total):
    """
    Compare the locations of the Y_A minimum and the n_A minimum on the centerline (y ≈ 0).
    """
    info_ya = get_centerline_YA_min(sol, fields_total)
    info_na = get_centerline_nA_min(sol, fields_total)
    Lm = float(sol["Lm"])

    dx_abs = abs(info_ya["x"] - info_na["x"])
    dix_abs = abs(int(info_ya["ix"]) - int(info_na["ix"]))
    dx_over_Lm = dx_abs / Lm if Lm != 0.0 else np.nan

    return {
        "YA_min": info_ya,
        "nA_min": info_na,
        "same_position": bool(
            info_ya["ix"] == info_na["ix"] and info_ya["iy"] == info_na["iy"]
        ),
        "dx_abs": float(dx_abs),
        "dx_over_Lm": float(dx_over_Lm),
        "dx_percent_of_Lm": float(dx_over_Lm * 100.0) if np.isfinite(dx_over_Lm) else np.nan,
        "dix_abs": int(dix_abs),
    }

def get_YA_min_curve(sol, fields_total):
    """
    For every y (row), find the point where Y_A is minimum along x.
    Returns: dict(y, x_min, YA_min, ix_min)
    """
    x = sol["x"]
    y = sol["y"]
    YA = fields_total["YA"]  # shape (ny, nx)

    ny = y.size
    x_min = np.empty(ny, dtype=float)
    YA_min = np.empty(ny, dtype=float)
    ix_min = np.empty(ny, dtype=int)

    for iy in range(ny):
        row = YA[iy, :]
        ix = int(np.nanargmin(row))
        ix_min[iy] = ix
        x_min[iy] = float(x[ix])
        YA_min[iy] = float(row[ix])

    return {"y": y.copy(), "x_min": x_min, "YA_min": YA_min, "ix_min": ix_min}


def get_nA_min_curve(sol, fields_total):
    """
    For every y (row), find the point where n_A is minimum along x.
    Returns: dict(y, x_min, nA_min, ix_min)
    """
    x = sol["x"]
    y = sol["y"]
    nA = fields_total["A"]["n"]  # shape (ny, nx)

    ny = y.size
    x_min = np.empty(ny, dtype=float)
    nA_min = np.empty(ny, dtype=float)
    ix_min = np.empty(ny, dtype=int)

    for iy in range(ny):
        row = nA[iy, :]
        ix = int(np.nanargmin(row))
        ix_min[iy] = ix
        x_min[iy] = float(x[ix])
        nA_min[iy] = float(row[ix])

    return {"y": y.copy(), "x_min": x_min, "nA_min": nA_min, "ix_min": ix_min}


def compare_YA_nA_min_curves(sol, fields_total):
    """
    Compare the Y_A minimum curve and the n_A minimum curve for every y (row).
    """
    ya_curve = get_YA_min_curve(sol, fields_total)
    na_curve = get_nA_min_curve(sol, fields_total)
    Lm = float(sol["Lm"])

    same_position = (ya_curve["ix_min"] == na_curve["ix_min"])
    dx_abs = np.abs(ya_curve["x_min"] - na_curve["x_min"])
    dx_over_Lm = dx_abs / Lm if Lm != 0.0 else np.full_like(dx_abs, np.nan)

    return {
        "y": ya_curve["y"].copy(),
        "ix_YA_min": ya_curve["ix_min"].copy(),
        "x_YA_min": ya_curve["x_min"].copy(),
        "YA_min": ya_curve["YA_min"].copy(),
        "ix_nA_min": na_curve["ix_min"].copy(),
        "x_nA_min": na_curve["x_min"].copy(),
        "nA_min": na_curve["nA_min"].copy(),
        "same_position": same_position.copy(),
        "dx_abs": dx_abs.copy(),
        "dx_over_Lm": dx_over_Lm.copy(),
        "dx_percent_of_Lm": (dx_over_Lm * 100.0).copy(),
    }

def save_centerline_YA_min_to_txt(save_dir, prefix, info):
    """
    Save the location of the centerline Y_A minimum to a txt file.
    Also saves percent = (Lm - x_min)/Lm*100.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_centerline_YA_min.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("Centerline Y_A minimum (at y≈0)\n")
        f.write(f"ix = {info['ix']}\n")
        f.write(f"iy = {info['iy']}\n")
        f.write(f"x  = {info['x']:.16e}\n")
        f.write(f"y  = {info['y']:.16e}\n")
        f.write(f"Y_A= {info['YA']:.16e}\n")
        f.write(f"percent = (Lm - x)/Lm*100 = {info['percent']:.10f}\n")

    return path

def save_YA_min_curve_to_txt(save_dir, prefix, curve):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_YA_min_curve.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("y, x_min(y), YA_min(y), ix_min\n")
        for yv, xv, yav, ix in zip(curve["y"], curve["x_min"], curve["YA_min"], curve["ix_min"]):
            f.write(f"{yv:.16e}\t{xv:.16e}\t{yav:.16e}\t{ix}\n")
    return path


def save_centerline_YA_nA_min_comparison_to_txt(save_dir, prefix, comparison):
    """
    Save the comparison of the centerline Y_A minimum and n_A minimum to a txt file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_centerline_YA_nA_min_comparison.txt")

    info_ya = comparison["YA_min"]
    info_na = comparison["nA_min"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("Centerline minima comparison (at y≈0)\n")
        f.write("\n")
        f.write("[Y_A minimum]\n")
        f.write(f"ix = {info_ya['ix']}\n")
        f.write(f"iy = {info_ya['iy']}\n")
        f.write(f"x  = {info_ya['x']:.16e}\n")
        f.write(f"y  = {info_ya['y']:.16e}\n")
        f.write(f"Y_A= {info_ya['YA']:.16e}\n")
        f.write(f"percent = (Lm - x)/Lm*100 = {info_ya['percent']:.10f}\n")
        f.write("\n")
        f.write("[n_A minimum]\n")
        f.write(f"ix = {info_na['ix']}\n")
        f.write(f"iy = {info_na['iy']}\n")
        f.write(f"x  = {info_na['x']:.16e}\n")
        f.write(f"y  = {info_na['y']:.16e}\n")
        f.write(f"n_A= {info_na['nA']:.16e}\n")
        f.write(f"percent = (Lm - x)/Lm*100 = {info_na['percent']:.10f}\n")
        f.write("\n")
        f.write("[Comparison]\n")
        f.write(f"same_position = {comparison['same_position']}\n")
        f.write(f"dix_abs = {comparison['dix_abs']}\n")
        f.write(f"dx_abs  = {comparison['dx_abs']:.16e}\n")
        f.write(f"dx_over_Lm = {comparison['dx_over_Lm']:.16e}\n")
        f.write(f"dx_percent_of_Lm = {comparison['dx_percent_of_Lm']:.10f}\n")

    return path


def save_YA_nA_min_curve_comparison_to_txt(save_dir, prefix, comparison_curve):
    """
    Save the comparison of the Y_A and n_A minimum curves for every y (row) to a txt file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_YA_nA_min_curve_comparison.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "iy\ty\tix_YA_min\tx_YA_min\tYA_min\tix_nA_min\tx_nA_min\tnA_min\t"
            "same_position\tdx_abs\tdx_over_Lm\tdx_percent_of_Lm\n"
        )
        for iy, yv, ix_ya, x_ya, ya_min, ix_na, x_na, na_min, same_pos, dx_abs, dx_rel, dx_pct in zip(
            np.arange(comparison_curve["y"].size, dtype=int),
            comparison_curve["y"],
            comparison_curve["ix_YA_min"],
            comparison_curve["x_YA_min"],
            comparison_curve["YA_min"],
            comparison_curve["ix_nA_min"],
            comparison_curve["x_nA_min"],
            comparison_curve["nA_min"],
            comparison_curve["same_position"],
            comparison_curve["dx_abs"],
            comparison_curve["dx_over_Lm"],
            comparison_curve["dx_percent_of_Lm"],
        ):
            f.write(
                f"{iy}\t{yv:.16e}\t{ix_ya}\t{x_ya:.16e}\t{ya_min:.16e}\t"
                f"{ix_na}\t{x_na:.16e}\t{na_min:.16e}\t{bool(same_pos)}\t"
                f"{dx_abs:.16e}\t{dx_rel:.16e}\t{dx_pct:.10f}\n"
            )

    return path


def print_YA_nA_min_curve_comparison_summary(prefix, comparison_curve):
    """
    Print a summary of the Y_A / n_A minimum curve comparison to stdout.

    Results for all rows are saved to txt; only
    - center row (y≈0)
    - the row with the largest dx_abs
    are printed here.
    """
    y = np.asarray(comparison_curve["y"], dtype=float)
    ix_ya = np.asarray(comparison_curve["ix_YA_min"], dtype=int)
    x_ya = np.asarray(comparison_curve["x_YA_min"], dtype=float)
    ya_min = np.asarray(comparison_curve["YA_min"], dtype=float)
    ix_na = np.asarray(comparison_curve["ix_nA_min"], dtype=int)
    x_na = np.asarray(comparison_curve["x_nA_min"], dtype=float)
    na_min = np.asarray(comparison_curve["nA_min"], dtype=float)
    same_pos = np.asarray(comparison_curve["same_position"], dtype=bool)
    dx_abs = np.asarray(comparison_curve["dx_abs"], dtype=float)
    dx_rel = np.asarray(comparison_curve["dx_over_Lm"], dtype=float)
    dx_pct = np.asarray(comparison_curve["dx_percent_of_Lm"], dtype=float)

    iy_center = int(np.argmin(np.abs(y - 0.0)))
    iy_max = int(np.nanargmax(dx_abs))

    print(f"[{prefix}] YA min curve vs nA min curve comparison summary")
    print(
        "row_tag\tiy\ty\tix_YA_min\tx_YA_min\tYA_min\tix_nA_min\tx_nA_min\tnA_min\t"
        "same_position\tdx_abs\tdx_over_Lm\tdx_percent_of_Lm"
    )

    def _print_row(tag, iy):
        print(
            f"{tag}\t{iy}\t{y[iy]:.16e}\t{ix_ya[iy]}\t{x_ya[iy]:.16e}\t{ya_min[iy]:.16e}\t"
            f"{ix_na[iy]}\t{x_na[iy]:.16e}\t{na_min[iy]:.16e}\t{bool(same_pos[iy])}\t"
            f"{dx_abs[iy]:.16e}\t{dx_rel[iy]:.16e}\t{dx_pct[iy]:.10f}"
        )

    _print_row("center", iy_center)
    if iy_max == iy_center:
        print("max_dx\t(same as center row)")
    else:
        _print_row("max_dx", iy_max)


def emit_YA_nA_min_diagnostics(prefix, sol, fields_total, *, emit_warning=True, print_summary=True):
    """
    Run the Y_A minimum / n_A minimum comparison diagnostics and, if requested, print a summary and emit warnings.

    Nothing is saved; kept separate so that it can be called quickly before plotting.
    """
    comparison = compare_centerline_YA_nA_min(sol, fields_total)
    comparison_curve = compare_YA_nA_min_curves(sol, fields_total)

    if print_summary and (not comparison["same_position"]):
        print_YA_nA_min_curve_comparison_summary(prefix, comparison_curve)

    if emit_warning and (not comparison["same_position"]):
        info_ya = comparison["YA_min"]
        info_na = comparison["nA_min"]
        warnings.warn(
            "Centerline Y_A minimum and n_A minimum are different "
            f"for '{prefix}': "
            f"ix_YA={info_ya['ix']}, ix_nA={info_na['ix']}, "
            f"x_YA={info_ya['x']:.16e}, x_nA={info_na['x']:.16e}, "
            f"dx={comparison['dx_abs']:.16e} m "
            f"({comparison['dx_percent_of_Lm']:.10f}% of Lm).",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "centerline_comparison": comparison,
        "curve_comparison": comparison_curve,
    }

def compute_min_diagnostics(sol, fields_total):
    """
    Compute only the min point / min line (= YA_min_curve); nothing is saved.
    Returns: (info_min, ya_curve, minline_common)
    """
    info_min = get_centerline_YA_min(sol, fields_total)
    ya_curve = get_YA_min_curve(sol, fields_total)
    minline_common = (ya_curve["x_min"], ya_curve["y"])
    return info_min, ya_curve, minline_common


def compute_inversion_zoom_xlim(x_coords, *minline_x_values, zoom_factor=3.0):
    x_coords = np.asarray(x_coords, dtype=float)
    if x_coords.ndim != 1 or x_coords.size == 0:
        raise ValueError("x_coords must be a non-empty 1D array.")

    dx = float(x_coords[1] - x_coords[0]) if x_coords.size > 1 else 0.0
    domain_left = float(np.min(x_coords) - 0.5 * dx)
    domain_right = float(np.max(x_coords) + 0.5 * dx)
    domain_width = domain_right - domain_left

    valid_mins = []
    for values in minline_x_values:
        if values is None:
            continue
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            valid_mins.append(float(np.min(arr)))

    if not valid_mins:
        return (domain_left, domain_right)

    inversion_left = min(valid_mins)
    inversion_left = min(max(inversion_left, domain_left), domain_right)
    min_width = dx if dx > 0.0 else domain_width
    inversion_width = max(domain_right - inversion_left, min_width)
    zoom_width = min(domain_width, float(zoom_factor) * inversion_width)
    zoom_left = max(domain_left, domain_right - zoom_width)
    return (float(zoom_left), float(domain_right))


def crop_contour_to_x_window(X, Y, Z, xlim=None):
    if xlim is None:
        return X, Y, Z

    xlim = tuple(float(v) for v in xlim)
    x_coords = np.asarray(X[0, :], dtype=float)
    tol = 1.0e-12 * max(1.0, abs(xlim[1] - xlim[0]))
    mask = (x_coords >= xlim[0] - tol) & (x_coords <= xlim[1] + tol)

    if not np.any(mask):
        ic = int(np.argmin(np.abs(x_coords - 0.5 * (xlim[0] + xlim[1]))))
        mask = np.zeros_like(x_coords, dtype=bool)
        mask[ic] = True

    return X[:, mask], Y[:, mask], Z[:, mask]


def build_powernorm(Z, powernorm_gamma=None):
    if powernorm_gamma is None:
        return None

    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    if np.isclose(zmin, zmax):
        return None

    return mcolors.PowerNorm(
        gamma=float(powernorm_gamma),
        vmin=zmin,
        vmax=zmax,
    )

def save_min_diagnostics(
    save_dir,
    prefix,
    sol,
    fields_total,
    *,
    diagnostics=None,
    emit_warning=True,
    print_summary=True,
):
    """
    Only save the min point / min line to txt (separate from plot_results).
    """
    info_min, ya_curve, _ = compute_min_diagnostics(sol, fields_total)
    if diagnostics is None:
        diagnostics = emit_YA_nA_min_diagnostics(
            prefix,
            sol,
            fields_total,
            emit_warning=emit_warning,
            print_summary=print_summary,
        )
    comparison = diagnostics["centerline_comparison"]
    comparison_curve = diagnostics["curve_comparison"]

    p1 = save_centerline_YA_min_to_txt(save_dir, prefix, info_min)
    p2 = save_YA_min_curve_to_txt(save_dir, prefix, ya_curve)
    p3 = save_centerline_YA_nA_min_comparison_to_txt(save_dir, prefix, comparison)
    p4 = save_YA_nA_min_curve_comparison_to_txt(save_dir, prefix, comparison_curve)

    return {
        "centerline_min_txt": p1,
        "ya_min_curve_txt": p2,
        "centerline_ya_na_min_comparison_txt": p3,
        "ya_na_min_curve_comparison_txt": p4,
    }


def compute_mass_fraction_gradients(sol, fields_total, species="A"):
    """
    Compute mass-fraction gradients with second-order central differences.

    np.gradient(..., edge_order=2) uses second-order central differences in the
    interior and second-order one-sided differences at the boundaries.
    """
    if species == "A":
        Ysp = fields_total["YA"]
    elif species == "B":
        Ysp = fields_total["YB"]
    else:
        raise ValueError(f"Unsupported species={species!r}. Use 'A' or 'B'.")

    x = sol["x"]
    y = sol["y"]
    dY_dy, dY_dx = np.gradient(Ysp, y, x, edge_order=2)
    return {"dx": dY_dx, "dy": dY_dy}


def compute_inversion_regions(sol, fields_total, diff_fields, species="A"):
    """
    Inversion is defined here as:
      x-direction: j_x * dY/dx > 0
      y-direction: j_y * dY/dy > 0

    For a binary mixture, using species A or B yields the same inversion mask
    because Y_B = 1 - Y_A and j_B = -j_A.
    """
    if diff_fields is None:
        raise ValueError("diff_fields is required to compute inversion regions.")

    gradients = compute_mass_fraction_gradients(sol, fields_total, species=species)
    mass_fraction_key = f"Y{species}"
    flux_x = diff_fields[species]["ju_diff"]
    flux_y = diff_fields[species]["jv_diff"]

    product_x = flux_x * gradients["dx"]
    product_y = flux_y * gradients["dy"]

    sign_x = np.sign(product_x).astype(int)
    sign_y = np.sign(product_y).astype(int)
    total_cells = int(sign_x.size)
    x = np.asarray(sol["x"], dtype=float)
    y = np.asarray(sol["y"], dtype=float)
    wx = mid_weights_1d(x)
    wy = mid_weights_1d(y)
    total_area = float(np.sum(wx) * np.sum(wy))

    if total_area <= 0.0:
        raise ValueError("Total area must be positive to compute inversion-region integrals.")

    def _positive_product_integral_over_total_area(product):
        product = np.asarray(product, dtype=float)
        positive_product = np.where(product > 0.0, product, 0.0)
        positive_integral = float(
            np.einsum("ij,j,i->", positive_product, wx, wy, optimize=True)
        )
        return float(positive_integral / total_area)

    def _stats(sign_map, product):
        inversion_cells = int(np.count_nonzero(sign_map > 0))
        zero_cells = int(np.count_nonzero(sign_map == 0))
        negative_cells = int(np.count_nonzero(sign_map < 0))
        return {
            "inversion_cells": inversion_cells,
            "zero_cells": zero_cells,
            "negative_cells": negative_cells,
            "non_inversion_cells": int(negative_cells + zero_cells),
            "total_cells": total_cells,
            "inversion_area_fraction": float(inversion_cells / total_cells),
            "inversion_area_percent": float(100.0 * inversion_cells / total_cells),
            "positive_product_integral_over_total_area": _positive_product_integral_over_total_area(product),
        }

    return {
        "species": species,
        "mass_fraction_key": mass_fraction_key,
        "gradients": {
            "dY_dx": gradients["dx"],
            "dY_dy": gradients["dy"],
        },
        "flux": {
            "j_x": flux_x,
            "j_y": flux_y,
        },
        "product": {
            "x": product_x,
            "y": product_y,
        },
        "sign": {
            "x": sign_x,
            "y": sign_y,
        },
        "stats": {
            "x": _stats(sign_x, product_x),
            "y": _stats(sign_y, product_y),
        },
    }


def save_inversion_region_summary(save_dir, prefix, inversion_regions):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_inversion_region_summary.txt")
    species = inversion_regions["species"]
    xstats = inversion_regions["stats"]["x"]
    ystats = inversion_regions["stats"]["y"]
    mass_fraction_key = inversion_regions["mass_fraction_key"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("Inversion region summary\n")
        f.write(f"species = {species}\n")
        f.write(f"mass_fraction = {mass_fraction_key}\n")
        f.write(f"definition_x = j_x * d{mass_fraction_key}/dx > 0\n")
        f.write(f"definition_y = j_y * d{mass_fraction_key}/dy > 0\n")
        f.write(f"total_cells = {xstats['total_cells']}\n")
        f.write("\n")
        f.write("[x-direction]\n")
        f.write(f"inversion_cells = {xstats['inversion_cells']}\n")
        f.write(f"zero_cells = {xstats['zero_cells']}\n")
        f.write(f"negative_cells = {xstats['negative_cells']}\n")
        f.write(f"inversion_area_fraction = {xstats['inversion_area_fraction']:.16e}\n")
        f.write(f"inversion_area_percent = {xstats['inversion_area_percent']:.10f}\n")
        f.write(
            "positive_product_integral_over_total_area = "
            f"{xstats['positive_product_integral_over_total_area']:.16e}\n"
        )
        f.write("\n")
        f.write("[y-direction]\n")
        f.write(f"inversion_cells = {ystats['inversion_cells']}\n")
        f.write(f"zero_cells = {ystats['zero_cells']}\n")
        f.write(f"negative_cells = {ystats['negative_cells']}\n")
        f.write(f"inversion_area_fraction = {ystats['inversion_area_fraction']:.16e}\n")
        f.write(f"inversion_area_percent = {ystats['inversion_area_percent']:.10f}\n")
        f.write(
            "positive_product_integral_over_total_area = "
            f"{ystats['positive_product_integral_over_total_area']:.16e}\n"
        )
        f.write("\n")
        f.write("note = For a binary mixture, species B gives the same inversion mask because Y_B = 1 - Y_A and J_B = -J_A.\n")

    return path


def plot_inversion_mask_contour(
    X, Y, sign_map,
    save_path, show, dpi,
    figsize,
    tick_length, tick_width,
    box_linewidth,
    y_nticks=3, x_nticks=6,
    ax_aspect=1,
    minline=None,
    minline_linewidth=2.0,
    Lref=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    sign_int = np.where(np.asarray(sign_map) > 0, 1, -1).astype(int)
    levels = [-1.5, 0.5, 1.5]
    norm = BoundaryNorm(levels, INVERSION_CMAP.N)

    # Non-dimensionalize spatial length axes by the full trench height 2h.
    # The mesh is CELL-CENTERED; reconstruct the TRUE physical domain (walls) from the
    # uniform cell spacing for all axis limits/ticks. Data arrays are plotted unchanged.
    dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 0.0
    dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 0.0
    if Lref is None:
        Lref = (float(np.max(Y)) - float(np.min(Y))) + dy  # true full trench height 2h
    Lref = float(Lref) if Lref else 1.0
    Lm_true = float(np.max(X)) + 0.5 * dx
    AR = Lm_true / Lref

    ax.contourf(X / Lref, Y / Lref, sign_int, levels=levels, cmap=INVERSION_CMAP, norm=norm)

    ax.set_xlabel(r"$x/(2h)$")
    ax.set_ylabel(r"$y/(2h)$")
    ax.set_title("")
    ax.set_aspect(ax_aspect, adjustable="box")

    # Exact physical non-dimensional domain: x in (0, AR), y in (-0.5, 0.5).
    ax.set_xlim(0.0, AR)
    ax.set_ylim(-0.5, 0.5)

    if minline is not None:
        xl, yl = minline
        ax.plot(
            np.asarray(xl) / Lref,
            np.asarray(yl) / Lref,
            color="k",
            linestyle="--",
            linewidth=minline_linewidth,
            zorder=20,
        )

    # y ticks: walls + centre.
    ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    nx = 3 if AR <= 1.5 else (4 if AR <= 3.5 else 6)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0.0, AR, nx)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    apply_axes_style(ax, box_linewidth, tick_length, tick_width, tick_pad=8)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if not show:
        plt.close(fig)

    return fig, ax


def inversion_product_expression(direction: str, species: str) -> str:
    if direction not in ("x", "y"):
        raise ValueError(f"Unsupported direction={direction!r}. Use 'x' or 'y'.")
    coord = direction
    flux = rf"j_{{{coord},\mathrm{{{species}}}}}"
    gradient = rf"\frac{{\partial Y_{{\mathrm{{{species}}}}}}}{{\partial {coord}}}"
    return rf"{flux}\,{gradient}"


def inversion_product_file_stem(direction: str, species: str) -> str:
    if direction not in ("x", "y"):
        raise ValueError(f"Unsupported direction={direction!r}. Use 'x' or 'y'.")
    return f"inversion_region_{direction}_j{direction}_partial_Y{species}_partial_{direction}"


def get_inversion_product_range_values(product_map, Y, range_source="field"):
    range_source = str(range_source).lower()
    product_arr = np.asarray(product_map, dtype=float)

    if range_source == "field":
        return product_arr
    if range_source != "centerline":
        raise ValueError(
            f"Unsupported inversion product range source={range_source!r}. "
            "Use 'field' or 'centerline'."
        )

    y_arr = np.asarray(Y, dtype=float)
    if y_arr.ndim != 2 or product_arr.ndim != 2:
        raise ValueError("Centerline-based product range requires 2D X/Y/product arrays.")

    y_coords = y_arr[:, 0]
    iy0 = int(np.argmin(np.abs(y_coords - 0.0)))
    centerline_values = product_arr[iy0, :]
    centerline_finite = centerline_values[np.isfinite(centerline_values)]
    if centerline_finite.size == 0:
        warnings.warn(
            "Centerline product values are all NaN; falling back to full-field product range.",
            RuntimeWarning,
            stacklevel=2,
        )
        return product_arr
    return centerline_values


def build_linear_product_levels(
    values,
    num_levels=900,
    negative_range_scale=INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
    positive_range_scale=INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot build contour levels from an all-NaN product field.")

    negative_range_scale = float(negative_range_scale)
    positive_range_scale = float(positive_range_scale)
    if negative_range_scale <= 0.0 or positive_range_scale <= 0.0:
        raise ValueError("Product contour range scales must be positive.")

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    scale = max(abs(data_min), abs(data_max), 1.0)
    eps = 1.0e-12 * scale

    if np.isclose(data_min, data_max):
        pad = max(1.0e-6 * scale, eps)
        data_min = float(data_min - pad)
        data_max = float(data_max + pad)

    if data_min < 0.0:
        vmin = float(data_min * negative_range_scale)
    else:
        vmin = -max(eps, 1.0e-6 * max(abs(data_max), 1.0))

    if data_max > 0.0:
        vmax = float(data_max * positive_range_scale)
    else:
        vmax = max(eps, 1.0e-6 * max(abs(data_min), 1.0))

    if not (vmin < 0.0):
        vmin = -max(eps, 1.0e-6 * max(abs(vmax), 1.0))
    if not (vmax > 0.0):
        vmax = max(eps, 1.0e-6 * max(abs(vmin), 1.0))

    level_count = max(3, int(num_levels))
    if vmin < 0.0 < vmax:
        neg_intervals = max(1, level_count // 2)
        pos_intervals = max(1, level_count - neg_intervals)
        neg_levels = np.linspace(vmin, 0.0, neg_intervals + 1, dtype=float)
        pos_levels = np.linspace(0.0, vmax, pos_intervals + 1, dtype=float)[1:]
        levels = np.concatenate([neg_levels, pos_levels])
    else:
        levels = np.linspace(vmin, vmax, level_count)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return levels, norm, vmin, vmax


def build_product_cbar_ticks(vmin: float, vmax: float, cbar_nticks: int):
    tick_count = max(2, int(cbar_nticks))
    vmin = float(vmin)
    vmax = float(vmax)

    if vmin < 0.0 < vmax:
        if tick_count <= 2:
            return np.asarray([vmin, vmax], dtype=float)
        neg_count = (tick_count - 1) // 2
        pos_count = (tick_count - 1) - neg_count
        neg_ticks = np.linspace(vmin, 0.0, neg_count + 1, dtype=float)[:-1]
        pos_ticks = np.linspace(0.0, vmax, pos_count + 1, dtype=float)[1:]
        return np.concatenate([neg_ticks, np.asarray([0.0]), pos_ticks])

    if vmax <= 0.0:
        return np.linspace(vmin, 0.0, tick_count, dtype=float)

    return np.linspace(0.0, vmax, tick_count, dtype=float)


def save_inversion_region_legend(
    save_path: str | None,
    show: bool,
    dpi: int,
    include_minline: bool = True,
):
    handles = [
        Patch(facecolor=INVERSION_CMAP.colors[0], edgecolor="none", label="non-inversion"),
        Patch(facecolor=INVERSION_CMAP.colors[1], edgecolor="none", label="inversion"),
    ]
    if include_minline:
        handles.append(
            Line2D([0], [0], color="k", linestyle="--", linewidth=2.0, label=r"$Y_{\mathrm{A}}$ min line")
        )

    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", frameon=False, ncol=1)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if not show:
        plt.close(fig)

    return fig, ax


def save_contour_colorbar_legend(
    save_path: str | None,
    show: bool,
    dpi: int,
    title: str,
    cmap,
    norm,
    vmin: float,
    vmax: float,
    tick_length: float,
    tick_width: float,
    box_linewidth: float,
    cbar_nticks: int = 5,
    cbar_scientific: bool = True,
    cbar_decimals: int = 2,
    cbar_ticks=None,
):
    fig = plt.figure(figsize=(3.0, 6.0))
    cax = fig.add_axes([0.36, 0.10, 0.24, 0.82])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if cbar_scientific:
        cfmt = ScalarFormatterFixed(decimals=cbar_decimals, useMathText=True)
        cfmt.set_scientific(True)
        cfmt.set_powerlimits((0, 0))
        cfmt.set_useOffset(False)
    else:
        cfmt = mticker.FormatStrFormatter(f"%.{cbar_decimals}f")

    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", format=cfmt)
    cbar.set_label(title, rotation=90, labelpad=12)

    if cbar_ticks is None:
        ticks = [float(vmin)] if np.isclose(vmin, vmax) else np.linspace(vmin, vmax, cbar_nticks)
    else:
        ticks = np.asarray(cbar_ticks, dtype=float)
    cbar.set_ticks(ticks)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_visible(True)

    apply_axes_style(cbar.ax, box_linewidth, tick_length, tick_width, tick_pad=8)
    if cbar_scientific:
        place_colorbar_offset_text_outside(cbar)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if not show:
        plt.close(fig)

    return fig, cbar.ax


def plot_inversion_product_contour(
    X, Y, product_map,
    save_path, legend_path, show, dpi,
    levels,
    figsize,
    tick_length, tick_width,
    box_linewidth,
    cbar_shrink,
    cbar_fraction,
    cbar_pad,
    cbar_aspect,
    y_nticks=3,
    x_nticks=6,
    cbar_nticks=5,
    cbar_decimals=2,
    ax_aspect=1,
    minline=None,
    minline_linewidth=2.0,
    species="A",
    direction="x",
    range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
    negative_range_scale=INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
    positive_range_scale=INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
):
    label = rf"${inversion_product_expression(direction, species)}$"
    range_values = get_inversion_product_range_values(
        product_map,
        Y,
        range_source=range_source,
    )
    contour_levels, norm, vmin, vmax = build_linear_product_levels(
        range_values,
        num_levels=levels,
        negative_range_scale=negative_range_scale,
        positive_range_scale=positive_range_scale,
    )
    cbar_ticks = build_product_cbar_ticks(vmin, vmax, cbar_nticks)
    product_plot = np.clip(np.asarray(product_map, dtype=float), vmin, vmax)

    plot_contour(
        X, Y, product_plot,
        title=label,
        levels=contour_levels,
        enable_probe=False,
        cmap=INVERSION_PRODUCT_CMAP,
        save_path=save_path,
        show=show,
        dpi=dpi,
        figsize=figsize,
        tick_length=tick_length,
        tick_width=tick_width,
        box_linewidth=box_linewidth,
        cbar_shrink=cbar_shrink,
        cbar_fraction=cbar_fraction,
        cbar_pad=cbar_pad,
        cbar_aspect=cbar_aspect,
        y_nticks=y_nticks,
        x_nticks=x_nticks,
        cbar_nticks=cbar_nticks,
        cbar_scientific=True,
        cbar_decimals=cbar_decimals,
        cbar_ticks=cbar_ticks,
        ax_aspect=ax_aspect,
        minline=minline,
        minline_kwargs={"linewidth": minline_linewidth},
        norm=norm,
    )
    save_contour_colorbar_legend(
        save_path=legend_path,
        show=show,
        dpi=dpi,
        title=label,
        cmap=INVERSION_PRODUCT_CMAP,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        tick_length=tick_length,
        tick_width=tick_width,
        box_linewidth=box_linewidth,
        cbar_nticks=cbar_nticks,
        cbar_scientific=True,
        cbar_decimals=cbar_decimals,
        cbar_ticks=cbar_ticks,
    )


def plot_inversion_region_contours(
    sol, fields_total, diff_fields,
    save_dir, prefix,
    show_inversion_region_plots, save_inversion_region_plots,
    save_inversion_region_txt,
    levels,
    dpi,
    figsize_contour,
    font_size,
    tick_length, tick_width,
    box_linewidth,
    cbar_shrink,
    cbar_fraction,
    cbar_pad,
    cbar_aspect,
    cbar_nticks,
    cbar_decimals,
    ax_aspect=1,
    x_nticks=6,
    minline_linewidth=2.0,
    inversion_product_range_source=DEFAULT_INVERSION_PRODUCT_RANGE_SOURCE,
    inversion_product_negative_range_scale=INVERSION_PRODUCT_NEGATIVE_RANGE_SCALE,
    inversion_product_positive_range_scale=INVERSION_PRODUCT_POSITIVE_RANGE_SCALE,
    species="A",
):
    inversion_regions = compute_inversion_regions(sol, fields_total, diff_fields, species=species)
    if save_inversion_region_txt:
        save_inversion_region_summary(save_dir, prefix, inversion_regions)

    inversion_dir = os.path.join(save_dir, "inversion_region_contours")
    if save_inversion_region_plots:
        os.makedirs(inversion_dir, exist_ok=True)

    X = sol["X"]
    Y = sol["Y"]
    _, _, minline_common = compute_min_diagnostics(sol, fields_total)

    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }):
        for direction in ["x", "y"]:
            save_path = None
            legend_path = None
            minline = minline_common if direction == "x" else None
            if save_inversion_region_plots:
                save_path = os.path.join(
                    inversion_dir,
                    f"{prefix}_inversion_region_{direction}.png",
                )
                legend_path = os.path.join(
                    inversion_dir,
                    f"{prefix}_inversion_region_{direction}_legend.png",
                )
            plot_inversion_mask_contour(
                X, Y, inversion_regions["sign"][direction],
                save_path=save_path,
                show=show_inversion_region_plots,
                dpi=dpi,
                figsize=figsize_contour,
                tick_length=tick_length,
                tick_width=tick_width,
                box_linewidth=box_linewidth,
                x_nticks=x_nticks,
                ax_aspect=ax_aspect,
                minline=minline,
                minline_linewidth=minline_linewidth,
            )
            save_inversion_region_legend(
                save_path=legend_path,
                show=show_inversion_region_plots,
                dpi=dpi,
                include_minline=(direction == "x"),
            )

        if save_inversion_region_plots:
            product_stem = inversion_product_file_stem("x", species)
            plot_inversion_product_contour(
                X,
                Y,
                inversion_regions["product"]["x"],
                save_path=os.path.join(inversion_dir, f"{prefix}_{product_stem}.png"),
                legend_path=os.path.join(inversion_dir, f"{prefix}_{product_stem}_legend.png"),
                show=show_inversion_region_plots,
                dpi=dpi,
                levels=levels,
                figsize=figsize_contour,
                tick_length=tick_length,
                tick_width=tick_width,
                box_linewidth=box_linewidth,
                cbar_shrink=cbar_shrink,
                cbar_fraction=cbar_fraction,
                cbar_pad=cbar_pad,
                cbar_aspect=cbar_aspect,
                cbar_nticks=cbar_nticks,
                cbar_decimals=cbar_decimals,
                x_nticks=x_nticks,
                ax_aspect=ax_aspect,
                minline=minline_common,
                minline_linewidth=minline_linewidth,
                species=species,
                direction="x",
                range_source=inversion_product_range_source,
                negative_range_scale=inversion_product_negative_range_scale,
                positive_range_scale=inversion_product_positive_range_scale,
            )

    return inversion_regions

# ============================================================
# Plotting field data (contours + click probe)
# ============================================================
def display_part_tag(part):
    mapping = {
        "jet": "open",
        "in": "open",
        "bottom": "lower",
        "bot": "lower",
        "top": "upper",
        "end": "end",
        "tot": "tot",
    }
    return mapping.get(part, part)

def centerline_part_label(part):
    mapping = {
        "tot": "field",
        "total": "field",
        "jet": "open",
        "in": "open",
        "bottom": "lower",
        "bot": "lower",
        "top": "upper",
        "end": "end",
    }
    return mapping.get(part, part)

def field_symbol(comp, sp=None, part=None):
    """
    For centerline legends / y labels.
      nu -> (nu)_{x,A}
      nv -> (nv)_{y,A}
      c -> c_A
      ju_diff -> j_{x,A}
      jv_diff -> j_{y,A}
      Ju_diff -> J_{x,A}
      Jv_diff -> J_{y,A}
    """
    if comp == "nu":
        base = r"\Gamma"
        subs = [r"\mathrm{x}"]   # component direction is a LABEL -> roman (Nature style)
    elif comp == "nv":
        base = r"\Gamma"
        subs = [r"\mathrm{y}"]   # component direction is a LABEL -> roman (Nature style)
    elif comp == "ju_diff":
        base = "j"
        subs = [r"\mathrm{x}"]   # component direction is a LABEL -> roman (Nature style)
    elif comp == "jv_diff":
        base = "j"
        subs = [r"\mathrm{y}"]   # component direction is a LABEL -> roman (Nature style)
    elif comp == "Ju_diff":
        base = "J"
        subs = [r"\mathrm{x}"]   # component direction is a LABEL -> roman (Nature style)
    elif comp == "Jv_diff":
        base = "J"
        subs = [r"\mathrm{y}"]   # component direction is a LABEL -> roman (Nature style)
    elif comp == "n":
        base = "n"
        subs = []
    elif comp == "c":
        base = "c"
        subs = []
    else:
        raise ValueError(f"Unknown comp={comp}")

    if sp is not None:
        subs.append(rf"\mathrm{{{sp}}}")
    sup = None if part in (None, "tot", "total") else rf"\mathrm{{{display_part_tag(part)}}}"

    out = base
    if subs:
        out += rf"_{{{','.join(subs)}}}"
    if sup is not None:
        out += rf"^{{{sup}}}"
    return rf"${out}$"
    
def enable_click_probe(fig, ax, X, Y, Z, fmt="x={x:.4e}, y={y:.4e}, z={z:.6e}"):
    """
    Clicking on the contourf plot prints the (x, y, z) of the nearest grid point
    to the console and shows it as text on the figure.
    X,Y,Z: meshgrid (ny,nx)
    """
    X1 = X[0, :]
    Y1 = Y[:, 0]

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def onclick(event):
        if event.inaxes != ax:
            return
        x0, y0 = event.xdata, event.ydata

        ix = int(np.clip(np.searchsorted(X1, x0), 0, len(X1)-1))
        iy = int(np.clip(np.searchsorted(Y1, y0), 0, len(Y1)-1))

        if ix > 0 and (abs(X1[ix-1]-x0) < abs(X1[ix]-x0)):
            ix -= 1
        if iy > 0 and (abs(Y1[iy-1]-y0) < abs(Y1[iy]-y0)):
            iy -= 1

        z0 = Z[iy, ix]
        msg = fmt.format(x=X1[ix], y=Y1[iy], z=z0)
        print(msg)
        txt.set_text(msg)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    return fig, ax

def apply_sci_offset(ax, axis="both", scilimits=(0, 0), useOffset=False):
    ax.ticklabel_format(axis=axis, style="sci", scilimits=scilimits,
                        useMathText=True, useOffset=useOffset)

    def _set_axis_formatter(axis_obj):
        f = mticker.ScalarFormatter(useMathText=True)
        f.set_scientific(True)
        f.set_powerlimits(scilimits)
        f.set_useOffset(useOffset)
        axis_obj.set_major_formatter(f)
        # font size is not fixed here (controlled by rcParams)
        return f

    if axis in ("x", "both"):
        _set_axis_formatter(ax.xaxis)
    if axis in ("y", "both"):
        _set_axis_formatter(ax.yaxis)

    ax.figure.canvas.draw_idle()

def field_core(comp, sp=None, part=None):
    """
    For contour titles / colorbar labels.
      nu -> (nu)_{A,tot}
      nv -> (nv)_{A,tot}
      n  -> n_{A,tot}
      c  -> c_{A,tot}
      ju_diff -> j_{x,A,tot}
      jv_diff -> j_{y,A,tot}
      Ju_diff -> J_{x,A,tot}
      Jv_diff -> J_{y,A,tot}
    """
    subs = []

    if comp == "nu":
        base = r"\Gamma"
        subs.append("x")
    elif comp == "nv":
        base = r"\Gamma"
        subs.append("y")
    elif comp == "ju_diff":
        base = "j"
        subs.append("x")
    elif comp == "jv_diff":
        base = "j"
        subs.append("y")
    elif comp == "Ju_diff":
        base = "J"
        subs.append("x")
    elif comp == "Jv_diff":
        base = "J"
        subs.append("y")
    elif comp == "n":
        base = "n"
    elif comp == "c":
        base = "c"
    else:
        raise ValueError(f"Unknown comp={comp}")

    if sp is not None:
        subs.append(rf"\mathrm{{{sp}}}")
    sup = None if part in (None, "tot", "total") else rf"\mathrm{{{display_part_tag(part)}}}"

    out = base
    if subs:
        out += rf"_{{{','.join(subs)}}}"
    if sup is not None:
        out += rf"^{{{sup}}}"
    return out

def field_latex(comp, sp=None, part=None):
    return rf"${field_core(comp, sp=sp, part=part)}$"

def field_title(comp, sp=None, tag="tot"):
    return field_latex(comp, sp=sp, part=tag)

def set_sci_axis_one_decimal(ax, axis="x", offset_x=0.98):
    fmt = ScalarFormatterFixed(decimals=1, useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    fmt.set_useOffset(False)

    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
        ax.figure.canvas.draw()
        offset_text = ax.xaxis.get_offset_text()
        offset_text.set_visible(True)
        offset_text.set_x(offset_x)
        offset_text.set_ha("left")
        offset_text.set_clip_on(False)
    else:
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_visible(True)

    ax.figure.canvas.draw_idle()


def set_linear_axis_one_decimal(ax, axis="y"):
    fmt = mticker.FormatStrFormatter("%.1f")
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)
    ax.figure.canvas.draw_idle()


def set_centerline_mass_fraction_axis(ax, axis="y"):
    ticks = np.round(np.arange(0.0, 1.0 + 0.2, 0.2), decimals=10)
    fmt = mticker.FormatStrFormatter("%.1f")
    if axis == "x":
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(fmt)
    ax.figure.canvas.draw_idle()


def _axis_limits(ax, axis="y"):
    limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
    return float(limits[0]), float(limits[1])


def _count_locator_ticks(vmin: float, vmax: float, step: float) -> int:
    lo = min(vmin, vmax)
    hi = max(vmin, vmax)
    if not np.isfinite(step) or step <= 0.0 or hi < lo:
        return 0
    start = int(np.ceil(lo / step - 1.0e-12))
    stop = int(np.floor(hi / step + 1.0e-12))
    return max(0, stop - start + 1)


def _choose_centerline_display_step(
    vmin: float,
    vmax: float,
    min_ticks: int = 3,
    max_ticks: int = 5,
) -> float:
    lo = min(vmin, vmax)
    hi = max(vmin, vmax)
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        return 0.01

    quantum = 0.01
    raw_step = float(span) / max(int(max_ticks) - 1, 1)
    raw_n = max(1, int(np.ceil(raw_step / quantum - 1.0e-12)))
    max_n = max(raw_n + 1, int(np.ceil(span / quantum + 1.0e-12)) + 200)
    target_ticks = 0.5 * (int(min_ticks) + int(max_ticks))
    valid = []
    fallback = []

    for n in range(1, max_n + 1):
        step = quantum * n
        count = _count_locator_ticks(lo, hi, step)
        score = (
            0 if int(min_ticks) <= count <= int(max_ticks) else 1,
            abs(count - target_ticks),
            abs(n - raw_n),
            n,
        )
        if int(min_ticks) <= count <= int(max_ticks):
            valid.append((score, step))
        fallback.append((score, step))

    if valid:
        return float(min(valid)[1])
    return float(min(fallback)[1])


def set_linear_axis_two_decimal(ax, axis="y"):
    fmt = mticker.FormatStrFormatter("%.2f")
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)
    ax.figure.canvas.draw_idle()


def set_linear_axis_two_decimal_ticks(ax, axis="y", min_ticks: int = 3, max_ticks: int = 5):
    vmin, vmax = _axis_limits(ax, axis=axis)
    step = _choose_centerline_display_step(vmin, vmax, min_ticks=min_ticks, max_ticks=max_ticks)
    locator = mticker.MultipleLocator(base=step)
    if axis == "x":
        ax.xaxis.set_major_locator(locator)
    else:
        ax.yaxis.set_major_locator(locator)
    set_linear_axis_two_decimal(ax, axis=axis)


def set_sci_axis_two_decimal(ax, axis="y", offset_x=0.98):
    fmt = ScalarFormatterFixed(decimals=2, useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    fmt.set_useOffset(False)

    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
        ax.figure.canvas.draw()
        offset_text = ax.xaxis.get_offset_text()
        offset_text.set_visible(True)
        offset_text.set_x(offset_x)
        offset_text.set_ha("left")
        offset_text.set_clip_on(False)
    else:
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_visible(True)

    ax.figure.canvas.draw_idle()


def set_sci_axis_two_decimal_ticks(
    ax,
    axis="y",
    min_ticks: int = 3,
    max_ticks: int = 5,
    offset_x: float = 0.98,
):
    vmin, vmax = _axis_limits(ax, axis=axis)
    max_abs = max(abs(vmin), abs(vmax))
    if max_abs > 0.0 and np.isfinite(max_abs):
        scale = 10.0 ** float(np.floor(np.log10(max_abs)))
    else:
        scale = 1.0
    display_step = _choose_centerline_display_step(
        vmin / scale,
        vmax / scale,
        min_ticks=min_ticks,
        max_ticks=max_ticks,
    )
    locator = mticker.MultipleLocator(base=display_step * scale)
    if axis == "x":
        ax.xaxis.set_major_locator(locator)
    else:
        ax.yaxis.set_major_locator(locator)
    set_sci_axis_two_decimal(ax, axis=axis, offset_x=offset_x)

def _finite_line_values(ax, axis="y"):
    values = []
    for line in ax.get_lines():
        data = line.get_xdata() if axis == "x" else line.get_ydata()
        arr = np.asarray(data, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)

def set_centerline_minmax_ticks(
    ax,
    axis="y",
    nticks: int = 5,
    scientific: bool = True,
    limit_pad_fraction: float = 0.05,
):
    values = _finite_line_values(ax, axis=axis)
    if values.size == 0:
        return

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-6
        vmin -= pad
        vmax += pad

    ticks = np.linspace(vmin, vmax, max(2, int(nticks)))
    pad = max(0.0, float(limit_pad_fraction)) * (vmax - vmin)
    lim_min = vmin - pad
    lim_max = vmax + pad
    locator = mticker.FixedLocator(ticks)
    if axis == "x":
        ax.set_xlim(lim_min, lim_max)
        ax.xaxis.set_major_locator(locator)
    else:
        ax.set_ylim(lim_min, lim_max)
        ax.yaxis.set_major_locator(locator)

    if scientific:
        set_sci_axis_two_decimal(ax, axis=axis)
    else:
        set_linear_axis_two_decimal(ax, axis=axis)

def inlet_number_density(sol, sp: str):
    sp = str(sp).upper()
    nested = sol.get("n_in") if isinstance(sol, dict) else None
    if isinstance(nested, dict) and sp in nested:
        return float(nested[sp])
    key = f"n_in_{sp}"
    if isinstance(sol, dict) and key in sol:
        return float(sol[key])
    return None

def apply_axes_style(ax, box_linewidth, tick_length, tick_width, tick_pad=6):
    for sp in ax.spines.values():
        sp.set_linewidth(box_linewidth)
    ax.tick_params(axis="both", which="major",
                   length=tick_length, width=tick_width,
                   pad=tick_pad)

def place_colorbar_offset_text_outside(cbar, x=0.5, y_points=7.0):
    """
    scientific notation position
    Move the scientific-notation offset text (e.g. x10^n) to the outer-right
    side of a vertical colorbar so it does not overlap the main domain.
    """
    cbar.ax.figure.canvas.draw()
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_visible(True)
    offset_text.set_x(x)
    offset_text.set_ha("left")
    offset_text.set_clip_on(False)
    offset_text.set_transform(
        offset_text.get_transform()
        + ScaledTranslation(0, y_points / 72.0, cbar.ax.figure.dpi_scale_trans)
    )
from matplotlib import colors as mcolors


def contour_rc_params(font_size):
    return {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }

def plot_contour(
    X, Y, Z, title, levels,
    enable_probe, cmap,
    save_path, show, dpi,
    figsize,
    tick_length, tick_width,
    box_linewidth,
    cbar_shrink,
    cbar_fraction,
    cbar_pad,
    cbar_aspect,
    y_nticks=3, x_nticks=6,
    cbar_nticks=5, cbar_scientific=True,
    cbar_decimals=2,          # default: 2 decimal places
    ax_aspect=1,              # x-direction compression (1 = equal aspect, AR = square axes)
    minline=None,             # (x_line, y_line)
    minline_kwargs=None,
    norm=None,
    xlim=None,
    ylim=None,
    x_ticks=None,
    cbar_ticks=None,
):
    # Non-dimensionalize spatial length axes by the full trench height 2h.
    # The contour mesh is CELL-CENTERED, so its extent is half a cell short of the walls.
    # Reconstruct the TRUE physical domain (walls) from the uniform cell spacing and use it
    # for all axis limits/ticks; the DATA arrays are plotted unchanged (still cell-centered).
    dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 0.0
    dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 0.0
    Lref = (float(np.max(Y)) - float(np.min(Y))) + dy  # true full trench height 2h
    Lref = Lref if Lref else 1.0
    Lm_true = float(np.max(X)) + 0.5 * dx              # true depth Lm (right wall)
    AR = Lm_true / Lref                                # true aspect ratio

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(X / Lref, Y / Lref, Z, levels=levels, cmap=cmap, norm=norm)

    ax.set_xlabel(r"$x/(2h)$")
    ax.set_ylabel(r"$y/(2h)$")
    ax.set_title("")
    ax.set_aspect(ax_aspect, adjustable="box")

    # Exact physical non-dimensional domain (no padded-edge distortion):
    # main view x in (0, AR); zoom view (xlim passed in SI) x in (zoom_left_nd, AR),
    # with zoom_left_nd rounded to one decimal. y in (-0.5, 0.5).
    is_zoom = xlim is not None
    zoom_left_nd = None
    if is_zoom:
        zoom_left_nd = round(float(xlim[0]) / Lref, 1)
        ax.set_xlim(zoom_left_nd, AR)
    else:
        ax.set_xlim(0.0, AR)
    ax.set_ylim(-0.5, 0.5)

    # min Y_A line
    if minline is not None:
        xl, yl = minline
        lk = dict(color="k", linestyle="--", linewidth=3.0, zorder=20)
        if minline_kwargs is not None:
            lk.update(minline_kwargs)
        ax.plot(np.asarray(xl) / Lref, np.asarray(yl) / Lref, **lk)

    # y-axis ticks: walls + centre.
    ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # x-axis ticks
    if is_zoom:
        _ZOOM_NTICKS_BY_AR = {3: 3, 5: 4, 10: 5}
        ar_key = int(round(AR))
        if ar_key in _ZOOM_NTICKS_BY_AR:
            nz = _ZOOM_NTICKS_BY_AR[ar_key]
        else:
            nz = max(3, min(6, int(round(AR - zoom_left_nd)) + 2))
        zoom_ticks = np.round(np.linspace(zoom_left_nd, AR, nz), 1)
        ax.xaxis.set_major_locator(mticker.FixedLocator(zoom_ticks))
    else:
        nx = 3 if AR <= 1.5 else (4 if AR <= 3.5 else 6)
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0.0, AR, nx)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # colorbar formatter
    if cbar_scientific:
        cfmt = ScalarFormatterFixed(decimals=cbar_decimals, useMathText=True)
        cfmt.set_scientific(True)
        cfmt.set_powerlimits((0, 0))
        cfmt.set_useOffset(False)
    else:
        cfmt = mticker.FormatStrFormatter(f"%.{cbar_decimals}f")

    cbar = fig.colorbar(
        cf, ax=ax,
        orientation="vertical",
        pad=cbar_pad,
        fraction=cbar_fraction,
        shrink=cbar_shrink,
        format=cfmt,
        aspect=cbar_aspect
    )

    cbar.set_label(title, rotation=90, labelpad=12)

    vmin = float(np.nanmin(Z))
    vmax = float(np.nanmax(Z))
    if cbar_ticks is None:
        ticks = [vmin] if np.isclose(vmin, vmax) else np.linspace(vmin, vmax, cbar_nticks)
    else:
        ticks = np.asarray(cbar_ticks, dtype=float)
    cbar.set_ticks(ticks)

    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_visible(True)

    apply_axes_style(ax, box_linewidth, tick_length, tick_width, tick_pad=8)
    apply_axes_style(cbar.ax, box_linewidth, tick_length, tick_width, tick_pad=8)
    if cbar_scientific:
        place_colorbar_offset_text_outside(cbar)

    fig.tight_layout()
    if cbar_scientific:
        place_colorbar_offset_text_outside(cbar)

    if enable_probe and show:
        enable_click_probe(fig, ax, X, Y, Z)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if not show:
        plt.close(fig)

    return fig, ax
    
def plot_centerline(
    sol, fields_total, diff_fields, contrib,
    save_dir, prefix,
    show, save, dpi,
    figsize,                 
    centerline_linewidth,    
    box_linewidth,           
    tick_length, tick_width  
):
    x = sol["x"]
    y = sol["y"]
    iy0 = int(np.argmin(np.abs(y - 0.0)))

    # Non-dimensionalize spatial length (x) axis by the full trench height 2h.
    # Scaled copy is used for plotting only; index computations use original x.
    Lref = 2.0 * float(sol["h"])
    x_s = np.asarray(x) / Lref

    centerline_dir = os.path.join(save_dir, "centerline_plots")
    if save:
        os.makedirs(centerline_dir, exist_ok=True)

    show_inline_legend = show and (not save)
    saved_legends = set()

    def _save_legend(handles, labels, name, ncol=1):
        if (not save) or (len(handles) == 0):
            return
        if name in saved_legends:
            return

        # keep the first occurrence of each legend entry
        unique_handles = []
        unique_labels = []
        seen = set()
        for h, lab in zip(handles, labels):
            if (lab in seen) or (lab == "_nolegend_"):
                continue
            seen.add(lab)
            unique_handles.append(h)
            unique_labels.append(lab)

        if len(unique_handles) == 0:
            return

        ncol = max(1, min(int(ncol), len(unique_handles)))
        nrows = int(np.ceil(len(unique_handles) / ncol))
        legend_fig, legend_ax = plt.subplots(
            figsize=(2.2 * ncol + 0.8, 0.7 * nrows + 0.8)
        )
        legend_ax.axis("off")
        legend_ax.legend(
            unique_handles, unique_labels,
            loc="center",
            ncol=ncol,
            frameon=False
        )
        legend_fig.savefig(
            os.path.join(centerline_dir, f"{prefix}_{name}_legend.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close(legend_fig)
        saved_legends.add(name)

    def _savefig(fig, name, ax=None, legend_name=None, legend_ncol=1):
        if save and (ax is not None) and (legend_name is not None):
            handles, labels = ax.get_legend_handles_labels()
            _save_legend(handles, labels, legend_name, ncol=legend_ncol)
        if save:
            fig.savefig(os.path.join(centerline_dir, f"{prefix}_{name}.png"),
                        dpi=dpi, bbox_inches="tight")
        if not show:
            plt.close(fig)

    line_colors = CUSTOM_COLORS

    def _centerline_linewidth(comp, part="tot"):
        linewidth = float(centerline_linewidth)
        if contrib is None:
            return linewidth
        if comp in {"n", "c", "nu"} and part == "bottom":
            linewidth += 2.0
        if comp == "nv":
            if part == "tot":
                linewidth += 4.0
            elif part == "jet":
                linewidth += 2.0
        return linewidth

    def _plot_number_density_centerline_with_inlet(sp, comp, total_line, sym):
        n_in = inlet_number_density(sol, sp)
        if n_in is None or not np.isfinite(n_in):
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            x_s,
            total_line,
            label=centerline_part_label("tot"),
            color=line_colors[0],
            linewidth=_centerline_linewidth(comp, "tot"),
        )

        if contrib is not None:
            parts = ["jet", "bottom", "top", "end"]
            for k, part in enumerate(parts, start=1):
                ax.plot(
                    x_s,
                    contrib[sp][part][comp][iy0, :],
                    label=centerline_part_label(part),
                    color=line_colors[k % len(line_colors)],
                    linewidth=_centerline_linewidth(comp, part),
                )

        ax.axhline(
            n_in,
            color="k",
            linestyle="--",
            linewidth=centerline_linewidth,
            label=r"$n_{\mathrm{open,res}}$",
        )

        ax.set_xlabel(r"$x/(2h)$")
        ax.set_ylabel(sym)
        # x-axis (x/(2h) depth, a length) -> plain decimal; y-axis (data) stays scientific.
        apply_sci_offset(ax, axis="y", scilimits=(0, 0))
        _AR = float(sol["Lm"]) / Lref
        ax.set_xlim(-0.05 * _AR, 1.05 * _AR)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, _AR / 2.0, _AR]))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
        apply_axes_style(ax, box_linewidth, tick_length, tick_width)
        fig.tight_layout()
        _savefig(
            fig,
            f"centerline_{sp}_{comp_tag_mol[comp]}_with_inlet",
            ax=ax,
            legend_name=f"centerline_{sp}_{comp_tag_mol[comp]}_with_inlet",
            legend_ncol=2,
        )

    def _plot_extreme_marker(ax, x_values, line_values, color, markersize, mode):
        if mode == "min":
            ix_extreme = int(np.nanargmin(line_values))
        elif mode == "max":
            ix_extreme = int(np.nanargmax(line_values))
        else:
            raise ValueError(f"Unsupported marker mode={mode!r}")
        ax.plot(
            x_values[ix_extreme],
            line_values[ix_extreme],
            marker="*",
            linestyle="None",
            markersize=markersize,
            color=color,
            label="_nolegend_",
        )

    # (1) mass fraction
    fig, ax = plt.subplots(figsize=figsize)
    YA_line = fields_total["YA"][iy0, :]
    YB_line = fields_total["YB"][iy0, :]

    ax.plot(x_s, YA_line, label=r"$Y_{\mathrm{A}}$", color=line_colors[0],
            linewidth=centerline_linewidth)
    ax.plot(x_s, YB_line, label=r"$Y_{\mathrm{B}}$", color=line_colors[4],
            linewidth=centerline_linewidth)

    # Add star markers for the Y_A minimum and Y_B maximum.
    ix_min_A = int(np.nanargmin(YA_line))
    ix_max_B = int(np.nanargmax(YB_line))

    ax.plot(x_s[ix_min_A], YA_line[ix_min_A],
            marker="*", linestyle="None",
            markersize=centerline_linewidth * 5, color=line_colors[0],
            label="_nolegend_")   # not added to the legend

    ax.plot(x_s[ix_max_B], YB_line[ix_max_B],
            marker="*", linestyle="None",
            markersize=centerline_linewidth * 5, color=line_colors[4],
            label="_nolegend_")   # not added to the legend

    ax.set_xlabel(r"$x/(2h)$")
    ax.set_ylabel("mass fraction")
    if show_inline_legend:
        ax.legend()
    # x-axis (x/(2h) depth, a length) -> plain decimal; y-axis is a fraction (unchanged).
    _AR = float(sol["Lm"]) / Lref
    ax.set_xlim(-0.05 * _AR, 1.05 * _AR)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, _AR / 2.0, _AR]))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    set_centerline_mass_fraction_axis(ax, axis="y")
    apply_axes_style(ax, box_linewidth, tick_length, tick_width)
    fig.tight_layout()
    _savefig(fig, "centerline_mass_fraction", ax=ax,
             legend_name="centerline_mass_fraction", legend_ncol=2)

    if "XA" in fields_total and "XB" in fields_total:
        fig, ax = plt.subplots(figsize=figsize)
        XA_line = fields_total["XA"][iy0, :]
        XB_line = fields_total["XB"][iy0, :]

        ax.plot(x_s, XA_line, label=r"$X_{\mathrm{A}}$", color=line_colors[0],
                linewidth=centerline_linewidth)
        ax.plot(x_s, XB_line, label=r"$X_{\mathrm{B}}$", color=line_colors[4],
                linewidth=centerline_linewidth)

        _plot_extreme_marker(ax, x_s, XA_line, line_colors[0], centerline_linewidth * 5, "min")
        _plot_extreme_marker(ax, x_s, XB_line, line_colors[4], centerline_linewidth * 5, "max")

        ax.set_xlabel(r"$x/(2h)$")
        ax.set_ylabel("mole fraction")
        if show_inline_legend:
            ax.legend()
        # x-axis (x/(2h) depth, a length) -> plain decimal; y-axis is a fraction (unchanged).
        _AR = float(sol["Lm"]) / Lref
        ax.set_xlim(-0.05 * _AR, 1.05 * _AR)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, _AR / 2.0, _AR]))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        set_centerline_mass_fraction_axis(ax, axis="y")
        apply_axes_style(ax, box_linewidth, tick_length, tick_width)
        fig.tight_layout()
        _savefig(fig, "centerline_mole_fraction", ax=ax,
                 legend_name="centerline_mole_fraction", legend_ncol=2)

    comp_tag_mol = {"nu": "nu", "nv": "nv", "n": "n", "c": "c"}
    comp_tag_diff = {"ju_diff": "j_x", "jv_diff": "j_y"}
    comp_tag_molar_diff = {"Ju_diff": "J_x", "Jv_diff": "J_y"}
    
    # (2) centerline fields (A/B)
    for sp in ["A", "B"]:
        for comp in ["nu", "nv", "n", "c"]:
            if comp not in fields_total[sp]:
                continue
            sym = field_symbol(comp, sp=sp)
            total_line = fields_total[sp][comp][iy0, :]

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(x_s, total_line,
                    label=centerline_part_label("tot"),
                    color=line_colors[0],
                    linewidth=_centerline_linewidth(comp, "tot"))

            if contrib is not None:
                parts = ["jet", "bottom", "top", "end"]
                for k, part in enumerate(parts, start=1):
                    ax.plot(x_s, contrib[sp][part][comp][iy0, :],
                            label=centerline_part_label(part),
                            color=line_colors[k % len(line_colors)],
                            linewidth=_centerline_linewidth(comp, part))

                if comp == "n":
                    if sp == "A":
                        ix_star = int(np.nanargmin(total_line))
                    else:
                        ix_star = int(np.nanargmax(total_line))
                    ax.plot(
                        x_s[ix_star],
                        total_line[ix_star],
                        marker="*",
                        linestyle="None",
                        markersize=_centerline_linewidth(comp, "tot") * 5,
                        color=line_colors[0],
                        label="_nolegend_",
                    )

            if comp == "c":
                _plot_extreme_marker(
                    ax,
                    x_s,
                    total_line,
                    line_colors[0],
                    _centerline_linewidth(comp, "tot") * 5,
                    "min" if sp == "A" else "max",
                )

            ax.set_xlabel(r"$x/(2h)$")
            ax.set_ylabel(sym)
            if show_inline_legend:
                ax.legend(ncol=2)  # font size not fixed
            # x-axis (x/(2h) depth, a length) -> plain decimal; y-axis (data) stays scientific.
            apply_sci_offset(ax, axis="y", scilimits=(0, 0))
            _AR = float(sol["Lm"]) / Lref
            ax.set_xlim(-0.05 * _AR, 1.05 * _AR)
            ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, _AR / 2.0, _AR]))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
            apply_axes_style(ax, box_linewidth, tick_length, tick_width)
            fig.tight_layout()
            _savefig(fig, f"centerline_{sp}_{comp_tag_mol[comp]}", ax=ax,
                     legend_name="centerline_contribution", legend_ncol=2)
            if sp == "A" and comp == "n":
                _plot_number_density_centerline_with_inlet(sp, comp, total_line, sym)

    # (3) centerline diffusive mass and molar flux fields (A/B)
    if diff_fields is not None:
        for sp in ["A", "B"]:
            for comp in ["ju_diff", "jv_diff", "Ju_diff", "Jv_diff"]:
                if comp not in diff_fields[sp]:
                    continue
                sym = field_symbol(comp, sp=sp)
                comp_tag = (
                    comp_tag_diff[comp]
                    if comp in comp_tag_diff
                    else comp_tag_molar_diff[comp]
                )

                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(x_s, diff_fields[sp][comp][iy0, :],
                        label=centerline_part_label("tot"),
                        color=line_colors[0],
                        linewidth=centerline_linewidth)

                ax.set_xlabel(r"$x/(2h)$")
                ax.set_ylabel(sym)
                if show_inline_legend:
                    ax.legend(ncol=2)
                # x-axis (x/(2h) depth, a length) -> plain decimal; y-axis (data) stays scientific.
                apply_sci_offset(ax, axis="y", scilimits=(0, 0))
                _AR = float(sol["Lm"]) / Lref
                ax.set_xlim(-0.05 * _AR, 1.05 * _AR)
                ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, _AR / 2.0, _AR]))
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
                set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
                apply_axes_style(ax, box_linewidth, tick_length, tick_width)
                fig.tight_layout()
                _savefig(fig, f"centerline_{sp}_{comp_tag}", ax=ax,
                         legend_name="centerline_contribution", legend_ncol=2)

        # (4) combined-species flux plots: A (solid) and B (dashed) on one axis,
        # mirroring the mass-fraction plot. One plot per component (j_x, j_y, J_x, J_y).
        for comp in ["ju_diff", "jv_diff", "Ju_diff", "Jv_diff"]:
            if comp not in diff_fields["A"] or comp not in diff_fields["B"]:
                continue
            comp_tag = (
                comp_tag_diff[comp]
                if comp in comp_tag_diff
                else comp_tag_molar_diff[comp]
            )
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(x_s, diff_fields["A"][comp][iy0, :], label=field_symbol(comp, sp="A"),
                    color=line_colors[0], linestyle="-", linewidth=centerline_linewidth)
            ax.plot(x_s, diff_fields["B"][comp][iy0, :], label=field_symbol(comp, sp="B"),
                    color=line_colors[4], linestyle="--", linewidth=centerline_linewidth)
            ax.set_xlabel(r"$x/(2h)$")
            ax.set_ylabel(field_symbol(comp, sp=None))
            if show_inline_legend:
                ax.legend(ncol=2)
            apply_sci_offset(ax, axis="y", scilimits=(0, 0))
            _AR = float(sol["Lm"]) / Lref
            ax.set_xlim(-0.05 * _AR, 1.05 * _AR)
            ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, _AR / 2.0, _AR]))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
            apply_axes_style(ax, box_linewidth, tick_length, tick_width)
            fig.tight_layout()
            _savefig(fig, f"centerline_{comp_tag}", ax=ax,
                     legend_name=f"centerline_AB_{comp_tag}", legend_ncol=2)

def plot_centerline_results(
    sol, fields_total, diff_fields, contrib,
    save_dir, prefix,
    show_centerline_plots, save_centerline_plots,
    dpi,
    figsize_centerline,
    font_size,
    tick_length, tick_width,
    box_linewidth,
    centerline_linewidth
):
    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }):
        plot_centerline(
            sol, fields_total, diff_fields, contrib,
            save_dir=save_dir, prefix=prefix,
            show=show_centerline_plots, save=save_centerline_plots, dpi=dpi,
            figsize=figsize_centerline,
            centerline_linewidth=centerline_linewidth,
            box_linewidth=box_linewidth,
            tick_length=tick_length, tick_width=tick_width
        )


def _plot_field_contour_group_worker(task):
    X = task["X"]
    Y = task["Y"]
    Z = task["Z"]
    title = task["title"]
    levels = task["levels"]
    enable_probe = task["enable_probe"]
    show = task["show"]
    dpi = task["dpi"]
    figsize = task["figsize"]
    tick_length = task["tick_length"]
    tick_width = task["tick_width"]
    box_linewidth = task["box_linewidth"]
    cbar_shrink = task["cbar_shrink"]
    cbar_pad = task["cbar_pad"]
    cbar_nticks = task["cbar_nticks"]
    cbar_scientific = task["cbar_scientific"]
    cbar_decimals = task["cbar_decimals"]
    ax_aspect = task["ax_aspect"]
    minline_linewidth = task["minline_linewidth"]
    powernorm_gamma = task["powernorm_gamma"]

    # True full trench height 2h (reconstructed from cell-centered Y) so the DATA crop window
    # matches the rounded zoom axis limit computed inside plot_contour.
    _dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 0.0
    Lref_full = (float(np.max(Y)) - float(np.min(Y))) + _dy
    Lref_full = Lref_full if Lref_full else 1.0

    saved = []
    for variant in task["variants"]:
        save_path = variant["save_path"]
        local_xlim = variant["xlim"]
        crop_xlim = local_xlim
        if local_xlim is not None:
            # Round the zoom start to one decimal in x/(2h) before cropping the DATA,
            # so it spans exactly [zoom_left_nd, AR] like the axis.
            zoom_left_nd = round(float(local_xlim[0]) / Lref_full, 1)
            crop_xlim = (zoom_left_nd * Lref_full, float(local_xlim[1]))
        X_plot, Y_plot, Z_plot = crop_contour_to_x_window(X, Y, Z, xlim=crop_xlim)
        contour_norm = build_powernorm(Z_plot, powernorm_gamma=powernorm_gamma)
        minline = variant["minline"]
        with plt.rc_context(contour_rc_params(variant["font_size"])):
            plot_contour(
                X_plot, Y_plot, Z_plot,
                title=title,
                levels=levels, enable_probe=enable_probe, cmap="jet",
                save_path=save_path, show=show, dpi=dpi,
                figsize=figsize,
                tick_length=tick_length, tick_width=tick_width,
                box_linewidth=box_linewidth,
                cbar_shrink=cbar_shrink,
                cbar_fraction=variant["cbar_fraction"],
                cbar_pad=cbar_pad,
                cbar_aspect=variant["cbar_aspect"],
                cbar_nticks=cbar_nticks, cbar_scientific=cbar_scientific,
                cbar_decimals=cbar_decimals,
                ax_aspect=ax_aspect,
                x_nticks=variant["x_nticks"],
                x_ticks=variant["x_ticks"],
                minline=minline,
                minline_kwargs={"linewidth": float(minline_linewidth)} if minline is not None else None,
                norm=contour_norm,
                xlim=local_xlim,
            )
        if save_path is not None:
            saved.append(str(save_path))
    return saved


def _run_plot_worker_tasks(tasks, max_workers: int):
    if not tasks:
        return
    workers = max(1, min(int(max_workers), len(tasks)))
    if workers == 1:
        for task in tasks:
            _plot_field_contour_group_worker(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_plot_field_contour_group_worker, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


def plot_field_contours(
    sol, fields_total, diff_fields, contrib,
    save_dir, prefix,
    show_field_contour_plots, save_field_contour_plots,
    levels, enable_probe, dpi,

    cbar_shrink,
    cbar_fraction,
    cbar_pad,
    cbar_aspect,

    figsize_contour,
    font_size,
    tick_length, tick_width,
    box_linewidth,

    cbar_nticks,
    cbar_decimals,
    minline_linewidth,
    ax_aspect=1,
    x_nticks=6,
    mass_fraction_powernorm_gamma=None,
    zoom_factor=3.0,
    zoom_xlim=None,
    zoom_font_size=None,
    zoom_cbar_fraction=None,
    zoom_cbar_aspect=None,
    zoom_x_nticks=None,
    x_ticks=None,
    zoom_x_ticks=None,
    parallel_save_workers=1,
):
    X = sol["X"]; Y = sol["Y"]
    os.makedirs(save_dir, exist_ok=True)
    if zoom_font_size is None:
        zoom_font_size = font_size
    if zoom_cbar_fraction is None:
        zoom_cbar_fraction = cbar_fraction
    if zoom_cbar_aspect is None:
        zoom_cbar_aspect = cbar_aspect
    if zoom_x_nticks is None:
        zoom_x_nticks = max(3, min(int(x_nticks), 5))
    else:
        zoom_x_nticks = max(2, int(zoom_x_nticks))
    if x_ticks is not None:
        x_ticks = tuple(float(value) for value in x_ticks)
    if zoom_x_ticks is not None:
        zoom_x_ticks = tuple(float(value) for value in zoom_x_ticks)

    # -----------------------------
    # 1) compute the minimum points (for saving) + save txt
    # -----------------------------
    info_min, ya_curve, minline_common = compute_min_diagnostics(sol, fields_total)

    # -----------------------------
    # 2) two output folders for contours
    # -----------------------------
    contour_dir_plain = os.path.join(save_dir, "contours_plain")
    contour_dir_line  = os.path.join(save_dir, "contours_with_YA_min_line")
    os.makedirs(contour_dir_plain, exist_ok=True)
    os.makedirs(contour_dir_line,  exist_ok=True)

    def spath_plain(group, name, zoom=False):
        if not save_field_contour_plots:
            return None
        parts = [contour_dir_plain]
        if zoom:
            parts.append("zoom")
        parts.append(group)
        return os.path.join(*parts, f"{prefix}_{name}.png")

    def spath_line(group, name, zoom=False):
        if not save_field_contour_plots:
            return None
        parts = [contour_dir_line]
        if zoom:
            parts.append("zoom")
        parts.append(group)
        return os.path.join(*parts, f"{prefix}_{name}.png")

    comp_tag_mol = {"nu": "nu", "nv": "nv", "n": "n", "c": "c"}
    comp_tag_diff = {"ju_diff": "j_x", "jv_diff": "j_y"}
    comp_tag_molar_diff = {"Ju_diff": "J_x", "Jv_diff": "J_y"}
    tag_map = {"bottom": "bot", "top": "top", "end": "end", "jet": "in"}
    if zoom_xlim is None:
        zoom_xlim = compute_inversion_zoom_xlim(sol["x"], minline_common[0], zoom_factor=zoom_factor)
    else:
        zoom_xlim = tuple(float(value) for value in zoom_xlim)
    parallel_save_workers = max(1, int(parallel_save_workers))
    use_parallel_save = (
        parallel_save_workers > 1
        and save_field_contour_plots
        and not show_field_contour_plots
        and not enable_probe
    )
    contour_tasks = []
    # -----------------------------
    # 3) wrapper that saves both the plain and the min-line versions
    # -----------------------------
    def plot_contour_both(
        Z, title, group, fname,
        enable_probe_local,
        cbar_scientific, cbar_nticks,
        powernorm_gamma=None
    ):
        task = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "title": title,
            "levels": levels,
            "enable_probe": enable_probe_local,
            "show": show_field_contour_plots,
            "dpi": dpi,
            "figsize": figsize_contour,
            "tick_length": tick_length,
            "tick_width": tick_width,
            "box_linewidth": box_linewidth,
            "cbar_shrink": cbar_shrink,
            "cbar_pad": cbar_pad,
            "cbar_nticks": cbar_nticks,
            "cbar_scientific": cbar_scientific,
            "cbar_decimals": cbar_decimals,
            "ax_aspect": ax_aspect,
            "minline_linewidth": minline_linewidth,
            "powernorm_gamma": powernorm_gamma,
            "variants": [
                {
                    "save_path": spath_plain(group, fname),
                    "minline": None,
                    "xlim": None,
                    "x_nticks": x_nticks,
                    "x_ticks": x_ticks,
                    "font_size": font_size,
                    "cbar_fraction": cbar_fraction,
                    "cbar_aspect": cbar_aspect,
                },
                {
                    "save_path": spath_line(group, fname),
                    "minline": minline_common,
                    "xlim": None,
                    "x_nticks": x_nticks,
                    "x_ticks": x_ticks,
                    "font_size": font_size,
                    "cbar_fraction": cbar_fraction,
                    "cbar_aspect": cbar_aspect,
                },
                {
                    "save_path": spath_plain(group, fname, zoom=True),
                    "minline": None,
                    "xlim": zoom_xlim,
                    "x_nticks": zoom_x_nticks,
                    "x_ticks": zoom_x_ticks,
                    "font_size": zoom_font_size,
                    "cbar_fraction": zoom_cbar_fraction,
                    "cbar_aspect": zoom_cbar_aspect,
                },
                {
                    "save_path": spath_line(group, fname, zoom=True),
                    "minline": minline_common,
                    "xlim": zoom_xlim,
                    "x_nticks": zoom_x_nticks,
                    "x_ticks": zoom_x_ticks,
                    "font_size": zoom_font_size,
                    "cbar_fraction": zoom_cbar_fraction,
                    "cbar_aspect": zoom_cbar_aspect,
                },
            ],
        }
        if use_parallel_save:
            contour_tasks.append(task)
        else:
            _plot_field_contour_group_worker(task)

    # total: mass fraction
    plot_contour_both(fields_total["YA"], r"$Y_{\mathrm{A}}$", "YA", "tot",
                      enable_probe_local=enable_probe,
                      cbar_scientific=False, cbar_nticks=cbar_nticks,
                      powernorm_gamma=mass_fraction_powernorm_gamma)

    plot_contour_both(fields_total["YB"], r"$Y_{\mathrm{B}}$", "YB", "tot",
                      enable_probe_local=enable_probe,
                      cbar_scientific=False, cbar_nticks=cbar_nticks,
                      powernorm_gamma=mass_fraction_powernorm_gamma)

    if "XA" in fields_total and "XB" in fields_total:
        plot_contour_both(fields_total["XA"], r"$X_{\mathrm{A}}$", "XA", "tot",
                          enable_probe_local=enable_probe,
                          cbar_scientific=False, cbar_nticks=cbar_nticks,
                          powernorm_gamma=mass_fraction_powernorm_gamma)

        plot_contour_both(fields_total["XB"], r"$X_{\mathrm{B}}$", "XB", "tot",
                          enable_probe_local=enable_probe,
                          cbar_scientific=False, cbar_nticks=cbar_nticks,
                          powernorm_gamma=mass_fraction_powernorm_gamma)

    # total: A/B molecular number density and molecular number flux
    for sp in ["A", "B"]:
        for comp in ["nu", "nv", "n", "c"]:
            if comp not in fields_total[sp]:
                continue
            title = field_title(comp, sp=sp, tag="tot")
            plot_contour_both(fields_total[sp][comp], title, comp_tag_mol[comp], f"{sp}_tot",
                              enable_probe_local=enable_probe,
                              cbar_scientific=True, cbar_nticks=cbar_nticks)

    # total: A/B diffusive mass and molar flux
    if diff_fields is not None:
        for sp in ["A", "B"]:
            for comp in ["ju_diff", "jv_diff", "Ju_diff", "Jv_diff"]:
                if comp not in diff_fields[sp]:
                    continue
                title = field_title(comp, sp=sp, tag="tot")
                comp_tag = (
                    comp_tag_diff[comp]
                    if comp in comp_tag_diff
                    else comp_tag_molar_diff[comp]
                )
                plot_contour_both(diff_fields[sp][comp], title, comp_tag, f"{sp}_tot",
                                  enable_probe_local=enable_probe,
                                  cbar_scientific=True, cbar_nticks=cbar_nticks)

    # contributions
    if contrib is not None:
        for sp in ["A", "B"]:
            for part in ["jet", "bottom", "top", "end"]:
                for comp in ["nu", "nv", "n", "c"]:
                    if comp not in contrib[sp][part]:
                        continue
                    title = field_title(comp, sp=sp, tag=tag_map[part])
                    plot_contour_both(contrib[sp][part][comp], title,
                                      comp_tag_mol[comp],
                                      f"{sp}_{display_part_tag(part)}",
                                      enable_probe_local=False,
                                      cbar_scientific=True, cbar_nticks=cbar_nticks)

    if use_parallel_save:
        _run_plot_worker_tasks(contour_tasks, parallel_save_workers)
                        

# ============================================================
# Plotting reaction / wall-flux data
# ============================================================
def save_line_legend(plot_dir, prefix, name, handles, labels, dpi, ncol=1):
    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen or label == "_nolegend_":
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if not unique_handles:
        return

    ncol = max(1, min(int(ncol), len(unique_handles)))
    nrows = int(np.ceil(len(unique_handles) / ncol))
    fig, ax = plt.subplots(figsize=(2.2 * ncol + 0.8, 0.7 * nrows + 0.8))
    ax.axis("off")
    ax.legend(unique_handles, unique_labels, loc="center", ncol=ncol, frameon=False)
    fig.savefig(os.path.join(plot_dir, f"{prefix}_{name}_legend.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def surface_wall_polyline_coordinates(sol):
    x = np.asarray(sol["x"], dtype=float)
    y = np.asarray(sol["y"], dtype=float)
    Lm = float(sol["Lm"])
    h = float(sol["h"])
    s_bot = x - (Lm + h)
    s_end = y
    s_top = h + (Lm - x[::-1])
    return s_bot, s_end, s_top, h


def plot_surface_quantity_along_wall(
    sol,
    total_values,
    component_values,
    save_dir,
    prefix,
    plot_dirname,
    stem,
    ylabel,
    show_plots,
    save_plots,
    dpi,
    figsize,
    zoom_figsize,
    font_size,
    tick_length,
    tick_width,
    box_linewidth,
    line_width,
    wall_dashed_width,
    end_zoom_pad=None,
    end_zoom_rate_pad_frac=0.08,
    end_zoom_rate_pad_frac_contrib=None,
    figsize_contrib=None,
    zoom_figsize_contrib=None,
    color=None,
    legend_ncol=3,
):
    x = np.asarray(sol["x"], dtype=float)
    y = np.asarray(sol["y"], dtype=float)
    s_bot, s_end, s_top, h = surface_wall_polyline_coordinates(sol)
    wall_marks = [-h, +h]

    # Non-dimensionalize the along-wall spatial coordinate s by the full trench height 2h.
    # Scaled copies are used for plotting only; data-reduction masks keep SI s-values.
    Lref = 2.0 * float(h)
    s_bot_s = s_bot / Lref
    s_end_s = s_end / Lref
    s_top_s = s_top / Lref
    # True-domain non-dimensional ticks (NOT from cell-center data max): side walls run along
    # depth [0, Lm] -> nd extent AR; end-wall corners (vertices) at s = +/- h -> +/- 0.5;
    # the open ends of the unwrapped polyline are at s = +/- (Lm + h) -> +/- (AR + 0.5).
    Lm = float(sol["Lm"])
    AR = Lm / Lref
    s_wall_end_nd = (Lm + float(h)) / Lref
    plot_dir = os.path.join(save_dir, plot_dirname)
    zoom_dir = os.path.join(plot_dir, "zoom")
    show_inline_legend = show_plots and (not save_plots)

    total_series = {
        "total": {
            "bottom": np.asarray(total_values["bottom"], dtype=float),
            "end": np.asarray(total_values["end"], dtype=float),
            "top": np.asarray(total_values["top"], dtype=float),
        }
    }

    contribution_series = None
    if component_values is not None:
        contribution_series = {"total": total_series["total"]}
        for part in ("jet", "bottom", "top", "end"):
            contribution_series[part] = {
                "bottom": np.asarray(component_values["bottom"][part], dtype=float),
                "end": np.asarray(component_values["end"][part], dtype=float),
                "top": np.asarray(component_values["top"][part], dtype=float),
            }

    line_colors = CUSTOM_COLORS
    series_colors = {
        "total": line_colors[0] if color is None else color,
        "jet": line_colors[1],
        "bottom": line_colors[2],
        "top": line_colors[3],
        "end": line_colors[4],
    }

    if zoom_figsize is None:
        zoom_figsize = figsize
    if figsize_contrib is None:
        figsize_contrib = figsize
    if zoom_figsize_contrib is None:
        zoom_figsize_contrib = zoom_figsize if zoom_figsize is not None else figsize_contrib
    if end_zoom_rate_pad_frac_contrib is None:
        end_zoom_rate_pad_frac_contrib = end_zoom_rate_pad_frac

    def _set_ylim_from_values(ax, values, pad_frac):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        if np.isclose(ymin, ymax):
            pad = pad_frac * max(abs(ymin), 1.0)
        else:
            pad = pad_frac * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    def _collect_region_values(
        series_map,
        xlim,
        ignore_structural_zero_segments=False,
        include_segments=None,
    ):
        structural_zero_segment = {
            "bottom": "bottom",
            "top": "top",
            "end": "end",
        }
        if include_segments is not None:
            include_segments = set(include_segments)
        vals = []
        for name, wall_series in series_map.items():
            segments = (
                ("bottom", s_bot, np.asarray(wall_series["bottom"], dtype=float)),
                ("end", s_end, np.asarray(wall_series["end"], dtype=float)),
                ("top", s_top, np.asarray(wall_series["top"][::-1], dtype=float)),
            )
            for seg_name, seg_coord, seg_values in segments:
                if include_segments is not None and seg_name not in include_segments:
                    continue
                if ignore_structural_zero_segments and structural_zero_segment.get(name) == seg_name:
                    continue
                mask = (seg_coord >= xlim[0]) & (seg_coord <= xlim[1])
                if np.any(mask):
                    vals.append(seg_values[mask])
        if not vals:
            return np.asarray([], dtype=float)
        return np.concatenate(vals)

    def _draw_series(ax, series_map):
        for name, wall_series in series_map.items():
            line_color = series_colors[name]
            label = centerline_part_label(name)
            ax.plot(
                s_bot_s,
                wall_series["bottom"],
                linewidth=line_width,
                color=line_color,
                label=label,
            )
            ax.plot(
                s_end_s,
                wall_series["end"],
                linewidth=line_width,
                color=line_color,
                label="_nolegend_",
            )
            ax.plot(
                s_top_s,
                wall_series["top"][::-1],
                linewidth=line_width,
                color=line_color,
                label="_nolegend_",
            )

        for sm in wall_marks:
            ax.axvline(sm / Lref, color="k", linestyle="--", linewidth=wall_dashed_width)

        ax.set_xlabel(r"$s/(2h)$ (along wall)")
        ax.set_ylabel(ylabel)
        # y-axis (reaction-rate magnitude) stays scientific; x-axis (s/(2h) length) -> plain decimal.
        # The unwrapped along-wall axis spans [-(AR+0.5), +(AR+0.5)] in nd; ticks include the
        # end-wall vertices (corners) at +/-0.5 and the open wall ends at +/-(AR+0.5).
        apply_sci_offset(ax, axis="y", scilimits=(0, 0), useOffset=False)
        ax.xaxis.set_major_locator(
            mticker.FixedLocator([-s_wall_end_nd, -0.5, 0.0, 0.5, s_wall_end_nd])
        )
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        apply_axes_style(ax, box_linewidth, tick_length, tick_width)

    def _save_or_close(fig, save_path=None):
        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if not show_plots:
            plt.close(fig)

    def _plot_variant(series_map, variant_stem, variant_figsize, variant_zoom_figsize, save_legend=False):
        out_path = os.path.join(plot_dir, f"{prefix}_{variant_stem}.png") if save_plots else None
        end_zoom_path = os.path.join(zoom_dir, f"{prefix}_{variant_stem}_endwall_zoom.png") if save_plots else None
        is_contribution_variant = contribution_series is not None and series_map is contribution_series

        fig, ax = plt.subplots(figsize=variant_figsize)
        _draw_series(ax, series_map)
        if show_inline_legend and len(series_map) > 1:
            ax.legend(ncol=legend_ncol, frameon=False)
        if save_legend:
            handles, labels = ax.get_legend_handles_labels()
            save_line_legend(plot_dir, prefix, variant_stem, handles, labels, dpi, ncol=legend_ncol)
        _save_or_close(fig, out_path)

        if not is_contribution_variant:
            fig_zoom, ax_zoom = plt.subplots(figsize=variant_zoom_figsize)
            _draw_series(ax_zoom, series_map)
            ax_zoom.set_xlim((-h - end_zoom_pad) / Lref, (h + end_zoom_pad) / Lref)
            # End-wall zoom: ticks at the two visible vertices and centre.
            ax_zoom.xaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
            ax_zoom.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            y_values = _collect_region_values(series_map, (-h - end_zoom_pad, h + end_zoom_pad))
            _set_ylim_from_values(ax_zoom, y_values, end_zoom_rate_pad_frac)
            _save_or_close(fig_zoom, end_zoom_path)
            return

        zoom_group_specs = [
            ("total", {"total": series_map["total"]}, False),
            (
                "bot_top",
                {
                    key: series_map[key]
                    for key in ("bottom", "top")
                    if key in series_map
                },
                True,
            ),
            ("in", {"jet": series_map["jet"]} if "jet" in series_map else {}, False),
            ("end", {"end": series_map["end"]} if "end" in series_map else {}, True),
        ]

        end_xlim = (-h - end_zoom_pad, h + end_zoom_pad)

        def _plot_endwall_zoom(
            save_path,
            draw_map,
            ylim_source_map,
            pad_frac,
            ignore_structural_zero_segments,
            include_segments=None,
        ):
            if not draw_map:
                return
            fig_zoom, ax_zoom = plt.subplots(figsize=variant_zoom_figsize)
            _draw_series(ax_zoom, draw_map)
            ax_zoom.set_xlim(end_xlim[0] / Lref, end_xlim[1] / Lref)
            # End-wall zoom: ticks at the two visible vertices and centre.
            ax_zoom.xaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
            ax_zoom.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            y_values = _collect_region_values(
                ylim_source_map,
                end_xlim,
                ignore_structural_zero_segments=ignore_structural_zero_segments,
                include_segments=include_segments,
            )
            _set_ylim_from_values(ax_zoom, y_values, pad_frac)
            _save_or_close(fig_zoom, save_path)

        for group_name, subset_map, ignore_structural_zero_segments in zoom_group_specs:
            _plot_endwall_zoom(
                os.path.join(zoom_dir, f"{prefix}_{variant_stem}_{group_name}_endwall_zoom.png") if save_plots else None,
                series_map,
                subset_map,
                end_zoom_rate_pad_frac_contrib,
                ignore_structural_zero_segments,
            )
            _plot_endwall_zoom(
                os.path.join(zoom_dir, f"{prefix}_{variant_stem}_{group_name}_only_endwall_zoom.png") if save_plots else None,
                subset_map,
                subset_map,
                end_zoom_rate_pad_frac_contrib,
                ignore_structural_zero_segments,
            )
            if group_name == "in":
                _plot_endwall_zoom(
                    os.path.join(
                        zoom_dir,
                        f"{prefix}_{variant_stem}_{group_name}_only_endsegment_endwall_zoom.png",
                    ) if save_plots else None,
                    subset_map,
                    subset_map,
                    14.0 * end_zoom_rate_pad_frac_contrib,
                    ignore_structural_zero_segments,
                    include_segments=("end",),
                )

    ds_ref = max(
        abs(x[1] - x[0]) if x.size > 1 else 0.0,
        abs(y[1] - y[0]) if y.size > 1 else 0.0,
    )
    if end_zoom_pad is None:
        end_zoom_pad = 0.5 * h
    end_zoom_pad = max(float(end_zoom_pad), ds_ref)

    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }):
        _plot_variant(total_series, stem, figsize, zoom_figsize, save_legend=False)
        if contribution_series is not None:
            _plot_variant(
                contribution_series,
                f"{stem}_contributions",
                figsize_contrib,
                zoom_figsize_contrib,
                save_legend=False,
            )
        if show_plots:
            plt.show()


def plot_surface_flux_along_wall(
    sol,
    surface_flux,
    save_dir,
    prefix,
    show_plots,
    save_plots,
    dpi,
    figsize,
    zoom_figsize,
    font_size,
    tick_length,
    tick_width,
    box_linewidth,
    line_width,
    wall_dashed_width,
    end_zoom_pad=None,
    end_zoom_rate_pad_frac=0.08,
    end_zoom_rate_pad_frac_contrib=None,
    figsize_contrib=None,
    zoom_figsize_contrib=None,
    color=None,
):
    for sp in ("A", "B"):
        plot_surface_quantity_along_wall(
            sol=sol,
            total_values=surface_flux["local_flux"][sp],
            component_values=surface_flux["local_flux_components"][sp],
            save_dir=save_dir,
            prefix=prefix,
            plot_dirname="surface_flux_plots",
            stem=f"surface_flux_{sp}",
            ylabel=rf"$\Gamma_{{\mathrm{{{sp}}}}}$",
            show_plots=show_plots,
            save_plots=save_plots,
            dpi=dpi,
            figsize=figsize,
            zoom_figsize=zoom_figsize,
            figsize_contrib=figsize_contrib,
            zoom_figsize_contrib=zoom_figsize_contrib,
            font_size=font_size,
            tick_length=tick_length,
            tick_width=tick_width,
            box_linewidth=box_linewidth,
            line_width=line_width,
            wall_dashed_width=wall_dashed_width,
            end_zoom_pad=end_zoom_pad,
            end_zoom_rate_pad_frac=end_zoom_rate_pad_frac,
            end_zoom_rate_pad_frac_contrib=end_zoom_rate_pad_frac_contrib,
            color=color,
        )


def plot_surface_reaction_rate_along_wall(
    sol, surface_rr,
    save_dir, prefix,
    show_plots, save_plots, dpi,
    figsize,
    zoom_figsize,
    font_size,
    tick_length, tick_width,
    box_linewidth,
    line_width,
    wall_dashed_width,
    end_zoom_pad=None,
    end_zoom_rate_pad_frac=0.08,
    end_zoom_rate_pad_frac_contrib=None,
    figsize_contrib=None,
    zoom_figsize_contrib=None,
    color=None
):
    """
    Plot surface reaction rate along the wall polyline: bottom -> end -> top.

    If surface_rr contains source-decomposed contributions, an additional
    total+contributions figure set is saved alongside the original total-only set.
    """
    plot_surface_quantity_along_wall(
        sol=sol,
        total_values=surface_rr["local_rate"],
        component_values=surface_rr.get("local_rate_components"),
        save_dir=save_dir,
        prefix=prefix,
        plot_dirname="surface_reaction_rate_plots",
        stem="surface_reaction_rate",
        ylabel=r"surface reaction rate",
        show_plots=show_plots,
        save_plots=save_plots,
        dpi=dpi,
        figsize=figsize,
        zoom_figsize=zoom_figsize,
        figsize_contrib=figsize_contrib,
        zoom_figsize_contrib=zoom_figsize_contrib,
        font_size=font_size,
        tick_length=tick_length,
        tick_width=tick_width,
        box_linewidth=box_linewidth,
        line_width=line_width,
        wall_dashed_width=wall_dashed_width,
        end_zoom_pad=end_zoom_pad,
        end_zoom_rate_pad_frac=end_zoom_rate_pad_frac,
        end_zoom_rate_pad_frac_contrib=end_zoom_rate_pad_frac_contrib,
        color=color,
    )

# ============================================================
# run
# ============================================================
if __name__ == "__main__":

    #property
    molecular_weight_A = 28.016 # kg/kmol
    molecular_weight_B = 28.016 # kg/kmol
    mass_A = 4.65E-26 # kg
    mass_B = 4.65E-26 # kg

    R_uni = 8.31446261815324 # J/K mol
    boltzmann_constant = 1.380658E-23 # m^2 kg / s^2 K
    R_A    = R_uni / molecular_weight_A * 1000 # J/(kg K)
    R_B    = R_uni / molecular_weight_B * 1000 # J/(kg K)

    # geometry
    Aspect_ratio = 10
    width = 50.0e-9
    h  = width/2 # width = 2h
    Lm = width * Aspect_ratio # depth

    # grid
    base_grid = 200
    nx = base_grid * Aspect_ratio + 1
    ny = base_grid + 1 # recomand odd number due to finding centerline
    n_quad_jet = 500 # number of node in integration at jet

    # boundary condition
    pressure = 1005
    temperature = 273.15 # assume isothermal condition
    reaction_probability = 0.01
    u_in = 0.0
    inlet_mole_frac_A = 1.0
    inlet_mole_frac_B = 0.0

    T_in = temperature
    T_w = temperature
    number_density = pressure / boltzmann_constant / temperature
    n_in_A = number_density * inlet_mole_frac_A
    n_in_B = number_density * inlet_mole_frac_B

    # convergence
    tol = 1e-15
    max_iter = 300000
    verbose = True

    # parallelization
    max_workers = 6

    # postprocessing
    # plot
    show_field_contour_plots = False      
    save_field_contour_plots = False      

    show_surface_reaction_plots = False      
    save_surface_reaction_plots = False      

    show_centerline_plots = False      
    save_centerline_plots = True     

    show_inversion_region_plots = False
    save_inversion_region_plots = False
    save_inversion_region_txt = False

    save_YA_minimum_plots = False

    save_dir   = "figures"
    prefix     = f"AR{Aspect_ratio}_rp{reaction_probability:.3f}_nx{nx}_ny{ny}"

    # ------------------------------------------------------------
    # Case-specific plotting setup
    # Edit module-level PLOT_CASE_SETTINGS in this file.
    # ------------------------------------------------------------

    if Aspect_ratio not in PLOT_CASE_SETTINGS:
        raise ValueError(
            f"Aspect_ratio={Aspect_ratio} is not configured in PLOT_CASE_SETTINGS. "
            "Add a case block for this AR first."
    )

    case_plot_cfg = get_case_plot_settings(Aspect_ratio)
    contour_plot_cfg = case_plot_cfg["contour"]
    centerline_plot_cfg = case_plot_cfg["centerline"]
    surface_plot_cfg = build_surface_plot_cfg(Aspect_ratio, h)

    sol = solve_reactive_channel(
        nx=nx, ny=ny, h=h, Lm=Lm,
        R_A=R_A, R_B=R_B, T_in=T_in, u_in=u_in,
        n_in_A=n_in_A, n_in_B=n_in_B,
        T_w=T_w, reaction_probability=reaction_probability,
        n_quad_jet=n_quad_jet,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        max_workers=max_workers
    )

    fields_total, contrib = compute_full_fields(
        sol,
        mass_A,
        mass_B,
        max_workers=max_workers,
        verbose=verbose,
    )
    min_diagnostics = emit_YA_nA_min_diagnostics(
        prefix,
        sol,
        fields_total,
        emit_warning=True,
        print_summary=True,
    )

    # total diffusive mass and molar flux
    diff_fields = compute_diffusive_mass_flux_fields(fields_total, mass_A, mass_B)

    surface_flux = compute_surface_flux(sol)
    surface_rr = compute_surface_reaction_rate(sol)
    # contour field data
    if show_field_contour_plots or save_field_contour_plots:
        plot_field_contours(
            sol, fields_total, diff_fields, contrib,
            save_dir=save_dir, prefix=prefix,
            show_field_contour_plots=show_field_contour_plots, save_field_contour_plots=save_field_contour_plots,
            **contour_plot_cfg,
        )

    if diff_fields is not None and (
        show_inversion_region_plots
        or save_inversion_region_plots
        or save_inversion_region_txt
    ):
        plot_inversion_region_contours(
            sol, fields_total, diff_fields,
            save_dir=save_dir, prefix=prefix,
            show_inversion_region_plots=show_inversion_region_plots,
            save_inversion_region_plots=save_inversion_region_plots,
            save_inversion_region_txt=save_inversion_region_txt,
            levels=contour_plot_cfg["levels"],
            dpi=contour_plot_cfg["dpi"],
            figsize_contour=contour_plot_cfg["figsize_contour"],
            font_size=contour_plot_cfg["font_size"],
            tick_length=contour_plot_cfg["tick_length"],
            tick_width=contour_plot_cfg["tick_width"],
            box_linewidth=contour_plot_cfg["box_linewidth"],
            cbar_shrink=contour_plot_cfg["cbar_shrink"],
            cbar_fraction=contour_plot_cfg["cbar_fraction"],
            cbar_pad=contour_plot_cfg["cbar_pad"],
            cbar_aspect=contour_plot_cfg["cbar_aspect"],
            cbar_nticks=contour_plot_cfg["cbar_nticks"],
            cbar_decimals=contour_plot_cfg["cbar_decimals"],
            ax_aspect=contour_plot_cfg.get("ax_aspect", 1),
            x_nticks=contour_plot_cfg.get("x_nticks", 6),
            minline_linewidth=contour_plot_cfg["minline_linewidth"],
            inversion_product_range_source=contour_plot_cfg["inversion_product_range_source"],
            inversion_product_negative_range_scale=contour_plot_cfg["inversion_product_negative_range_scale"],
            inversion_product_positive_range_scale=contour_plot_cfg["inversion_product_positive_range_scale"],
        )

    # centerline data
    if show_centerline_plots or save_centerline_plots:
        plot_centerline_results(
            sol, fields_total, diff_fields, contrib,
            save_dir=save_dir, prefix=prefix,
            show_centerline_plots=show_centerline_plots,
            save_centerline_plots=save_centerline_plots,
            **centerline_plot_cfg,
        )

    if show_field_contour_plots or show_centerline_plots:
        plt.show()

    # surface data
    if show_surface_reaction_plots or save_surface_reaction_plots:
        plot_surface_flux_along_wall(
            sol, surface_flux,
            save_dir=save_dir, prefix=prefix,
            show_plots=show_surface_reaction_plots, save_plots=save_surface_reaction_plots,
            **surface_plot_cfg,
        )
        plot_surface_reaction_rate_along_wall(
            sol, surface_rr,
            save_dir=save_dir, prefix=prefix,
            show_plots=show_surface_reaction_plots, save_plots=save_surface_reaction_plots,
            **surface_plot_cfg,
        )

    # mininum point
    if save_YA_minimum_plots:
        save_min_diagnostics(
            save_dir,
            prefix,
            sol,
            fields_total,
            diagnostics=min_diagnostics,
            emit_warning=False,
            print_summary=False,
        )


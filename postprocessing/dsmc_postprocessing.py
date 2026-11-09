#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DSMC + analytic postprocessing for inversion-in-trench validation cases.

What this script does
---------------------
- converts the DSMC coordinates to the same analytical-model coordinates
  used in ``inversion_analytic.py``
- crops the field to the trench interior only
- reconstructs the full channel by symmetry when the DSMC case is half-channel
- computes total DSMC fields for:
    n, c, nu, nv, Y_A, Y_B, X_A, X_B, j_x, j_y, J_x, J_y
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: save figures without a GUI/X display (avoids Qt/xcb crash)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402


ROOT = Path(__file__).resolve().parent
# Data folder (the published dataset). By default it sits next to the code/ folder as data/;
# set the environment variable COUNTER_GRADIENT_DATA to use another location.
DATA_ROOT = Path(os.environ.get("COUNTER_GRADIENT_DATA", ROOT.parents[1] / "data")).resolve()
CASE_STUDY_ROOT = DATA_ROOT / "case_studies"
NO_SOLID_CASE_STUDY_ROOT = DATA_ROOT / "dsmc"
R_UNIVERSAL = 8.31446261815324
BOLTZMANN_CONSTANT = 1.380658e-23
AVOGADRO_CONSTANT = 6.02214076e23
DSMC_COMPARE_COLOR = np.array([0.6350, 0.0780, 0.1840], dtype=float)
ANALYTIC_COMPARE_COLOR = np.array([0.0000, 0.4470, 0.7410], dtype=float)
KN_REF = 0.05
PRESSURE_REF_AT_KN_REF = 2_010_000.0


PLOT_CASE_SETTINGS = analytic.PLOT_CASE_SETTINGS


# ============================================================
# User configuration
# ============================================================

# Analytic comparison solver used inside DSMC postprocessing
ANALYTIC_BASE_GRID = 200
ANALYTIC_N_QUAD_JET = 500
ANALYTIC_TOL = 1.0e-15
ANALYTIC_MAX_ITER = 30000
DEFAULT_CASE_MAX_WORKERS = 6
VERBOSE_ANALYTIC_SOLVER = True


def figure_output_root(root_suffix: str = "") -> Path:
    return ROOT / f"figure_analytic_grid{int(ANALYTIC_BASE_GRID)}{root_suffix}"

# Plot visibility
SHOW_CONTOURS = False
SHOW_CENTERLINES = False
SHOW_SURFACE = False
SAVE_INVERSION_REGION_PLOTS = True
SAVE_INVERSION_REGION_TXT = True
PLOT_SAVE_WORKERS = 24

# Study switches
RUN_AR_STUDY = True
RUN_KN_STUDY = True
RUN_PROBABILITY_STUDY = False

POSTPROCESS_DATASET = "no_solid"
NO_SOLID_FIGURE_ROOT_SUFFIX = "_no_solid"

# Case selection:
# - Use "all" to process every completed case in that study.
# - Or provide an explicit tuple/list, e.g. (1, 3, 5) or (1, 10, 100).
AR_CASE_SELECTION = "all"
KN_CASE_SELECTION = "all"
PROBABILITY_CASE_SELECTION = ()  # "all"

# Parallel case execution:
# - Each batch runs sequentially.
# - Cases inside one batch run in parallel processes.
# - case_max_workers controls the internal analytic solver thread count per case.
# - Batches can include cases that are not present yet; they are ignored unless requested.
AR_STUDY_BATCHES = (
    {"cases": (1, 3, 5), "parallel_cases": 96, "case_max_workers": 24},
    {"cases": (10, 20), "parallel_cases": 1, "case_max_workers": 24},
    {"cases": (50,), "parallel_cases": 1, "case_max_workers": 24},
    {"cases": (100,), "parallel_cases": 1, "case_max_workers": 24},
)

KN_STUDY_BATCHES = (
    {"cases": (0.05, 0.1, 1), "parallel_cases": 1, "case_max_workers": 24},
    {"cases": (10,), "parallel_cases": 1, "case_max_workers": 24},
    {"cases": (100,), "parallel_cases": 1, "case_max_workers": 24},
)

PROBABILITY_STUDY_BATCHES = ()


# Ensemble averaging:
# - DSMC fields are averaged over the statistically independent runs of each case,
#   stored in the dataset as dsmc/<case>/run_<n>.grid (SPARTA grid dumps).
# - The plot layout remains the same; the case output folder is suffixed with
#   ENSEMBLE_AVERAGING_OUTPUT_SUFFIX_TEMPLATE.
USE_ENSEMBLE_AVERAGING = True
ENSEMBLE_AVERAGING_ROOT = DATA_ROOT / "dsmc"
ENSEMBLE_AVERAGING_OUTPUT_SUFFIX_TEMPLATE = "_ensemble_average_n{count}"
SINGLE_DSMC_OUTPUT_SUFFIX_TEMPLATE = "_n{count}"
ENSEMBLE_AVERAGING_REQUIRE_COMPLETION = True
ENSEMBLE_AVERAGING_MAX_CASES: int | None = None
ENSEMBLE_AVERAGING_MIN_CASES = 1
ENSEMBLE_AVERAGING_EXCLUDE_SEEDS: dict[str, tuple[int, ...]] = {}
# Dataset folder of each case. Keys are "Study/case".
ENSEMBLE_AVERAGING_CASE_ROOTS: dict[str, Path | str] = {
    "AR/1": ENSEMBLE_AVERAGING_ROOT / "Kn100_AR1",
    "AR/3": ENSEMBLE_AVERAGING_ROOT / "Kn100_AR3",
    "AR/5": ENSEMBLE_AVERAGING_ROOT / "Kn100_AR5",
    "Kn/10": ENSEMBLE_AVERAGING_ROOT / "Kn10_AR5",
    "Kn/1": ENSEMBLE_AVERAGING_ROOT / "Kn1_AR5",
    "Kn/0.05": ENSEMBLE_AVERAGING_ROOT / "Kn0.05_AR5",
}


ENSEMBLE_AVERAGING_METHOD_TEXT = (
    "Ensemble averaging is performed on conservative DSMC moments first: "
    "n_A, n_B, n_A u_A, n_A v_A, n_B u_B, and n_B v_B are averaged over seed cases. "
    "Then c, rho, Y_A/Y_B, X_A/X_B, and diffusive mass/molar fluxes are recomputed "
    "from those averaged moments. The code does not use mean(Y_A_i) or mean(X_A_i) "
    "as the final ensemble composition field."
)


@dataclass(frozen=True)
class CaseConfig:
    case_dir: Path
    study: str
    case_name: str
    aspect_ratio: int
    width: float
    depth: float
    half_width: float
    reaction_probability: float
    trench_direction: str
    trench_mode: str
    pressure: float
    temperature: float
    mass_A: float
    mass_B: float
    molecular_weight_A: float
    molecular_weight_B: float
    timestep: float
    fnum: float
    averaging: int
    knudsen_number: float
    output_suffix: str = ""
    has_surface_dump: bool = True


@dataclass(frozen=True)
class EnsembleCaseInfo:
    seed: int
    case_dir: Path
    grid_path: Path
    timestep: int


@dataclass(frozen=True)
class EnsembleAverageInfo:
    enabled: bool
    source_root: Path
    case_count: int
    cases: tuple[EnsembleCaseInfo, ...]
    method: str


@dataclass(frozen=True)
class EnsembleAverageResult:
    sol: dict
    fields: dict
    diff: dict
    info: EnsembleAverageInfo


class NoEnsembleCasesError(RuntimeError):
    """Raised only when ensemble averaging was requested but no usable runs exist."""


def normalize_case_name(study: str, case_name) -> str:
    if isinstance(case_name, str):
        return case_name
    if study == "AR":
        return str(int(case_name))
    return f"{float(case_name):g}"


def load_case_module(main_py: Path):
    module_name = f"dsmc_case_{main_py.parent.parent.name}_{main_py.parent.name}_{main_py.stem}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_species_properties(species_file: Path) -> Dict[str, Dict[str, float]]:
    props: Dict[str, Dict[str, float]] = {}
    for line in species_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        species = parts[0]
        props[species] = {
            "molecular_weight": float(parts[1]),
            "mass": float(parts[2]),
        }
    return props


def read_species_masses(species_file: Path) -> Tuple[float, float]:
    props = read_species_properties(species_file)
    try:
        return props["A"]["mass"], props["B"]["mass"]
    except KeyError as exc:
        raise RuntimeError(f"Could not read species masses A/B from {species_file}") from exc


def parse_dsmc_input(dsmc_in: Path) -> Dict[str, float]:
    text = dsmc_in.read_text(encoding="utf-8")

    patterns = {
        "timestep": r"^\s*timestep\s+([0-9eE+\-.]+)\s*$",
        "fnum": r"^\s*global\s+nrho\s+\$\{number_density\}\s+fnum\s+([0-9eE+\-.]+)\s+temp\s+\$\{temperature\}\s*$",
        "averaging": r"^\s*variable\s+averaging\s+equal\s+([0-9eE+\-.]+)\s*$",
    }

    values: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Could not parse '{key}' from {dsmc_in}")
        values[key] = float(match.group(1))

    return values


def load_case_meta(case_dir: Path) -> dict | None:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def infer_knudsen_number(case_dir: Path, pressure: float) -> float:
    case_meta = load_case_meta(case_dir)
    if case_meta is not None:
        meta = case_meta.get("meta", {})
        if "Kn" in meta:
            return float(meta["Kn"])
        if "Kn_fixed" in meta:
            return float(meta["Kn_fixed"])
    return float(KN_REF * PRESSURE_REF_AT_KN_REF / pressure)


def load_case_config(
    study: str,
    case_name: str,
    case_root: Path = CASE_STUDY_ROOT,
    output_suffix: str = "",
    has_surface_dump: bool = True,
) -> CaseConfig:
    case_dir = Path(case_root) / study / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    case_main = load_case_module(case_dir / "main.py")
    species_props = read_species_properties(case_dir / "air.species")
    mass_A = species_props["A"]["mass"]
    mass_B = species_props["B"]["mass"]
    dsmc_values = parse_dsmc_input(case_dir / "dsmc.in")

    width = float(case_main.W)
    depth = float(case_main.L)

    return CaseConfig(
        case_dir=case_dir,
        study=study,
        case_name=case_name,
        aspect_ratio=int(case_main.Aspect_ratio),
        width=width,
        depth=depth,
        half_width=0.5 * width,
        reaction_probability=float(case_main.REACTION_STICKING_COEF),
        trench_direction=str(getattr(case_main, "TRENCH_DIRECTION", "x")).lower(),
        trench_mode=str(getattr(case_main, "TRENCH_MODE", "half")).lower(),
        pressure=float(case_main.PRESSURE),
        temperature=float(case_main.TEMPERATURE),
        mass_A=mass_A,
        mass_B=mass_B,
        molecular_weight_A=float(species_props["A"]["molecular_weight"]),
        molecular_weight_B=float(species_props["B"]["molecular_weight"]),
        timestep=float(dsmc_values["timestep"]),
        fnum=float(dsmc_values["fnum"]),
        averaging=int(round(dsmc_values["averaging"])),
        knudsen_number=infer_knudsen_number(case_dir, float(case_main.PRESSURE)),
        output_suffix=str(output_suffix),
        has_surface_dump=bool(has_surface_dump),
    )


def latest_grid_file(case_dir: Path) -> Path:
    grid_dir = case_dir / "output_grid"
    candidates = []
    for path in grid_dir.glob("tmp.grid.*"):
        match = re.fullmatch(r"tmp\.grid\.(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No tmp.grid.* files found in {grid_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def latest_surface_reaction_file(case_dir: Path) -> Path:
    surf_dir = case_dir / "output_surf"
    candidates = []
    for path in surf_dir.glob("tmp.surf_react.*"):
        match = re.fullmatch(r"tmp\.surf_react\.(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No tmp.surf_react.* files found in {surf_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def latest_positive_snapshot(case_dir: Path, kind: str) -> Path | None:
    if kind == "grid":
        pattern = "tmp.grid.*"
        regex = r"tmp\.grid\.(\d+)"
        subdir = "output_grid"
    elif kind == "surf":
        pattern = "tmp.surf_react.*"
        regex = r"tmp\.surf_react\.(\d+)"
        subdir = "output_surf"
    else:
        raise ValueError(f"Unsupported snapshot kind: {kind}")

    candidates = []
    for path in (case_dir / subdir).glob(pattern):
        match = re.fullmatch(regex, path.name)
        if match and int(match.group(1)) > 0:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def read_grid_dump(path: Path) -> Tuple[int, Dict[str, np.ndarray]]:
    timestep = None
    columns = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(next(handle).strip())
            elif line.startswith("ITEM: CELLS"):
                columns = line.split()[2:]
                data = np.loadtxt(handle)
                break
    if timestep is None or columns is None:
        raise RuntimeError(f"Failed to parse SPARTA grid dump: {path}")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != len(columns):
        raise RuntimeError(
            f"Column mismatch while reading {path}: data has {data.shape[1]} columns, "
            f"header declares {len(columns)}"
        )
    arrays = {name: data[:, i] for i, name in enumerate(columns)}
    return timestep, arrays


def read_surf_dump(path: Path) -> Tuple[int, Dict[str, np.ndarray]]:
    timestep = None
    columns = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(next(handle).strip())
            elif line.startswith("ITEM: SURFS"):
                columns = line.split()[2:]
                data = np.loadtxt(handle)
                break
    if timestep is None or columns is None:
        raise RuntimeError(f"Failed to parse SPARTA surface dump: {path}")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != len(columns):
        raise RuntimeError(
            f"Column mismatch while reading {path}: data has {data.shape[1]} columns, "
            f"header declares {len(columns)}"
        )
    arrays = {name: data[:, i] for i, name in enumerate(columns)}
    return timestep, arrays


def require_columns(columns: Iterable[str], arrays: Dict[str, np.ndarray]) -> None:
    missing = [name for name in columns if name not in arrays]
    if missing:
        raise RuntimeError(f"Missing required columns in grid dump: {missing}")


def map_to_analytic_axes(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    depth: float,
    trench_direction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if trench_direction == "x":
        return x_raw.copy(), y_raw.copy()
    if trench_direction == "y":
        return depth - y_raw, x_raw.copy()
    raise ValueError(f"Unsupported trench direction: {trench_direction}")


def crop_channel_mask(
    x_model: np.ndarray,
    y_model: np.ndarray,
    depth: float,
    half_width: float,
    trench_mode: str,
) -> np.ndarray:
    tol = 1.0e-15
    mask_x = (x_model >= -tol) & (x_model <= depth + tol)
    if trench_mode.startswith("half"):
        mask_y = (y_model >= -tol) & (y_model <= half_width + tol)
    else:
        mask_y = np.abs(y_model) <= half_width + tol
    return mask_x & mask_y


def gridify(
    x_points: np.ndarray,
    y_points: np.ndarray,
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    field = np.full((y_coords.size, x_coords.size), np.nan, dtype=float)
    ix = np.searchsorted(x_coords, x_points)
    iy = np.searchsorted(y_coords, y_points)
    field[iy, ix] = values
    if np.isnan(field).any():
        raise RuntimeError("Grid reconstruction produced missing cells.")
    return field


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def compute_half_fields(arrays: Dict[str, np.ndarray], case_cfg: CaseConfig):
    needed = (
        "xc",
        "yc",
        "f_ftot[5]",
        "f_ftot[6]",
        "f_fA[2]",
        "f_fA[3]",
        "f_fA[4]",
        "f_fA[5]",
        "f_fB[2]",
        "f_fB[3]",
        "f_fB[4]",
        "f_fB[5]",
    )
    require_columns(needed, arrays)

    x_model, y_model = map_to_analytic_axes(
        arrays["xc"], arrays["yc"], case_cfg.depth, case_cfg.trench_direction
    )
    mask = crop_channel_mask(
        x_model,
        y_model,
        case_cfg.depth,
        case_cfg.half_width,
        case_cfg.trench_mode,
    )

    x_pts = x_model[mask]
    y_pts = y_model[mask]
    x_coords = np.unique(x_pts)
    y_coords = np.unique(y_pts)

    nA = gridify(x_pts, y_pts, arrays["f_fA[2]"][mask], x_coords, y_coords)
    nB = gridify(x_pts, y_pts, arrays["f_fB[2]"][mask], x_coords, y_coords)

    rhoA = gridify(x_pts, y_pts, arrays["f_fA[3]"][mask], x_coords, y_coords)
    rhoB = gridify(x_pts, y_pts, arrays["f_fB[3]"][mask], x_coords, y_coords)

    uA = gridify(x_pts, y_pts, arrays["f_fA[4]"][mask], x_coords, y_coords)
    vA = gridify(x_pts, y_pts, arrays["f_fA[5]"][mask], x_coords, y_coords)
    uB = gridify(x_pts, y_pts, arrays["f_fB[4]"][mask], x_coords, y_coords)
    vB = gridify(x_pts, y_pts, arrays["f_fB[5]"][mask], x_coords, y_coords)

    u_mix_dump = gridify(x_pts, y_pts, arrays["f_ftot[5]"][mask], x_coords, y_coords)
    v_mix_dump = gridify(x_pts, y_pts, arrays["f_ftot[6]"][mask], x_coords, y_coords)

    nuA = nA * uA
    nvA = nA * vA
    nuB = nB * uB
    nvB = nB * vB

    rho_total = rhoA + rhoB
    YA = safe_divide(rhoA, rho_total)
    YB = safe_divide(rhoB, rho_total)

    cA = nA / AVOGADRO_CONSTANT
    cB = nB / AVOGADRO_CONSTANT
    c_total = cA + cB
    XA = safe_divide(cA, c_total)
    XB = safe_divide(cB, c_total)

    mass_flux_mix_u = case_cfg.mass_A * nuA + case_cfg.mass_B * nuB
    mass_flux_mix_v = case_cfg.mass_A * nvA + case_cfg.mass_B * nvB

    jxA = case_cfg.mass_A * nuA - YA * mass_flux_mix_u
    jyA = case_cfg.mass_A * nvA - YA * mass_flux_mix_v
    jxB = case_cfg.mass_B * nuB - YB * mass_flux_mix_u
    jyB = case_cfg.mass_B * nvB - YB * mass_flux_mix_v

    molar_flux_A_u = nuA / AVOGADRO_CONSTANT
    molar_flux_A_v = nvA / AVOGADRO_CONSTANT
    molar_flux_B_u = nuB / AVOGADRO_CONSTANT
    molar_flux_B_v = nvB / AVOGADRO_CONSTANT
    molar_flux_mix_u = molar_flux_A_u + molar_flux_B_u
    molar_flux_mix_v = molar_flux_A_v + molar_flux_B_v

    JxA = molar_flux_A_u - XA * molar_flux_mix_u
    JyA = molar_flux_A_v - XA * molar_flux_mix_v
    JxB = molar_flux_B_u - XB * molar_flux_mix_u
    JyB = molar_flux_B_v - XB * molar_flux_mix_v

    user_jxA = rhoA * (uA - u_mix_dump)
    user_jyA = rhoA * (vA - v_mix_dump)
    user_jxB = rhoB * (uB - u_mix_dump)
    user_jyB = rhoB * (vB - v_mix_dump)

    formula_check = {
        "A_jx_max_abs_diff": float(np.nanmax(np.abs(jxA - user_jxA))),
        "A_jy_max_abs_diff": float(np.nanmax(np.abs(jyA - user_jyA))),
        "B_jx_max_abs_diff": float(np.nanmax(np.abs(jxB - user_jxB))),
        "B_jy_max_abs_diff": float(np.nanmax(np.abs(jyB - user_jyB))),
    }

    fields_total_half = {
        "A": {"n": nA, "c": cA, "nu": nuA, "nv": nvA},
        "B": {"n": nB, "c": cB, "nu": nuB, "nv": nvB},
        "YA": YA,
        "YB": YB,
        "XA": XA,
        "XB": XB,
    }
    diff_fields_half = {
        "A": {"ju_diff": jxA, "jv_diff": jyA, "Ju_diff": JxA, "Jv_diff": JyA},
        "B": {"ju_diff": jxB, "jv_diff": jyB, "Ju_diff": JxB, "Jv_diff": JyB},
    }
    return x_coords, y_coords, fields_total_half, diff_fields_half, formula_check


def mirror_field_half_to_full(field_half: np.ndarray, parity: str) -> np.ndarray:
    if parity not in {"even", "odd"}:
        raise ValueError(f"Unsupported parity: {parity}")
    upper = field_half
    lower = field_half[::-1, :]
    if parity == "even":
        center = field_half[0:1, :]
    else:
        center = np.zeros((1, field_half.shape[1]), dtype=field_half.dtype)
        lower = -lower
    return np.vstack((lower, center, upper))


def mirror_half_channel(
    y_half: np.ndarray,
    fields_total_half,
    diff_fields_half,
):
    y_full = np.concatenate((-y_half[::-1], np.array([0.0]), y_half))
    fields_total = {
        "A": {
            "n": mirror_field_half_to_full(fields_total_half["A"]["n"], "even"),
            "c": mirror_field_half_to_full(fields_total_half["A"]["c"], "even"),
            "nu": mirror_field_half_to_full(fields_total_half["A"]["nu"], "even"),
            "nv": mirror_field_half_to_full(fields_total_half["A"]["nv"], "odd"),
        },
        "B": {
            "n": mirror_field_half_to_full(fields_total_half["B"]["n"], "even"),
            "c": mirror_field_half_to_full(fields_total_half["B"]["c"], "even"),
            "nu": mirror_field_half_to_full(fields_total_half["B"]["nu"], "even"),
            "nv": mirror_field_half_to_full(fields_total_half["B"]["nv"], "odd"),
        },
        "YA": mirror_field_half_to_full(fields_total_half["YA"], "even"),
        "YB": mirror_field_half_to_full(fields_total_half["YB"], "even"),
        "XA": mirror_field_half_to_full(fields_total_half["XA"], "even"),
        "XB": mirror_field_half_to_full(fields_total_half["XB"], "even"),
    }
    diff_fields = {
        "A": {
            "ju_diff": mirror_field_half_to_full(diff_fields_half["A"]["ju_diff"], "even"),
            "jv_diff": mirror_field_half_to_full(diff_fields_half["A"]["jv_diff"], "odd"),
            "Ju_diff": mirror_field_half_to_full(diff_fields_half["A"]["Ju_diff"], "even"),
            "Jv_diff": mirror_field_half_to_full(diff_fields_half["A"]["Jv_diff"], "odd"),
        },
        "B": {
            "ju_diff": mirror_field_half_to_full(diff_fields_half["B"]["ju_diff"], "even"),
            "jv_diff": mirror_field_half_to_full(diff_fields_half["B"]["jv_diff"], "odd"),
            "Ju_diff": mirror_field_half_to_full(diff_fields_half["B"]["Ju_diff"], "even"),
            "Jv_diff": mirror_field_half_to_full(diff_fields_half["B"]["Jv_diff"], "odd"),
        },
    }
    return y_full, fields_total, diff_fields


def build_full_fields(case_cfg: CaseConfig, arrays: Dict[str, np.ndarray]):
    x_coords, y_coords_half, fields_total_half, diff_fields_half, formula_check = compute_half_fields(
        arrays, case_cfg
    )

    if case_cfg.trench_mode.startswith("half"):
        y_coords, fields_total, diff_fields = mirror_half_channel(
            y_coords_half,
            fields_total_half,
            diff_fields_half,
        )
    else:
        y_coords = y_coords_half
        fields_total = fields_total_half
        diff_fields = diff_fields_half

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")
    n_in_A = case_cfg.pressure / BOLTZMANN_CONSTANT / case_cfg.temperature
    n_in_B = 0.0
    sol = {
        "x": x_coords,
        "y": y_coords,
        "X": X,
        "Y": Y,
        "h": case_cfg.half_width,
        "Lm": case_cfg.depth,
        "n_in_A": float(n_in_A),
        "n_in_B": float(n_in_B),
        "n_in": {"A": float(n_in_A), "B": float(n_in_B)},
    }
    return sol, fields_total, diff_fields, formula_check


def format_ensemble_kn_value(value: float) -> str:
    rounded = round(float(value))
    if math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=1.0e-9):
        return str(int(rounded))
    return f"{float(value):g}"


def ensemble_case_key(case_cfg: CaseConfig) -> str:
    return f"{case_cfg.study}/{case_cfg.case_name}"


def resolve_ensemble_cases_root(case_cfg: CaseConfig) -> Path:
    key = ensemble_case_key(case_cfg)
    if key in ENSEMBLE_AVERAGING_CASE_ROOTS:
        return Path(ENSEMBLE_AVERAGING_CASE_ROOTS[key]).resolve()

    kn_name = format_ensemble_kn_value(case_cfg.knudsen_number)
    if case_cfg.study in {"AR", "Kn"}:
        folder = f"Kn{kn_name}_AR{case_cfg.aspect_ratio}_ensemble"
    else:
        raise RuntimeError(
            f"No automatic ensemble folder rule is defined for {key}. "
            "Add ENSEMBLE_AVERAGING_CASE_ROOTS override."
        )
    return (Path(ENSEMBLE_AVERAGING_ROOT) / folder / "cases").resolve()


def load_ensemble_case_meta(case_dir: Path) -> dict:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def seed_from_ensemble_case(case_dir: Path, meta: dict) -> int:
    if "seed" in meta:
        return int(meta["seed"])
    match = re.fullmatch(r"seed_(\d+)", case_dir.name)
    if match:
        return int(match.group(1))
    raise RuntimeError(f"Could not infer seed from {case_dir}")


def ensemble_meta_value(meta: dict, *keys: str):
    sources = (
        meta.get("params", {}),
        meta.get("params_applied_to_main_py", {}),
        meta.get("meta", {}),
    )
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def ensemble_case_matches(case_cfg: CaseConfig, case_dir: Path, meta: dict) -> bool:
    checks: list[tuple[str, float | None, float]] = []
    meta_kn = ensemble_meta_value(meta, "Kn", "Kn_fixed")
    if meta_kn is not None:
        checks.append(("Kn", float(meta_kn), float(case_cfg.knudsen_number)))
    meta_ar = ensemble_meta_value(meta, "Aspect_ratio", "AR")
    if meta_ar is not None:
        checks.append(("Aspect_ratio", float(meta_ar), float(case_cfg.aspect_ratio)))
    meta_prob = ensemble_meta_value(meta, "REACTION_STICKING_COEF", "reaction_probability")
    if meta_prob is not None:
        checks.append(("REACTION_STICKING_COEF", float(meta_prob), float(case_cfg.reaction_probability)))

    main_py = case_dir / "main.py"
    if main_py.exists():
        case_main = load_case_module(main_py)
        checks.extend(
            [
                ("Aspect_ratio", float(case_main.Aspect_ratio), float(case_cfg.aspect_ratio)),
                ("REACTION_STICKING_COEF", float(case_main.REACTION_STICKING_COEF), float(case_cfg.reaction_probability)),
                ("PRESSURE", float(case_main.PRESSURE), float(case_cfg.pressure)),
            ]
        )

    for name, actual, expected in checks:
        if actual is None:
            continue
        tolerance = 1.0e-12 if name != "PRESSURE" else 1.0e-9
        if not math.isclose(float(actual), float(expected), rel_tol=1.0e-9, abs_tol=tolerance):
            print(
                f"Skipping ensemble case with mismatched {name}: "
                f"{case_dir} has {actual}, expected {expected}",
                flush=True,
            )
            return False
    return True


def discover_ensemble_grid_cases(case_cfg: CaseConfig) -> tuple[Path, tuple[EnsembleCaseInfo, ...]]:
    cases_root = resolve_ensemble_cases_root(case_cfg)
    if not cases_root.exists():
        raise NoEnsembleCasesError(
            f"Ensemble cases folder not found: {cases_root}. Place the dataset folder data/ next to "
            "the code/ folder or set COUNTER_GRADIENT_DATA to its location."
        )

    # Each run of a case is one grid dump, dsmc/<case>/run_<n>.grid; the run number <n> takes
    # the place of the seed and sets the averaging order.
    cases: list[EnsembleCaseInfo] = []
    for grid_path in cases_root.glob("run_*.grid"):
        match = re.fullmatch(r"run_(\d+)\.grid", grid_path.name)
        if match is None:
            continue
        timestep, _arrays = read_grid_dump(grid_path)
        cases.append(EnsembleCaseInfo(seed=int(match.group(1)), case_dir=cases_root, grid_path=grid_path, timestep=timestep))

    cases.sort(key=lambda item: item.seed)
    excluded = set(ENSEMBLE_AVERAGING_EXCLUDE_SEEDS.get(cases_root.parent.name, ()))
    if excluded:
        cases = [c for c in cases if c.seed not in excluded]
    if ENSEMBLE_AVERAGING_MAX_CASES is not None:
        cases = cases[: int(ENSEMBLE_AVERAGING_MAX_CASES)]
    if len(cases) < int(ENSEMBLE_AVERAGING_MIN_CASES):
        raise NoEnsembleCasesError(
            f"Only {len(cases)} completed ensemble grid case(s) found in {cases_root}; "
            f"minimum is {ENSEMBLE_AVERAGING_MIN_CASES}"
        )
    return cases_root, tuple(cases)


def init_ensemble_moment_sums(fields: dict) -> dict:
    return {
        "A": {
            "n": np.zeros_like(fields["A"]["n"], dtype=float),
            "nu": np.zeros_like(fields["A"]["nu"], dtype=float),
            "nv": np.zeros_like(fields["A"]["nv"], dtype=float),
        },
        "B": {
            "n": np.zeros_like(fields["B"]["n"], dtype=float),
            "nu": np.zeros_like(fields["B"]["nu"], dtype=float),
            "nv": np.zeros_like(fields["B"]["nv"], dtype=float),
        },
    }


def accumulate_ensemble_moments(sums: dict, fields: dict) -> None:
    for sp in ("A", "B"):
        for comp in ("n", "nu", "nv"):
            sums[sp][comp] += fields[sp][comp]


def compute_fields_from_ensemble_moments(sums: dict, count: int, case_cfg: CaseConfig):
    nA = sums["A"]["n"] / count
    nB = sums["B"]["n"] / count
    nuA = sums["A"]["nu"] / count
    nuB = sums["B"]["nu"] / count
    nvA = sums["A"]["nv"] / count
    nvB = sums["B"]["nv"] / count

    rhoA = case_cfg.mass_A * nA
    rhoB = case_cfg.mass_B * nB
    rho_total = rhoA + rhoB
    YA = safe_divide(rhoA, rho_total)
    YB = safe_divide(rhoB, rho_total)

    cA = nA / AVOGADRO_CONSTANT
    cB = nB / AVOGADRO_CONSTANT
    c_total = cA + cB
    XA = safe_divide(cA, c_total)
    XB = safe_divide(cB, c_total)

    mass_flux_mix_u = case_cfg.mass_A * nuA + case_cfg.mass_B * nuB
    mass_flux_mix_v = case_cfg.mass_A * nvA + case_cfg.mass_B * nvB
    jxA = case_cfg.mass_A * nuA - YA * mass_flux_mix_u
    jyA = case_cfg.mass_A * nvA - YA * mass_flux_mix_v
    jxB = case_cfg.mass_B * nuB - YB * mass_flux_mix_u
    jyB = case_cfg.mass_B * nvB - YB * mass_flux_mix_v

    molar_flux_A_u = nuA / AVOGADRO_CONSTANT
    molar_flux_A_v = nvA / AVOGADRO_CONSTANT
    molar_flux_B_u = nuB / AVOGADRO_CONSTANT
    molar_flux_B_v = nvB / AVOGADRO_CONSTANT
    molar_flux_mix_u = molar_flux_A_u + molar_flux_B_u
    molar_flux_mix_v = molar_flux_A_v + molar_flux_B_v
    JxA = molar_flux_A_u - XA * molar_flux_mix_u
    JyA = molar_flux_A_v - XA * molar_flux_mix_v
    JxB = molar_flux_B_u - XB * molar_flux_mix_u
    JyB = molar_flux_B_v - XB * molar_flux_mix_v

    fields = {
        "A": {"n": nA, "c": cA, "nu": nuA, "nv": nvA},
        "B": {"n": nB, "c": cB, "nu": nuB, "nv": nvB},
        "YA": YA,
        "YB": YB,
        "XA": XA,
        "XB": XB,
    }
    diff = {
        "A": {"ju_diff": jxA, "jv_diff": jyA, "Ju_diff": JxA, "Jv_diff": JyA},
        "B": {"ju_diff": jxB, "jv_diff": jyB, "Ju_diff": JxB, "Jv_diff": JyB},
    }
    return fields, diff


def build_ensemble_average_result(case_cfg: CaseConfig) -> EnsembleAverageResult:
    cases_root, ensemble_cases = discover_ensemble_grid_cases(case_cfg)
    sol_ref = None
    sums = None
    for info in ensemble_cases:
        _timestep, arrays = read_grid_dump(info.grid_path)
        sol, fields, _diff, _formula_check = build_full_fields(case_cfg, arrays)
        if sol_ref is None:
            sol_ref = sol
            sums = init_ensemble_moment_sums(fields)
        else:
            if not np.allclose(sol_ref["x"], sol["x"]) or not np.allclose(sol_ref["y"], sol["y"]):
                raise RuntimeError(f"Grid mismatch while ensemble-averaging {info.case_dir}")
        accumulate_ensemble_moments(sums, fields)

    if sol_ref is None or sums is None:
        raise NoEnsembleCasesError(f"No usable ensemble cases found in {cases_root}")

    fields, diff = compute_fields_from_ensemble_moments(sums, len(ensemble_cases), case_cfg)
    info = EnsembleAverageInfo(
        enabled=True,
        source_root=cases_root,
        case_count=len(ensemble_cases),
        cases=ensemble_cases,
        method=ENSEMBLE_AVERAGING_METHOD_TEXT,
    )
    return EnsembleAverageResult(sol=sol_ref, fields=fields, diff=diff, info=info)


def write_ensemble_average_summary(save_dir: Path, case_cfg: CaseConfig, info: EnsembleAverageInfo) -> None:
    payload = {
        "enabled": info.enabled,
        "study": case_cfg.study,
        "case_name": case_cfg.case_name,
        "case_count": info.case_count,
        "source_root": str(info.source_root),
        "method": info.method,
        "cases": [
            {
                "seed": item.seed,
                "case_dir": str(item.case_dir),
                "grid_path": str(item.grid_path),
                "timestep": item.timestep,
            }
            for item in info.cases
        ],
    }
    (save_dir / "ensemble_average_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "DSMC ensemble averaging summary",
        f"study = {case_cfg.study}",
        f"case = {case_cfg.case_name}",
        f"case_count = {info.case_count}",
        f"source_root = {info.source_root}",
        "",
        "method:",
        info.method,
        "",
        "seed timestep grid_path",
    ]
    for item in info.cases:
        lines.append(f"{item.seed} {item.timestep} {item.grid_path}")
    (save_dir / "ensemble_average_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analytic_grid_shape(case_cfg: CaseConfig) -> Tuple[int, int]:
    ny = ANALYTIC_BASE_GRID + 1
    nx = int(round(ny * case_cfg.aspect_ratio))
    return nx, ny


def solve_analytic_case(case_cfg: CaseConfig, case_max_workers: int):
    R_A = R_UNIVERSAL / case_cfg.molecular_weight_A * 1000.0
    R_B = R_UNIVERSAL / case_cfg.molecular_weight_B * 1000.0
    nx, ny = analytic_grid_shape(case_cfg)
    number_density = case_cfg.pressure / BOLTZMANN_CONSTANT / case_cfg.temperature
    log_prefix = f"[analytic {case_cfg.study}/{case_cfg.case_name}]"

    sol = analytic.solve_reactive_channel(
        nx=nx,
        ny=ny,
        h=case_cfg.half_width,
        Lm=case_cfg.depth,
        R_A=R_A,
        R_B=R_B,
        T_in=case_cfg.temperature,
        u_in=0.0,
        n_in_A=number_density,
        n_in_B=0.0,
        T_w=case_cfg.temperature,
        reaction_probability=case_cfg.reaction_probability,
        n_quad_jet=ANALYTIC_N_QUAD_JET,
        tol=ANALYTIC_TOL,
        max_iter=ANALYTIC_MAX_ITER,
        verbose=VERBOSE_ANALYTIC_SOLVER,
        max_workers=case_max_workers,
        log_prefix=log_prefix,
    )
    print(analytic.format_solver_status(sol, label=f"analytic {case_cfg.study}/{case_cfg.case_name}"), flush=True)
    fields_total, contrib = analytic.compute_full_fields(
        sol,
        case_cfg.mass_A,
        case_cfg.mass_B,
        max_workers=case_max_workers,
        verbose=VERBOSE_ANALYTIC_SOLVER,
        log_prefix=log_prefix,
    )
    diff_fields = analytic.compute_diffusive_mass_flux_fields(
        fields_total, case_cfg.mass_A, case_cfg.mass_B
    )

    surface_rr = analytic.compute_surface_reaction_rate(sol)
    return sol, fields_total, diff_fields, contrib, surface_rr


def build_surface_plot_cfg(case_cfg: CaseConfig) -> Dict[str, float]:
    return analytic.build_surface_plot_cfg(case_cfg.aspect_ratio, case_cfg.half_width)


def field_contour_kwargs(contour_cfg: Dict[str, float]) -> Dict[str, float]:
    allowed_keys = (
        "levels",
        "enable_probe",
        "dpi",
        "cbar_shrink",
        "cbar_fraction",
        "cbar_pad",
        "cbar_aspect",
        "figsize_contour",
        "font_size",
        "tick_length",
        "tick_width",
        "box_linewidth",
        "cbar_nticks",
        "cbar_decimals",
        "minline_linewidth",
        "ax_aspect",
        "x_nticks",
        "x_ticks",
        "mass_fraction_powernorm_gamma",
        "zoom_factor",
        "zoom_xlim",
        "zoom_font_size",
        "zoom_cbar_fraction",
        "zoom_cbar_aspect",
        "zoom_x_nticks",
        "zoom_x_ticks",
    )
    return {key: contour_cfg[key] for key in allowed_keys if key in contour_cfg}


def split_sol_field(sol, Z: np.ndarray, upper: bool):
    y = sol["y"]
    mask = y >= 0.0 if upper else y <= 0.0
    return sol["X"][mask, :], sol["Y"][mask, :], Z[mask, :]


def compare_level_values(z_top: np.ndarray, z_bottom: np.ndarray, levels: int | np.ndarray) -> Tuple[np.ndarray, float, float]:
    vmin = min(float(np.nanmin(z_top)), float(np.nanmin(z_bottom)))
    vmax = max(float(np.nanmax(z_top)), float(np.nanmax(z_bottom)))
    if np.isscalar(levels):
        nlevels = int(levels)
        if np.isclose(vmin, vmax):
            delta = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 1.0e-9
            level_values = np.linspace(vmin - delta, vmax + delta, max(2, nlevels))
        else:
            level_values = np.linspace(vmin, vmax, max(2, nlevels))
    else:
        level_values = np.asarray(levels, dtype=float)
    return level_values, vmin, vmax


def _scale_minline(minline, Lref: float):
    """Return a (x, y) minline with both coordinates divided by Lref for plotting.

    Only the plotted coordinates are rescaled; the original arrays are untouched.
    """
    if minline is None:
        return None
    x_line, y_line = minline
    return (np.asarray(x_line) / Lref, np.asarray(y_line) / Lref)


def write_contour_axes_meta(fig, ax, save_path: Path) -> None:
    """Write a sidecar JSON describing, for a contour saved with bbox_inches='tight',
    the data axis limits and the data-axes rectangle as a fraction of the saved image
    (origin at the lower-left). This lets a downstream tool place a zoom box exactly on
    the rasterized PNG without re-rendering the field."""
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)  # inches
        axin = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())  # inches
        W = float(tight.width) or 1.0
        H = float(tight.height) or 1.0
        meta = {
            "axes_frac": [
                (float(axin.x0) - float(tight.x0)) / W,
                (float(axin.y0) - float(tight.y0)) / H,
                (float(axin.x1) - float(tight.x0)) / W,
                (float(axin.y1) - float(tight.y0)) / H,
            ],
            "data_xlim": [float(v) for v in ax.get_xlim()],
            "data_ylim": [float(v) for v in ax.get_ylim()],
        }
        with open(str(save_path) + ".meta.json", "w") as handle:
            json.dump(meta, handle)
    except Exception as exc:  # metadata is best-effort; never break plotting
        print(f"[contour-meta] skipped {save_path}: {exc}", flush=True)


def plot_half_minline(ax, minline, upper: bool, linewidth: float, color: str = "k") -> None:
    if minline is None:
        return
    x_line, y_line = minline
    y_line = np.asarray(y_line)
    x_line = np.asarray(x_line)
    mask = y_line >= 0.0 if upper else y_line <= 0.0
    ax.plot(
        x_line[mask],
        y_line[mask],
        color=color,
        linestyle="--",
        linewidth=linewidth,
        zorder=20,
    )


def plot_compare_contour_split(
    sol_dsmc,
    Z_dsmc: np.ndarray,
    sol_analytic,
    Z_analytic: np.ndarray,
    title: str,
    levels,
    save_path: Path | None,
    show: bool,
    dpi: int,
    figsize,
    tick_length: float,
    tick_width: float,
    box_linewidth: float,
    cbar_shrink: float,
    cbar_fraction: float,
    cbar_pad: float,
    cbar_aspect: float,
    x_nticks: int,
    cbar_nticks: int,
    cbar_decimals: int,
    cbar_scientific: bool,
    ax_aspect: float,
    y_nticks: int = 3,
    minline_dsmc=None,
    minline_analytic=None,
    minline_linewidth: float = 2.0,
    powernorm_gamma: float | None = None,
    xlim=None,
    x_ticks=None,
):
    X_top, Y_top, Z_top = split_sol_field(sol_dsmc, Z_dsmc, upper=True)
    X_bottom, Y_bottom, Z_bottom = split_sol_field(sol_analytic, Z_analytic, upper=False)

    # Non-dimensionalize spatial length axes by the full trench height 2h.
    Lref = 2.0 * float(sol_dsmc["h"])
    AR = float(sol_dsmc["Lm"]) / Lref  # true aspect ratio from physical domain (walls)

    # Zoom view (xlim passed in SI): round the non-dimensional start to one decimal and
    # span exactly [zoom_left_nd, AR] in x/(2h). The DATA is cropped to that rounded
    # window; the axis limit/ticks use the exact true-domain values.
    is_zoom = xlim is not None
    zoom_left_nd = None
    if is_zoom:
        zoom_left_nd = round(float(xlim[0]) / Lref, 1)
        crop_window = (zoom_left_nd * Lref, float(sol_dsmc["Lm"]))
    else:
        crop_window = None
    X_top, Y_top, Z_top = analytic.crop_contour_to_x_window(X_top, Y_top, Z_top, xlim=crop_window)
    X_bottom, Y_bottom, Z_bottom = analytic.crop_contour_to_x_window(X_bottom, Y_bottom, Z_bottom, xlim=crop_window)

    level_values, vmin, vmax = compare_level_values(Z_top, Z_bottom, levels)
    cmap_name = "jet"
    if powernorm_gamma is None or np.isclose(vmin, vmax):
        norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    else:
        norm = mcolors.PowerNorm(
            gamma=float(powernorm_gamma),
            vmin=float(vmin),
            vmax=float(vmax),
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(X_top / Lref, Y_top / Lref, Z_top, levels=level_values, cmap=cmap_name, norm=norm)
    ax.contourf(X_bottom / Lref, Y_bottom / Lref, Z_bottom, levels=level_values, cmap=cmap_name, norm=norm)
    ax.axhline(0.0, color="k", linewidth=box_linewidth * 0.5, linestyle="-", zorder=10)

    ax.set_xlabel(r"$x/(2h)$")
    ax.set_ylabel(r"$y/(2h)$")
    ax.set_title("")
    ax.set_aspect(ax_aspect, adjustable="box")
    # Limits are the exact physical non-dimensional domain (no padded-edge distortion):
    # main view x in (0, AR); zoom view x in (zoom_left_nd, AR). y in (-0.5, 0.5) since Lref = 2h.
    if is_zoom:
        ax.set_xlim(zoom_left_nd, AR)
    else:
        ax.set_xlim(0.0, AR)
    ax.set_ylim(-sol_dsmc["h"] / Lref, sol_dsmc["h"] / Lref)

    plot_half_minline(ax, _scale_minline(minline_dsmc, Lref), upper=True, linewidth=minline_linewidth)
    plot_half_minline(ax, _scale_minline(minline_analytic, Lref), upper=False, linewidth=minline_linewidth)

    # y ticks: walls + centre, both for main and zoom contour views.
    ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    if is_zoom:
        # Zoom x ticks span [zoom_left_nd, AR]; count from a true-AR lookup with a fallback rule.
        _ZOOM_NTICKS_BY_AR = {3: 3, 5: 4, 10: 5}
        ar_key = int(round(AR))
        if ar_key in _ZOOM_NTICKS_BY_AR:
            nz = _ZOOM_NTICKS_BY_AR[ar_key]
        else:
            nz = max(3, min(6, int(round(AR - zoom_left_nd)) + 2))
        zoom_ticks = np.round(np.linspace(zoom_left_nd, AR, nz), 1)
        ax.xaxis.set_major_locator(mticker.FixedLocator(zoom_ticks))
    else:
        # Main view x ticks: endpoints 0 and AR exact, count by AR.
        nx = 3 if AR <= 1.5 else (4 if AR <= 3.5 else 6)
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0.0, AR, nx)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    if cbar_scientific:
        cfmt = analytic.ScalarFormatterFixed(decimals=cbar_decimals, useMathText=True)
        cfmt.set_scientific(True)
        cfmt.set_powerlimits((0, 0))
        cfmt.set_useOffset(False)
    else:
        cfmt = mticker.FormatStrFormatter(f"%.{cbar_decimals}f")

    sm = ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        pad=cbar_pad,
        fraction=cbar_fraction,
        shrink=cbar_shrink,
        format=cfmt,
        aspect=cbar_aspect,
    )
    cbar.set_label(title, rotation=90, labelpad=12)
    ticks = [vmin] if np.isclose(vmin, vmax) else np.linspace(vmin, vmax, cbar_nticks)
    cbar.set_ticks(ticks)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_visible(True)

    analytic.apply_axes_style(ax, box_linewidth, tick_length, tick_width, tick_pad=8)
    analytic.apply_axes_style(cbar.ax, box_linewidth, tick_length, tick_width, tick_pad=8)
    if cbar_scientific:
        analytic.place_colorbar_offset_text_outside(cbar)

    fig.tight_layout()
    if cbar_scientific:
        analytic.place_colorbar_offset_text_outside(cbar)

    if save_path is not None:
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        write_contour_axes_meta(fig, ax, save_path)
    if not show:
        plt.close(fig)
    return fig, ax


def compare_contour_paths(save_dir: Path, with_line: bool, group: str, filename: str) -> Path:
    base = "contours_with_YA_min_line" if with_line else "contours_plain"
    return save_dir / base / group / f"{filename}.png"


def compare_zoom_contour_paths(save_dir: Path, with_line: bool, group: str, filename: str) -> Path:
    base = "contours_with_YA_min_line" if with_line else "contours_plain"
    return save_dir / base / "zoom" / group / f"{filename}.png"


def _plot_compare_field_contour_group_worker(task: dict):
    saved = []
    for variant in task["variants"]:
        kwargs = dict(task["common_kwargs"])
        kwargs.update(variant["kwargs"])
        with plt.rc_context(analytic.contour_rc_params(variant["font_size"])):
            plot_compare_contour_split(**kwargs)
        save_path = kwargs.get("save_path")
        if save_path is not None:
            saved.append(str(save_path))
    return saved


def _run_compare_plot_worker_tasks(tasks: list[dict], max_workers: int):
    if not tasks:
        return
    workers = max(1, min(int(max_workers), len(tasks)))
    if workers == 1:
        for task in tasks:
            _plot_compare_field_contour_group_worker(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_plot_compare_field_contour_group_worker, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


def plot_compare_field_contours(
    sol_dsmc,
    fields_dsmc,
    diff_dsmc,
    sol_analytic,
    fields_analytic,
    diff_analytic,
    save_dir: Path,
    prefix: str,
    contour_cfg: Dict[str, float],
    parallel_save_workers: int = 1,
):
    info_dsmc = analytic.compute_min_diagnostics(sol_dsmc, fields_dsmc)
    info_analytic = analytic.compute_min_diagnostics(sol_analytic, fields_analytic)
    minline_dsmc = info_dsmc[2]
    minline_analytic = info_analytic[2]
    zoom_factor = float(contour_cfg.get("zoom_factor", 3.0))
    zoom_font_size = float(contour_cfg.get("zoom_font_size", contour_cfg["font_size"]))
    zoom_cbar_fraction = float(contour_cfg.get("zoom_cbar_fraction", contour_cfg["cbar_fraction"]))
    zoom_cbar_aspect = float(contour_cfg.get("zoom_cbar_aspect", contour_cfg["cbar_aspect"]))
    zoom_x_nticks = int(contour_cfg.get("zoom_x_nticks", max(3, min(int(contour_cfg["x_nticks"]), 5))))
    x_ticks = contour_cfg.get("x_ticks")
    if x_ticks is not None:
        x_ticks = tuple(float(value) for value in x_ticks)
    zoom_x_ticks = contour_cfg.get("zoom_x_ticks")
    if zoom_x_ticks is not None:
        zoom_x_ticks = tuple(float(value) for value in zoom_x_ticks)
    if contour_cfg.get("zoom_xlim") is not None:
        zoom_xlim = tuple(float(value) for value in contour_cfg["zoom_xlim"])
    else:
        zoom_xlim = analytic.compute_inversion_zoom_xlim(
            sol_dsmc["x"],
            minline_dsmc[0],
            minline_analytic[0],
            zoom_factor=zoom_factor,
        )

    comp_tag_mol = {"nu": "nu", "nv": "nv", "n": "n", "c": "c"}
    comp_tag_diff = {"ju_diff": "j_x", "jv_diff": "j_y"}
    comp_tag_molar_diff = {"Ju_diff": "J_x", "Jv_diff": "J_y"}
    parallel_save_workers = max(1, int(parallel_save_workers))
    use_parallel_save = parallel_save_workers > 1
    contour_tasks: list[dict] = []

    def _plot_both(Z_dsmc, Z_analytic, title, group, stem, scientific):
        powernorm_gamma = contour_cfg.get("mass_fraction_powernorm_gamma")
        if group not in {"YA", "YB", "XA", "XB"}:
            powernorm_gamma = None
        common_kwargs = dict(
            sol_dsmc=sol_dsmc,
            Z_dsmc=Z_dsmc,
            sol_analytic=sol_analytic,
            Z_analytic=Z_analytic,
            title=title,
            levels=contour_cfg["levels"],
            show=False,
            dpi=contour_cfg["dpi"],
            figsize=contour_cfg["figsize_contour"],
            tick_length=contour_cfg["tick_length"],
            tick_width=contour_cfg["tick_width"],
            box_linewidth=contour_cfg["box_linewidth"],
            cbar_shrink=contour_cfg["cbar_shrink"],
            cbar_fraction=contour_cfg["cbar_fraction"],
            cbar_pad=contour_cfg["cbar_pad"],
            cbar_aspect=contour_cfg["cbar_aspect"],
            x_nticks=contour_cfg["x_nticks"],
            x_ticks=x_ticks,
            cbar_nticks=contour_cfg["cbar_nticks"],
            cbar_decimals=contour_cfg["cbar_decimals"],
            cbar_scientific=scientific,
            ax_aspect=contour_cfg["ax_aspect"],
            powernorm_gamma=powernorm_gamma,
        )
        zoom_kwargs = dict(
            xlim=zoom_xlim,
            x_nticks=zoom_x_nticks,
            x_ticks=zoom_x_ticks,
            cbar_fraction=zoom_cbar_fraction,
            cbar_aspect=zoom_cbar_aspect,
        )
        task = {
            "common_kwargs": common_kwargs,
            "variants": [
                {
                    "font_size": contour_cfg["font_size"],
                    "kwargs": {
                        "save_path": compare_contour_paths(save_dir, False, group, f"{prefix}_{stem}"),
                        "minline_dsmc": None,
                        "minline_analytic": None,
                    },
                },
                {
                    "font_size": contour_cfg["font_size"],
                    "kwargs": {
                        "save_path": compare_contour_paths(save_dir, True, group, f"{prefix}_{stem}"),
                        "minline_dsmc": minline_dsmc,
                        "minline_analytic": minline_analytic,
                        "minline_linewidth": contour_cfg["minline_linewidth"],
                    },
                },
                {
                    "font_size": zoom_font_size,
                    "kwargs": {
                        **zoom_kwargs,
                        "save_path": compare_zoom_contour_paths(save_dir, False, group, f"{prefix}_{stem}"),
                        "minline_dsmc": None,
                        "minline_analytic": None,
                    },
                },
                {
                    "font_size": zoom_font_size,
                    "kwargs": {
                        **zoom_kwargs,
                        "save_path": compare_zoom_contour_paths(save_dir, True, group, f"{prefix}_{stem}"),
                        "minline_dsmc": minline_dsmc,
                        "minline_analytic": minline_analytic,
                        "minline_linewidth": contour_cfg["minline_linewidth"],
                    },
                },
            ],
        }
        if use_parallel_save:
            contour_tasks.append(task)
        else:
            _plot_compare_field_contour_group_worker(task)

    _plot_both(fields_dsmc["YA"], fields_analytic["YA"], r"$Y_{\mathrm{A}}$", "YA", "tot", scientific=False)
    _plot_both(fields_dsmc["YB"], fields_analytic["YB"], r"$Y_{\mathrm{B}}$", "YB", "tot", scientific=False)
    if "XA" in fields_dsmc and "XB" in fields_dsmc and "XA" in fields_analytic and "XB" in fields_analytic:
        _plot_both(fields_dsmc["XA"], fields_analytic["XA"], r"$X_{\mathrm{A}}$", "XA", "tot", scientific=False)
        _plot_both(fields_dsmc["XB"], fields_analytic["XB"], r"$X_{\mathrm{B}}$", "XB", "tot", scientific=False)

    for sp in ["A", "B"]:
        for comp in ["n", "c", "nu", "nv"]:
            if comp not in fields_dsmc[sp] or comp not in fields_analytic[sp]:
                continue
            title = analytic.field_title(comp, sp=sp, tag="tot")
            _plot_both(
                fields_dsmc[sp][comp],
                fields_analytic[sp][comp],
                title,
                comp_tag_mol[comp],
                f"{sp}_tot",
                scientific=True,
            )
        for comp in ["ju_diff", "jv_diff", "Ju_diff", "Jv_diff"]:
            if comp not in diff_dsmc[sp] or comp not in diff_analytic[sp]:
                continue
            title = analytic.field_title(comp, sp=sp, tag="tot")
            comp_tag = (
                comp_tag_diff[comp]
                if comp in comp_tag_diff
                else comp_tag_molar_diff[comp]
            )
            _plot_both(
                diff_dsmc[sp][comp],
                diff_analytic[sp][comp],
                title,
                comp_tag,
                f"{sp}_tot",
                scientific=True,
            )

    if use_parallel_save:
        _run_compare_plot_worker_tasks(contour_tasks, parallel_save_workers)


def compute_ya_min_based_inversion_x(sol, fields_total):
    ya_curve = analytic.get_YA_min_curve(sol, fields_total)
    ix_min = np.asarray(ya_curve["ix_min"], dtype=int)
    nx = int(sol["x"].size)
    ix_grid = np.broadcast_to(np.arange(nx, dtype=int), (ix_min.size, nx))
    sign_x = np.where(ix_grid < ix_min[:, None], -1, 1).astype(int)
    total_cells = int(sign_x.size)
    inversion_cells = int(np.count_nonzero(sign_x > 0))
    zero_cells = int(np.count_nonzero(sign_x == 0))
    negative_cells = int(np.count_nonzero(sign_x < 0))
    return {
        "species": "A",
        "mass_fraction_key": "YA",
        "ya_min_curve": ya_curve,
        "sign": {"x": sign_x},
        "stats": {
            "x": {
                "inversion_cells": inversion_cells,
                "zero_cells": zero_cells,
                "negative_cells": negative_cells,
                "non_inversion_cells": int(negative_cells + zero_cells),
                "total_cells": total_cells,
                "inversion_area_fraction": float(inversion_cells / total_cells),
                "inversion_area_percent": float(100.0 * inversion_cells / total_cells),
            }
        },
    }


def save_ya_min_based_inversion_summary(save_dir: Path, prefix: str, inversion_regions) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{prefix}_inversion_region_summary.txt"
    xstats = inversion_regions["stats"]["x"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("DSMC inversion region summary (YA-min heuristic)\n")
        f.write("species = A\n")
        f.write("mass_fraction = YA\n")
        f.write("definition_x = x >= x_min_YA(y)\n")
        f.write("note = For each y-row, the YA minimum location is used as the inversion boundary.\n")
        f.write(f"total_cells = {xstats['total_cells']}\n")
        f.write("\n")
        f.write("[x-direction]\n")
        f.write(f"inversion_cells = {xstats['inversion_cells']}\n")
        f.write(f"zero_cells = {xstats['zero_cells']}\n")
        f.write(f"negative_cells = {xstats['negative_cells']}\n")
        f.write(f"inversion_area_fraction = {xstats['inversion_area_fraction']:.16e}\n")
        f.write(f"inversion_area_percent = {xstats['inversion_area_percent']:.10f}\n")
    return path


def plot_dsmc_ya_min_inversion_region(
    sol_dsmc,
    fields_dsmc,
    save_dir: Path,
    prefix: str,
    show: bool,
    contour_cfg: Dict[str, float],
):
    inversion_regions = compute_ya_min_based_inversion_x(sol_dsmc, fields_dsmc)
    inversion_dir = save_dir / "inversion_region_contours"
    inversion_dir.mkdir(parents=True, exist_ok=True)
    save_ya_min_based_inversion_summary(save_dir, prefix, inversion_regions)
    minline = (
        inversion_regions["ya_min_curve"]["x_min"],
        inversion_regions["ya_min_curve"]["y"],
    )

    rc = {
        "font.size": contour_cfg["font_size"],
        "axes.titlesize": contour_cfg["font_size"],
        "axes.labelsize": contour_cfg["font_size"],
        "xtick.labelsize": contour_cfg["font_size"],
        "ytick.labelsize": contour_cfg["font_size"],
        "legend.fontsize": contour_cfg["font_size"],
    }

    with plt.rc_context(rc):
        analytic.plot_inversion_mask_contour(
            sol_dsmc["X"],
            sol_dsmc["Y"],
            inversion_regions["sign"]["x"],
            save_path=str(inversion_dir / f"{prefix}_inversion_region_x.png"),
            show=show,
            dpi=contour_cfg["dpi"],
            figsize=contour_cfg["figsize_contour"],
            tick_length=contour_cfg["tick_length"],
            tick_width=contour_cfg["tick_width"],
            box_linewidth=contour_cfg["box_linewidth"],
            x_nticks=contour_cfg.get("x_nticks", 6),
            ax_aspect=contour_cfg.get("ax_aspect", 1),
            minline=minline,
            minline_linewidth=contour_cfg["minline_linewidth"],
            Lref=2.0 * float(sol_dsmc["h"]),
        )
        analytic.save_inversion_region_legend(
            save_path=str(inversion_dir / f"{prefix}_inversion_region_x_legend.png"),
            show=show,
            dpi=contour_cfg["dpi"],
        )


def plot_compare_inversion_region_x(
    sol_dsmc,
    fields_dsmc,
    sol_analytic,
    fields_analytic,
    diff_analytic,
    save_dir: Path,
    prefix: str,
    contour_cfg: Dict[str, float],
):
    dsmc_inversion = compute_ya_min_based_inversion_x(sol_dsmc, fields_dsmc)
    analytic_inversion = analytic.compute_inversion_regions(sol_analytic, fields_analytic, diff_analytic, species="A")
    minline_dsmc = (
        dsmc_inversion["ya_min_curve"]["x_min"],
        dsmc_inversion["ya_min_curve"]["y"],
    )
    _, _, minline_analytic = analytic.compute_min_diagnostics(sol_analytic, fields_analytic)
    rc = {
        "font.size": contour_cfg["font_size"],
        "axes.titlesize": contour_cfg["font_size"],
        "axes.labelsize": contour_cfg["font_size"],
        "xtick.labelsize": contour_cfg["font_size"],
        "ytick.labelsize": contour_cfg["font_size"],
        "legend.fontsize": contour_cfg["font_size"],
    }
    inversion_dir = save_dir / "inversion_region_contours"
    inversion_dir.mkdir(parents=True, exist_ok=True)

    def _binary_sign(sign_map: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(sign_map) > 0, 1, -1).astype(int)

    sign_dsmc = _binary_sign(dsmc_inversion["sign"]["x"])
    sign_analytic = _binary_sign(analytic_inversion["sign"]["x"])
    cmap = mcolors.ListedColormap([analytic.CUSTOM_COLORS[4], analytic.CUSTOM_COLORS[0]])
    levels = [-1.5, 0.5, 1.5]

    with plt.rc_context(rc):
        X_top, Y_top, Z_top = split_sol_field(sol_dsmc, sign_dsmc, upper=True)
        X_bottom, Y_bottom, Z_bottom = split_sol_field(sol_analytic, sign_analytic, upper=False)

        # Non-dimensionalize spatial length axes by the full trench height 2h.
        Lref = 2.0 * float(sol_dsmc["h"])

        fig, ax = plt.subplots(figsize=contour_cfg["figsize_contour"])
        ax.contourf(X_top / Lref, Y_top / Lref, Z_top, levels=levels, cmap=cmap)
        ax.contourf(X_bottom / Lref, Y_bottom / Lref, Z_bottom, levels=levels, cmap=cmap)
        ax.axhline(0.0, color="k", linewidth=contour_cfg["box_linewidth"] * 0.5, linestyle="-", zorder=10)

        ax.set_xlabel(r"$x/(2h)$")
        ax.set_ylabel(r"$y/(2h)$")
        ax.set_title("")
        ax.set_aspect(contour_cfg["ax_aspect"], adjustable="box")
        # Exact physical non-dimensional domain: x in (0, AR), y in (-0.5, 0.5).
        ax.set_xlim(0.0, float(sol_dsmc["Lm"]) / Lref)
        ax.set_ylim(-sol_dsmc["h"] / Lref, sol_dsmc["h"] / Lref)

        plot_half_minline(ax, _scale_minline(minline_dsmc, Lref), upper=True, linewidth=contour_cfg["minline_linewidth"])
        plot_half_minline(ax, _scale_minline(minline_analytic, Lref), upper=False, linewidth=contour_cfg["minline_linewidth"])

        ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        AR = float(sol_dsmc["Lm"]) / Lref
        nx = 3 if AR <= 1.5 else (4 if AR <= 3.5 else 6)
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0.0, AR, nx)))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        analytic.apply_axes_style(
            ax,
            contour_cfg["box_linewidth"],
            contour_cfg["tick_length"],
            contour_cfg["tick_width"],
            tick_pad=8,
        )

        fig.tight_layout()
        fig.savefig(
            inversion_dir / f"{prefix}_inversion_region_x.png",
            dpi=contour_cfg["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)

        analytic.save_inversion_region_legend(
            save_path=str(inversion_dir / f"{prefix}_inversion_region_x_legend.png"),
            show=False,
            dpi=contour_cfg["dpi"],
        )


def surface_value_column(arrays: Dict[str, np.ndarray]) -> str:
    matches = [name for name in arrays if name.startswith("f_surface_reaction")]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one surface reaction value column in surf dump, "
            f"got {matches}"
        )
    return matches[0]


def sort_by_coordinate(coord: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(coord)
    return coord[order], values[order]


def even_reflect_positive_half(coord_half: np.ndarray, values_half: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coord_full = np.concatenate((-coord_half[::-1], coord_half))
    value_full = np.concatenate((values_half[::-1], values_half))
    return coord_full, value_full


def extract_surface_reaction_rate(case_cfg: CaseConfig, surf_arrays: Dict[str, np.ndarray]):
    required = ("v1x", "v1y", "v2x", "v2y")
    require_columns(required, surf_arrays)
    value_key = surface_value_column(surf_arrays)

    x1, y1 = map_to_analytic_axes(
        surf_arrays["v1x"],
        surf_arrays["v1y"],
        case_cfg.depth,
        case_cfg.trench_direction,
    )
    x2, y2 = map_to_analytic_axes(
        surf_arrays["v2x"],
        surf_arrays["v2y"],
        case_cfg.depth,
        case_cfg.trench_direction,
    )

    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    dl = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    count_avg = surf_arrays[value_key]

    # fix ave/surf already outputs the time-averaged reaction count per sampled timestep.
    # To obtain the physical 2D line rate, divide only by segment length and dt.
    local_rate = count_avg * case_cfg.fnum / (dl * case_cfg.timestep)

    tol = 1.0e-12
    is_horizontal = np.abs(y2 - y1) <= tol
    is_vertical = np.abs(x2 - x1) <= tol

    top_mask = (
        is_horizontal
        & np.isclose(yc, case_cfg.half_width, atol=tol, rtol=0.0)
        & (xc >= -tol)
        & (xc <= case_cfg.depth + tol)
    )
    bottom_mask = (
        is_horizontal
        & np.isclose(yc, -case_cfg.half_width, atol=tol, rtol=0.0)
        & (xc >= -tol)
        & (xc <= case_cfg.depth + tol)
    )
    end_mask = (
        is_vertical
        & np.isclose(xc, case_cfg.depth, atol=tol, rtol=0.0)
        & (np.abs(yc) <= case_cfg.half_width + tol)
    )

    x_top, r_top = sort_by_coordinate(xc[top_mask], local_rate[top_mask])
    x_bot, r_bot = sort_by_coordinate(xc[bottom_mask], local_rate[bottom_mask])
    y_end, r_end = sort_by_coordinate(yc[end_mask], local_rate[end_mask])

    if x_top.size == 0 and x_bot.size == 0:
        raise RuntimeError("Could not identify trench top/bottom wall segments in surf dump.")
    if y_end.size == 0:
        raise RuntimeError("Could not identify trench end-wall segments in surf dump.")

    if case_cfg.trench_mode.startswith("half"):
        if x_top.size == 0:
            x_top, r_top = x_bot.copy(), r_bot.copy()
        if x_bot.size == 0:
            x_bot, r_bot = x_top.copy(), r_top.copy()
        if np.all(y_end >= -tol):
            y_end, r_end = even_reflect_positive_half(y_end, r_end)
    else:
        if x_top.size == 0 or x_bot.size == 0:
            raise RuntimeError("Full-trench case expected both top and bottom wall segments.")

    if x_top.size != x_bot.size or not np.allclose(x_top, x_bot):
        raise RuntimeError("Top/bottom wall coordinates are inconsistent.")

    wx = np.diff(np.concatenate(([0.0], 0.5 * (x_top[1:] + x_top[:-1]), [case_cfg.depth])))
    wy_edges = np.empty(y_end.size + 1, dtype=float)
    wy_edges[1:-1] = 0.5 * (y_end[1:] + y_end[:-1])
    wy_edges[0] = -case_cfg.half_width
    wy_edges[-1] = case_cfg.half_width
    wy = np.diff(wy_edges)

    surface_rr = {
        "kappa": case_cfg.reaction_probability,
        "local_rate": {
            "bottom": r_bot,
            "top": r_top,
            "end": r_end,
        },
        "total_number_rate_2D_unit_depth": {
            "bottom": float(np.sum(r_bot * wx)),
            "top": float(np.sum(r_top * wx)),
            "end": float(np.sum(r_end * wy)),
        },
        "units": {
            "local_rate": "[#/ (m·s)]",
            "count_avg_from_fix_ave_surf": "[reactions / timestep / surf element]",
        },
        "dsmc_conversion": {
            "fnum": case_cfg.fnum,
            "timestep": case_cfg.timestep,
            "averaging": case_cfg.averaging,
            "formula": "local_rate = count_avg * fnum / (segment_length * dt)",
        },
    }
    surface_rr["total_number_rate_2D_unit_depth"]["total"] = (
        surface_rr["total_number_rate_2D_unit_depth"]["bottom"]
        + surface_rr["total_number_rate_2D_unit_depth"]["top"]
        + surface_rr["total_number_rate_2D_unit_depth"]["end"]
    )

    sol_surface = {
        "x": x_top,
        "y": y_end,
        "Lm": case_cfg.depth,
        "h": case_cfg.half_width,
    }
    return sol_surface, surface_rr


def save_compare_legend(centerline_dir: Path, prefix: str, name: str, handles, labels, dpi: int, ncol: int = 1):
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
    fig, ax = plt.subplots(figsize=(2.4 * ncol + 0.8, 0.8 * nrows + 0.8))
    ax.axis("off")
    ax.legend(unique_handles, unique_labels, loc="center", ncol=ncol, frameon=False)
    fig.savefig(centerline_dir / f"{prefix}_{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def line_cell_widths(coord: np.ndarray, lower: float, upper: float) -> np.ndarray:
    coord = np.asarray(coord, dtype=float)
    if coord.ndim != 1 or coord.size == 0:
        return np.array([], dtype=float)
    if coord.size == 1:
        return np.array([float(upper) - float(lower)], dtype=float)

    edges = np.empty(coord.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    edges[0] = float(lower)
    edges[-1] = float(upper)
    return np.diff(edges)


def inlet_nA_diagnostics(sol, fields_total, label: str) -> dict:
    x = np.asarray(sol["x"], dtype=float)
    y = np.asarray(sol["y"], dtype=float)
    nA = np.asarray(fields_total["A"]["n"], dtype=float)
    n_in_A = analytic.inlet_number_density(sol, "A")
    if n_in_A is None or not np.isfinite(n_in_A):
        raise RuntimeError(f"n_in_A is not available for {label} inlet diagnostics.")

    ix_inlet = 0
    iy_center = int(np.argmin(np.abs(y - 0.0)))
    profile = nA[:, ix_inlet]
    diff = profile - float(n_in_A)
    rel = diff / float(n_in_A)
    percent = 100.0 * rel
    weights = line_cell_widths(y, -float(sol["h"]), float(sol["h"]))

    finite = np.isfinite(profile) & np.isfinite(rel) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        raise RuntimeError(f"No finite inlet n_A values found for {label}.")

    wf = weights[finite]
    wf_sum = float(np.sum(wf))
    profile_f = profile[finite]
    diff_f = diff[finite]
    percent_f = percent[finite]

    max_abs_idx_local = int(np.argmax(np.abs(percent_f)))
    max_abs_idx = int(np.nonzero(finite)[0][max_abs_idx_local])

    return {
        "label": label,
        "ix_inlet": ix_inlet,
        "x_inlet": float(x[ix_inlet]),
        "iy_center": iy_center,
        "y_center": float(y[iy_center]),
        "n_in_A": float(n_in_A),
        "center_nA": float(profile[iy_center]),
        "center_abs_difference": float(diff[iy_center]),
        "center_percent_difference": float(percent[iy_center]),
        "simple_mean_nA": float(np.mean(profile_f)),
        "simple_mean_percent_difference": float(np.mean(percent_f)),
        "simple_mean_abs_percent_difference": float(np.mean(np.abs(percent_f))),
        "weighted_mean_nA": float(np.sum(profile_f * wf) / wf_sum),
        "weighted_mean_abs_difference": float(np.sum(diff_f * wf) / wf_sum),
        "weighted_mean_percent_difference": float(np.sum(percent_f * wf) / wf_sum),
        "weighted_mean_abs_percent_difference": float(np.sum(np.abs(percent_f) * wf) / wf_sum),
        "weighted_rms_percent_difference": float(np.sqrt(np.sum(percent_f * percent_f * wf) / wf_sum)),
        "max_abs_percent_difference": float(percent[max_abs_idx]),
        "max_abs_difference_y": float(y[max_abs_idx]),
        "max_abs_difference_nA": float(profile[max_abs_idx]),
        "profile": {
            "y": y.copy(),
            "weight": weights.copy(),
            "nA": profile.copy(),
            "abs_difference": diff.copy(),
            "percent_difference": percent.copy(),
        },
    }


def save_inlet_nA_diagnostics_compare(
    sol_dsmc,
    fields_dsmc,
    sol_analytic,
    fields_analytic,
    save_dir: Path,
    prefix: str,
    centerline_cfg: Dict[str, float],
) -> None:
    inlet_dir = save_dir / "inlet_diagnostics"
    inlet_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = [
        inlet_nA_diagnostics(sol_dsmc, fields_dsmc, "DSMC"),
        inlet_nA_diagnostics(sol_analytic, fields_analytic, "Analytic"),
    ]

    summary_path = inlet_dir / f"{prefix}_inlet_nA_difference_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Inlet n_A diagnostics\n")
        f.write("definition = inlet column is the first cell-center column, ix=0\n")
        f.write("centerline = row with minimum |y|\n")
        f.write("percent_difference = 100 * (n_A - n_in_A) / n_in_A\n")
        f.write("note = weighted statistics use y-cell widths; for uniform spacing this equals the simple mean apart from reconstructed center-row weighting.\n")
        for diag in diagnostics:
            f.write("\n")
            f.write(f"[{diag['label']}]\n")
            f.write(f"n_in_A = {diag['n_in_A']:.16e}\n")
            f.write(f"ix_inlet = {diag['ix_inlet']}\n")
            f.write(f"x_inlet_cell_center = {diag['x_inlet']:.16e}\n")
            f.write(f"iy_center = {diag['iy_center']}\n")
            f.write(f"y_center = {diag['y_center']:.16e}\n")
            f.write(f"center_nA = {diag['center_nA']:.16e}\n")
            f.write(f"center_abs_difference = {diag['center_abs_difference']:.16e}\n")
            f.write(f"center_percent_difference = {diag['center_percent_difference']:.10f}\n")
            f.write(f"simple_mean_nA = {diag['simple_mean_nA']:.16e}\n")
            f.write(f"simple_mean_percent_difference = {diag['simple_mean_percent_difference']:.10f}\n")
            f.write(f"simple_mean_abs_percent_difference = {diag['simple_mean_abs_percent_difference']:.10f}\n")
            f.write(f"weighted_mean_nA = {diag['weighted_mean_nA']:.16e}\n")
            f.write(f"weighted_mean_abs_difference = {diag['weighted_mean_abs_difference']:.16e}\n")
            f.write(f"weighted_mean_percent_difference = {diag['weighted_mean_percent_difference']:.10f}\n")
            f.write(f"weighted_mean_abs_percent_difference = {diag['weighted_mean_abs_percent_difference']:.10f}\n")
            f.write(f"weighted_rms_percent_difference = {diag['weighted_rms_percent_difference']:.10f}\n")
            f.write(f"max_abs_percent_difference = {diag['max_abs_percent_difference']:.10f}\n")
            f.write(f"max_abs_difference_y = {diag['max_abs_difference_y']:.16e}\n")
            f.write(f"max_abs_difference_nA = {diag['max_abs_difference_nA']:.16e}\n")

    profile_path = inlet_dir / f"{prefix}_inlet_nA_difference_profile.tsv"
    with profile_path.open("w", encoding="utf-8") as f:
        f.write("dataset\ty\tweight\tn_in_A\tnA\tabs_difference\tpercent_difference\n")
        for diag in diagnostics:
            profile = diag["profile"]
            for yv, wv, nAv, ev, pv in zip(
                profile["y"],
                profile["weight"],
                profile["nA"],
                profile["abs_difference"],
                profile["percent_difference"],
            ):
                f.write(
                    f"{diag['label']}\t{yv:.16e}\t{wv:.16e}\t{diag['n_in_A']:.16e}\t"
                    f"{nAv:.16e}\t{ev:.16e}\t{pv:.16e}\n"
                )

    rc = {
        "font.size": centerline_cfg["font_size"],
        "axes.titlesize": centerline_cfg["font_size"],
        "axes.labelsize": centerline_cfg["font_size"],
        "xtick.labelsize": centerline_cfg["font_size"],
        "ytick.labelsize": centerline_cfg["font_size"],
        "legend.fontsize": centerline_cfg["font_size"],
    }

    # Non-dimensionalize spatial length axis by the full trench height 2h.
    Lref = 2.0 * float(sol_dsmc["h"])

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
        colors = {"DSMC": DSMC_COMPARE_COLOR, "Analytic": ANALYTIC_COMPARE_COLOR}
        for diag in diagnostics:
            profile = diag["profile"]
            ax.plot(
                np.asarray(profile["y"]) / Lref,
                profile["percent_difference"],
                color=colors[diag["label"]],
                linewidth=centerline_cfg["centerline_linewidth"],
                label=diag["label"],
            )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=centerline_cfg["centerline_linewidth"], label="_nolegend_")
        ax.set_xlabel(r"$y/(2h)$")
        ax.set_ylabel(r"$(n_{\mathrm{A}}-n_{\mathrm{open,res}})/n_{\mathrm{open,res}}$ [%]")
        # x-axis (y/(2h) length) -> plain decimal
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        analytic.set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=False)
        analytic.apply_axes_style(
            ax,
            centerline_cfg["box_linewidth"],
            centerline_cfg["tick_length"],
            centerline_cfg["tick_width"],
        )
        handles, labels = ax.get_legend_handles_labels()
        save_compare_legend(inlet_dir, prefix, "inlet_nA_difference_legend", handles, labels, centerline_cfg["dpi"], ncol=2)
        fig.tight_layout()
        fig.savefig(inlet_dir / f"{prefix}_inlet_nA_percent_difference_profile.png", dpi=centerline_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)


def plot_compare_centerlines(
    sol_dsmc,
    fields_dsmc,
    diff_dsmc,
    sol_analytic,
    fields_analytic,
    diff_analytic,
    save_dir: Path,
    prefix: str,
    centerline_cfg: Dict[str, float],
):
    centerline_dir = save_dir / "centerline_plots"
    centerline_dir.mkdir(parents=True, exist_ok=True)

    x_d = sol_dsmc["x"]
    y_d = sol_dsmc["y"]
    iy_d = int(np.argmin(np.abs(y_d - 0.0)))

    x_a = sol_analytic["x"]
    y_a = sol_analytic["y"]
    iy_a = int(np.argmin(np.abs(y_a - 0.0)))

    # Non-dimensionalize spatial length (x) axis by the full trench height 2h.
    # Use scaled copies for plotting only; index computations use original arrays.
    Lref = 2.0 * float(sol_dsmc["h"])
    x_d_s = np.asarray(x_d) / Lref
    x_a_s = np.asarray(x_a) / Lref

    dsmc_color = DSMC_COMPARE_COLOR
    analytic_color = ANALYTIC_COMPARE_COLOR

    rc = {
        "font.size": centerline_cfg["font_size"],
        "axes.titlesize": centerline_cfg["font_size"],
        "axes.labelsize": centerline_cfg["font_size"],
        "xtick.labelsize": centerline_cfg["font_size"],
        "ytick.labelsize": centerline_cfg["font_size"],
        "legend.fontsize": centerline_cfg["font_size"],
    }

    def _finish(fig, ax, stem: str, sci_axis: str, legend_name: str | None = None, legend_ncol: int = 1):
        # The centerline x-axis is always the depth "x/(2h)" (a length) -> plain decimal,
        # matching the contour treatment. The y-axis (data magnitude) keeps scientific
        # notation only when requested via sci_axis.
        if sci_axis in ("y", "both"):
            analytic.apply_sci_offset(ax, axis="y", scilimits=(0, 0))
        # x-axis is the depth x/(2h) over the TRUE physical domain (walls), not the
        # cell-center data max. Use exactly 3 ticks: both endpoints + centre.
        AR = float(sol_dsmc["Lm"]) / Lref
        ax.set_xlim(-0.05 * AR, 1.05 * AR)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, AR / 2.0, AR]))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        if stem in {"centerline_mass_fraction", "centerline_mole_fraction"}:
            analytic.set_centerline_mass_fraction_axis(ax, axis="y")
        elif sci_axis in ("y", "both"):
            analytic.set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
        else:
            analytic.set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=False)
        analytic.apply_axes_style(
            ax,
            centerline_cfg["box_linewidth"],
            centerline_cfg["tick_length"],
            centerline_cfg["tick_width"],
        )
        fig.tight_layout()
        if legend_name is not None:
            handles, labels = ax.get_legend_handles_labels()
            save_compare_legend(
                centerline_dir,
                prefix,
                legend_name,
                handles,
                labels,
                centerline_cfg["dpi"],
                ncol=legend_ncol,
            )
        fig.savefig(centerline_dir / f"{prefix}_{stem}.png", dpi=centerline_cfg["dpi"], bbox_inches="tight")
        plt.close(fig)

    def _plot_compare_nA_with_inlet():
        n_in_A = analytic.inlet_number_density(sol_analytic, "A")
        if n_in_A is None:
            n_in_A = analytic.inlet_number_density(sol_dsmc, "A")
        if n_in_A is None or not np.isfinite(n_in_A):
            return

        fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
        ax.plot(
            x_d_s,
            fields_dsmc["A"]["n"][iy_d, :],
            color=dsmc_color,
            linewidth=centerline_cfg["centerline_linewidth"] + 1,
            label="DSMC",
        )
        ax.plot(
            x_a_s,
            fields_analytic["A"]["n"][iy_a, :],
            color=analytic_color,
            linewidth=centerline_cfg["centerline_linewidth"],
            label="Analytic",
        )
        ax.axhline(
            n_in_A,
            color="k",
            linestyle="--",
            linewidth=centerline_cfg["centerline_linewidth"],
            label=r"$n_{\mathrm{open,res}}$",
        )
        ax.set_xlabel(r"$x/(2h)$")
        ax.set_ylabel(analytic.field_symbol("n", sp="A"))
        _finish(
            fig,
            ax,
            "centerline_A_n_with_inlet",
            sci_axis="both",
            legend_name="centerline_A_n_with_inlet_legend",
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

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
        ax.plot(x_d_s, fields_dsmc["YA"][iy_d, :], color=dsmc_color, linestyle="-", linewidth=centerline_cfg["centerline_linewidth"]+1, label=r"DSMC $Y_{\mathrm{A}}$")
        ax.plot(x_d_s, fields_dsmc["YB"][iy_d, :], color=dsmc_color, linestyle="--", linewidth=centerline_cfg["centerline_linewidth"]+1, label=r"DSMC $Y_{\mathrm{B}}$")
        ax.plot(x_a_s, fields_analytic["YA"][iy_a, :], color=analytic_color, linestyle="-", linewidth=centerline_cfg["centerline_linewidth"], label=r"Analytic $Y_{\mathrm{A}}$")
        ax.plot(x_a_s, fields_analytic["YB"][iy_a, :], color=analytic_color, linestyle="--", linewidth=centerline_cfg["centerline_linewidth"], label=r"Analytic $Y_{\mathrm{B}}$")
        _msize = centerline_cfg["centerline_linewidth"] * 5
        ix_min_A_d = int(np.nanargmin(fields_dsmc["YA"][iy_d, :]))
        ix_max_B_d = int(np.nanargmax(fields_dsmc["YB"][iy_d, :]))
        ix_min_A_a = int(np.nanargmin(fields_analytic["YA"][iy_a, :]))
        ix_max_B_a = int(np.nanargmax(fields_analytic["YB"][iy_a, :]))
        ax.plot(x_d_s[ix_min_A_d], fields_dsmc["YA"][iy_d, ix_min_A_d], marker="*", linestyle="None", markersize=_msize+1, color=dsmc_color, label="_nolegend_")
        ax.plot(x_d_s[ix_max_B_d], fields_dsmc["YB"][iy_d, ix_max_B_d], marker="*", linestyle="None", markersize=_msize+1, color=dsmc_color, label="_nolegend_")
        ax.plot(x_a_s[ix_min_A_a], fields_analytic["YA"][iy_a, ix_min_A_a], marker="*", linestyle="None", markersize=_msize, color=analytic_color, label="_nolegend_")
        ax.plot(x_a_s[ix_max_B_a], fields_analytic["YB"][iy_a, ix_max_B_a], marker="*", linestyle="None", markersize=_msize, color=analytic_color, label="_nolegend_")
        ax.set_xlabel(r"$x/(2h)$")
        ax.set_ylabel("mass fraction")
        _finish(fig, ax, "centerline_mass_fraction", sci_axis="x", legend_name="centerline_mass_fraction_legend", legend_ncol=2)

        if "XA" in fields_dsmc and "XB" in fields_dsmc and "XA" in fields_analytic and "XB" in fields_analytic:
            fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
            ax.plot(x_d_s, fields_dsmc["XA"][iy_d, :], color=dsmc_color, linestyle="-", linewidth=centerline_cfg["centerline_linewidth"]+1, label=r"DSMC $X_{\mathrm{A}}$")
            ax.plot(x_d_s, fields_dsmc["XB"][iy_d, :], color=dsmc_color, linestyle="--", linewidth=centerline_cfg["centerline_linewidth"]+1, label=r"DSMC $X_{\mathrm{B}}$")
            ax.plot(x_a_s, fields_analytic["XA"][iy_a, :], color=analytic_color, linestyle="-", linewidth=centerline_cfg["centerline_linewidth"], label=r"Analytic $X_{\mathrm{A}}$")
            ax.plot(x_a_s, fields_analytic["XB"][iy_a, :], color=analytic_color, linestyle="--", linewidth=centerline_cfg["centerline_linewidth"], label=r"Analytic $X_{\mathrm{B}}$")
            _plot_extreme_marker(ax, x_d_s, fields_dsmc["XA"][iy_d, :], dsmc_color, _msize+1, "min")
            _plot_extreme_marker(ax, x_d_s, fields_dsmc["XB"][iy_d, :], dsmc_color, _msize+1, "max")
            _plot_extreme_marker(ax, x_a_s, fields_analytic["XA"][iy_a, :], analytic_color, _msize, "min")
            _plot_extreme_marker(ax, x_a_s, fields_analytic["XB"][iy_a, :], analytic_color, _msize, "max")
            ax.set_xlabel(r"$x/(2h)$")
            ax.set_ylabel("mole fraction")
            _finish(fig, ax, "centerline_mole_fraction", sci_axis="x", legend_name="centerline_mole_fraction_legend", legend_ncol=2)

        comp_tag_mol = {"nu": "nu", "nv": "nv", "n": "n", "c": "c"}
        comp_tag_diff = {"ju_diff": "j_x", "jv_diff": "j_y"}
        comp_tag_molar_diff = {"Ju_diff": "J_x", "Jv_diff": "J_y"}

        for sp in ["A", "B"]:
            for comp in ["nu", "nv", "n", "c"]:
                if comp not in fields_dsmc[sp] or comp not in fields_analytic[sp]:
                    continue
                fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
                ax.plot(
                    x_d_s,
                    fields_dsmc[sp][comp][iy_d, :],
                    color=dsmc_color,
                    linewidth=centerline_cfg["centerline_linewidth"]+1,
                    label="DSMC",
                )
                ax.plot(
                    x_a_s,
                    fields_analytic[sp][comp][iy_a, :],
                    color=analytic_color,
                    linewidth=centerline_cfg["centerline_linewidth"],
                    label="Analytic",
                )
                ax.set_xlabel(r"$x/(2h)$")
                ax.set_ylabel(analytic.field_symbol(comp, sp=sp))
                if comp == "c":
                    marker_mode = "min" if sp == "A" else "max"
                    _plot_extreme_marker(ax, x_d_s, fields_dsmc[sp][comp][iy_d, :], dsmc_color, _msize+1, marker_mode)
                    _plot_extreme_marker(ax, x_a_s, fields_analytic[sp][comp][iy_a, :], analytic_color, _msize, marker_mode)
                _finish(fig, ax, f"centerline_{sp}_{comp_tag_mol[comp]}", sci_axis="both", legend_name="centerline_compare_legend", legend_ncol=2)
                if sp == "A" and comp == "n":
                    _plot_compare_nA_with_inlet()

        for sp in ["A", "B"]:
            for comp in ["ju_diff", "jv_diff", "Ju_diff", "Jv_diff"]:
                if comp not in diff_dsmc[sp] or comp not in diff_analytic[sp]:
                    continue
                comp_tag = (
                    comp_tag_diff[comp]
                    if comp in comp_tag_diff
                    else comp_tag_molar_diff[comp]
                )
                fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
                ax.plot(
                    x_d_s,
                    diff_dsmc[sp][comp][iy_d, :],
                    color=dsmc_color,
                    linewidth=centerline_cfg["centerline_linewidth"]+1,
                    label="DSMC",
                )
                ax.plot(
                    x_a_s,
                    diff_analytic[sp][comp][iy_a, :],
                    color=analytic_color,
                    linewidth=centerline_cfg["centerline_linewidth"],
                    label="Analytic",
                )
                ax.set_xlabel(r"$x/(2h)$")
                ax.set_ylabel(analytic.field_symbol(comp, sp=sp))
                _finish(fig, ax, f"centerline_{sp}_{comp_tag}", sci_axis="both", legend_name="centerline_compare_legend", legend_ncol=2)

        # Combined-species flux plots: both species A (solid) and B (dashed) on one
        # axis, DSMC vs analytic by color -- mirrors the mass-fraction plot above.
        # Produced for each diffusive-flux component (j_x, j_y, J_x, J_y).
        for comp in ["ju_diff", "jv_diff", "Ju_diff", "Jv_diff"]:
            if any(comp not in src["A"] or comp not in src["B"] for src in (diff_dsmc, diff_analytic)):
                continue
            comp_tag = comp_tag_diff[comp] if comp in comp_tag_diff else comp_tag_molar_diff[comp]
            fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
            sym_A = analytic.field_symbol(comp, sp="A")
            sym_B = analytic.field_symbol(comp, sp="B")
            ax.plot(x_d_s, diff_dsmc["A"][comp][iy_d, :], color=dsmc_color, linestyle="-", linewidth=centerline_cfg["centerline_linewidth"]+1, label=rf"DSMC {sym_A}")
            ax.plot(x_d_s, diff_dsmc["B"][comp][iy_d, :], color=dsmc_color, linestyle="--", linewidth=centerline_cfg["centerline_linewidth"]+1, label=rf"DSMC {sym_B}")
            ax.plot(x_a_s, diff_analytic["A"][comp][iy_a, :], color=analytic_color, linestyle="-", linewidth=centerline_cfg["centerline_linewidth"], label=rf"Analytic {sym_A}")
            ax.plot(x_a_s, diff_analytic["B"][comp][iy_a, :], color=analytic_color, linestyle="--", linewidth=centerline_cfg["centerline_linewidth"], label=rf"Analytic {sym_B}")
            ax.set_xlabel(r"$x/(2h)$")
            ax.set_ylabel(analytic.field_symbol(comp, sp=None))
            _finish(fig, ax, f"centerline_{comp_tag}", sci_axis="both", legend_name=f"centerline_AB_{comp_tag}_legend", legend_ncol=2)


def plot_compare_YA_min_curves(
    sol_dsmc,
    fields_dsmc,
    sol_analytic,
    fields_analytic,
    save_dir: Path,
    prefix: str,
    centerline_cfg: Dict[str, float],
):
    ymin_dir = save_dir / "YA_min_plots"
    ymin_dir.mkdir(parents=True, exist_ok=True)

    curve_dsmc = analytic.get_YA_min_curve(sol_dsmc, fields_dsmc)
    curve_analytic = analytic.get_YA_min_curve(sol_analytic, fields_analytic)

    dsmc_color = DSMC_COMPARE_COLOR
    analytic_color = ANALYTIC_COMPARE_COLOR
    linewidth = centerline_cfg["centerline_linewidth"]

    rc = {
        "font.size": centerline_cfg["font_size"],
        "axes.titlesize": centerline_cfg["font_size"],
        "axes.labelsize": centerline_cfg["font_size"],
        "xtick.labelsize": centerline_cfg["font_size"],
        "ytick.labelsize": centerline_cfg["font_size"],
        "legend.fontsize": centerline_cfg["font_size"],
    }

    # Non-dimensionalize spatial length axes by the full trench height 2h.
    Lref = 2.0 * float(sol_dsmc["h"])

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=centerline_cfg["figsize_centerline"])
        ax.plot(
            np.asarray(curve_dsmc["y"]) / Lref,
            np.asarray(curve_dsmc["x_min"]) / Lref,
            color=dsmc_color,
            linewidth=linewidth + 1,
            label=r"DSMC $x(Y_{\mathrm{A},\min})$",
        )
        ax.plot(
            np.asarray(curve_analytic["y"]) / Lref,
            np.asarray(curve_analytic["x_min"]) / Lref,
            color=analytic_color,
            linewidth=linewidth,
            label=r"Analytic $x(Y_{\mathrm{A},\min})$",
        )

        ax.set_xlabel(r"$y/(2h)$")
        ax.set_ylabel(r"$x(Y_{\mathrm{A},\min})/(2h)$")
        # Both axes are lengths (y/(2h) and x(Y_min)/(2h)) -> plain decimal.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        analytic.apply_axes_style(
            ax,
            centerline_cfg["box_linewidth"],
            centerline_cfg["tick_length"],
            centerline_cfg["tick_width"],
        )
        fig.tight_layout()

        handles, labels = ax.get_legend_handles_labels()
        save_compare_legend(
            ymin_dir,
            prefix,
            "YA_min_comparison_legend",
            handles,
            labels,
            centerline_cfg["dpi"],
            ncol=2,
        )
        fig.savefig(
            ymin_dir / f"{prefix}_YA_min_comparison.png",
            dpi=centerline_cfg["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)


def wall_path_coordinates(sol, surface_rr):
    x = sol["x"]
    y = sol["y"]
    Lm = float(sol["Lm"])
    h = float(sol["h"])
    r_bot = surface_rr["local_rate"]["bottom"]
    r_end = surface_rr["local_rate"]["end"]
    r_top = surface_rr["local_rate"]["top"]

    s_bot = x - (Lm + h)
    s_end = y
    x_rev = x[::-1]
    r_top_rev = r_top[::-1]
    s_top = h + (Lm - x_rev)
    return s_bot, r_bot, s_end, r_end, s_top, r_top_rev


def plot_surface_reaction_rate_compare(
    sol_dsmc,
    surface_rr_dsmc,
    sol_analytic,
    surface_rr_analytic,
    save_dir: Path,
    prefix: str,
    show: bool,
    surface_cfg: Dict[str, float],
):
    sr_dir = save_dir / "surface_reaction_rate_plots"
    sr_dir.mkdir(parents=True, exist_ok=True)

    sbd, rbd, sed, red, std, rtd = wall_path_coordinates(sol_dsmc, surface_rr_dsmc)
    sba, rba, sea, rea, sta, rta = wall_path_coordinates(sol_analytic, surface_rr_analytic)
    h = float(sol_analytic["h"])

    # Non-dimensionalize the along-wall spatial coordinate s by the full trench height 2h.
    Lref = 2.0 * h
    # True-domain non-dimensional values for ticks (NOT from cell-center data max):
    #   side walls run along depth [0, Lm] -> nd extent AR; end-wall corners at s = +/- h -> +/- 0.5.
    Lm = float(sol_analytic["Lm"])
    AR = Lm / Lref
    s_wall_end_nd = (Lm + h) / Lref  # = AR + 0.5, the open ends of the unwrapped polyline
    sbd_s, sed_s, std_s = sbd / Lref, sed / Lref, std / Lref
    sba_s, sea_s, sta_s = sba / Lref, sea / Lref, sta / Lref

    rc = {
        "font.size": surface_cfg["font_size"],
        "axes.titlesize": surface_cfg["font_size"],
        "axes.labelsize": surface_cfg["font_size"],
        "xtick.labelsize": surface_cfg["font_size"],
        "ytick.labelsize": surface_cfg["font_size"],
        "legend.fontsize": surface_cfg["font_size"],
    }

    def _draw(ax):
        ax.plot(sba_s, rba, color=ANALYTIC_COMPARE_COLOR, linewidth=surface_cfg["line_width"], label="Analytic")
        ax.plot(sea_s, rea, color=ANALYTIC_COMPARE_COLOR, linewidth=surface_cfg["line_width"], label="_nolegend_")
        ax.plot(sta_s, rta, color=ANALYTIC_COMPARE_COLOR, linewidth=surface_cfg["line_width"], label="_nolegend_")

        marker_size = surface_cfg["line_width"] * 0.5
        ax.plot(sbd_s, rbd, linestyle="None", marker="o", markersize=marker_size, color=DSMC_COMPARE_COLOR, label="DSMC")
        ax.plot(sed_s, red, linestyle="None", marker="o", markersize=marker_size, color=DSMC_COMPARE_COLOR, label="_nolegend_")
        ax.plot(std_s, rtd, linestyle="None", marker="o", markersize=marker_size, color=DSMC_COMPARE_COLOR, label="_nolegend_")

        for wall_mark in (-h, h):
            ax.axvline(wall_mark / Lref, color="k", linestyle="--", linewidth=surface_cfg["wall_dashed_width"])
        ax.set_xlabel(r"$s/(2h)$ (along wall)")
        ax.set_ylabel(r"surface reaction rate [$\#/(m\cdot s)$]")
        # y-axis (reaction-rate magnitude) stays scientific; x-axis (s/(2h) length) -> plain decimal.
        # The unwrapped along-wall axis spans [-(AR+0.5), +(AR+0.5)] in nd; ticks must include
        # the end-wall vertices (corners) at +/-0.5 and the open wall ends at +/-(AR+0.5).
        analytic.apply_sci_offset(ax, axis="y", scilimits=(0, 0), useOffset=False)
        ax.xaxis.set_major_locator(
            mticker.FixedLocator([-s_wall_end_nd, -0.5, 0.0, 0.5, s_wall_end_nd])
        )
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        analytic.apply_axes_style(
            ax,
            surface_cfg["box_linewidth"],
            surface_cfg["tick_length"],
            surface_cfg["tick_width"],
        )

    def _set_ylim_from_values(ax, values, pad_frac):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        ymin = float(np.min(values))
        ymax = float(np.max(values))
        if np.isclose(ymin, ymax):
            pad = pad_frac * max(abs(ymin), 1.0)
        else:
            pad = pad_frac * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    def _save(fig, path: Path):
        fig.tight_layout()
        fig.savefig(path, dpi=surface_cfg["dpi"], bbox_inches="tight")
        if not show:
            plt.close(fig)

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=surface_cfg["figsize"])
        _draw(ax)
        handles, labels = ax.get_legend_handles_labels()
        save_compare_legend(sr_dir, prefix, "surface_reaction_rate_legend", handles, labels, surface_cfg["dpi"], ncol=1)
        _save(fig, sr_dir / f"{prefix}_surface_reaction_rate.png")

        fig, ax = plt.subplots(figsize=surface_cfg["zoom_figsize"])
        _draw(ax)
        end_values = [rbd[-1], red[0], red[-1], rtd[0], rba[-1], rea[0], rea[-1], rta[0]]
        ax.set_xlim((-h - surface_cfg["end_zoom_pad"]) / Lref, (h + surface_cfg["end_zoom_pad"]) / Lref)
        # End-wall zoom: ticks at the two visible vertices and centre.
        ax.xaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        _set_ylim_from_values(ax, end_values, surface_cfg["end_zoom_rate_pad_frac"])
        _save(fig, sr_dir / f"{prefix}_surface_reaction_rate_endwall_zoom.png")


def make_prefix(case_cfg: CaseConfig, sol) -> str:
    nx = int(sol["x"].size)
    ny = int(sol["y"].size)
    return f"AR{case_cfg.aspect_ratio}_rp{case_cfg.reaction_probability:.3f}_nx{nx}_ny{ny}"


def make_compare_prefix(case_cfg: CaseConfig) -> str:
    return f"AR{case_cfg.aspect_ratio}_rp{case_cfg.reaction_probability:.3f}_compare"


def resolve_save_dir(case_cfg: CaseConfig, figure_root: Path | None = None) -> Path:
    root = figure_output_root() if figure_root is None else Path(figure_root)
    case_folder = f"{case_cfg.case_name}{case_cfg.output_suffix}"
    return (root / case_cfg.study / case_folder).resolve()


def case_name_sort_key(study: str, name: str):
    if study == "AR":
        return int(name)
    return float(name)


def case_has_completion_marker(case_dir: Path) -> bool:
    if (case_dir / "job_done").exists():
        return True
    log_path = case_dir / "log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "Loop time of" in text


def discover_completed_cases(
    study_filter: str | None = None,
    case_root: Path = CASE_STUDY_ROOT,
    output_suffix: str = "",
    require_surface_dump: bool = True,
    require_completion_marker: bool = False,
):
    discovered = []
    studies = [study_filter] if study_filter is not None else ["AR", "Kn", "Probability"]
    for study in studies:
        study_dir = Path(case_root) / study
        if not study_dir.exists():
            continue
        cases = sorted(
            [path for path in study_dir.iterdir() if path.is_dir()],
            key=lambda path: case_name_sort_key(study, path.name),
        )
        for case_dir in cases:
            if require_completion_marker and not case_has_completion_marker(case_dir):
                continue
            grid_path = latest_positive_snapshot(case_dir, "grid")
            if grid_path is None:
                continue
            surf_path = latest_positive_snapshot(case_dir, "surf")
            if require_surface_dump and surf_path is None:
                continue
            case_cfg = load_case_config(
                study,
                case_dir.name,
                case_root=Path(case_root),
                output_suffix=output_suffix,
                has_surface_dump=surf_path is not None,
            )
            discovered.append((case_cfg, grid_path, surf_path))
    return discovered


def load_case_manager_duplicates(study_filter: str | None = None) -> list[tuple[str, str]]:
    case_manager_path = DATA_ROOT / "case_study_manager.py"
    if not case_manager_path.exists():
        return []

    spec = importlib.util.spec_from_file_location("case_study_manager_for_post", case_manager_path)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "prepare_cases_with_dedupe"):
        return []

    duplicate_pairs = []
    for case in module.prepare_cases_with_dedupe():
        if not case.get("is_duplicate"):
            continue
        if study_filter is not None and case["study"] != study_filter:
            continue
        duplicate_key = f"{case['study']}/{case['case']}"
        canonical_key = case["canonical_key"]
        duplicate_pairs.append((duplicate_key, canonical_key))
    return duplicate_pairs


def replicate_duplicate_case_figures(study_filter: str | None = None) -> list[tuple[str, str]]:
    root = figure_output_root()
    replicated = []
    for duplicate_key, canonical_key in load_case_manager_duplicates(study_filter=study_filter):
        duplicate_dir = root / duplicate_key
        canonical_dir = root / canonical_key
        if not canonical_dir.exists():
            continue
        if duplicate_dir.exists():
            shutil.rmtree(duplicate_dir)
        duplicate_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(canonical_dir, duplicate_dir)
        replicated.append((duplicate_key, canonical_key))
    return replicated


def unique_case_names(case_names: Iterable[str]) -> list[str]:
    unique_names = []
    seen = set()
    for case_name in case_names:
        if case_name in seen:
            continue
        seen.add(case_name)
        unique_names.append(case_name)
    return unique_names


def resolve_requested_case_names(
    study: str,
    case_selection,
    discovered_by_name: dict[str, tuple[CaseConfig, Path, Path]],
) -> tuple[str, ...]:
    available_case_names = list(discovered_by_name.keys())
    if case_selection == "all":
        return tuple(available_case_names)
    if case_selection is None:
        return tuple()

    raw_case_names = case_selection
    if isinstance(case_selection, (str, int, float)):
        raw_case_names = (case_selection,)

    requested_case_names = unique_case_names(
        normalize_case_name(study, case_name) for case_name in raw_case_names
    )
    missing = sorted(
        [case_name for case_name in requested_case_names if case_name not in discovered_by_name],
        key=lambda name: case_name_sort_key(study, name),
    )
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Requested completed case(s) were not found for study {study}: {missing_str}"
        )
    return tuple(requested_case_names)


def validate_study_batches(
    study: str,
    requested_case_names: tuple[str, ...],
    batch_defs: tuple[dict, ...] | list[dict],
) -> tuple[dict, ...]:
    if not requested_case_names:
        return tuple()
    if not batch_defs:
        return (
            {
                "case_names": tuple(requested_case_names),
                "parallel_cases": 1,
                "case_max_workers": DEFAULT_CASE_MAX_WORKERS,
            },
        )

    configured = []
    covered: list[str] = []
    requested = set(requested_case_names)

    for batch in batch_defs:
        batch_case_names = []
        for case_name in batch.get("cases", ()):
            normalized_name = normalize_case_name(study, case_name)
            if normalized_name in requested and normalized_name not in batch_case_names:
                batch_case_names.append(normalized_name)
        if not batch_case_names:
            continue
        configured.append(
            {
                "case_names": tuple(batch_case_names),
                "parallel_cases": int(batch.get("parallel_cases", len(batch_case_names))),
                "case_max_workers": int(batch.get("case_max_workers", DEFAULT_CASE_MAX_WORKERS)),
            }
        )
        covered.extend(batch_case_names)

    missing = sorted(
        set(requested_case_names).difference(covered),
        key=lambda name: case_name_sort_key(study, name),
    )
    if missing:
        raise ValueError(f"Missing batch definitions for {study} cases: {missing}")
    return tuple(configured)


def plot_case(
    case_cfg: CaseConfig,
    grid_path: Path,
    surf_path: Path | None,
    save_dir: Path,
    show_contours: bool,
    show_centerlines: bool,
    show_surface: bool,
    case_max_workers: int,
    ensemble_average: EnsembleAverageResult | None = None,
) -> None:
    if ensemble_average is None:
        timestep, arrays = read_grid_dump(grid_path)
        sol_dsmc, fields_dsmc, diff_dsmc, formula_check = build_full_fields(case_cfg, arrays)
        ensemble_info = None
    else:
        timestep = None
        sol_dsmc = ensemble_average.sol
        fields_dsmc = ensemble_average.fields
        diff_dsmc = ensemble_average.diff
        formula_check = {}
        ensemble_info = ensemble_average.info
    surf_timestep = None
    sol_surface_dsmc = None
    surface_rr_dsmc = None
    if surf_path is not None:
        surf_timestep, surf_arrays = read_surf_dump(surf_path)
        sol_surface_dsmc, surface_rr_dsmc = extract_surface_reaction_rate(case_cfg, surf_arrays)
    (
        sol_analytic,
        fields_analytic,
        diff_analytic,
        contrib_analytic,
        surface_rr_analytic,
    ) = solve_analytic_case(case_cfg, case_max_workers=case_max_workers)

    dsmc_prefix = make_prefix(case_cfg, sol_dsmc)
    analytic_prefix = make_prefix(case_cfg, sol_analytic)
    compare_prefix = make_compare_prefix(case_cfg)
    case_plot_cfg = analytic.get_case_plot_settings(case_cfg.aspect_ratio)
    contour_cfg = case_plot_cfg["contour"]
    field_contour_cfg = field_contour_kwargs(contour_cfg)
    centerline_cfg = case_plot_cfg["centerline"]
    surface_cfg = build_surface_plot_cfg(case_cfg)

    save_dir.mkdir(parents=True, exist_ok=True)
    dsmc_dir = save_dir / "dsmc"
    analytic_dir = save_dir / "analytic"
    both_dir = save_dir / "both"
    dsmc_dir.mkdir(parents=True, exist_ok=True)
    analytic_dir.mkdir(parents=True, exist_ok=True)
    both_dir.mkdir(parents=True, exist_ok=True)
    if ensemble_info is not None:
        write_ensemble_average_summary(save_dir, case_cfg, ensemble_info)

    analytic.plot_field_contours(
        sol_dsmc,
        fields_dsmc,
        diff_dsmc,
        contrib=None,
        save_dir=str(dsmc_dir),
        prefix=dsmc_prefix,
        show_field_contour_plots=show_contours,
        save_field_contour_plots=True,
        parallel_save_workers=PLOT_SAVE_WORKERS,
        **field_contour_cfg,
    )

    analytic.plot_centerline_results(
        sol_dsmc,
        fields_dsmc,
        diff_dsmc,
        contrib=None,
        save_dir=str(dsmc_dir),
        prefix=dsmc_prefix,
        show_centerline_plots=show_centerlines,
        save_centerline_plots=True,
        **centerline_cfg,
    )

    analytic.save_min_diagnostics(
        str(dsmc_dir),
        dsmc_prefix,
        sol_dsmc,
        fields_dsmc,
        emit_warning=False,
    )
    plot_dsmc_ya_min_inversion_region(
        sol_dsmc,
        fields_dsmc,
        dsmc_dir,
        dsmc_prefix,
        show_contours,
        contour_cfg,
    )

    if sol_surface_dsmc is not None and surface_rr_dsmc is not None:
        analytic.plot_surface_reaction_rate_along_wall(
            sol_surface_dsmc,
            surface_rr_dsmc,
            save_dir=str(dsmc_dir),
            prefix=dsmc_prefix,
            show_plots=show_surface,
            save_plots=True,
            **surface_cfg,
        )

    analytic.plot_field_contours(
        sol_analytic,
        fields_analytic,
        diff_analytic,
        contrib_analytic,
        save_dir=str(analytic_dir),
        prefix=analytic_prefix,
        show_field_contour_plots=show_contours,
        save_field_contour_plots=True,
        parallel_save_workers=PLOT_SAVE_WORKERS,
        **field_contour_cfg,
    )

    analytic.plot_centerline_results(
        sol_analytic,
        fields_analytic,
        diff_analytic,
        contrib_analytic,
        save_dir=str(analytic_dir),
        prefix=analytic_prefix,
        show_centerline_plots=show_centerlines,
        save_centerline_plots=True,
        **centerline_cfg,
    )

    analytic.save_min_diagnostics(str(analytic_dir), analytic_prefix, sol_analytic, fields_analytic)

    analytic.plot_inversion_region_contours(
        sol_analytic,
        fields_analytic,
        diff_analytic,
        save_dir=str(analytic_dir),
        prefix=analytic_prefix,
        show_inversion_region_plots=show_contours,
        save_inversion_region_plots=SAVE_INVERSION_REGION_PLOTS,
        save_inversion_region_txt=SAVE_INVERSION_REGION_TXT,
        levels=contour_cfg["levels"],
        dpi=contour_cfg["dpi"],
        figsize_contour=contour_cfg["figsize_contour"],
        font_size=contour_cfg["font_size"],
        tick_length=contour_cfg["tick_length"],
        tick_width=contour_cfg["tick_width"],
        box_linewidth=contour_cfg["box_linewidth"],
        cbar_shrink=contour_cfg["cbar_shrink"],
        cbar_fraction=contour_cfg["cbar_fraction"],
        cbar_pad=contour_cfg["cbar_pad"],
        cbar_aspect=contour_cfg["cbar_aspect"],
        cbar_nticks=contour_cfg["cbar_nticks"],
        cbar_decimals=contour_cfg["cbar_decimals"],
        ax_aspect=contour_cfg.get("ax_aspect", 1),
        x_nticks=contour_cfg.get("x_nticks", 6),
        minline_linewidth=contour_cfg["minline_linewidth"],
        inversion_product_range_source=contour_cfg["inversion_product_range_source"],
        inversion_product_negative_range_scale=contour_cfg["inversion_product_negative_range_scale"],
        inversion_product_positive_range_scale=contour_cfg["inversion_product_positive_range_scale"],
    )

    analytic.plot_surface_reaction_rate_along_wall(
        sol_analytic,
        surface_rr_analytic,
        save_dir=str(analytic_dir),
        prefix=analytic_prefix,
        show_plots=show_surface,
        save_plots=True,
        **surface_cfg,
    )
    analytic.plot_surface_flux_along_wall(
        sol_analytic,
        analytic.compute_surface_flux(sol_analytic),
        save_dir=str(analytic_dir),
        prefix=analytic_prefix,
        show_plots=show_surface,
        save_plots=True,
        **surface_cfg,
    )

    plot_compare_field_contours(
        sol_dsmc,
        fields_dsmc,
        diff_dsmc,
        sol_analytic,
        fields_analytic,
        diff_analytic,
        both_dir,
        compare_prefix,
        contour_cfg,
        parallel_save_workers=PLOT_SAVE_WORKERS,
    )
    plot_compare_centerlines(
        sol_dsmc,
        fields_dsmc,
        diff_dsmc,
        sol_analytic,
        fields_analytic,
        diff_analytic,
        both_dir,
        compare_prefix,
        centerline_cfg,
    )
    plot_compare_YA_min_curves(
        sol_dsmc,
        fields_dsmc,
        sol_analytic,
        fields_analytic,
        both_dir,
        compare_prefix,
        centerline_cfg,
    )
    save_inlet_nA_diagnostics_compare(
        sol_dsmc,
        fields_dsmc,
        sol_analytic,
        fields_analytic,
        both_dir,
        compare_prefix,
        centerline_cfg,
    )
    plot_compare_inversion_region_x(
        sol_dsmc,
        fields_dsmc,
        sol_analytic,
        fields_analytic,
        diff_analytic,
        both_dir,
        compare_prefix,
        contour_cfg,
    )
    if sol_surface_dsmc is not None and surface_rr_dsmc is not None:
        plot_surface_reaction_rate_compare(
            sol_surface_dsmc,
            surface_rr_dsmc,
            sol_analytic,
            surface_rr_analytic,
            both_dir,
            compare_prefix,
            show_surface,
            surface_cfg,
        )

    print(f"Case directory     : {case_cfg.case_dir}")
    if ensemble_info is None:
        print(f"Grid snapshot      : {grid_path}")
    else:
        print(f"Grid snapshot      : ensemble average ({ensemble_info.case_count} completed seed cases)")
        print(f"Ensemble source    : {ensemble_info.source_root}")
    print(f"Surface snapshot   : {surf_path if surf_path is not None else 'none (no_solid boundary case)'}")
    if timestep is not None:
        print(f"SPARTA timestep    : {timestep}")
    if ensemble_info is not None:
        timesteps = ", ".join(str(item.timestep) for item in ensemble_info.cases)
        seeds = ", ".join(str(item.seed) for item in ensemble_info.cases)
        print(f"Ensemble count     : {ensemble_info.case_count}")
        print(f"Ensemble seeds     : {seeds}")
        print(f"Ensemble timesteps : {timesteps}")
        print("Ensemble method    :")
        print(f"  {ensemble_info.method}")
    if surf_timestep is not None:
        print(f"Surface timestep   : {surf_timestep}")
    print(f"DSMC axes grid     : nx={sol_dsmc['x'].size}, ny={sol_dsmc['y'].size}")
    print(f"Analytic axes grid : nx={sol_analytic['x'].size}, ny={sol_analytic['y'].size}")
    print(f"Saved figures to   : {save_dir}")
    print(f"DSMC parameters    : fnum={case_cfg.fnum}, dt={case_cfg.timestep:.6e}, averaging={case_cfg.averaging}")
    if ensemble_info is None:
        print("Formula check against the user-provided J = rho_i (V_i - V_mix):")
        for key, value in formula_check.items():
            print(f"  {key} = {value:.6e}")
    else:
        print("Formula check      : skipped for ensemble fields; fluxes were recomputed from ensemble-mean moments.")
    if surface_rr_dsmc is not None:
        print("Surface reaction conversion:")
        print("  local_rate = count_avg_from_fix_ave_surf * fnum / (segment_length * dt)")
        print("  averaging is not divided again because fix ave/surf has already averaged the counts.")
        for wall, value in surface_rr_dsmc["total_number_rate_2D_unit_depth"].items():
            print(f"  Ndot_{wall:>6s} = {value:.6e} [#/s]")
    else:
        print("Surface reaction conversion: skipped (no_solid boundary output has no spatial surf dump).")
    if case_cfg.trench_mode.startswith("half"):
        print("Centerline note    : y=0 row was reconstructed by symmetry.")
        if surface_rr_dsmc is not None:
            print("Surface note       : bottom wall and negative-y end wall were reconstructed by symmetry.")


def run_postprocess_case_job(
    case_cfg: CaseConfig,
    grid_path: str,
    surf_path: str | None,
    figure_root: str,
    show_contours: bool,
    show_centerlines: bool,
    show_surface: bool,
    case_max_workers: int,
) -> str:
    ensemble_average = None
    output_case_cfg = case_cfg
    single_dsmc_suffix = SINGLE_DSMC_OUTPUT_SUFFIX_TEMPLATE.format(count=1)
    if USE_ENSEMBLE_AVERAGING:
        try:
            ensemble_average = build_ensemble_average_result(case_cfg)
        except NoEnsembleCasesError as exc:
            print(
                f"{case_cfg.study}/{case_cfg.case_name}: no ensemble average available "
                f"({exc}); using the single DSMC grid as n=1.",
                flush=True,
            )
            output_case_cfg = replace(
                case_cfg,
                output_suffix=f"{case_cfg.output_suffix}{single_dsmc_suffix}",
            )
        else:
            suffix = ENSEMBLE_AVERAGING_OUTPUT_SUFFIX_TEMPLATE.format(
                count=ensemble_average.info.case_count,
            )
            output_case_cfg = replace(case_cfg, output_suffix=f"{case_cfg.output_suffix}{suffix}")
    else:
        output_case_cfg = replace(
            case_cfg,
            output_suffix=f"{case_cfg.output_suffix}{single_dsmc_suffix}",
        )

    save_dir = resolve_save_dir(output_case_cfg, figure_root=Path(figure_root))
    plot_case(
        case_cfg,
        Path(grid_path),
        Path(surf_path) if surf_path else None,
        save_dir,
        show_contours,
        show_centerlines,
        show_surface,
        case_max_workers=case_max_workers,
        ensemble_average=ensemble_average,
    )
    if ensemble_average is not None:
        return (
            f"{case_cfg.study}/{case_cfg.case_name} ensemble_n={ensemble_average.info.case_count} "
            f"-> {save_dir}"
        )
    return f"{case_cfg.study}/{case_cfg.case_name} dsmc_n=1 -> {save_dir}"


def run_parallel_batch(
    study: str,
    batch_cfg: dict,
    discovered_by_name: dict[str, tuple[CaseConfig, Path, Path | None]],
    job_index_start: int,
    total_jobs: int,
    figure_root: Path,
) -> tuple[int, list[tuple[str, str]]]:
    case_names = tuple(batch_cfg["case_names"])
    parallel_cases = max(1, min(int(batch_cfg["parallel_cases"]), len(case_names)))
    case_max_workers = int(batch_cfg["case_max_workers"])

    print()
    print(
        f"Batch start: study={study}, cases={case_names}, "
        f"parallel_cases={parallel_cases}, case_max_workers={case_max_workers}"
    )

    failures: list[tuple[str, str]] = []
    futures = {}
    with ProcessPoolExecutor(max_workers=parallel_cases) as executor:
        for offset, case_name in enumerate(case_names):
            case_cfg, grid_path, surf_path = discovered_by_name[case_name]
            job_index = job_index_start + offset + 1
            print(f"[{job_index}/{total_jobs}] submit {study}/{case_name}")
            future = executor.submit(
                run_postprocess_case_job,
                case_cfg,
                str(grid_path),
                str(surf_path) if surf_path is not None else None,
                str(figure_root),
                SHOW_CONTOURS,
                SHOW_CENTERLINES,
                SHOW_SURFACE,
                case_max_workers,
            )
            futures[future] = (job_index, case_name)

        for future in as_completed(futures):
            job_index, case_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((case_name, str(exc)))
                print(f"[{job_index}/{total_jobs}] FAILED {study}/{case_name}: {exc}")
                continue
            print(f"[{job_index}/{total_jobs}] done {study}/{case_name}")
            print(f"  {result}")

    return job_index_start + len(case_names), failures


def run_study(
    study: str,
    case_selection,
    batch_defs,
    *,
    dataset_label: str = "solid",
    case_root: Path = CASE_STUDY_ROOT,
    figure_root: Path | None = None,
    output_suffix: str = "",
    require_surface_dump: bool = True,
    require_completion_marker: bool = False,
) -> None:
    figure_root = figure_output_root() if figure_root is None else Path(figure_root)
    discovered = discover_completed_cases(
        study_filter=study,
        case_root=case_root,
        output_suffix=output_suffix,
        require_surface_dump=require_surface_dump,
        require_completion_marker=require_completion_marker,
    )
    if not discovered:
        expected = "grid/surface outputs" if require_surface_dump else "completed grid outputs"
        print(f"{dataset_label} {study}: no completed DSMC cases with {expected} were found. Skipping.")
        return

    discovered_by_name = {
        case_cfg.case_name: (case_cfg, grid_path, surf_path)
        for case_cfg, grid_path, surf_path in discovered
    }
    requested_case_names = resolve_requested_case_names(study, case_selection, discovered_by_name)
    if not requested_case_names:
        print(f"{study}: no cases selected. Skipping.")
        return

    batch_cfgs = validate_study_batches(study, requested_case_names, batch_defs)

    print(f"{dataset_label} {study} study configuration")
    print(f"  cases                 = {requested_case_names}")
    print(f"  case_root             = {Path(case_root).resolve()}")
    print(f"  output_root           = {figure_root}")
    print(f"  output_suffix         = {output_suffix or '(none)'}")
    print(f"  analytic_base_grid    = {ANALYTIC_BASE_GRID}")
    print(f"  default_case_workers  = {DEFAULT_CASE_MAX_WORKERS}")
    print(f"  show_contours         = {SHOW_CONTOURS}")
    print(f"  show_centerlines      = {SHOW_CENTERLINES}")
    print(f"  show_surface          = {SHOW_SURFACE}")
    print(f"  plot_save_workers     = {PLOT_SAVE_WORKERS}")
    print(f"  ensemble_averaging    = {USE_ENSEMBLE_AVERAGING}")
    print(f"  single_dsmc_suffix    = {SINGLE_DSMC_OUTPUT_SUFFIX_TEMPLATE}")
    if USE_ENSEMBLE_AVERAGING:
        print(f"  ensemble_root         = {Path(ENSEMBLE_AVERAGING_ROOT).resolve()}")
        print(f"  ensemble_min_cases    = {ENSEMBLE_AVERAGING_MIN_CASES}")
        max_cases = ENSEMBLE_AVERAGING_MAX_CASES if ENSEMBLE_AVERAGING_MAX_CASES is not None else "all"
        print(f"  ensemble_max_cases    = {max_cases}")
        print(f"  ensemble_output_suffix= {ENSEMBLE_AVERAGING_OUTPUT_SUFFIX_TEMPLATE}")
        print(f"  ensemble_method       = {ENSEMBLE_AVERAGING_METHOD_TEXT}")
    print(f"  batches               = {batch_cfgs}")

    total_jobs = len(requested_case_names)
    job_index = 0
    failures: list[tuple[str, str]] = []

    for batch_cfg in batch_cfgs:
        job_index, batch_failures = run_parallel_batch(
            study=study,
            batch_cfg=batch_cfg,
            discovered_by_name=discovered_by_name,
            job_index_start=job_index,
            total_jobs=total_jobs,
            figure_root=figure_root,
        )
        failures.extend(batch_failures)

    print()
    print(f"{dataset_label} {study}: completed {total_jobs - len(failures)} / {total_jobs} cases.")
    if failures:
        print("Failures:")
        for case_name, message in failures:
            print(f"  - {dataset_label} {study}/{case_name}: {message}")
        raise SystemExit(1)

    replicated = []
    if dataset_label == "solid":
        replicated = replicate_duplicate_case_figures(study_filter=study)
    if replicated:
        print("Replicated duplicate figure directories:")
        for duplicate_key, canonical_key in replicated:
            print(f"  - {duplicate_key} <= {canonical_key}")


def main() -> None:
    any_run = False
    dataset = str(POSTPROCESS_DATASET).strip().lower()
    valid_datasets = {"solid", "no_solid", "both"}
    if dataset not in valid_datasets:
        raise ValueError(f"POSTPROCESS_DATASET must be one of {sorted(valid_datasets)}, got {POSTPROCESS_DATASET!r}")

    def run_enabled_dataset(study: str, case_selection, batch_defs) -> None:
        nonlocal any_run
        if dataset in {"solid", "both"}:
            any_run = True
            run_study(
                study,
                case_selection,
                batch_defs,
                dataset_label="solid",
                case_root=CASE_STUDY_ROOT,
                figure_root=figure_output_root(),
                output_suffix="",
                require_surface_dump=True,
                require_completion_marker=False,
            )
        if dataset in {"no_solid", "both"}:
            any_run = True
            run_study(
                study,
                case_selection,
                batch_defs,
                dataset_label="no_solid",
                case_root=NO_SOLID_CASE_STUDY_ROOT,
                figure_root=figure_output_root(NO_SOLID_FIGURE_ROOT_SUFFIX),
                output_suffix="",
                require_surface_dump=False,
                require_completion_marker=True,
            )

    if RUN_AR_STUDY:
        run_enabled_dataset("AR", AR_CASE_SELECTION, AR_STUDY_BATCHES)

    if RUN_KN_STUDY:
        run_enabled_dataset("Kn", KN_CASE_SELECTION, KN_STUDY_BATCHES)

    if RUN_PROBABILITY_STUDY:
        run_enabled_dataset("Probability", PROBABILITY_CASE_SELECTION, PROBABILITY_STUDY_BATCHES)

    if not any_run:
        print("All RUN_* flags are False, nothing to do.")


if __name__ == "__main__":
    main()

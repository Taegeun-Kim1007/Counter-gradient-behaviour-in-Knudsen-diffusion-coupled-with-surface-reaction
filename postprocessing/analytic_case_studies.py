#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analytic-only case-study runner.

Edit the configuration block below, then run:
    python3 analytic_case_studies.py
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402
import dsmc_postprocessing as post


ROOT = Path(__file__).resolve().parent


# ============================================================
# User configuration
# ============================================================

# Geometry / physics
WIDTH = 50.0e-9
TEMPERATURE = 273.15
MOLECULAR_WEIGHT_A = 28.016
MOLECULAR_WEIGHT_B = 28.016
MASS_A = 4.65e-26
MASS_B = 4.65e-26
TRENCH_DIRECTION = "x"
TRENCH_MODE = "half"

# Analytic solver
BASE_GRID = 200
N_QUAD_JET = 500
TOL = 1.0e-15
MAX_ITER = 300000
DEFAULT_CASE_MAX_WORKERS = 6
VERBOSE_SOLVER = True


def analytic_figure_root() -> Path:
    return ROOT / f"figure_analytic_grid{int(BASE_GRID)}"

# Plot visibility
SHOW_CONTOURS = False
SHOW_CENTERLINES = False
SHOW_SURFACE = False
SAVE_INVERSION_REGION_PLOTS = True
SAVE_INVERSION_REGION_TXT = True
PLOT_SAVE_WORKERS = 48

# AR study:
# Set exactly one of AR_STUDY_KN_FIXED / AR_STUDY_PRESSURE_FIXED.
RUN_AR_STUDY = True
AR_STUDY_ASPECT_RATIOS = (1, 3, 5, 10, 20)
AR_STUDY_REACTION_PROBABILITIES = (0.01, 0.05)
AR_STUDY_KN_FIXED = 100.0
AR_STUDY_PRESSURE_FIXED = None

# Parallel case execution:
# - Each batch runs sequentially.
# - Cases inside one batch run in parallel processes.
# - case_max_workers controls the internal thread count per case.
AR_STUDY_BATCHES = (
    {"aspect_ratios": (1, 3, 5), "parallel_cases": 48, "case_max_workers": 24},
    {"aspect_ratios": (10, 20), "parallel_cases": 5, "case_max_workers": 24},
    {"aspect_ratios": (50,), "parallel_cases": 1, "case_max_workers": 24},
    {"aspect_ratios": (100,), "parallel_cases": 1, "case_max_workers": 24},
)


def format_probability_folder(probability: float) -> str:
    return f"prob_{probability:g}"


def resolve_fixed_pressure_and_kn() -> tuple[float, float]:
    use_kn = AR_STUDY_KN_FIXED is not None
    use_pressure = AR_STUDY_PRESSURE_FIXED is not None
    if use_kn == use_pressure:
        raise ValueError("Set exactly one of AR_STUDY_KN_FIXED or AR_STUDY_PRESSURE_FIXED.")

    if use_pressure:
        pressure = float(AR_STUDY_PRESSURE_FIXED)
        knudsen_number = float(post.KN_REF * post.PRESSURE_REF_AT_KN_REF / pressure)
    else:
        knudsen_number = float(AR_STUDY_KN_FIXED)
        pressure = float(post.KN_REF * post.PRESSURE_REF_AT_KN_REF / knudsen_number)
    return pressure, knudsen_number


def build_ar_case_config(
    aspect_ratio: float,
    reaction_probability: float,
    pressure: float,
    knudsen_number: float,
) -> post.CaseConfig:
    width = float(WIDTH)
    depth = width * float(aspect_ratio)
    return post.CaseConfig(
        case_dir=ROOT,
        study="AR",
        case_name=f"{float(aspect_ratio):g}",
        aspect_ratio=float(aspect_ratio),
        width=width,
        depth=depth,
        half_width=0.5 * width,
        reaction_probability=float(reaction_probability),
        trench_direction=TRENCH_DIRECTION,
        trench_mode=TRENCH_MODE,
        pressure=float(pressure),
        temperature=float(TEMPERATURE),
        mass_A=float(MASS_A),
        mass_B=float(MASS_B),
        molecular_weight_A=float(MOLECULAR_WEIGHT_A),
        molecular_weight_B=float(MOLECULAR_WEIGHT_B),
        timestep=0.0,
        fnum=0.0,
        averaging=0,
        knudsen_number=float(knudsen_number),
    )


def analytic_grid_shape(case_cfg: post.CaseConfig) -> tuple[int, int]:
    nx = max(2, int(math.ceil(float(BASE_GRID) * float(case_cfg.aspect_ratio))) + 1)
    ny = int(BASE_GRID) + 1
    return nx, ny


def solve_analytic_case(case_cfg: post.CaseConfig, case_max_workers: int):
    R_A = post.R_UNIVERSAL / case_cfg.molecular_weight_A * 1000.0
    R_B = post.R_UNIVERSAL / case_cfg.molecular_weight_B * 1000.0
    nx, ny = analytic_grid_shape(case_cfg)
    number_density = case_cfg.pressure / post.BOLTZMANN_CONSTANT / case_cfg.temperature
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
        n_quad_jet=N_QUAD_JET,
        tol=TOL,
        max_iter=MAX_ITER,
        verbose=VERBOSE_SOLVER,
        max_workers=case_max_workers,
        log_prefix=log_prefix,
    )
    print(analytic.format_solver_status(sol, label=f"analytic {case_cfg.study}/{case_cfg.case_name}"), flush=True)
    fields_total, contrib = analytic.compute_full_fields(
        sol,
        case_cfg.mass_A,
        case_cfg.mass_B,
        max_workers=case_max_workers,
        verbose=VERBOSE_SOLVER,
        log_prefix=log_prefix,
    )
    diff_fields = analytic.compute_diffusive_mass_flux_fields(
        fields_total,
        case_cfg.mass_A,
        case_cfg.mass_B,
    )
    surface_flux = analytic.compute_surface_flux(sol)
    surface_rr = analytic.compute_surface_reaction_rate(sol)
    return sol, fields_total, diff_fields, contrib, surface_flux, surface_rr


def plot_analytic_case(case_cfg: post.CaseConfig, save_dir: Path, case_max_workers: int) -> None:
    (
        sol,
        fields_total,
        diff_fields,
        contrib,
        surface_flux,
        surface_rr,
    ) = solve_analytic_case(case_cfg, case_max_workers=case_max_workers)

    case_plot_cfg = analytic.get_case_plot_settings(case_cfg.aspect_ratio)
    contour_cfg = case_plot_cfg["contour"]
    field_contour_cfg = post.field_contour_kwargs(contour_cfg)
    centerline_cfg = case_plot_cfg["centerline"]
    surface_cfg = analytic.build_surface_plot_cfg(case_cfg.aspect_ratio, case_cfg.half_width)
    prefix = post.make_prefix(case_cfg, sol)

    save_dir.mkdir(parents=True, exist_ok=True)
    min_diagnostics = analytic.emit_YA_nA_min_diagnostics(
        prefix,
        sol,
        fields_total,
        emit_warning=True,
        print_summary=True,
    )

    analytic.plot_field_contours(
        sol,
        fields_total,
        diff_fields,
        contrib,
        save_dir=str(save_dir),
        prefix=prefix,
        show_field_contour_plots=SHOW_CONTOURS,
        save_field_contour_plots=True,
        parallel_save_workers=PLOT_SAVE_WORKERS,
        **field_contour_cfg,
    )
    analytic.plot_centerline_results(
        sol,
        fields_total,
        diff_fields,
        contrib,
        save_dir=str(save_dir),
        prefix=prefix,
        show_centerline_plots=SHOW_CENTERLINES,
        save_centerline_plots=True,
        **centerline_cfg,
    )
    analytic.save_min_diagnostics(
        str(save_dir),
        prefix,
        sol,
        fields_total,
        diagnostics=min_diagnostics,
        emit_warning=False,
        print_summary=False,
    )
    analytic.plot_inversion_region_contours(
        sol,
        fields_total,
        diff_fields,
        save_dir=str(save_dir),
        prefix=prefix,
        show_inversion_region_plots=SHOW_CONTOURS,
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
        sol,
        surface_rr,
        save_dir=str(save_dir),
        prefix=prefix,
        show_plots=SHOW_SURFACE,
        save_plots=True,
        **surface_cfg,
    )
    analytic.plot_surface_flux_along_wall(
        sol,
        surface_flux,
        save_dir=str(save_dir),
        prefix=prefix,
        show_plots=SHOW_SURFACE,
        save_plots=True,
        **surface_cfg,
    )


def run_analytic_case_job(case_cfg: post.CaseConfig, save_dir: str, case_max_workers: int) -> str:
    save_path = Path(save_dir)
    plot_analytic_case(case_cfg, save_path, case_max_workers=case_max_workers)
    return f"{case_cfg.study}/{case_cfg.case_name} -> {save_path}"


def validate_ar_batches(aspect_ratios: tuple[int, ...]) -> tuple[dict, ...]:
    configured = []
    covered: list[int] = []
    requested = set(int(v) for v in aspect_ratios)

    for batch in AR_STUDY_BATCHES:
        batch_ars = tuple(int(v) for v in batch["aspect_ratios"] if int(v) in requested)
        if not batch_ars:
            continue
        configured.append(
            {
                "aspect_ratios": batch_ars,
                "parallel_cases": int(batch.get("parallel_cases", len(batch_ars))),
                "case_max_workers": int(batch.get("case_max_workers", DEFAULT_CASE_MAX_WORKERS)),
            }
        )
        covered.extend(batch_ars)

    missing = sorted(requested.difference(covered))
    if missing:
        raise ValueError(f"Missing AR batch definitions for aspect ratios: {missing}")
    return tuple(configured)


def run_parallel_batch(
    probabilities: tuple[float, ...],
    pressure: float,
    knudsen_number: float,
    batch_cfg: dict,
    job_index_start: int,
    total_jobs: int,
) -> int:
    aspect_ratios = tuple(int(v) for v in batch_cfg["aspect_ratios"])
    jobs = [(float(probability), aspect_ratio) for probability in probabilities for aspect_ratio in aspect_ratios]
    parallel_cases = max(1, min(int(batch_cfg["parallel_cases"]), len(jobs)))
    case_max_workers = int(batch_cfg["case_max_workers"])

    print()
    print(
        f"Batch start: probs={probabilities}, AR={aspect_ratios}, "
        f"parallel_cases={parallel_cases}, case_max_workers={case_max_workers}"
    )

    futures = {}
    with ProcessPoolExecutor(max_workers=parallel_cases) as executor:
        for offset, (probability, aspect_ratio) in enumerate(jobs):
            case_cfg = build_ar_case_config(
                aspect_ratio=aspect_ratio,
                reaction_probability=probability,
                pressure=pressure,
                knudsen_number=knudsen_number,
            )
            save_dir = analytic_figure_root() / format_probability_folder(probability) / "AR" / str(aspect_ratio)
            job_index = job_index_start + offset + 1
            print(f"[{job_index}/{total_jobs}] submit AR={aspect_ratio}, prob={probability:g}")
            future = executor.submit(
                run_analytic_case_job,
                case_cfg,
                str(save_dir),
                case_max_workers,
            )
            futures[future] = (job_index, probability, aspect_ratio)

        for future in as_completed(futures):
            job_index, probability, aspect_ratio = futures[future]
            result = future.result()
            print(f"[{job_index}/{total_jobs}] done AR={aspect_ratio}, prob={probability:g}")
            print(f"  {result}")

    return job_index_start + len(jobs)


def run_ar_study() -> None:
    pressure, knudsen_number = resolve_fixed_pressure_and_kn()
    probabilities = tuple(float(v) for v in AR_STUDY_REACTION_PROBABILITIES)
    aspect_ratios = tuple(int(v) for v in AR_STUDY_ASPECT_RATIOS)
    batch_cfgs = validate_ar_batches(aspect_ratios)

    print("AR study configuration")
    print(f"  aspect_ratios = {aspect_ratios}")
    print(f"  probabilities = {probabilities}")
    print(f"  pressure      = {pressure:.12g} Pa")
    print(f"  Kn            = {knudsen_number:.12g}")
    print(f"  base_grid     = {BASE_GRID}")
    print(f"  default_case_max_workers = {DEFAULT_CASE_MAX_WORKERS}")
    print(f"  plot_save_workers = {PLOT_SAVE_WORKERS}")
    print(f"  batches       = {batch_cfgs}")

    total_jobs = len(probabilities) * len(aspect_ratios)
    job_index = 0

    for batch_cfg in batch_cfgs:
        job_index = run_parallel_batch(
            probabilities=probabilities,
            pressure=pressure,
            knudsen_number=knudsen_number,
            batch_cfg=batch_cfg,
            job_index_start=job_index,
            total_jobs=total_jobs,
        )


def main() -> None:
    if not RUN_AR_STUDY:
        print("RUN_AR_STUDY=False, nothing to do.")
        return
    run_ar_study()


if __name__ == "__main__":
    main()

"""Standalone reproduction of the four inclination reconstructions.

The script is intentionally self-contained: it reads one Gaia NSS/TI row,
rebuilds the ABFG covariance, compares two Gaussian-ABFG Monte-Carlo methods,
constructs the angular-momentum proposal on the sphere, folds it into U-space,
and applies the face-on rotation correction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_ID = 1988288559178163840
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "gaia_1988288559178163840_twobody_row.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
TI_PARAMS = ["a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"]
PARAM_ORDER = [
    ("ra", "ra_error"),
    ("dec", "dec_error"),
    ("parallax", "parallax_error"),
    ("pmra", "pmra_error"),
    ("pmdec", "pmdec_error"),
    ("a_thiele_innes", "a_thiele_innes_error"),
    ("b_thiele_innes", "b_thiele_innes_error"),
    ("f_thiele_innes", "f_thiele_innes_error"),
    ("g_thiele_innes", "g_thiele_innes_error"),
    ("c_thiele_innes", "c_thiele_innes_error"),
    ("h_thiele_innes", "h_thiele_innes_error"),
    ("period", "period_error"),
    ("t_periastron", "t_periastron_error"),
    ("eccentricity", "eccentricity_error"),
    ("center_of_mass_velocity", "center_of_mass_velocity_error"),
    ("semi_amplitude_primary", "semi_amplitude_primary_error"),
    ("semi_amplitude_secondary", "semi_amplitude_secondary_error"),
    ("mass_ratio", "mass_ratio_error"),
    ("fill_factor_primary", "fill_factor_primary_error"),
    ("fill_factor_secondary", "fill_factor_secondary_error"),
    ("inclination", "inclination_error"),
    ("arg_periastron", "arg_periastron_error"),
    ("temperature_ratio", "temperature_ratio_error"),
]
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone face-on Gaia 1988288559178163840 inclination demo."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help="One-row Gaia NSS/TI CSV. Defaults to the CSV shipped next to this script.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for plots and CSV outputs. Defaults to ./outputs next to this script.",
    )
    parser.add_argument("--source-id", type=int, default=SOURCE_ID, help="Used only if the CSV contains more than one row.")
    parser.add_argument("--n-dir", type=int, default=120000, help="Uniform sphere directions for U-space posterior.")
    parser.add_argument("--tau-grid", type=int, default=180, help="Grid size for marginalizing the in-plane angle.")
    parser.add_argument("--mc-draws", type=int, default=25000, help="Gaussian ABFG draws for diagonal/full-cov MC.")
    parser.add_argument("--posterior-samples", type=int, default=12000, help="Resampled directions for cloud plots.")
    parser.add_argument("--seed", type=int, default=20260428)
    return parser.parse_args()


def read_target_row(path: Path, source_id: int) -> pd.Series:
    df = pd.read_csv(path, low_memory=False)
    if "source_id" in df.columns and len(df) > 1:
        sub = df.loc[pd.to_numeric(df["source_id"], errors="coerce") == int(source_id)]
        if len(sub) == 0:
            raise ValueError(f"source_id={source_id} was not found in {path}")
        return sub.iloc[0]
    return df.iloc[0]


def active_param_names(row: pd.Series) -> list[str]:
    names: list[str] = []
    for param, err in PARAM_ORDER:
        if err in row and pd.notna(row.get(err)):
            names.append(param)
    return names


def parse_corr_vec(text: object) -> np.ndarray:
    if not isinstance(text, str):
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in NUM_RE.findall(text)], dtype=np.float64)


def build_corr_matrix_column_major(n: int, packed: np.ndarray) -> np.ndarray:
    need = n * (n - 1) // 2
    if len(packed) != need:
        raise ValueError(f"corr_vec length mismatch: got {len(packed)}, expected {need}")
    corr = np.eye(n, dtype=np.float64)
    k = 0
    for j in range(1, n):
        for i in range(j):
            rho = float(np.clip(packed[k], -0.999999, 0.999999))
            corr[i, j] = rho
            corr[j, i] = rho
            k += 1
    return corr


def extract_abfg_and_cov(row: pd.Series, cov_jitter: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = active_param_names(row)
    if not all(name in active for name in TI_PARAMS):
        raise ValueError("The input row does not contain complete A,B,F,G values and errors.")

    mu = np.array([float(row[name]) for name in TI_PARAMS], dtype=np.float64)
    sig = np.array([float(row[f"{name}_error"]) for name in TI_PARAMS], dtype=np.float64)

    packed = parse_corr_vec(row.get("corr_vec", None))
    if len(packed) == 0:
        cov = np.diag(sig * sig)
    else:
        corr_full = build_corr_matrix_column_major(len(active), packed)
        idx = [active.index(name) for name in TI_PARAMS]
        corr_ti = corr_full[np.ix_(idx, idx)]
        cov = corr_ti * np.outer(sig, sig)

    eigmin = float(np.min(np.linalg.eigvalsh(cov)))
    if eigmin <= 0.0:
        cov = cov + np.eye(4) * (abs(eigmin) + cov_jitter)
    return mu, sig, cov


def abfg_to_cosi_phi2(abfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(abfg, dtype=np.float64)
    a, b, f, g = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    k = 0.5 * (a * a + b * b + f * f + g * g)
    m = a * g - b * f
    disc = np.maximum(k * k - m * m, 0.0)
    a0_sq = k + np.sqrt(disc)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosi = m / a0_sq
    cosi = np.clip(cosi, -1.0, 1.0)

    q = (a * a + f * f) - (b * b + g * g)
    u = 2.0 * (a * b + f * g)
    phi2 = np.mod(np.arctan2(u, q), 2.0 * np.pi)
    return cosi, phi2


def standard_i_from_cosi(cosi: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.asarray(cosi, dtype=np.float64), -1.0, 1.0)))


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = normalize_weights(weights)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    return float(np.interp(float(q), np.cumsum(ws), xs, left=xs[0], right=xs[-1]))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        return np.full(max(len(w), 1), 1.0 / max(len(w), 1), dtype=np.float64)
    return w / total


def hist_density(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    total = float(np.sum(counts))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if total <= 0.0:
        return centers, np.zeros_like(centers)
    return centers, counts / (total * np.diff(edges))


def fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic near-uniform points on the unit sphere."""
    i = np.arange(int(n), dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = np.mod(i * golden_angle, 2.0 * math.pi)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def build_uniform_sphere_proposal(n_dir: int) -> np.ndarray:
    """Proposal grid for the unit angular-momentum vector n.

    This replaces a random proposal/MCMC step with a deterministic spherical
    quadrature grid, so the demo is reproducible and has no hidden sampler.
    """
    return fibonacci_sphere(n_dir)


def build_tau_proposal(tau_grid_size: int) -> np.ndarray:
    """Uniform proposal grid for the nuisance in-plane rotation angle tau."""
    return np.linspace(0.0, 2.0 * math.pi, int(tau_grid_size), endpoint=False)


def logmeanexp(log_values: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(log_values, axis=axis, keepdims=True)
    return np.squeeze(mx, axis=axis) + np.log(np.mean(np.exp(log_values - mx), axis=axis))


def abfg_template_from_n_tau(n_vec: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Unit-amplitude ABFG template k(n,tau).

    For each angular-momentum direction n and in-plane angle tau, this returns
    the ABFG vector for a0=1. The scale a0 is profiled out later, because only
    the angular direction is needed for the inclination reconstruction.
    """
    nx, ny, nz = n_vec[:, 0], n_vec[:, 1], n_vec[:, 2]
    omega = np.arctan2(nx, -ny)
    cO, sO = np.cos(omega)[:, None], np.sin(omega)[:, None]
    ci = nz[:, None]
    ct, st = np.cos(tau)[None, :], np.sin(tau)[None, :]
    return np.stack(
        [
            cO * ct - sO * st * ci,
            sO * ct + cO * st * ci,
            -cO * st - sO * ct * ci,
            -sO * st + cO * ct * ci,
        ],
        axis=-1,
    )


def profiled_abfg_loglike_for_direction_block(
    n_block: np.ndarray,
    tau: np.ndarray,
    y: np.ndarray,
    cov_inv: np.ndarray,
    ycy: float,
) -> np.ndarray:
    """Evaluate log p(y|n), marginalizing tau and profiling the amplitude a0."""
    templ = abfg_template_from_n_tau(n_block, tau)
    ky = np.einsum("...i,i->...", templ, cov_inv @ y)
    kk = np.einsum("...i,ij,...j->...", templ, cov_inv, templ)
    amp = np.maximum(ky / np.maximum(kk, 1e-300), 0.0)
    chi2 = ycy - 2.0 * amp * ky + amp * amp * kk
    return logmeanexp(-0.5 * chi2, axis=1)


def sphere_posterior_abfg(
    y: np.ndarray,
    cov: np.ndarray,
    n_dir: int,
    tau_grid_size: int,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior over angular-momentum directions from the ABFG likelihood."""
    directions = build_uniform_sphere_proposal(n_dir)
    tau = build_tau_proposal(tau_grid_size)
    cov_inv = np.linalg.inv(cov)
    ycy = float(y @ cov_inv @ y)
    log_like = np.empty(len(directions), dtype=np.float64)

    for start in range(0, len(directions), chunk_size):
        block = directions[start : start + chunk_size]
        log_like[start : start + len(block)] = profiled_abfg_loglike_for_direction_block(
            block,
            tau,
            y,
            cov_inv,
            ycy,
        )

    log_like -= np.max(log_like)
    weights = normalize_weights(np.exp(log_like))
    return directions, weights


def phi2_from_n(n_vec: np.ndarray) -> np.ndarray:
    omega_axial_deg = np.degrees(np.arctan2(n_vec[:, 0], -n_vec[:, 1])) % 180.0
    return np.mod(2.0 * omega_axial_deg, 360.0)


def build_u_space_from_angular_momentum(n_vec: np.ndarray) -> np.ndarray:
    """Fold angular-momentum directions into U=(sin i cos2Omega, sin i sin2Omega, cos i)."""
    phi2 = np.deg2rad(phi2_from_n(n_vec))
    rho = np.hypot(n_vec[:, 0], n_vec[:, 1])
    return np.column_stack([rho * np.cos(phi2), rho * np.sin(phi2), n_vec[:, 2]])


def ufold_from_n(n_vec: np.ndarray) -> np.ndarray:
    return build_u_space_from_angular_momentum(n_vec)


def rotation_axis_from_phi2(phi2_deg: float) -> np.ndarray:
    ph = math.radians(float(phi2_deg))
    axis = np.array([-math.sin(ph), math.cos(ph), 0.0], dtype=np.float64)
    return axis / np.linalg.norm(axis)


def rotate_cloud(u_cloud: np.ndarray, axis: np.ndarray, delta_deg: float) -> np.ndarray:
    v = np.asarray(u_cloud, dtype=np.float64)
    k = np.asarray(axis, dtype=np.float64)
    d = math.radians(float(delta_deg))
    c, s = math.cos(d), math.sin(d)
    return v * c + np.cross(np.broadcast_to(k, v.shape), v) * s + np.outer(v @ k, k) * (1.0 - c)


def method_from_cosi(cosi: np.ndarray, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    c = np.clip(np.asarray(cosi, dtype=np.float64), -1.0, 1.0)
    w = np.ones_like(c) if weights is None else np.asarray(weights, dtype=np.float64)
    return {"cosi": c, "i_deg": standard_i_from_cosi(c), "weights": normalize_weights(w)}


def gaussian_campbell_methods(y: np.ndarray, sig: np.ndarray, cov: np.ndarray, draws: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    diag = rng.normal(loc=y, scale=sig, size=(int(draws), 4))
    full = rng.multivariate_normal(mean=y, cov=cov, size=int(draws))

    out: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in [("diagonal_mc", diag), ("covariance_mc", full)]:
        cosi, _ = abfg_to_cosi_phi2(arr)
        mask = np.isfinite(cosi)
        out[name] = method_from_cosi(cosi[mask])
    return out


def uspace_methods(
    y: np.ndarray,
    cov: np.ndarray,
    n_dir: int,
    tau_grid: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_vec, weights = sphere_posterior_abfg(y, cov, n_dir=n_dir, tau_grid_size=tau_grid)
    u = build_u_space_from_angular_momentum(n_vec)
    ordinary = method_from_cosi(u[:, 2], weights)

    nominal_cosi, nominal_phi2 = abfg_to_cosi_phi2(y.reshape(1, 4))
    nominal_std = float(standard_i_from_cosi(nominal_cosi)[0])
    phi2_deg = float(np.rad2deg(nominal_phi2[0]) % 360.0)
    target_std = 45.0 if float(nominal_cosi[0]) >= 0.0 else 135.0
    delta = target_std - nominal_std

    rotated_u = rotate_cloud(u, rotation_axis_from_phi2(phi2_deg), delta)
    rotated_i_unwrapped = standard_i_from_cosi(rotated_u[:, 2]) - delta
    mask = np.isfinite(rotated_i_unwrapped) & (rotated_i_unwrapped >= 0.0) & (rotated_i_unwrapped <= 180.0)
    rotated_cosi = np.cos(np.deg2rad(rotated_i_unwrapped[mask]))
    rotated = method_from_cosi(rotated_cosi, weights[mask])

    meta = {
        "nominal_cosi": float(nominal_cosi[0]),
        "nominal_i_deg": nominal_std,
        "nominal_phi2_deg": phi2_deg,
        "rotation_target_i_deg": target_std,
        "rotation_delta_deg": float(delta),
        "rotated_kept_fraction": float(np.sum(weights[mask])),
    }
    return {"original_uspace": ordinary, "rotated_uspace": rotated}, meta, n_vec, u, rotated_u, weights


def summarize_methods(methods: dict[str, dict[str, np.ndarray]], meta: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, vals in methods.items():
        w = normalize_weights(vals["weights"])
        rows.append(
            {
                "method": name,
                "i_q16_deg": weighted_quantile(vals["i_deg"], 0.16, w),
                "i_q50_deg": weighted_quantile(vals["i_deg"], 0.50, w),
                "i_q84_deg": weighted_quantile(vals["i_deg"], 0.84, w),
                "cosi_q16": weighted_quantile(vals["cosi"], 0.16, w),
                "cosi_q50": weighted_quantile(vals["cosi"], 0.50, w),
                "cosi_q84": weighted_quantile(vals["cosi"], 0.84, w),
                "posterior_mass_abs_cosi_gt_0p99": float(np.sum(w[np.abs(vals["cosi"]) >= 0.99])),
                "n_samples": int(len(w)),
            }
        )
    df = pd.DataFrame(rows)
    for key, value in meta.items():
        df[key] = value
    return df


def plot_four_methods(methods: dict[str, dict[str, np.ndarray]], nominal_i_deg: float, out_png: Path) -> None:
    colors = {
        "diagonal_mc": "#8ecae6",
        "covariance_mc": "#6a3d9a",
        "original_uspace": "#1f77b4",
        "rotated_uspace": "#d62728",
    }
    labels = {
        "diagonal_mc": "diagonal MC",
        "covariance_mc": "covariance MC",
        "original_uspace": "original U-space",
        "rotated_uspace": "rotated U-space",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for name, vals in methods.items():
        w = normalize_weights(vals["weights"])
        x_i, y_i = hist_density(vals["i_deg"], w, np.linspace(0.0, 90.0, 121))
        x_c, y_c = hist_density(vals["cosi"], w, np.linspace(0.0, 1.0, 121))
        axes[0].plot(x_c, y_c, color=colors[name], lw=2.0, label=labels[name])
        axes[1].plot(x_i, y_i, color=colors[name], lw=2.0, label=labels[name])

    axes[1].axvline(nominal_i_deg, color="black", ls=":", lw=1.4, label="nominal ABFG")
    axes[0].set_xlabel(r"$\cos i$")
    axes[0].set_ylabel("density")
    axes[0].set_xlim(0.0, 1.0)
    axes[1].set_xlabel(r"$i$ [deg]")
    axes[1].set_ylabel("density")
    axes[1].set_xlim(0.0, 90.0)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def weighted_resample_indices(weights: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = normalize_weights(weights)
    return rng.choice(len(w), size=min(int(n), len(w)), replace=True, p=w)


def plot_uspace_cloud(u: np.ndarray, rotated_u: np.ndarray, weights: np.ndarray, n_keep: int, out_png: Path) -> None:
    idx = weighted_resample_indices(weights, n_keep, seed=20260429)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.7), constrained_layout=True)
    panels = [(0, 2, r"$U_x$", r"$U_z$"), (1, 2, r"$U_y$", r"$U_z$")]
    for ax, (ix, iy, xl, yl) in zip(axes, panels):
        ax.scatter(u[idx, ix], u[idx, iy], s=5, alpha=0.20, color="#1f77b4", edgecolors="none", label="original")
        ax.scatter(rotated_u[idx, ix], rotated_u[idx, iy], s=5, alpha=0.20, color="#d62728", edgecolors="none", label="rotated")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_density_csv(methods: dict[str, dict[str, np.ndarray]], out_csv: Path) -> None:
    rows = []
    edges_i = np.linspace(0.0, 90.0, 121)
    edges_c = np.linspace(0.0, 1.0, 121)
    for name, vals in methods.items():
        w = normalize_weights(vals["weights"])
        xi, yi = hist_density(vals["i_deg"], w, edges_i)
        xc, yc = hist_density(vals["cosi"], w, edges_c)
        rows.append(pd.DataFrame({"method": name, "space": "i_deg", "grid": xi, "density": yi}))
        rows.append(pd.DataFrame({"method": name, "space": "cosi", "grid": xc, "density": yc}))
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)


def write_uspace_proposal_csv(
    n_vec: np.ndarray,
    u: np.ndarray,
    rotated_u: np.ndarray,
    weights: np.ndarray,
    out_csv: Path,
) -> None:
    pd.DataFrame(
        {
            "n_x": n_vec[:, 0],
            "n_y": n_vec[:, 1],
            "n_z": n_vec[:, 2],
            "u_x": u[:, 0],
            "u_y": u[:, 1],
            "u_z_cosi": u[:, 2],
            "rotated_u_x": rotated_u[:, 0],
            "rotated_u_y": rotated_u[:, 1],
            "rotated_u_z": rotated_u[:, 2],
            "posterior_weight": normalize_weights(weights),
        }
    ).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = read_target_row(Path(args.input), args.source_id)
    y, sig, cov = extract_abfg_and_cov(row)

    gauss = gaussian_campbell_methods(y, sig, cov, args.mc_draws, args.seed)
    uspace, meta, n_vec, u, rotated_u, weights = uspace_methods(y, cov, args.n_dir, args.tau_grid)
    methods = {**gauss, **uspace}

    summary = summarize_methods(methods, meta)
    summary.to_csv(out_dir / "method_summary.csv", index=False)
    write_density_csv(methods, out_dir / "posterior_density_curves.csv")
    write_uspace_proposal_csv(n_vec, u, rotated_u, weights, out_dir / "uspace_proposal_and_rotation.csv")
    plot_four_methods(methods, meta["nominal_i_deg"], out_dir / "1988_four_methods_i_cosi.png")
    plot_uspace_cloud(u, rotated_u, weights, args.posterior_samples, out_dir / "1988_original_vs_rotated_uspace.png")

    config = {
        "source_id": int(args.source_id),
        "input": str(Path(args.input)),
        "output_dir": str(out_dir),
        "n_dir": int(args.n_dir),
        "tau_grid": int(args.tau_grid),
        "mc_draws": int(args.mc_draws),
        "posterior_samples": int(args.posterior_samples),
        "seed": int(args.seed),
        "abfg": dict(zip(["A", "B", "F", "G"], [float(x) for x in y])),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary[["method", "i_q16_deg", "i_q50_deg", "i_q84_deg", "cosi_q50"]].to_string(index=False))
    print(f"\nWrote demo outputs to: {out_dir}")


if __name__ == "__main__":
    main()

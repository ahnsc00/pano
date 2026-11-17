"""PANO와 CRTM을 실행해 Figure 1의 세 패널을 생성한다."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from pano import PANO
from xue_crtm import XueCRTM

ROOT = Path(__file__).resolve().parent
METHODS = ("pano", "xue_crtm")
PAPER = {
    "horizon": 1_000_000,
    "chunk_size": 2000,
    "actions": 128,
    "scenarios": {
        "synthetic": {
            "scenario": "exact_d8_K128",
            "kind": "exact",
            "context_key": "synthetic_contexts",
            "theta_init": [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "theta_star": None,
            "pano": {
                "radius": 0.1,
                "score_low": -1.6,
                "score_high": -1.4,
                "delta": 0.01,
                "ridge_scale": 1e-08,
                "beta_scale": 0.01,
            },
            "crtm": {
                "epsilon": 0.17999999999999997,
                "moment_bound_u": 6.363636363636364,
                "kappa": 2.7777777777777772,
                "parameter_radius_S": 0.1,
                "mean_bound_U": 3.5000000000000004,
                "confidence_scale": 1e-08,
                "confidence_mode": "experimental",
                "delta": 0.01,
            },
            "link_offset": -1.5,
            "seed_start": 6000,
            "seed_count": 20,
            "checkpoint_divisor": 40,
            "figure_name": "synthetic_d8_K128.pdf",
            "perturbation_norm": 0.09000000000000001,
        },
        "fre": {
            "scenario": "fre_d8_K128",
            "kind": "semi",
            "context_key": "fre_contexts",
            "theta_init": [-2.5276135083360116, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "theta_star": [
                -2.5276135083360116, 0.013112526552204873, 0.026741376794832624,
                -0.0849474098362863, -0.03952810756429257, -0.010198068458980886,
                -0.045842495380613656, -0.018876033833450018,
            ],
            "pano": {
                "radius": 0.1271642780924181,
                "score_low": -1.3909710322604238,
                "score_high": -1.1366424760755878,
                "delta": 0.01,
                "ridge_scale": 1e-08,
                "beta_scale": 0.01,
            },
            "crtm": {
                "epsilon": 0.0614891142340145,
                "moment_bound_u": 15.124306461122572,
                "kappa": 6.542004494011121,
                "parameter_radius_S": 0.1271642780924181,
                "mean_bound_U": 8.318368553617404,
                "confidence_scale": 1e-08,
                "confidence_mode": "experimental",
                "delta": 0.01,
            },
            "link_offset": -1.2638067541680058,
            "seed_start": 16000,
            "seed_count": 20,
            "checkpoint_divisor": 50,
            "figure_name": "fre_semisynthetic.pdf",
        },
        "mendeley": {
            "scenario": "mendeley_d8_K128",
            "kind": "semi",
            "context_key": "mendeley_contexts",
            "theta_init": [-3.387616713678815, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "theta_star": [
                -3.387616713678815, -0.17656054013798528, 0.059237694007106004,
                -0.010630794100886083, -0.1276397830508575, 0.005685139159809919,
                0.16935523250623785, 0.17822324558824,
            ],
            "pano": {
                "radius": 0.38411497308362313,
                "score_low": -2.077923329923031,
                "score_high": -1.3096933837557845,
                "delta": 0.01,
                "ridge_scale": 1e-08,
                "beta_scale": 0.01,
            },
            "crtm": {
                "epsilon": 0.13936202269010303,
                "moment_bound_u": 7.689091283963583,
                "kappa": 0.8606454096411593,
                "parameter_radius_S": 0.38411497308362313,
                "mean_bound_U": 4.229000206179968,
                "confidence_scale": 1e-08,
                "confidence_mode": "experimental",
                "delta": 0.01,
            },
            "link_offset": -1.6938083568394076,
            "seed_start": 16000,
            "seed_count": 20,
            "checkpoint_divisor": 50,
            "figure_name": "mendeley_semisynthetic.pdf",
        },
    },
    "inputs": {
        "synthetic_contexts": {
            "shape": [128, 8],
            "dtype": "float64",
            "sha256": "e9146761f02b4276b73c55c573a21bb1ddbfa169758bfc9a439987b59d23a1eb",
        },
        "fre_contexts": {
            "shape": [1909, 8],
            "dtype": "float64",
            "sha256": "7a45e967cae6a47d2503cb14cea6513ca3d8b051a4af60f4b89ef22bee8db0d0",
        },
        "mendeley_contexts": {
            "shape": [2694, 8],
            "dtype": "float64",
            "sha256": "a1408688c2038dfdafe9735bbdb0911ed7fcf3b8350908b91046d531267b6c32",
        },
    },
    "inputs_file_sha256": "ddc6951bfbfd2f568a41dd6638cc1285f45ba7855a1d22c1ab5b85a7eb4ca850",
}
CONFIG = {}
CONTEXTS = {}


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def synthetic_contexts():
    signs = np.array([[1.0 if (row >> j) & 1 else -1.0 for j in range(7)]
                      for row in range(128)])
    contexts = np.empty((128, 8), dtype=float)
    contexts[:, 0] = 0.5
    contexts[:, 1:] = math.sqrt(1.0 - 0.5 * 0.5) * (signs / math.sqrt(7))
    return contexts


def load_inputs(labels=None):
    config = PAPER
    labels = set(config["scenarios"]) if labels is None else set(labels)
    if not labels or not labels.issubset(config["scenarios"]):
        raise ValueError("unknown or empty scenario selection")
    contexts = {"synthetic_contexts": synthetic_contexts()}
    if labels - {"synthetic"}:
        path = ROOT / "paper_inputs.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["inputs_file_sha256"]:
            raise ValueError("paper input file checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"fre_contexts", "mendeley_contexts"}:
                raise ValueError("unexpected input arrays")
            contexts.update({key: np.ascontiguousarray(data[key]) for key in data.files})
    for key, value in contexts.items():
        expected = config["inputs"][key]
        if (list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]
                or array_hash(value) != expected["sha256"] or not np.isfinite(value).all()):
            raise ValueError(f"invalid context pool: {key}")
    return config, contexts

def initialize_worker(config, contexts):
    global CONFIG, CONTEXTS
    CONFIG, CONTEXTS = config, contexts


def mean_reward(score):
    z = np.asarray(score)
    return 1.0 * z / (z + 1.0)


def simulate(task):
    label, seed, horizon = task
    started = time.monotonic()
    scene = CONFIG["scenarios"][label]
    pool = CONTEXTS[scene["context_key"]]
    center = np.asarray(scene["theta_init"], dtype=float)
    pano = PANO(theta_init=center, seed=seed, **scene["pano"])

    def link(score):
        return mean_reward(np.asarray(score) + scene["link_offset"])

    crtm = XueCRTM(dimension=len(center), link=link, horizon=horizon,
                   seed=200_000 + seed, **scene["crtm"])
    exact = scene["kind"] == "exact"
    reward_rng = np.random.default_rng(seed if exact else 710_000 + seed)
    context_rng = np.random.default_rng(610_000 + seed)
    if exact:
        direction = np.zeros(len(center))
        direction[1:] = reward_rng.choice(np.array([-1.0, 1.0]), size=len(center) - 1) / math.sqrt(len(center) - 1)
        theta_star = center + scene["perturbation_norm"] * direction
        fixed_scores = pool @ theta_star
        fixed_means = mean_reward(fixed_scores)
        fixed_oracle = float(np.max(fixed_means))
        candidate_digest = array_hash(pool)
    else:
        theta_star = np.asarray(scene["theta_star"], dtype=float)

    interval = max(1, horizon // scene["checkpoint_divisor"])
    checkpoints = {1, horizon, *range(interval, horizon + 1, interval)}
    cumulative = {method: 0.0 for method in METHODS}
    curves = []
    context_hash, reward_hash = hashlib.sha256(), hashlib.sha256()
    for start in range(0, horizon, CONFIG["chunk_size"]):
        size = min(CONFIG["chunk_size"], horizon - start)
        if exact:
            logs = reward_rng.exponential(scale=1.0 / (-fixed_scores[None, :]),
                                          size=(size, CONFIG["actions"]))
        else:
            indices = np.vstack([context_rng.choice(len(pool), size=CONFIG["actions"], replace=False)
                                 for _ in range(size)]).astype(np.int64, copy=False)
            contexts = pool[indices]
            scores = contexts @ theta_star
            if float(np.max(scores)) >= -1.05:
                raise ValueError("generated scores left the configured finite-mean domain")
            logs = -np.log(reward_rng.uniform(np.nextafter(0.0, 1.0), 1.0, size=scores.shape)) / (-scores)
            means = mean_reward(scores)
            oracle = np.max(means, axis=1)
            context_hash.update(np.ascontiguousarray(indices).view(np.uint8))
        reward_hash.update(np.ascontiguousarray(logs).view(np.uint8))
        for j in range(size):
            x = pool if exact else contexts[j]
            mu = fixed_means if exact else means[j]
            best = fixed_oracle if exact else float(oracle[j])
            a, b = pano.select(x), crtm.select(x)
            cumulative["pano"] += best - float(mu[a])
            cumulative["xue_crtm"] += best - float(mu[b])
            pano.update(x[a], float(logs[j, a]))
            crtm.update(x[b], math.exp(float(logs[j, b])))
            t = start + j + 1
            if t in checkpoints:
                curves.extend({"scenario": scene["scenario"], "seed": seed, "method": method,
                               "round": t, "cumulative_regret": cumulative[method]} for method in METHODS)
    if not exact:
        candidate_digest = context_hash.hexdigest()
    rows = [{"scenario": scene["scenario"], "seed": seed, "method": method,
             "horizon": horizon, "terminal_regret": cumulative[method],
             "candidate_sha256": candidate_digest, "reward_sha256": reward_hash.hexdigest()}
            for method in METHODS]
    if not all(math.isfinite(row["terminal_regret"]) for row in rows):
        raise ValueError("nonfinite experiment result")
    return {"label": label, "seed": seed, "rows": rows, "curves": curves,
            "elapsed_seconds": time.monotonic() - started}


def draw_panel(frame, destination):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    colors = {"pano": "#c62828", "xue_crtm": "#1565c0"}
    labels = {"pano": "PANO", "xue_crtm": "CRTM"}
    fig, ax = plt.subplots(figsize=(5.25, 3.95))
    top = 0.0
    for method in METHODS:
        stats = frame[frame.method == method].groupby("round").value.agg(["mean", "sem"])
        x = stats.index.to_numpy(float)
        mean = stats["mean"].to_numpy(float)
        sem = stats["sem"].fillna(0.0).to_numpy(float)
        ax.plot(x, mean, color=colors[method], linewidth=3.1, label=labels[method])
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[method], alpha=0.12, linewidth=0)
        top = max(top, float(np.max(mean + 1.96 * sem)))
    horizon = int(frame["round"].max())
    ax.set_xticks(np.linspace(0, horizon, 6))
    if horizon >= 1_000_000:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1e6:.1f}"))
        ax.set_xlabel(r"Round ($10^6$)", fontsize=20, labelpad=7)
    else:
        ax.set_xlabel("Round", fontsize=20, labelpad=7)
    exponent = int(math.floor(math.log10(max(float(top), 1e-12))))
    scale = 10.0 ** exponent
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / scale:.1f}"))
    ax.set_ylabel(rf"Cumulative regret ($10^{{{exponent}}}$)", fontsize=20, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=17, width=1.3, length=5)
    ax.grid(alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#bbbbbb",
              fontsize=16, handlelength=2.4, borderpad=0.35, labelspacing=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.20, top=0.985)
    fig.savefig(destination, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(destination.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def export_results(output, config, rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import pandas as pd

    by_seed = pd.DataFrame(rows).sort_values(["scenario", "seed", "method"])
    curve_frame = pd.DataFrame(curves).sort_values(["scenario", "seed", "method", "round"])
    by_seed.to_csv(output / "by_seed.csv", index=False)
    curve_frame.to_csv(output / "curves.csv", index=False)
    summary = by_seed.groupby(["scenario", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_regret=("terminal_regret", "mean"),
        se_regret=("terminal_regret", "sem"))
    summary.to_csv(output / "summary.csv", index=False)
    for scene in config["scenarios"].values():
        frame = curve_frame[curve_frame.scenario == scene["scenario"]]
        if not frame.empty:
            draw_panel(frame.rename(columns={"cumulative_regret": "value"}), output / "figures" / scene["figure_name"])
    return summary.astype(object).where(summary.notna(), None).to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper")
    parser.add_argument("--horizon", type=int, default=PAPER["horizon"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scenarios", nargs="+", choices=("synthetic", "fre", "mendeley"),
                        default=["synthetic", "fre", "mendeley"])
    args = parser.parse_args()
    if args.horizon <= 0 or not 1 <= args.seeds <= 20 or args.workers <= 0:
        parser.error("horizon/workers must be positive; seeds must be 1..20")
    labels = list(dict.fromkeys(args.scenarios))
    config, contexts = load_inputs(labels)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty")
    if output == ROOT or output in ROOT.parents:
        parser.error("output cannot be the source directory or its parents")
    output.mkdir(parents=True, exist_ok=True)
    (output / "jobs").mkdir()
    jobs = [(label, config["scenarios"][label]["seed_start"] + i, args.horizon)
            for label in labels for i in range(args.seeds)]
    protocol = {"horizon": args.horizon, "seeds_per_scenario": args.seeds, "scenarios": labels,
                "workers": args.workers, "jobs": len(jobs),
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "inputs_sha256": config["inputs_file_sha256"] if set(labels) - {"synthetic"} else None,
                "pano_sha256": hashlib.sha256((ROOT / "pano.py").read_bytes()).hexdigest(),
                "crtm_sha256": hashlib.sha256((ROOT / "xue_crtm.py").read_bytes()).hexdigest()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, curves = [], []
    print(f"PANO/CRTM: {len(jobs)} jobs, {args.horizon:,} steps per seed", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(config, contexts)) as pool:
        futures = {pool.submit(simulate, job): job for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = output / "jobs" / f"{result['label']}_seed{result['seed']}.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            rows.extend(result["rows"])
            curves.extend(result["curves"])
            (output / "progress.json").write_text(json.dumps({"completed": count, "total": len(jobs)}) + "\n")
            print(f"{count}/{len(jobs)} {result['label']} seed={result['seed']} "
                  f"PANO={result['rows'][0]['terminal_regret']:.3f} "
                  f"CRTM={result['rows'][1]['terminal_regret']:.3f}", flush=True)
    summary = export_results(output, config, rows, curves)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved curves and figures: {output}", flush=True)


if __name__ == "__main__":
    main()

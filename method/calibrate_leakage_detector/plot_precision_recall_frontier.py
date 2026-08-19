import argparse
import json
from pathlib import Path


def parse_args():
    base = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, nargs="+", default=[base / "all_experiments.jsonl"])
    parser.add_argument("--output-html", type=Path, default=base / "precision_recall_frontier.html")
    parser.add_argument("--output-png", type=Path, default=base / "precision_recall_frontier.png")
    parser.add_argument("--model", default=None, help="Optional model key filter.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def experiment_name(path: Path) -> str:
    return path.parent.name if path.parent.name != "output" else path.stem


def read_experiments(paths: list[Path], model: str | None = None) -> dict[str, list[dict]]:
    experiments = {}
    for path in paths:
        rows = read_jsonl(path)
        if model:
            rows = [row for row in rows if row.get("model") == model]
        rows = [row for row in rows if "precision" in row and "recall" in row]
        if rows:
            name = experiment_name(path)
            experiments[name] = [{**row, "experiment": name} for row in rows]
    return experiments


def pareto_frontier(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: (float(r["recall"]), float(r["precision"])))
    frontier, best_precision = [], -1.0
    for row in reversed(rows):
        precision = float(row["precision"])
        if precision > best_precision:
            frontier.append(row)
            best_precision = precision
    return sorted(frontier, key=lambda r: float(r["recall"]))


def hover_text(row: dict) -> str:
    fields = [
        ("experiment", row.get("experiment")),
        ("model", row.get("model")),
        ("top_k", row.get("top_k")),
        ("window_size", row.get("window_size")),
        ("threshold", row.get("threshold")),
        ("span_precision", row.get("precision")),
        ("span_recall", row.get("recall")),
        ("full_recall", row.get("full_recall")),
        ("token_precision", row.get("token_precision")),
        ("avg_pred_spans", row.get("avg_pred_spans_per_sample")),
    ]
    return "<br>".join(f"{key}: {value}" for key, value in fields if value is not None)


def plot_html(experiments: dict[str, list[dict]], path: Path):
    import plotly.graph_objects as go

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    for i, (name, rows) in enumerate(experiments.items()):
        color = colors[i % len(colors)]
        sizes = [7 + 3 * 1 ** 0.5 for r in rows]
        frontier = pareto_frontier(rows)

        fig.add_trace(
            go.Scatter(
                x=[r["recall"] for r in rows],
                y=[r["precision"] for r in rows],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.58,
                    line=dict(width=0.4, color="white"),
                ),
                text=[hover_text(r) for r in rows],
                hoverinfo="text",
                legendgroup=name,
                name=name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["recall"] for r in frontier],
                y=[r["precision"] for r in frontier],
                mode="lines",
                line=dict(color=color, width=2),
                text=[hover_text(r) for r in frontier],
                hoverinfo="text",
                legendgroup=name,
                name=f"{name} frontier",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Precision-Recall Frontier Comparison",
        xaxis_title="span recall",
        yaxis_title="span precision",
        template="plotly_white",
    )
    fig.write_html(path)


def plot_png(experiments: dict[str, list[dict]], path: Path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, rows in experiments.items():
        color = None
        x = [float(r["recall"]) for r in rows]
        y = [float(r["precision"]) for r in rows]
        s = [18 + 8 * 1 ** 0.5 for r in rows]
        scatter = ax.scatter(x, y, s=s, alpha=0.58, edgecolors="white", linewidths=0.3, label=name)
        color = scatter.get_facecolors()[0]

        frontier = pareto_frontier(rows)
        ax.plot(
            [float(r["recall"]) for r in frontier],
            [float(r["precision"]) for r in frontier],
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("span recall")
    ax.set_ylabel("span precision")
    ax.set_title("Precision-Recall Frontier Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main():
    args = parse_args()
    experiments = read_experiments(args.input_path, args.model)
    if not experiments:
        raise ValueError("No experiment rows with precision/recall were found.")

    try:
        plot_html(experiments, args.output_html)
        print(f"wrote {args.output_html}")
    except ImportError:
        print("plotly is not installed; skipped interactive HTML.")

    plot_png(experiments, args.output_png)
    print(f"wrote {args.output_png}")
    for name, rows in experiments.items():
        print(f"{name}: rows={len(rows)} frontier_points={len(pareto_frontier(rows))}")


if __name__ == "__main__":
    main()

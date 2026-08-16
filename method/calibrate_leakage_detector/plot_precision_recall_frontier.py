import argparse
import json
from pathlib import Path


def parse_args():
    base = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=base / "all_experiments.jsonl")
    parser.add_argument("--output-html", type=Path, default=base / "precision_recall_frontier.html")
    parser.add_argument("--output-png", type=Path, default=base / "precision_recall_frontier.png")
    parser.add_argument("--model", default=None, help="Optional model key filter.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def plot_html(rows: list[dict], frontier: list[dict], path: Path):
    import plotly.graph_objects as go

    path.parent.mkdir(parents=True, exist_ok=True)
    color = [float(r.get("avg_pred_spans_per_sample", 0.0)) for r in rows]
    sizes = [7 + 3 * float(r.get("top_k", 1)) ** 0.5 for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r["recall"] for r in rows],
            y=[r["precision"] for r in rows],
            mode="markers",
            marker=dict(
                size=sizes,
                color=color,
                colorscale="Viridis",
                colorbar=dict(title="avg spans/sample"),
                opacity=0.72,
                line=dict(width=0.4, color="white"),
            ),
            text=[hover_text(r) for r in rows],
            hoverinfo="text",
            name="all configs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[r["recall"] for r in frontier],
            y=[r["precision"] for r in frontier],
            mode="lines+markers",
            line=dict(color="crimson", width=2),
            marker=dict(size=8, color="crimson"),
            text=[hover_text(r) for r in frontier],
            hoverinfo="text",
            name="precision-recall frontier",
        )
    )
    fig.update_layout(
        title="Precision-Recall Frontier over top-k / threshold / window-size",
        xaxis_title="span recall",
        yaxis_title="span precision",
        template="plotly_white",
    )
    fig.write_html(path)


def plot_png(rows: list[dict], frontier: list[dict], path: Path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = [float(r["recall"]) for r in rows]
    y = [float(r["precision"]) for r in rows]
    c = [float(r.get("avg_pred_spans_per_sample", 0.0)) for r in rows]
    s = [18 + 8 * float(r.get("top_k", 1)) ** 0.5 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x, y, c=c, s=s, cmap="viridis", alpha=0.72, edgecolors="white", linewidths=0.3)
    ax.plot(
        [float(r["recall"]) for r in frontier],
        [float(r["precision"]) for r in frontier],
        color="crimson",
        marker="o",
        linewidth=2,
        label="frontier",
    )
    ax.set_xlabel("span recall")
    ax.set_ylabel("span precision")
    ax.set_title("Precision-Recall Frontier")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="avg predicted spans/sample")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main():
    args = parse_args()
    rows = read_jsonl(args.input_path)
    if args.model:
        rows = [row for row in rows if row.get("model") == args.model]
    rows = [row for row in rows if "precision" in row and "recall" in row]
    if not rows:
        raise ValueError("No experiment rows with precision/recall were found.")

    frontier = pareto_frontier(rows)
    try:
        plot_html(rows, frontier, args.output_html)
        print(f"wrote {args.output_html}")
    except ImportError:
        print("plotly is not installed; skipped interactive HTML.")

    plot_png(rows, frontier, args.output_png)
    print(f"wrote {args.output_png}")
    print(f"rows={len(rows)} frontier_points={len(frontier)}")


if __name__ == "__main__":
    main()

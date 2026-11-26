"""Static scientific plots backed by the exported graph tables, never fabricated curves."""
from pathlib import Path

import numpy as np
import pandas as pd


def render_plots(result, output, *, qini=None, strategy=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True)

    def canvas():
        return plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)

    def finish(fig, name):
        fig.savefig(output / f"{name}.png", dpi=160)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    def unavailable(ax, text):
        ax.text(.5, .5, "Unavailable\n"+str(text or "Required observations unavailable"), ha="center", va="center", wrap=True, transform=ax.transAxes)
        ax.set_axis_off()

    fig, ax = canvas()
    data = result.graph_data["churn_risk"].churn_prob.dropna()
    if len(data):
        ax.hist(data, bins=np.linspace(0, 1, 11), color="#3b6ea8", edgecolor="white")
        ax.set(xlabel="Predicted 4-week churn probability", ylabel="Customers", xlim=(0, 1),
               title=f"Churn risk: {len(data)}/{len(result.customers)} predictions available")
    else:
        unavailable(ax, "No aligned churn predictions")
    finish(fig, "churn_risk_distribution")

    fig, ax = canvas()
    data = result.graph_data["uplift_quadrants"]
    colors = {"Persuadables": "#2a9d8f", "Sure Things": "#3b6ea8", "Lost Causes": "#c49b42", "Sleeping Dogs": "#c65353"}
    drawn = 0
    for name, color in colors.items():
        group = data.loc[data.response_group.eq(name).fillna(False)].dropna(subset=["p0", "uplift_score"])
        if len(group):
            ax.scatter(group.p0, group.uplift_score, s=20, alpha=.75, label=name, color=color)
            drawn += len(group)
    if drawn:
        config = result.metadata["configuration"]
        ax.axvline(config["baseline_purchase_threshold"], color="grey", lw=.8)
        for cut in (-config["uplift_tolerance"], config["uplift_tolerance"]):
            ax.axhline(cut, color="grey", lw=.8, linestyle="--")
        ax.set(xlabel="Purchase probability without coupon (p0)", ylabel="Purchase p1 - p0 (4 weeks)", xlim=(0, 1),
               title="Operational uplift groups; not latent response truth")
        ax.legend(fontsize=8)
    else:
        unavailable(ax, "No aligned uplift and churn predictions")
    finish(fig, "uplift_quadrants")

    fig, ax = canvas()
    data = result.graph_data["segments6"].dropna(subset=["uplift_score", "clv", "churn_prob", "segment"])
    if len(data):
        shapes = ["o", "s", "^", "D", "P", "X"]
        artist = None
        for marker, (name, group) in zip(shapes, data.groupby("segment", sort=True)):
            artist = ax.scatter(group.uplift_score, group.clv, c=group.churn_prob, cmap="viridis", vmin=0, vmax=1,
                                marker=marker, alpha=.8, s=25, label=name)
        fig.colorbar(artist, ax=ax, label="Predicted churn probability")
        ax.set(xlabel="Purchase uplift (4 weeks)", ylabel="Predicted paid CLV (52 weeks; declared minor units)",
               title="Six configured business segments")
        ax.legend(fontsize=6)
    else:
        unavailable(ax, "Aligned monetary CLV/segment predictions unavailable")
    finish(fig, "segments6_scatter")

    fig, ax = canvas()
    data = qini.get("frame") if qini else None
    if qini and qini.get("status") == "available" and data is not None and len(data):
        for name, group in data.groupby("model", sort=False):
            ax.plot(group.fraction, group.cumulative_gain, label=name)
        reference = next(iter(data.groupby("model", sort=False)))[1]
        ax.plot(reference.fraction, reference.random_targeting_gain, linestyle="--", color="black", label="Random targeting reference")
        ax.set(xlabel="Targeted customer fraction", ylabel="IPW cumulative purchase gain / cohort size", title="Held-out factual Qini curves")
        ax.legend(fontsize=8)
    else:
        unavailable(ax, (qini or {}).get("reason", "Held-out randomized factual purchase outcomes not supplied"))
    finish(fig, "qini_curves")

    fig, ax = canvas()
    data = result.what_if.dropna(subset=["expected_roi"])
    if len(data):
        ax.plot(data.total_budget, data.expected_roi, "o-", color="#2a9d8f")
        for _, row in data.iterrows():
            ax.annotate(f"{int(row.budget_ratio*100)}%", (row.total_budget, row.expected_roi), xytext=(4, 5), textcoords="offset points")
        ax.set(xlabel="Budget (declared minor units)", ylabel="Modeled incremental net value / expected coupon cost", title="Budget what-if: 50%, 100%, 200%")
    else:
        unavailable(ax, "Expected monetary inputs or positive-cost selections unavailable")
    finish(fig, "budget_roi")

    fig, ax = canvas()
    data = strategy.get("frame") if strategy else None
    if data is not None and len(data) and data.primary_value.notna().any():
        selected = data.loc[data.primary_value.notna()].copy()
        ax.bar(selected.strategy_group, selected.primary_value, color="#3b6ea8")
        bounded = selected.dropna(subset=["primary_ci95_lower", "primary_ci95_upper"])
        if len(bounded):
            positions = np.array([selected.strategy_group.tolist().index(x) for x in bounded.strategy_group])
            ax.errorbar(positions, bounded.primary_value,
                        yerr=np.vstack([bounded.primary_value-bounded.primary_ci95_lower, bounded.primary_ci95_upper-bounded.primary_value]),
                        fmt="none", color="black", capsize=4)
        ax.set(xlabel="Actual strategy group", ylabel=strategy["primary_metric"],
               title="Strategy primary metric with 95% intervals" if len(bounded) else "Descriptive strategy metric; inference unavailable")
    else:
        unavailable(ax, (strategy or {}).get("reason", "Factual strategy evaluation not supplied"))
    finish(fig, "strategy_primary_metric")

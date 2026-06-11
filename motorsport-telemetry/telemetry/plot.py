"""
telemetry/plot.py
=================
Comparison chart — speed, throttle, brake, delta vs distance.
"""
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT = True
except ImportError:
    PLOT = False

def plot_comparison(df_a, df_b, cum_delta, common_dist,
                    label_a="A", label_b="B", out_path=None, show=False):
    if not PLOT:
        log.warning("matplotlib not available — chart skipped.")
        return None

    fig = plt.figure(figsize=(16,10), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(4,1,hspace=0.06,figure=fig,top=0.94,bottom=0.06,left=0.07,right=0.97)
    RED,BLUE,WHITE,GRAY = "#e8002d","#1e6fe0","#ffffff","#333333"

    panels = [("speed","Speed (km/h)"),("throttle","Throttle"),
              ("brake","Brake"),("ers_deployment_kw","ERS Deploy (kW)")]
    axes = []
    for idx,(col,ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors=WHITE,labelbottom=False,labelsize=7)
        ax.set_ylabel(ylabel,color=WHITE,fontsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRAY)
        ax.grid(True,color=GRAY,alpha=0.4,linewidth=0.5)
        for df,colour,label in [(df_a,RED,label_a),(df_b,BLUE,label_b)]:
            if col in df.columns:
                v = np.interp(common_dist, df["distance"].values, df[col].values)
                ax.plot(common_dist, v, color=colour, linewidth=0.8, label=label)
        axes.append(ax)

    axes[0].legend(loc="upper right",fontsize=8,facecolor="#1a1a1a",labelcolor=WHITE,edgecolor=GRAY)
    axes[0].set_title(f"Comparison: {label_a} vs {label_b}",color=WHITE,fontsize=11,pad=8,fontweight="bold")
    axes[-1].tick_params(labelbottom=True)

    ax_d = fig.add_subplot(gs[3])
    ax_d.set_facecolor("#0d0d0d")
    ax_d.tick_params(colors=WHITE,labelsize=7)
    ax_d.set_ylabel("Δ Time (s)",color=WHITE,fontsize=7)
    ax_d.set_xlabel("Distance (m)",color=WHITE,fontsize=8)
    for sp in ax_d.spines.values(): sp.set_edgecolor(GRAY)
    ax_d.axhline(0,color=WHITE,linewidth=0.6,linestyle="--",alpha=0.5)
    ax_d.fill_between(common_dist,cum_delta,0,where=cum_delta>0,alpha=0.4,color=RED,label=f"{label_a} faster")
    ax_d.fill_between(common_dist,cum_delta,0,where=cum_delta<0,alpha=0.4,color=BLUE,label=f"{label_b} faster")
    ax_d.plot(common_dist,cum_delta,color=WHITE,linewidth=0.7)
    ax_d.legend(loc="upper right",fontsize=7,facecolor="#1a1a1a",labelcolor=WHITE,edgecolor=GRAY)
    ax_d.grid(True,color=GRAY,alpha=0.4,linewidth=0.5)

    if out_path:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(out_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
        log.info("Chart → %s", out_path)
    if show: plt.show()
    plt.close(fig)
    return out_path

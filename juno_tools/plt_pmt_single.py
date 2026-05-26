import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi  # ridge_points encodes adjacency (Delaunay edges)
def plt_pmt_sing(highlight_csv):
    # -----------------------------
    # Fixed file paths (no args)
    # -----------------------------
    MAP_CSV = "pmt_bec_rmu_map.csv"
    
    OUT_PNG = "pmt_bec_rmu_map_voronoi.png"

    # -----------------------------
    # 1) Load map, filter, and sanitize types
    # -----------------------------
    df = pd.read_csv(MAP_CSV)

    # Keep valid BECs
    df = df[df["BECID"] != -1].copy()

    # Enforce integer types where expected; coerce then drop invalid rows
    for col in ["PMT_ID", "BECID", "RMU_ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PMT_ID", "BECID", "RMU_ID"])
    df["PMT_ID"] = df["PMT_ID"].astype(int)
    df["BECID"] = df["BECID"].astype(int)
    df["RMU_ID"] = df["RMU_ID"].astype(int)

    # Deduplicate by PMT_ID (keep first)
    df = df.drop_duplicates(subset=["PMT_ID"])

    # Ensure Phi/Theta exist and are numeric
    for col in ["Phi", "Theta"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Phi", "Theta"])

    # -----------------------------
    # 2) Build Voronoi adjacency from PMT (Phi, Theta) coordinates
    #    (Note: using ridge_points as Delaunay edges between neighboring PMTs)
    # -----------------------------
    points = df[["Phi", "Theta"]].to_numpy()
    vor = Voronoi(points)

    # -----------------------------
    # 3) Discrete colormap for RMU_ID
    # -----------------------------
    unique_rmu = np.sort(df["RMU_ID"].unique())
    cmap = plt.colormaps["jet"].resampled(len(unique_rmu))
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(0.5, len(unique_rmu) + 1.5),
        ncolors=len(unique_rmu)
    )

    # Map each row index -> RMU_ID and BECID for quick lookup
    rmu_arr = df["RMU_ID"].to_numpy()
    bec_arr = df["BECID"].to_numpy()

    # -----------------------------
    # 4) Scatter base layer (light) colored by RMU_ID
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["Phi"], df["Theta"],
        s=5, c=df["RMU_ID"], cmap=cmap, norm=norm, alpha=0.25, edgecolors="none"
    )

    # -----------------------------
    # 5) Collect only edges crossing different BECIDs
    #    Color each edge by the RMU_ID of the first point (consistent and fast)
    # -----------------------------
    lines = []
    edge_colors = []

    for p1, p2 in vor.ridge_points:
        # Guard against out-of-bounds (shouldn't happen, but be safe)
        if p1 >= len(points) or p2 >= len(points):
            continue

        if bec_arr[p1] != bec_arr[p2]:
            lines.append([(points[p1, 0], points[p1, 1]),
                        (points[p2, 0], points[p2, 1])])
            edge_colors.append(rmu_arr[p1])  # choose a deterministic side for color

    if lines:
        lc = LineCollection(lines, cmap=cmap, norm=norm, linewidths=1.0, alpha=0.9)
        lc.set_array(np.asarray(edge_colors))
        ax.add_collection(lc)

    # -----------------------------
    # 6) Colorbar for discrete RMU_ID
    # -----------------------------
    cbar = plt.colorbar(lc if lines else scatter, ax=ax, ticks=unique_rmu)
    cbar.set_label("RMU_ID")
    cbar.set_ticks(unique_rmu)
    cbar.set_ticklabels(unique_rmu)

    # -----------------------------
    # 7) Overlay highlighted PMTs from an external CSV (one column: PMT_ID)
    # -----------------------------
    if os.path.exists(highlight_csv):
        try:
            df_high = pd.read_csv(highlight_csv, header=None, names=["PMT_ID"])
            df_high = df_high.dropna(subset=["PMT_ID"])
            df_high["PMT_ID"] = pd.to_numeric(df_high["PMT_ID"], errors="coerce")
            df_high = df_high.dropna(subset=["PMT_ID"])
            df_high["PMT_ID"] = df_high["PMT_ID"].astype(int)

            # Join to get Phi/Theta for those PMTs
            merged = pd.merge(df_high, df[["PMT_ID", "Phi", "Theta"]], on="PMT_ID", how="inner")
            if not merged.empty:
                ax.scatter(
                    merged["Phi"], merged["Theta"],
                    s=10, color="black", alpha=0.8, label="Highlighted PMTs"
                )
                ax.legend(loc="upper right", frameon=True)
        except Exception as e:
            print(f"[WARN] Failed to overlay highlighted points: {e}")

    # -----------------------------
    # 8) Final cosmetics and export
    # -----------------------------
    ax.set_xlabel("Phi (°)")
    ax.set_ylabel("Theta (°)")
    ax.set_title("Theta vs Phi — Adjacent Edges Across Different BECIDs (colored by RMU_ID)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(df["Phi"].min() - 2, df["Phi"].max() + 2)
    ax.set_ylim(df["Theta"].min() - 2, df["Theta"].max() + 2)

    plt.tight_layout()
    #plt.savefig(OUT_PNG, dpi=200)
    return fig



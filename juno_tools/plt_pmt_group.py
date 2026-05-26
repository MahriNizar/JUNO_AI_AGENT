import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import Voronoi

def plt_pmt_grp(highlight_dir):

    # ---------------------------------------------------------
    # Fixed I/O
    # ---------------------------------------------------------
    MAP_CSV = "pmt_bec_rmu_map.csv"               # mapping file with columns: PMT_ID,BECID,RMU_ID,Phi,Theta
    OUT_PDF = "plots_larger_theta_vs_phi.pdf"      # output PDF (multi-page)

    # ---------------------------------------------------------
    # 1) Load and sanitize mapping table
    # ---------------------------------------------------------
    df = pd.read_csv(MAP_CSV)

    # Keep valid BECs
    df = df[df["BECID"] != -1].copy()

    # Enforce numeric types and drop invalid rows
    for col in ["PMT_ID", "BECID", "RMU_ID", "Phi", "Theta"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PMT_ID", "BECID", "RMU_ID", "Phi", "Theta"])
    df["PMT_ID"] = df["PMT_ID"].astype(int)
    df["BECID"]  = df["BECID"].astype(int)
    df["RMU_ID"] = df["RMU_ID"].astype(int)

    # Deduplicate by PMT_ID (keep first)
    df = df.drop_duplicates(subset=["PMT_ID"]).reset_index(drop=True)

    # Prepare arrays for fast access
    points  = df[["Phi", "Theta"]].to_numpy()
    bec_arr = df["BECID"].to_numpy()
    rmu_arr = df["RMU_ID"].to_numpy()

    # ---------------------------------------------------------
    # 2) Build Voronoi once and collect edges crossing different BECIDs
    #    ridge_points represents adjacency (Delaunay edges) among sites
    # ---------------------------------------------------------
    vor = Voronoi(points)
    lines = []
    edge_colors = []
    for p1, p2 in vor.ridge_points:
        # Only draw boundaries where neighboring PMTs belong to different BECs
        if bec_arr[p1] != bec_arr[p2]:
            lines.append([(points[p1, 0], points[p1, 1]),
                        (points[p2, 0], points[p2, 1])])
            # Color each edge deterministically by RMU_ID of endpoint p1
            edge_colors.append(rmu_arr[p1])

    # ---------------------------------------------------------
    # 3) Discrete colormap for RMU_ID
    # ---------------------------------------------------------
    unique_rmu = np.sort(df["RMU_ID"].unique())
    cmap = plt.colormaps["jet"].resampled(len(unique_rmu))
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(0.5, len(unique_rmu) + 1.5),
        ncolors=len(unique_rmu)
    )

    # ---------------------------------------------------------
    # 4) Iterate input files and render each page into a single PDF
    # ---------------------------------------------------------
    csv_files = sorted(glob.glob(os.path.join(highlight_dir, "*.csv")))
    if not csv_files:
        print(f"[INFO] No CSV files found in: {highlight_dir}")

    with PdfPages(OUT_PDF) as pdf:
        for i, file in enumerate(csv_files, 1):
            # Read event PMT list (expected: one column PMT_ID, no header)
            try:
                df_ev = pd.read_csv(file, header=None, names=["PMT_ID"])
            except Exception as e:
                print(f"[WARN] Skip unreadable file: {file} ({e})")
                continue

            # Clean and deduplicate PMT_IDs
            df_ev = df_ev.dropna(subset=["PMT_ID"])
            df_ev["PMT_ID"] = pd.to_numeric(df_ev["PMT_ID"], errors="coerce")
            df_ev = df_ev.dropna(subset=["PMT_ID"])
            df_ev["PMT_ID"] = df_ev["PMT_ID"].astype(int)
            df_ev = df_ev.drop_duplicates(subset=["PMT_ID"])

            # Join to get Phi/Theta of event PMTs
            merged = pd.merge(df_ev, df[["PMT_ID", "Phi", "Theta"]], on="PMT_ID", how="inner")

            # Start a new figure per file
            fig, ax = plt.subplots(figsize=(10, 6))

            # Base scatter: all PMTs colored by RMU_ID (light)
            ax.scatter(
                df["Phi"], df["Theta"],
                s=5, c=df["RMU_ID"], cmap=cmap, norm=norm, alpha=0.25, edgecolors="none"
            )

            # Voronoi edges across different BECIDs (colored by RMU_ID)
            if lines:
                lc = LineCollection(lines, cmap=cmap, norm=norm, linewidths=1.0, alpha=0.9)
                lc.set_array(np.asarray(edge_colors))
                ax.add_collection(lc)
                cbar = plt.colorbar(lc, ax=ax, ticks=unique_rmu)
            else:
                # Fallback: colorbar from scatter if no edges
                cbar = plt.colorbar(ax.collections[0], ax=ax, ticks=unique_rmu)

            cbar.set_label("RMU_ID")
            cbar.set_ticks(unique_rmu)
            cbar.set_ticklabels(unique_rmu)

            # Overlay highlighted PMTs from this event file
            if not merged.empty:
                ax.scatter(
                    merged["Phi"], merged["Theta"],
                    s=8, color="black", alpha=0.85, label="Event PMTs"
                )
                ax.legend(loc="upper right", frameon=True)

            # Cosmetics
            ax.set_xlabel("Phi (°)")
            ax.set_ylabel("Theta (°)")
            ax.set_title(f"Theta vs Phi — Voronoi Boundaries by BEC, File: {os.path.basename(file)}")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(df["Phi"].min() - 2, df["Phi"].max() + 2)
            ax.set_ylim(df["Theta"].min() - 2, df["Theta"].max() + 2)
            plt.tight_layout()

            # Save page and close figure
            pdf.savefig(fig)
            plt.close(fig)

            if i % 10 == 0 or i == len(csv_files):
                print(f"[INFO] Processed {i}/{len(csv_files)} files")

    
    return f"[OK] All plots saved to {OUT_PDF}"

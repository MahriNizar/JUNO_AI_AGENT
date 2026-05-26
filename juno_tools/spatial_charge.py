import pandas as pd
import numpy as np
from typing import Optional

# ============================================================
# JUNO SPATIAL CHARGE UNIFORMITY ANALYZER
# Uses the real hardware map loaded by get_hardware_connections
# instead of arithmetic approximations.
#
# Dependency: HARDWARE_MAP_DF and get_hardware_connections must
# be loaded in the same module (from juno_tools.hardware_check).
# ============================================================

UNIT_ASYMMETRY_FLAG_THRESHOLD = 0.10  # 10% deviation flags a unit
TOP_BOTTOM_FLAG_THRESHOLD     = 0.05  # 5% top/bottom imbalance
MAP_FILE_PATH = "pmt_bec_rmu_map.csv"


def _to_int_safe(x):
    """Convert values like 19, '19', '19.0' -> 19; leaves nan as np.nan."""
    if pd.isna(x):
        return np.nan
    try:
        # handles '19.0' / '5.0' / '2'
        return int(float(str(x).strip()))
    except Exception:
        return np.nan

def _normalize_rmu2nd_label(x: str) -> str:
    """
    Normalize RMU_2nd strings like '5.0-1.0' -> '5-1'.
    If already like '5-1', keep as-is. If numeric, cast to int string.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a_i = _to_int_safe(a)
        b_i = _to_int_safe(b)
        if pd.isna(a_i) or pd.isna(b_i):
            return s  # fallback: return original
        return f"{int(a_i)}-{int(b_i)}"
    # not hyphenated -> try int
    v = _to_int_safe(s)
    return "" if pd.isna(v) else str(int(v))









try:
    HARDWARE_MAP_DF = pd.read_csv(MAP_FILE_PATH)
    # Convert key columns to integers for clean, reliable matching
    HARDWARE_MAP_DF['PMT_ID'] = HARDWARE_MAP_DF['PMT_ID'].map(_to_int_safe)
    HARDWARE_MAP_DF['GCU'] = HARDWARE_MAP_DF['GCU'].map(_to_int_safe)
    HARDWARE_MAP_DF['BECID'] = HARDWARE_MAP_DF['BECID'].map(_to_int_safe)
    HARDWARE_MAP_DF['RMU_2nd'] = HARDWARE_MAP_DF['RMU_2nd'].map(_normalize_rmu2nd_label)
    HARDWARE_MAP_DF.dropna(subset=['PMT_ID', 'GCU', 'BECID'], inplace=True)
    # For RMU_2nd, we must explicitly check for the empty string ''
    HARDWARE_MAP_DF = HARDWARE_MAP_DF[HARDWARE_MAP_DF['RMU_2nd'] != '']
    print("--- Hardware Map loaded successfully. ---")
except FileNotFoundError:
    print(f"--- WARNING: '{MAP_FILE_PATH}' not found. The 'get_hardware_connections' tool will fail. ---")
    HARDWARE_MAP_DF = pd.DataFrame() # Create empty DF so script doesn't crash
except Exception as e:
    print(f"--- ERROR loading hardware map: {e} ---")
    HARDWARE_MAP_DF = pd.DataFrame()

# A mapping from the agent's simple names to the CSV's column names
COLUMN_MAP = {
    "PMT": "PMT_ID",
    "GCU": "GCU",
    "BEC": "BECID",
    "RMU": "RMU_2nd"
}

def _build_pmt_to_hardware_map() -> Optional[pd.DataFrame]:
    """
    Constructs a clean PMT_ID -> (GCU, BEC, RMU) lookup DataFrame
    directly from the already-loaded HARDWARE_MAP_DF.
    Returns None if the map is not available.
    """
    if HARDWARE_MAP_DF.empty:
        return None

    # Select only the columns we need and deduplicate
    # Each PMT_ID should map to exactly one GCU, one BEC, one RMU
    hw_map = (
    HARDWARE_MAP_DF[['PMT_ID', 'GCU', 'BECID', 'RMU_2nd']]
    .drop_duplicates(subset=['PMT_ID'])
    .rename(columns={'GCU': 'GCU_ID', 'BECID': 'BEC_ID', 'RMU_2nd': 'RMU_ID'})  # add GCU -> GCU_ID
    .reset_index(drop=True)
    )
    return hw_map


def analyze_spatial_charge_uniformity(
    run_data_path: str,
    min_events: int = 300,
    asymmetry_threshold: float = UNIT_ASYMMETRY_FLAG_THRESHOLD,
    top_bottom_threshold: float = TOP_BOTTOM_FLAG_THRESHOLD,
    top_k_anomalies: int = 10
) -> dict:
    """
    Analyzes spatial charge non-uniformity across the JUNO detector
    by aggregating hits at the GCU, BEC, and RMU hardware levels.

    Uses the real JUNO hardware map (from HARDWARE_MAP_DF) to assign
    each PMT to its physical readout chain: GCU -> BEC -> RMU.
    This allows detection of hardware-block-level failures (dead GCU,
    dead BEC, dead RMU) that per-PMT outlier tools would dilute.

    Args:
        run_data_path:        Path to CSV (Event, PMTID, Charge, Time, Theta, Phi).
        min_events:           Minimum number of events for reliable statistics.
        asymmetry_threshold:  Fractional deviation from mean to flag a unit (default 0.10).
        top_bottom_threshold: Top/bottom hemisphere charge ratio deviation (default 0.05).
        top_k_anomalies:      Number of most anomalous units to return per level.

    Returns:
        Structured dict with charge uniformity at GCU, BEC, RMU levels,
        top/bottom asymmetry, flagged anomalous units, and a diagnosis.
    """
    try:
        df = pd.read_csv(run_data_path)

        # --- Input validation ---
        required_cols = ['Event', 'PMTID', 'Charge', 'Theta']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return {"error": f"CSV missing required columns: {missing}"}

        n_events = df['Event'].nunique()
        if n_events < min_events:
            return {
                "error": (
                    f"Insufficient events for reliable spatial statistics. "
                    f"Found {n_events}, minimum required is {min_events}."
                )
            }

        # --- Load and validate hardware map ---
        hw_map = _build_pmt_to_hardware_map()
        if hw_map is None:
            return {"error": "Hardware map (HARDWARE_MAP_DF) is not loaded. Cannot perform analysis."}

        # --- Join run data with hardware map ---
        # Merge on PMT ID. PMTs not in the map (e.g. 3-inch PMTs) are dropped.
        df = df.merge(
            hw_map,
            left_on='PMTID',
            right_on='PMT_ID',
            how='inner'
        )

        if df.empty:
            return {"error": "No PMT IDs in the run data matched the hardware map. Check PMT ID format."}

        n_mapped_pmts = df['PMTID'].nunique()

        # -------------------------------------------------------
        # LEVEL 1: Top/Bottom Hemisphere Asymmetry
        # Theta < 90 = top, Theta >= 90 = bottom
        # For isotropic radioactive background: ratio ~ 1.0
        # During LS filling this will be legitimately asymmetric.
        # -------------------------------------------------------
        top_df    = df[df['Theta'] < 90]
        bottom_df = df[df['Theta'] >= 90]

        top_charge    = float(top_df['Charge'].sum())
        bottom_charge = float(bottom_df['Charge'].sum())
        top_nhit      = int(len(top_df))
        bottom_nhit   = int(len(bottom_df))
        total_charge  = top_charge + bottom_charge

        top_bottom_ratio     = top_charge / bottom_charge if bottom_charge > 0 else float('inf')
        top_bottom_asymmetry = abs(top_charge - bottom_charge) / total_charge if total_charge > 0 else 0.0
        flag_top_bottom      = top_bottom_asymmetry > top_bottom_threshold

        # -------------------------------------------------------
        # LEVEL 2: GCU-level analysis (3 PMTs per GCU / UWBox)
        # Smallest hardware unit. A flagged GCU = 1 dead UWBox.
        # -------------------------------------------------------
        gcu_stats = (
            df.groupby('GCU_ID')
            .agg(
                total_charge=('Charge', 'sum'),
                total_hits=('PMTID', 'count'),
                n_active_pmts=('PMTID', 'nunique'),
                bec_id=('BEC_ID', 'first'),
                rmu_id=('RMU_ID', 'first'),
            )
            .reset_index()
        )

        gcu_mean   = float(gcu_stats['total_charge'].mean())
        gcu_std    = float(gcu_stats['total_charge'].std())
        gcu_stats['charge_deviation'] = (
            (gcu_stats['total_charge'] - gcu_mean) / gcu_mean
        ).round(4)
        gcu_stats['flagged'] = gcu_stats['charge_deviation'].abs() > asymmetry_threshold

        anomalous_gcu_list = _build_anomaly_list(
            gcu_stats[gcu_stats['flagged']],
            id_col='GCU_ID',
            extra_cols={'bec_id': 'bec_id', 'rmu_id': 'rmu_id', 'n_active_pmts': 'n_active_pmts'},
            top_k=top_k_anomalies
        )

        # -------------------------------------------------------
        # LEVEL 3: BEC-level analysis (up to 48 GCUs = 144 PMTs)
        # A flagged BEC is the most actionable finding:
        # it corresponds to a single physical Back-End Card.
        # -------------------------------------------------------
        bec_stats = (
            df.groupby('BEC_ID')
            .agg(
                total_charge=('Charge', 'sum'),
                total_hits=('PMTID', 'count'),
                n_active_pmts=('PMTID', 'nunique'),
                n_active_gcus=('GCU_ID', 'nunique'),
                rmu_id=('RMU_ID', 'first'),
            )
            .reset_index()
        )

        bec_mean = float(bec_stats['total_charge'].mean())
        bec_stats['charge_deviation'] = (
            (bec_stats['total_charge'] - bec_mean) / bec_mean
        ).round(4)
        bec_stats['flagged'] = bec_stats['charge_deviation'].abs() > asymmetry_threshold

        anomalous_bec_list = _build_anomaly_list(
            bec_stats[bec_stats['flagged']],
            id_col='BEC_ID',
            extra_cols={'rmu_id': 'rmu_id', 'n_active_pmts': 'n_active_pmts', 'n_active_gcus': 'n_active_gcus'},
            top_k=top_k_anomalies
        )

        # -------------------------------------------------------
        # LEVEL 4: RMU-level analysis (16 BECs = ~2304 PMTs)
        # A flagged RMU is a critical fault: entire detector sector.
        # RMU IDs are strings like '5-1' from the hardware map.
        # -------------------------------------------------------
        rmu_stats = (
            df.groupby('RMU_ID')
            .agg(
                total_charge=('Charge', 'sum'),
                total_hits=('PMTID', 'count'),
                n_active_pmts=('PMTID', 'nunique'),
                n_active_becs=('BEC_ID', 'nunique'),
                n_active_gcus=('GCU_ID', 'nunique'),
            )
            .reset_index()
        )

        rmu_mean = float(rmu_stats['total_charge'].mean())
        rmu_stats['charge_deviation'] = (
            (rmu_stats['total_charge'] - rmu_mean) / rmu_mean
        ).round(4)
        rmu_stats['flagged'] = rmu_stats['charge_deviation'].abs() > asymmetry_threshold

        anomalous_rmu_list = _build_anomaly_list(
            rmu_stats[rmu_stats['flagged']],
            id_col='RMU_ID',
            extra_cols={
                'n_active_pmts': 'n_active_pmts',
                'n_active_becs': 'n_active_becs',
                'n_active_gcus': 'n_active_gcus'
            },
            top_k=top_k_anomalies
        )

        # -------------------------------------------------------
        # DIAGNOSIS
        # -------------------------------------------------------
        diagnosis_notes = _build_diagnosis(
            flag_top_bottom=flag_top_bottom,
            top_bottom_asymmetry=top_bottom_asymmetry,
            top_charge=top_charge,
            bottom_charge=bottom_charge,
            n_flagged_bec=len(anomalous_bec_list),
            n_flagged_rmu=len(anomalous_rmu_list),
            n_flagged_gcu=len(anomalous_gcu_list),
            asymmetry_threshold=asymmetry_threshold,
        )

        return {
            "run_summary": {
                "total_events":    int(n_events),
                "total_hits":      int(len(df)),
                "total_charge":    round(total_charge, 2),
                "mapped_pmts":     int(n_mapped_pmts),
                "active_gcus":     int(df['GCU_ID'].nunique()),
                "active_becs":     int(df['BEC_ID'].nunique()),
                "active_rmus":     int(df['RMU_ID'].nunique()),
            },
            "top_bottom_asymmetry": {
                "top_charge":         round(top_charge, 2),
                "bottom_charge":      round(bottom_charge, 2),
                "top_nhit":           top_nhit,
                "bottom_nhit":        bottom_nhit,
                "top_bottom_ratio":   round(top_bottom_ratio, 4),
                "asymmetry_fraction": round(top_bottom_asymmetry, 4),
                "flagged":            bool(flag_top_bottom),
            },
            "gcu_level": {
                "mean_charge_per_gcu": round(gcu_mean, 2),
                "std_charge_per_gcu":  round(gcu_std, 2),
                "n_flagged_gcus":      len(anomalous_gcu_list),
                "anomalous_gcus":      anomalous_gcu_list,
            },
            "bec_level": {
                "mean_charge_per_bec": round(bec_mean, 2),
                "n_flagged_becs":      len(anomalous_bec_list),
                "anomalous_becs":      anomalous_bec_list,
            },
            "rmu_level": {
                "mean_charge_per_rmu": round(rmu_mean, 2),
                "n_flagged_rmus":      len(anomalous_rmu_list),
                "anomalous_rmus":      anomalous_rmu_list,
            },
            "diagnosis": diagnosis_notes,
        }

    except FileNotFoundError:
        return {"error": f"File not found: {run_data_path}"}
    except Exception as e:
        return {"error": f"Spatial uniformity analysis failed: {str(e)}"}


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def _build_anomaly_list(
    flagged_df: pd.DataFrame,
    id_col: str,
    extra_cols: dict,
    top_k: int
) -> list:
    """
    Sorts flagged hardware units by absolute charge deviation
    and returns a clean list of dicts for the top-K worst offenders.
    extra_cols maps output key -> dataframe column name.
    """
    if flagged_df.empty:
        return []

    sorted_df = flagged_df.reindex(
        flagged_df['charge_deviation'].abs().sort_values(ascending=False).index
    ).head(top_k)

    result = []
    for _, row in sorted_df.iterrows():
        entry = {
            id_col.lower():       row[id_col],
            "total_charge":       round(float(row['total_charge']), 2),
            "total_hits":         int(row['total_hits']),
            "charge_deviation":   round(float(row['charge_deviation']), 4),
            "type":               "deficit" if row['charge_deviation'] < 0 else "excess",
        }
        for out_key, df_col in extra_cols.items():
            val = row[df_col]
            # RMU IDs are strings; everything else is numeric
            entry[out_key] = val if isinstance(val, str) else (
                int(val) if not pd.isna(val) else None
            )
        result.append(entry)

    return result


def _build_diagnosis(
    flag_top_bottom: bool,
    top_bottom_asymmetry: float,
    top_charge: float,
    bottom_charge: float,
    n_flagged_bec: int,
    n_flagged_rmu: int,
    n_flagged_gcu: int,
    asymmetry_threshold: float,
) -> list:
    """Generates plain-language diagnostic notes for the agent."""
    notes = []

    if flag_top_bottom:
        dominant = "top" if top_charge > bottom_charge else "bottom"
        notes.append(
            f"Top/bottom charge asymmetry of {top_bottom_asymmetry:.1%} exceeds threshold. "
            f"The {dominant} hemisphere dominates. "
            f"Possible causes: partial detector fill (LS filling run), or dead PMT cluster "
            f"in the {'bottom' if dominant == 'top' else 'top'} hemisphere."
        )

    if n_flagged_rmu > 0:
        notes.append(
            f"{n_flagged_rmu} RMU(s) flagged with charge deviation > {asymmetry_threshold:.0%}. "
            f"An RMU covers ~2304 PMTs across 16 BECs. This is a critical fault "
            f"affecting an entire detector sector. Immediate hardware inspection required."
        )

    if n_flagged_bec > 0:
        notes.append(
            f"{n_flagged_bec} BEC(s) flagged with charge deviation > {asymmetry_threshold:.0%}. "
            f"Each BEC covers up to 144 PMTs (48 GCUs). A deficit BEC is consistent "
            f"with a dead or disconnected Back-End Card."
        )

    if n_flagged_gcu > 0 and n_flagged_bec == 0 and n_flagged_rmu == 0:
        notes.append(
            f"{n_flagged_gcu} isolated GCU(s) flagged with no BEC or RMU-level pattern. "
            f"Consistent with individual UWBox failures (3 PMTs each) rather than "
            f"a systemic hardware fault."
        )

    if not notes:
        notes.append(
            "No significant spatial charge non-uniformity detected. "
            "Detector response is spatially uniform within the defined threshold."
        )

    return notes

#print(analyze_spatial_charge_uniformity("test_data/your_run_first600.csv"))
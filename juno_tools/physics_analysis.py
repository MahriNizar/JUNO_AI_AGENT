import pandas as pd
import numpy as np
import json
import os
def compute_run_kpis(run_data_path: str) -> dict:
    """
    Computes key performance indicators (KPIs) for a run to provide a quick global summary.
    """
    try:
        df = pd.read_csv(run_data_path)
        
        # Basic validation
        required_cols = ['Event', 'Charge']
        if not all(col in df.columns for col in required_cols):
            return {"error": f"CSV missing required columns: {required_cols}"}
        
        if df.empty:
            return {"error": "Run data file is empty."}

        # Global stats
        num_hits = len(df)
        num_events = df['Event'].nunique()
        
        # Event-level stats (grouping takes time, so we do it once)
        hits_per_event = df.groupby('Event').size()
        
        # Charge stats
        avg_charge = float(df['Charge'].mean())


        return {
            "summary": "Run KPI Summary",
            "num_events": int(num_events),
            "total_hits": int(num_hits),
            "avg_hits_per_event": float(hits_per_event.mean()),
            "std_hits_per_event": float(hits_per_event.std()),
            "avg_charge_per_hit": round(avg_charge, 4),
            "nhit_p95": float(hits_per_event.quantile(0.95)), # 95th percentile to detect high-Nhit tails
            "status": "Normal" if num_events > 0 else "Empty"
        }

    except FileNotFoundError:
        return {"error": f"File not found: {run_data_path}"}
    except Exception as e:
        return {"error": f"KPI computation failed: {str(e)}"}
    

def get_event_multiplicity_distribution(run_data_path: str, num_bins: int = 20) -> dict:

    """
    Generates a histogram of the number of hits (multiplicity) per event.
    Useful for finding outliers, double-peaks, or abnormal tails.
    """
    try:
        df = pd.read_csv(run_data_path)
        
        if 'Event' not in df.columns:
             return {"error": "CSV missing 'Event' column."}

        # Group by event to get Nhit per event
        nhit_series = df.groupby('Event').size()
        
        if nhit_series.empty:
            return {"error": "No events found in data."}

        # Calculate simple stats
        stats = {
            "mean": float(nhit_series.mean()),
            "median": float(nhit_series.median()),
            "max": int(nhit_series.max()),
            "min": int(nhit_series.min()),
            "p99": float(nhit_series.quantile(0.99))
        }

        # Compute Histogram
        counts, bin_edges = np.histogram(nhit_series, bins=num_bins)
        
        # Format histogram for the Agent (list of readable dicts)
        histogram_data = []
        for i in range(len(counts)):
            histogram_data.append({
                "bin_range": f"{int(bin_edges[i])}-{int(bin_edges[i+1])}",
                "count": int(counts[i])
            })

        return {
            "summary_stats": stats,
            "histogram_bins": histogram_data,
            "note": "Bins with 0 counts are included to show gaps."
        }

    except FileNotFoundError:
        return {"error": f"File not found: {run_data_path}"}
    except Exception as e:
        return {"error": f"Distribution calculation failed: {str(e)}"}
    

def detect_global_occupancy_outliers(
    run_data_path: str, 
    noisy_z_threshold: float = 5.0, 
    quiet_z_threshold: float = 3.0, 
    top_k: int = 20
) -> dict:
    """
    Identifies globally noisy (flashing) or quiet (dead) PMTs using robust Z-scores (MAD-based).
    Adapts to run length: 'Dead' is defined statistically relative to the median occupancy.
    """
    try:
        df = pd.read_csv(run_data_path)
        if 'PMTID' not in df.columns:
             return {"error": "CSV missing 'PMTID' column."}
        
        # 1. Calculate Hits per PMT (Occupancy)
        pmt_counts = df['PMTID'].value_counts()
        
        if pmt_counts.empty:
             return {"error": "No hits found in file."}

        # 2. Robust Statistics (Median & MAD)
        median_hits = float(pmt_counts.median())
        
        # Constant 1.4826 scales MAD to be consistent with Sigma for Gaussian distributions
        mad = float(np.median(np.abs(pmt_counts - median_hits))) * 1.4826
        
        if mad == 0:
            mad = 1.0  # Avoid division by zero for very stable runs

        # 3. Calculate Z-scores
        z_scores = (pmt_counts - median_hits) / mad
        
        # 4. Identify Noisy PMTs (High Z-score)
        noisy_pmts = z_scores[z_scores > noisy_z_threshold].head(top_k)
        noisy_list = []
        for pmt_id, z in noisy_pmts.items():
            noisy_list.append({
                "pmt_id": int(pmt_id),
                "hits": int(pmt_counts[pmt_id]),
                "z_score": round(z, 2)
            })

        # 5. Identify Quiet/Dead PMTs (Low Z-score)
        # Logic: Find PMTs significantly *below* the median.
        # e.g., if Median is 100 and MAD is 10, threshold=3.0 -> cutoff is 70.
        # Any PMT with < 70 hits is "Quiet".
        
        quiet_mask = z_scores < -quiet_z_threshold
        quiet_pmts = pmt_counts[quiet_mask].tail(top_k) # Get the lowest counts
        
        quiet_list = []
        for pmt_id, count in quiet_pmts.items():
            # Calculate the specific Z-score for this PMT
            z = (count - median_hits) / mad
            quiet_list.append({
                "pmt_id": int(pmt_id),
                "hits": int(count),
                "z_score": round(z, 2)
            })
            
        # Calculate the theoretical "Dead Threshold" for context
        dead_threshold_hits = max(0, median_hits - (quiet_z_threshold * mad))

        return {
            "summary_stats": {
                "total_active_pmts": int(len(pmt_counts)),
                "occupancy_median": median_hits,
                "occupancy_mad": round(mad, 2),
                "noisy_count": int((z_scores > noisy_z_threshold).sum()),
                "quiet_count": int(quiet_mask.sum()),
                "quiet_threshold_hits": round(dead_threshold_hits, 1)
            },
            "noisy_pmts": noisy_list,
            "quiet_pmts": quiet_list,
            "note": f"PMTs with fewer than {round(dead_threshold_hits, 1)} hits are flagged as quiet."
        }

    except FileNotFoundError:
        return {"error": f"File not found: {run_data_path}"}
    except Exception as e:
        return {"error": f"Outlier detection failed: {str(e)}"}

def scan_run_for_occupancy_anomalies(
    run_data_path: str,
    events_per_window: int = 20,
    flashing_z_threshold: float = 7.0,
    top_k_windows: int = 20
) -> dict:
    """
    Slides a window across the run to find local bursts or flashing PMTs.
    Saves full results to a JSON file and returns a summary.
    """
    try:
        df = pd.read_csv(run_data_path)
        if 'Event' not in df.columns or 'PMTID' not in df.columns:
            return {"error": "CSV missing columns."}

        # 1. Setup Windows
        min_event = int(df['Event'].min())
        max_event = int(df['Event'].max())
        
        anomalous_windows = []
        window_id = 0
        
        # Global median for reference (lazy approximation)
        global_median_hits = df['PMTID'].value_counts().median()
        
        # 2. Slide Window
        for start_e in range(min_event, max_event, events_per_window):
            end_e = start_e + events_per_window
            
            # Filter (efficiently)
            window_df = df[(df['Event'] >= start_e) & (df['Event'] < end_e)]
            
            if window_df.empty:
                continue
                
            # 3. Calculate Window Stats
            hits_in_window = len(window_df)
            pmt_counts = window_df['PMTID'].value_counts()
            
            # Detect Flashing (Local Z-score vs Window Mean)
            # Note: In a small window, MAD might be 0 if most PMTs have 0 hits.
            # So we compare against the GLOBAL median scaled down to the window size?
            # Simpler approach: Absolute hit count threshold for now, or Z-score if enough data.
            
            # Let's use a simplified robust check for the window:
            w_median = pmt_counts.median()
            w_std = pmt_counts.std()
            if np.isnan(w_std) or w_std == 0: w_std = 1
            
            flashing_pmts = pmt_counts[pmt_counts > (w_median + flashing_z_threshold * w_std)]
            
            # Check for High Activity Burst (e.g. 3x average density)
            # (This is a heuristic)
            is_burst = hits_in_window > (global_median_hits * (events_per_window / (max_event-min_event)) * 5)

            if not flashing_pmts.empty or is_burst:
                
                flashers_list = [{"pmt_id": int(pid), "hits": int(h)} for pid, h in flashing_pmts.items()]
                
                anomalous_windows.append({
                    "window_id": window_id,
                    "event_start": start_e,
                    "event_end": end_e,
                    "total_hits": int(hits_in_window),
                    "num_flashers": len(flashers_list),
                    "flashers": flashers_list, # In a real app, maybe don't put full list in summary if huge
                    "is_burst": bool(is_burst)
                })
            
            window_id += 1

        # 4. Save Results
        results_filename = f"{run_data_path}_scan_results.json"
        with open(results_filename, 'w') as f:
            json.dump(anomalous_windows, f, indent=2)

        # 5. Return Summary
        top_windows = sorted(anomalous_windows, key=lambda x: x['num_flashers'], reverse=True)[:top_k_windows]
        
        # Strip the detailed 'flashers' list from the summary to save tokens
        # The agent must use 'read_scan_result' tools to see details (not implemented here, but implied)
        summary_windows = []
        for w in top_windows:
            w_copy = w.copy()
            w_copy['flashers_preview'] = [f['pmt_id'] for f in w['flashers'][:3]] # Just show first 3 IDs
            del w_copy['flashers'] 
            summary_windows.append(w_copy)

        return {
            "status": "Scan Complete",
            "total_windows_scanned": window_id,
            "anomalous_windows_found": len(anomalous_windows),
            "results_saved_to": results_filename,
            "top_anomalous_windows": summary_windows
        }

    except Exception as e:
        return {"error": f"Scan failed: {str(e)}"}
    

def get_pmt_profile(run_data_path: str, pmt_id: int, max_points: int = 50) -> dict:
    """
    Retrieves a detailed profile for a single PMT, including a downsampled time series 
    of its hits and charge. Useful for checking if a PMT is drifting or flashing.
    """
    try:
        df = pd.read_csv(run_data_path)
        
        # Filter for specific PMT
        pmt_df = df[df['PMTID'] == pmt_id]
        
        if pmt_df.empty:
            return {
                "pmt_id": pmt_id,
                "status": "No hits found (Dead or Invalid ID)",
                "total_hits": 0
            }

        # 1. Compute Aggregate Stats
        total_hits = len(pmt_df)
        avg_charge = float(pmt_df['Charge'].mean())
        max_charge = float(pmt_df['Charge'].max())
        
        # 2. Generate Time Series (Hits per Event)
        # We group by Event to see when it fired.
        # Since events are sequential, this acts as a timeline.
        hits_per_event = pmt_df.groupby('Event')['Charge'].agg(['count', 'sum'])
        hits_per_event.columns = ['hits', 'total_charge']
        
        # 3. Downsample for LLM Context Safety
        # We want to return ~max_points. If we have 1000 active events, we skip rows.
        num_active_events = len(hits_per_event)
        step = max(1, num_active_events // max_points)
        
        # Slice the dataframe
        sampled_df = hits_per_event.iloc[::step]
        
        # Convert to list of dicts
        time_series = []
        for event_id, row in sampled_df.iterrows():
            time_series.append({
                "event_id": int(event_id),
                "hits": int(row['hits']),
                "charge": round(float(row['total_charge']), 2)
            })

        return {
            "pmt_id": pmt_id,
            "status": "Active",
            "total_hits": total_hits,
            "num_active_events": num_active_events,
            "avg_charge_per_hit": round(avg_charge, 2),
            "sampling_note": f"Showing 1 out of every {step} active events.",
            "time_series_sample": time_series
        }

    except FileNotFoundError:
        return {"error": f"File not found: {run_data_path}"}
    except Exception as e:
        return {"error": f"Profile generation failed: {str(e)}"}
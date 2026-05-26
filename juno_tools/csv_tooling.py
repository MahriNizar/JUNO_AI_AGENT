import os
import pandas as pd

def inspect_csv(run_data_path: str) -> dict:
    """
    Returns a lightweight summary of a JUNO run CSV without loading
    the full dataset into memory.  The agent should call this FIRST
    on any new file to decide whether filtering / batching is needed.

    Parameters
    ----------
    run_data_path : str
        Path to the CSV file (must contain at least an 'Event' column).

    Returns
    -------
    dict
        file_size_mb        – size on disk
        total_rows          – number of hit-level rows
        num_events          – number of unique events
        event_range         – [min_event, max_event]
        columns             – list of column names present
        avg_hits_per_event  – mean hits per event (float)
        recommendation      – short text hint for the agent
    """
    try:
        if not os.path.exists(run_data_path):
            return {"error": f"File not found: {run_data_path}"}

        file_size_mb = round(os.path.getsize(run_data_path) / 1e6, 2)

        # Read only the header to get column names cheaply
        columns = list(pd.read_csv(run_data_path, nrows=0).columns)

        if "Event" not in columns:
            return {"error": "CSV missing required 'Event' column."}

        # Load only the Event column — keeps memory proportional to rows,
        # but avoids loading Charge / Time / PMTID arrays.
        events = pd.read_csv(run_data_path, usecols=["Event"])["Event"]
        total_rows = len(events)
        num_events = int(events.nunique())
        event_min = int(events.min())
        event_max = int(events.max())
        avg_hits = round(total_rows / num_events, 1) if num_events > 0 else 0.0

        # Heuristic guidance for the agent
        if num_events <= 500:
            recommendation = "Small file. Analyse directly — no batching needed."
        elif num_events <= 5000:
            recommendation = (
                "Medium file. Consider filtering to a sub-range if you "
                "only need a specific event window."
            )
        else:
            recommendation = (
                f"Large file ({num_events} events). You SHOULD filter into "
                f"batches of ~1 000–2 000 events using 'filter_events_by_range' "
                f"before running analysis tools."
            )

        return {
            "file_size_mb": file_size_mb,
            "total_rows": total_rows,
            "num_events": num_events,
            "event_range": [event_min, event_max],
            "columns": columns,
            "avg_hits_per_event": avg_hits,
            "recommendation": recommendation,
        }

    except Exception as e:
        return {"error": f"Inspection failed: {str(e)}"}


def filter_events_by_range(

    run_data_path: str,
    start_event: int,
    end_event: int,
    output_path: str = "",
) -> dict:
    """
    Extracts all hits whose Event ID falls in [start_event, end_event]
    (inclusive on both ends) and writes them to a new CSV file.

    Use this to create a manageable slice of a large run before calling
    analysis tools such as detect_global_occupancy_outliers or
    scan_run_for_occupancy_anomalies.

    Parameters
    ----------
    run_data_path : str
        Path to the source CSV.
    start_event : int
        First event ID to include (inclusive).
    end_event : int
        Last event ID to include (inclusive).
    output_path : str, optional
        Where to write the filtered CSV.  If empty, a default path is
        generated automatically:
            <original_name>_events_<start>_<end>.csv

    Returns
    -------
    dict
        output_path   – path to the new CSV
        num_events    – unique events in the slice
        total_rows    – hit-level rows written
        event_range   – [actual_min, actual_max] after filtering
    """
    try:
        if not os.path.exists(run_data_path):
            return {"error": f"File not found: {run_data_path}"}

        if start_event > end_event:
            return {"error": f"start_event ({start_event}) > end_event ({end_event})."}

        # Build default output path if not provided
        if not output_path:
            base, ext = os.path.splitext(run_data_path)
            output_path = f"{base}_events_{start_event}_{end_event}{ext}"

        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Read, filter, write
        df = pd.read_csv(run_data_path)

        if "Event" not in df.columns:
            return {"error": "CSV missing required 'Event' column."}

        mask = (df["Event"] >= start_event) & (df["Event"] <= end_event)
        filtered = df.loc[mask]

        if filtered.empty:
            return {
                "error": (
                    f"No events found in range [{start_event}, {end_event}]. "
                    f"File event range is [{int(df['Event'].min())}, {int(df['Event'].max())}]."
                )
            }

        filtered.to_csv(output_path, index=False)

        num_events = int(filtered["Event"].nunique())
        total_rows = len(filtered)
        actual_min = int(filtered["Event"].min())
        actual_max = int(filtered["Event"].max())

        return {
            "output_path": output_path,
            "num_events": num_events,
            "total_rows": total_rows,
            "event_range": [actual_min, actual_max],
        }

    except Exception as e:
        return {"error": f"Filtering failed: {str(e)}"}


import pandas as pd
import json
import numpy as np
from typing import Optional, Set,Union

MAP_FILE_PATH = "pmt_bec_rmu_map.csv"
# FILE_DATA_PATH = "test_data/scenario8_one.csv"
#FILE_DATA_PATH = "test_data/one_output_3398_200.csv"
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

def get_hardware_connections(element_type: str, element_id: str) -> dict:
    """
    Finds all connected hardware components for a given element type and ID.
    Also gives the number of pmt connected to the given element.
    Valid 'element_type' values are: 'PMT', 'GCU', 'BEC', 'RMU'.
    'element_id' should be a string (e.g., "1925" for a GCU, "5-1" for an RMU).
    This tool helps localize issues and focus the analysis scope.

    Args:
        element_type (str): The type of hardware element (e.g., "GCU", "RMU").
        element_id (str): The specific ID of the element (e.g. "1925", "5-1").

    Returns:
        dict: A structured dictionary of all connected components.
    """
    print(f"--- TOOL: Running get_hardware_connections for {element_type} {element_id} ---")
    
    # Check if the hardware map was loaded successfully
    if HARDWARE_MAP_DF.empty:
        return {"error": "Hardware map is not loaded. Cannot perform lookup."}

    # 1. Validate element_type and get the DataFrame column name
    element_type_upper = element_type.upper()
    query_col = COLUMN_MAP.get(element_type_upper)
    if not query_col:
        return {"error": f"Invalid element_type '{element_type}'. Must be one of {list(COLUMN_MAP.keys())}."}

    # 2. Normalize the input 'element_id' to match the DataFrame's format
    normalized_id = None
    if element_type_upper == "RMU":
        normalized_id = _normalize_rmu2nd_label(element_id)
        if normalized_id == "" or normalized_id=='None': # _normalize_rmu2nd_label returns "" for NaNs/invalid
            return {"error": f"Invalid RMU ID format '{element_id}'. Expected '5-1' or '5'."}
    elif element_type_upper in ["PMT", "GCU", "BEC"]:
        normalized_id = _to_int_safe(element_id)
        if pd.isna(normalized_id):
            return {"error": f"Invalid {element_type_upper} ID '{element_id}'. Must be an integer."}
        normalized_id = int(normalized_id) # Ensure it's a clean int for lookup
    
    # 3. Filter the DataFrame to find all matching rows
    try:
        filtered_df = HARDWARE_MAP_DF[HARDWARE_MAP_DF[query_col] == normalized_id]
    except Exception as e:
        return {"error": f"Error during lookup: {e}"}

    if filtered_df.empty:
        return {"error": f"No hardware found for {element_type} {element_id}."}

    # 4. Extract all unique connected components from the filtered rows
    # Use .dropna() to remove any NaNs before collecting unique values
    connected_pmts = sorted([int(x) for x in filtered_df['PMT_ID'].dropna().unique()])
    connected_gcus = sorted([int(x) for x in filtered_df['GCU'].dropna().unique()])
    connected_becs = sorted([int(x) for x in filtered_df['BECID'].dropna().unique()])
    connected_rmus = sorted(filtered_df['RMU_2nd'].dropna().unique().tolist()) # These are strings

    # 5. Build the structured result
    result = {
        "query": f"{element_type.upper()} {element_id}",
        "connections": {
            "pmt_ids": connected_pmts,
            "gcu_ids": connected_gcus,
            "bec_ids": connected_becs,
            "rmu_ids": connected_rmus, # This will be a list of strings
        },
        "pmt_match_count": len(filtered_df)
    }

    #print(f"--- TOOL: Found connections (first 5 PMTs): {result['connections']['pmt_ids'][:5]}... ---")
    return result

# --- Tool 3: Component Activity Checker ---
def check_component_activity(
    run_data_path: str, 
    element_type: str, 
    start_event: Optional[int] = None, 
    end_event: Optional[int] = None, 
    element_id: Optional[str] = None
) -> dict:
    """
    Checks for silent (inactive) components within a specific event window.
    Can check for a specific component ID or all components of a given type.
    Valid 'element_type' values are: 'PMT', 'GCU', 'BEC', 'RMU'.
    'element_id' is an Optional string (e.g., "1925" or "5-1").
    'start_event' and 'end_event' define the inclusive window. If None, they
    default to the beginning or end of the run, respectively.

    Args:
        run_data_path (str): The path to the aggregated run data CSV file.
        element_type (str): The type of hardware to check (e.g., "RMU").
        start_event (Optional[int]): The starting event ID of the window.
        end_event (Optional[int]): The ending event ID of the window.
        element_id (Optional[str]): If provided, check only this specific component ID.
                                    If None, check all components of 'element_type'.

    Returns:
        dict: A report on component activity.
    """
    print(f"--- TOOL: Running check_component_activity for {element_type} (Events: {start_event}-{end_event}) ---")

    # --- 1. Basic Validations ---
    if HARDWARE_MAP_DF.empty:
        return {"error": "Hardware map is not loaded. Cannot perform lookup."}
    if isinstance(element_id, str) and element_id.strip().lower() == "none":
        element_id = None
    element_type_upper = element_type.upper()
    query_col = COLUMN_MAP.get(element_type_upper)
    if not query_col:
        return {"error": f"Invalid element_type '{element_type}'. Must be one of {list(COLUMN_MAP.keys())}."}

    # --- 2. Load Run Data & Filter by Event Window ---
    try:
        df = pd.read_csv(run_data_path)
        if 'Event' not in df.columns or 'PMTID' not in df.columns:
            return {"error": "Run data file must contain 'Event' and 'PMTID' columns."}
        
        # --- START MODIFICATION ---
        
        # Step 2a: Ensure the 'Event' column is numeric.
        try:
            event_ids_numeric = pd.to_numeric(df['Event'])
        except ValueError:
            return {"error": f"The 'Event' column contains non-numeric values (e.g., '{df['Event'].iloc[0]}') and cannot be filtered."}

        # Step 2b: Build the event window filter
        
        # Start with a filter that selects all rows
        window_filter = pd.Series(True, index=df.index)
        
        # Get actual min/max event IDs for reporting
        min_event_id = event_ids_numeric.min()
        max_event_id = event_ids_numeric.max()

        # Handle user-provided start_event
        if start_event is not None:
            if start_event > max_event_id:
                return {"error": f"start_event {start_event} is after the last event {max_event_id}."}
            window_filter &= (event_ids_numeric >= start_event)
            event_window_start = start_event
        else:
            event_window_start = min_event_id # Default to actual start

        # Handle user-provided end_event
        if end_event is not None:
            if end_event < min_event_id:
                return {"error": f"end_event {end_event} is before the first event {min_event_id}."}
            window_filter &= (event_ids_numeric <= end_event)
            event_window_end = end_event
        else:
            event_window_end = max_event_id # Default to actual end
            
        # Final validation
        if event_window_start > event_window_end:
             return {"error": f"Invalid window: start_event {event_window_start} is greater than end_event {event_window_end}."}

        # Step 2c: Filter the DataFrame to the specified window
        recent_hits_df = df[window_filter]
        
        # --- END MODIFICATION ---
        
        # Get the set of all PMTs that had at least one hit in this window
        active_pmt_ids: Set[int] = set(recent_hits_df['PMTID'].unique())
        
        if recent_hits_df.empty:
             print(f"--- TOOL INFO: No hits of any kind were recorded in the event window {event_window_start}-{event_window_end}.")
        
    except FileNotFoundError:
        return {"error": f"Run data file not found at {run_data_path}"}
    except Exception as e:
        return {"error": f"Error loading run data: {e}"}

    # --- 3. Perform Analysis (This section is unchanged, but return dicts are updated) ---
    
    # Case A: Check for a *specific* component ID
    if element_id :
        normalized_id: Union[int, str, float] = np.nan
        if element_type_upper == "RMU":
            normalized_id = _normalize_rmu2nd_label(element_id)
        elif element_type_upper in ["PMT", "GCU", "BEC"]:
            normalized_id = _to_int_safe(element_id)
            if not pd.isna(normalized_id):
                normalized_id = int(normalized_id)

        if pd.isna(normalized_id) or (isinstance(normalized_id, str) and normalized_id == ""):
             return {"error": f"Invalid {element_type_upper} ID format '{element_id}'."}
        
        component_pmts_df = HARDWARE_MAP_DF[HARDWARE_MAP_DF[query_col] == normalized_id]
        expected_pmt_ids: Set[int] = set(component_pmts_df['PMT_ID'].unique())
        
        if not expected_pmt_ids:
            return {"error": f"Component {element_type} {element_id} not found in hardware map."}
        
        active_pmts_for_component = expected_pmt_ids.intersection(active_pmt_ids)
        
        if not active_pmts_for_component:
            return {
                "query": f"{element_type} {element_id}",
                "status": "SILENT",
                "event_window_checked": f"[{event_window_start}, {event_window_end}]",
                "details": f"Component {element_type} {element_id} was silent. None of its {len(expected_pmt_ids)} PMTs were active in this window."
            }
        else:
            return {
                "query": f"{element_type} {element_id}",
                "status": "ACTIVE",
                "event_window_checked": f"[{event_window_start}, {event_window_end}]",
                "active_pmts": len(active_pmts_for_component),
                "total_pmts": len(expected_pmt_ids),
                "details": f"Component {element_type} {element_id} was active. {len(active_pmts_for_component)} of its {len(expected_pmt_ids)} PMTs were active in this window."
            }

    # Case B: Check *all* components of a given type
    else:
        all_component_ids: Set = set(HARDWARE_MAP_DF[query_col].unique())
        
        active_map_df = HARDWARE_MAP_DF[HARDWARE_MAP_DF['PMT_ID'].isin(active_pmt_ids)]
        active_component_ids: Set = set(active_map_df[query_col].unique())
        
        missing_component_ids = all_component_ids - active_component_ids
        
        return {
            "query": f"Global check for {element_type}",
            "status": "GLOBAL_CHECK",
            "event_window_checked": f"[{event_window_start}, {event_window_end}]",
            "element_type": element_type,
            "missing_components": sorted(list(missing_component_ids)),
            "missing_count": len(missing_component_ids),
            "active_count": len(active_component_ids),
            "total_count": len(all_component_ids)
        }

def find_events_with_missing_components(run_data_path: str, element_type: str) -> dict:
    """
    Audits the entire run for missing components and saves a full report to a JSON file.
    Returns a high-level summary to the agent, NOT the full list of events.
    Valid 'element_type' values are: 'PMT', 'GCU', 'BEC', 'RMU'.

    Args:
        run_data_path (str): The path to the aggregated run data CSV file.
        element_type (str): The type of hardware to check (e.g., "RMU").

    Returns:
        dict: A high-level summary of the audit, including the path to the full report file.
    """ 

    print(f"--- TOOL: Running find_events_with_missing_components for {element_type} ---")

    # --- 1. Validations ---
    if HARDWARE_MAP_DF.empty:
        return {"error": "Hardware map is not loaded. Cannot perform lookup."}
    
    element_type_upper = element_type.upper()
    query_col = COLUMN_MAP.get(element_type_upper)
    if not query_col:
        return {"error": f"Invalid element_type '{element_type}'. Must be one of {list(COLUMN_MAP.keys())}."}

    # --- 2. Load Data and Map ---
    try:
        df = pd.read_csv(run_data_path)
        if 'Event' not in df.columns or 'PMTID' not in df.columns:
            return {"error": "Run data file must contain 'Event' and 'PMTID' columns."}
    except FileNotFoundError:
        return {"error": f"Run data file not found at {run_data_path}"}
    except Exception as e:
        return {"error": f"Error loading run data: {e}"}

    # --- 3. Perform Analysis (Inverted Logic) ---
    
    # Get the set of ALL components that *should* exist
    all_component_ids: Set = set(HARDWARE_MAP_DF[query_col].unique())
    if not all_component_ids:
         return {"error": f"No components of type {element_type} found in hardware map."}

    # Create a simple PMT -> Component map for fast lookup
    pmt_to_component_map = HARDWARE_MAP_DF.set_index('PMT_ID', drop=False)[query_col]

    # Map the 'PMTID' in the run data to its component
    df['Component'] = df['PMTID'].map(pmt_to_component_map)
    
    # Drop any hits that don't belong to a component of this type
    df_mapped = df.dropna(subset=['Component'])

    # Get a set of all unique event IDs in the entire run
    all_event_ids_in_run = set(df['Event'].unique().astype(str))

    # Find all active components for each event
    # This gives: { 'event_1': {'RMU-1', 'RMU-2'}, 'event_2': {'RMU-1'} }
    events_with_active_components = df_mapped.groupby('Event')['Component'].apply(set)
    
    # Get all events that had at least one *mapped* hit
    active_event_ids = set(events_with_active_components.index.astype(str))
    
    # Find events that had NO mapped hits at all
    # These events are "missing" ALL components
    silent_event_ids = all_event_ids_in_run - active_event_ids
    
    # --- 4. Invert the Map to be Component-Centric ---
    
    # Initialize a report map where each component is missing from the silent events
    # We must use list() so we can append later
    inverted_missing_map = {
        str(comp): list(silent_event_ids) for comp in all_component_ids
    }

    # Now, iterate through the *active* events and find partial misses
    for event_id, active_set in events_with_active_components.items():
        missing_for_this_event = all_component_ids - active_set
        
        event_id_str = str(event_id)
        for comp in missing_for_this_event:
            inverted_missing_map[str(comp)].append(event_id_str)

    # --- 5. Format Final Report ---
    full_report_dict = {}
    total_missing_component_count = 0
    for component, events in inverted_missing_map.items():
        if events:
            total_missing_component_count += 1
            full_report_dict[component] = {
                "missing_count": len(events),
                "missing_events_list": sorted(events)
            }
    report_filename = f"{run_data_path.replace('.csv', '')}_{element_type}_audit_report.json"
    
    try:
        with open(report_filename, 'w') as f:
            json.dump(full_report_dict, f, indent=2)
        print(f"--- TOOL: Full audit report saved to {report_filename} ---")
    except Exception as e:
        return {"error": f"Could not save full report to {report_filename}. Error: {e}"}

    # This is the small summary that goes back to the LLM
    summary_report = {
        "query": f"Audit of missing {element_type} components",
        "element_type": element_type,
        "total_events_in_run": len(all_event_ids_in_run),
        "components_with_misses_count": total_missing_component_count,
        "full_report_saved_to": report_filename
    }
    
    return summary_report

#print(check_component_activity('test_data/scenario7_one.csv', 'RMU','31','50','None'))
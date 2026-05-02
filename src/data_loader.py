"""
Data Loading — Memory-efficient PeMS .txt file parser
Two-pass approach: 1) Find best stations  2) Load only those stations
"""
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import (DATA_DIR, RAW_USECOLS, RAW_COLNAMES,
                        TOP_N_STATIONS, MIN_PCT_OBSERVED, PROCESSED_DIR)


def discover_files():
    """Find all PeMS daily .txt files in DATA_DIR."""
    pattern = os.path.join(DATA_DIR, 'd04_text_station_5min_*.txt')
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} data files:")
    for f in files:
        mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f):>45s}  {mb:>7.1f} MB")
    return files


def _read_chunks(file_path, usecols, chunksize=150_000):
    """Yield DataFrame chunks from a single PeMS file."""
    return pd.read_csv(
        file_path, header=None, usecols=usecols,
        chunksize=chunksize, on_bad_lines='skip', low_memory=False
    )


def analyze_stations(files):
    """
    Pass 1 — Lightweight scan to compute per-station statistics.
    Only reads Station (col 1), Pct_Observed (col 8), Total_Flow (col 9).
    Returns a DataFrame with avg_flow, avg_pct, count per station.
    """
    stats = {}
    print("\n📊 Pass 1: Scanning stations across all files...")
    for fpath in tqdm(files, desc="Files"):
        for chunk in _read_chunks(fpath, usecols=[1, 8, 9]):
            chunk.columns = ['Station', 'Pct_Observed', 'Total_Flow']
            chunk['Station'] = chunk['Station'].astype(str)
            chunk['Pct_Observed'] = pd.to_numeric(chunk['Pct_Observed'], errors='coerce').fillna(0)
            chunk['Total_Flow'] = pd.to_numeric(chunk['Total_Flow'], errors='coerce').fillna(0)

            for station, grp in chunk.groupby('Station'):
                if station not in stats:
                    stats[station] = {'flow_sum': 0.0, 'pct_sum': 0.0, 'count': 0}
                stats[station]['flow_sum'] += grp['Total_Flow'].sum()
                stats[station]['pct_sum'] += grp['Pct_Observed'].sum()
                stats[station]['count'] += len(grp)

    df = pd.DataFrame(stats).T
    df['avg_flow'] = df['flow_sum'] / df['count']
    df['avg_pct'] = df['pct_sum'] / df['count']
    df.index.name = 'Station'
    return df


def select_stations(station_stats, top_n=TOP_N_STATIONS, min_pct=MIN_PCT_OBSERVED):
    """Select top-N stations by avg_flow with sufficient data quality."""
    qualified = station_stats[station_stats['avg_pct'] >= min_pct].copy()
    selected = qualified.nlargest(top_n, 'avg_flow')
    print(f"\n✅ Selected {len(selected)} stations  "
          f"(avg_pct ≥ {min_pct}%, top by flow)")
    print(f"   Flow range : {selected['avg_flow'].min():.0f} – "
          f"{selected['avg_flow'].max():.0f} vehicles/5min")
    print(f"   Pct range  : {selected['avg_pct'].min():.1f}% – "
          f"{selected['avg_pct'].max():.1f}%")
    return selected.index.tolist()


def load_station_data(files, selected_stations):
    """
    Pass 2 — Load full feature columns for selected stations only.
    Returns a single DataFrame sorted by Station + Timestamp.
    """
    selected_set = set(str(s) for s in selected_stations)
    frames = []
    print(f"\n📥 Pass 2: Loading data for {len(selected_set)} stations...")

    for fpath in tqdm(files, desc="Files"):
        for chunk in _read_chunks(fpath, usecols=RAW_USECOLS):
            chunk.columns = RAW_COLNAMES
            chunk['Station'] = chunk['Station'].astype(str)
            filtered = chunk[chunk['Station'].isin(selected_set)]
            if len(filtered) > 0:
                frames.append(filtered.copy())

    df = pd.concat(frames, ignore_index=True)

    # Parse types
    for col in ['Pct_Observed', 'Total_Flow', 'Avg_Occupancy', 'Avg_Speed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M:%S')
    df.sort_values(['Station', 'Timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"   Loaded {len(df):,} rows  |  "
          f"{df['Station'].nunique()} stations  |  "
          f"{df['Timestamp'].dt.date.nunique()} days")
    return df


def load_data(use_cache=True):
    """
    Full pipeline: discover → analyze → select → load.
    Caches processed DataFrame as parquet for fast re-runs.
    """
    cache_path = os.path.join(PROCESSED_DIR, 'selected_stations_raw.parquet')

    if use_cache and os.path.exists(cache_path):
        print("♻️  Loading cached data from", cache_path)
        df = pd.read_parquet(cache_path)
        print(f"   {len(df):,} rows  |  {df['Station'].nunique()} stations")
        return df

    files = discover_files()
    station_stats = analyze_stations(files)
    selected = select_stations(station_stats)
    df = load_station_data(files, selected)

    df.to_parquet(cache_path, index=False)
    print(f"💾 Cached to {cache_path}")
    return df

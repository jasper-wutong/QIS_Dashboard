"""QIS SubBook Dashboard - Flask backend.

Serves a single-page dashboard that displays QIS SubBook data from
the latest EDSLib Excel file.  Single-ticker research is delegated
to ``research_cli.py`` via subprocess (avoids Copilot SDK + Flask
conflicts).
"""

import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, render_template
from ticker_mapping import populate_names

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__)

# -- Helpers -------------------------------------------------------------------

def safe_val(v):
    """Convert a pandas value to a JSON-safe Python scalar."""
    if pd.isna(v):
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return v
    return str(v)


def to_records(frame: pd.DataFrame, columns: list) -> list:
    """Convert a DataFrame to a list-of-lists matching *columns*."""
    return [[safe_val(row.get(c)) for c in columns] for _, row in frame.iterrows()]


# -- Data loading & processing ------------------------------------------------

DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "VSCodePyScripts", "QIS_Dashboard")
FILE_PREFIX = "EDSLib Realtime Result as of"


def load_data():
    """Read the most recent EDSLib Excel file and return (df_qis, date_str)."""
    files = sorted(
        (f for f in os.listdir(DATA_DIR) if f.startswith(FILE_PREFIX) and f.endswith(".xlsx")),
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No '{FILE_PREFIX}*.xlsx' files found in {DATA_DIR}")
    filepath = os.path.join(DATA_DIR, files[0])
    print(f"Reading: {files[0]}")

    df = pd.read_excel(filepath, engine="openpyxl", header=2)
    df_qis = df[df["SubBook"].astype(str).str.contains("QIS", case=False, na=False)].reset_index(drop=True)
    date_str = files[0].replace(f"{FILE_PREFIX} ", "").replace(".xlsx", "")
    return df_qis, date_str


def aggregate_by_wind_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows sharing the same Wind Ticker (sum numerics, first for rest)."""
    if df.empty:
        return df
    sum_cols = [
        "Delta($)", "SOD Delta($)", "PDE Delta($)", "%Gamma($)", "PDE Gamma(%)",
        "Vega($)", "FX Vega($)", "Theta",
        "现货市值", "期货市值", "风险敞口", "T+1 风险敞口", "风险敞口(PDE)",
        "当日损益", "当日损益(PDE)", "当日损益(合约端)", "当日损益(对冲端)",
        "存续名义本金", "Delta Shares", "Open Shares", "Need To Trade",
        "Exposure pnl", "Gamma pnl", "Theta pnl", "Vega pnl",
        "Borrow pnl", "Residual",
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]
    first_cols = [c for c in df.columns if c not in sum_cols and c != "Wind Ticker"]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in first_cols})
    return df.groupby("Wind Ticker", as_index=False).agg(agg)


def calc_summary(frame: pd.DataFrame) -> dict:
    """Compute aggregate summary metrics for a DataFrame slice."""
    def s(col):
        return float(frame[col].sum()) if col in frame.columns else 0.0
    return {
        "count": int(len(frame)),
        "delta": s("Delta($)"),
        "sod_delta": s("SOD Delta($)"),
        "vega": s("Vega($)"),
        "theta": s("Theta"),
        "gamma": s("%Gamma($)"),
        "exposure": s("风险敞口"),
        "pnl": s("当日损益"),
        "pnl_contract": s("当日损益(合约端)"),
        "pnl_hedge": s("当日损益(对冲端)"),
        "notional": s("存续名义本金"),
    }


def build_subbook_summaries(frame: pd.DataFrame, subbooks: list) -> dict:
    return {
        sb: calc_summary(frame[frame["SubBook"] == sb])
        for sb in subbooks
        if (frame["SubBook"] == sb).any()
    }


def process_data(df_qis: pd.DataFrame, date_str: str) -> dict:
    """Split, aggregate, summarise and return the full dashboard payload."""
    populate_names(df_qis)

    is_index = df_qis["Wind Ticker"].isna() & df_qis["涨跌幅"].notna()
    df_index = df_qis[is_index].reset_index(drop=True)
    df_other_raw = df_qis[~is_index].reset_index(drop=True)
    df_other = aggregate_by_wind_ticker(df_other_raw)

    columns = list(df_qis.columns)
    subbooks = sorted(df_qis["SubBook"].unique().tolist())

    for col in columns:
        if col not in df_index.columns:
            df_index[col] = None
        if col not in df_other.columns:
            df_other[col] = None

    return {
        "columns": columns,
        "subbooks": subbooks,
        "index_records": to_records(df_index, columns),
        "other_records": to_records(df_other, columns),
        "index_summary": calc_summary(df_index),
        "other_summary": calc_summary(df_other_raw),
        "total_summary": calc_summary(df_qis),
        "index_sb_summary": build_subbook_summaries(df_index, subbooks),
        "other_sb_summary": build_subbook_summaries(df_other_raw, subbooks),
        "index_count": len(df_index),
        "other_raw_count": len(df_other_raw),
        "other_merged_count": len(df_other),
    }


# -- Research cache -----------------------------------------------------------
RESEARCH_CACHE: dict = {}   # {name: {ok, name, model, content, ...}}
CACHE_DATE: str = ""

# -- Load data on startup -----------------------------------------------------
print("Loading data...")
df_qis, DATE_STR = load_data()
DATA = process_data(df_qis, DATE_STR)
CACHE_DATE = DATE_STR
print(f"Data loaded: {DATE_STR}")
print(f"  指数: {DATA['index_count']} rows")
print(f"  其他标的物: {DATA['other_raw_count']} raw -> {DATA['other_merged_count']} merged")


# -- Routes --------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/data")
def api_data():
    return jsonify({
        "columns": DATA["columns"],
        "subbooks": DATA["subbooks"],
        "index_data": DATA["index_records"],
        "other_data": DATA["other_records"],
        "index_summary": DATA["index_summary"],
        "other_summary": DATA["other_summary"],
        "total_summary": DATA["total_summary"],
        "index_sb": DATA["index_sb_summary"],
        "other_sb": DATA["other_sb_summary"],
        "date_str": DATE_STR,
    })


@app.route("/api/refresh")
def refresh():
    global df_qis, DATE_STR, DATA, RESEARCH_CACHE, CACHE_DATE
    df_qis, DATE_STR = load_data()
    DATA = process_data(df_qis, DATE_STR)
    RESEARCH_CACHE.clear()
    CACHE_DATE = DATE_STR
    return jsonify({"status": "ok", "date_str": DATE_STR})


# -- Single-ticker research via subprocess ------------------------------------

RESEARCH_CLI = str(Path(__file__).resolve().parent / "research_cli.py")


@app.route("/api/research")
def api_research():
    """Delegate to research_cli.py for Copilot SDK research on one ticker."""
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "missing 'name' parameter", "content": ""}), 400

    # Check cache
    if name in RESEARCH_CACHE:
        cached = dict(RESEARCH_CACHE[name])
        cached["cached"] = True
        return jsonify(cached)

    price = request.args.get("price", "NA").strip() or "NA"
    change = request.args.get("change", "NA").strip() or "NA"
    exposure = request.args.get("exposure", "NA").strip() or "NA"

    cmd = [
        sys.executable, RESEARCH_CLI, name,
        "--price", price, "--change", change, "--exposure", exposure,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parent),
        )
        if proc.returncode != 0:
            return jsonify({
                "ok": False, "name": name,
                "error": f"research_cli exited {proc.returncode}: {proc.stderr[:500]}",
                "content": "",
            })

        # Last stdout line should be JSON
        lines = proc.stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                result = json.loads(line.strip())
                # Cache successful results
                if result.get("ok") and result.get("content"):
                    RESEARCH_CACHE[name] = result
                return jsonify(result)
            except json.JSONDecodeError:
                continue
        return jsonify({
            "ok": False, "name": name,
            "error": "Failed to parse CLI output",
            "content": proc.stdout[:1000],
        })

    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "name": name, "error": "Research timed out (180s)", "content": ""})
    except Exception as exc:
        return jsonify({"ok": False, "name": name, "error": str(exc), "content": ""})


# -- Batch research via subprocess --------------------------------------------

@app.route("/api/research/batch", methods=["POST"])
def api_research_batch():
    """Batch research for multiple tickers in one Copilot SDK call."""
    body = request.get_json(silent=True) or {}
    tickers = body.get("tickers", [])
    if not tickers:
        return jsonify({"ok": False, "error": "missing 'tickers' array", "results": []}), 400

    names = [t.get("name", "?") for t in tickers]
    print(f"[BATCH] Received {len(tickers)} tickers: {', '.join(names)}")

    # Separate cached vs uncached
    cached_results = []
    uncached_tickers = []
    for t in tickers:
        name = t.get("name", "").strip()
        if not name:
            continue
        if name in RESEARCH_CACHE:
            entry = dict(RESEARCH_CACHE[name])
            entry["cached"] = True
            cached_results.append(entry)
        else:
            uncached_tickers.append(t)

    if cached_results:
        print(f"[BATCH]   {len(cached_results)} cached, {len(uncached_tickers)} to analyze")

    # If all cached, return immediately
    if not uncached_tickers:
        print("[BATCH]   All cached, returning immediately")
        return jsonify({"ok": True, "results": cached_results, "all_cached": True})

    # Call research_cli.py in batch mode
    batch_json = json.dumps(uncached_tickers, ensure_ascii=False)
    cmd = [sys.executable, RESEARCH_CLI, "--batch-json", batch_json]
    uncached_names = [t.get("name", "?") for t in uncached_tickers]
    print(f"[BATCH]   Calling research_cli.py for: {', '.join(uncached_names)}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parent),
        )

        # Log stderr (contains progress info from research_cli.py)
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                if line.strip():
                    print(f"[BATCH/CLI] {line.strip()}")

        if proc.returncode != 0:
            print(f"[BATCH]   ERROR: research_cli exited {proc.returncode}")
            return jsonify({
                "ok": False,
                "error": f"research_cli exited {proc.returncode}: {proc.stderr[:500]}",
                "results": cached_results,
            })

        # Parse JSON from last stdout line
        lines = proc.stdout.strip().split("\n")
        batch_result = None
        for line in reversed(lines):
            try:
                batch_result = json.loads(line.strip())
                break
            except json.JSONDecodeError:
                continue

        if not batch_result:
            print("[BATCH]   ERROR: Failed to parse CLI output")
            return jsonify({
                "ok": False,
                "error": "Failed to parse batch CLI output",
                "results": cached_results,
            })

        # Cache successful individual results
        new_results = batch_result.get("results", [])
        ok_count = 0
        for r in new_results:
            if r.get("ok") and r.get("content"):
                RESEARCH_CACHE[r["name"]] = r
                ok_count += 1
        print(f"[BATCH]   Done: {ok_count}/{len(new_results)} succeeded, model={batch_result.get('model','?')}")

        return jsonify({
            "ok": True,
            "model": batch_result.get("model", ""),
            "results": cached_results + new_results,
        })

    except subprocess.TimeoutExpired:
        print("[BATCH]   ERROR: Timed out (300s)")
        return jsonify({"ok": False, "error": "Batch research timed out (300s)", "results": cached_results})
    except Exception as exc:
        print(f"[BATCH]   ERROR: {exc}")
        return jsonify({"ok": False, "error": str(exc), "results": cached_results})


# -- Entrypoint ----------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5050)

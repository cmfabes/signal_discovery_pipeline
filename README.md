# Signal Discovery Analytics Pipeline

This repository implements a production‑ready pipeline for discovering leading operational signals from time‑series data.  The goal is to ingest raw operational and market data (e.g. vessel port calls, queue lengths, inventories), compute robust anomaly scores, evaluate lead/lag relationships with market instruments, and produce actionable insights for executives and investors.

## Overview

The pipeline performs the following steps:

1. **Data ingestion** – Load multiple time‑series sources (operational metrics and market data) and align them on a common daily frequency.
2. **Transforms** – Compute rolling means, percentage changes, z‑scores, and anomaly flags on each series.
3. **Signal construction** – Define custom metrics such as the "transshipment ratio" (containers routed via feeder hubs divided by total arrivals) per product class.
4. **Statistical analysis** – Compute pairwise and lagged correlations, using bootstrapped confidence intervals to assess statistical significance and effect sizes.
5. **Visualization and reporting** – Generate high‑resolution plots and a concise one‑pager PDF summarizing what changed, why it matters, and recommended actions.
6. **Streamlit UI** – Provide a simple web interface that allows users to upload their own data, select tickers/date ranges, and view results.

## Getting Started

### Prerequisites

* Python 3.9 or higher
* Git (for cloning the repo)
* A C compiler (required by some dependencies)

### Setting up the environment

To get started quickly, run the `init_project.sh` script.  It creates a fresh virtual environment, installs the required Python packages, and runs a basic diagnostics script to ensure everything is in place.

```bash
bash init_project.sh
```

If the script completes without errors, you are ready to begin developing and running the pipeline.  Subsequent steps will guide you through implementing the transforms, statistical analyses, and Streamlit interface.

### Run the Streamlit App (One Command)

The app provides a non-technical, repeatable interface with an executive summary and downloadable PDF report.

1. Activate your virtual environment (if not already activated).
2. Launch the app:

```
streamlit run app.py
```

3. In the UI:
   - Upload one or more operational CSVs (with a `date` column and one or more numeric columns).
   - Choose market data source:
     - Yahoo Finance (online) — enter tickers and date range; or
     - Upload CSV (offline) — supply a CSV with `Date` and one or more ticker columns.
   - Enter lags (e.g., `-10,-5,0,5,10`).
   - Click Run Analysis to see the executive summary, plots (normalized overlays + lag heatmap), and download a polished PDF.

Tip: Use `examples/sample_operational.csv` as a quick test file.

### Port Signals Predictor (Ports → Market Impact)

Use the second workflow in the app to backtest whether port activity signals predict market returns:

- Upload a long-format CSV with columns like `Date, port, anchored_count, arrivals_count, ...`.
- Select a port and features; choose feature lags (e.g., `0,1,2,3,5,10`).
- Choose market source (Yahoo or upload a CSV with `Date` + ticker column) and set a forward return horizon (e.g., 10 days).
- Run the backtest to get out-of-sample R², MAE, and directional accuracy, plus a predictions-vs-actual chart and a CSV export.

Live Mode with DuckDB
- Set source to DuckDB (Live Mode) and specify DB path (e.g., `data/ports.duckdb`) and table name (default `port_features`).
- Click “Load from DuckDB”, choose a port, features, and run the backtest.
- After training, the app will also produce a “Live Prediction” for the next horizon using the latest features row.

Templates and Geofences
- Example features CSV: `examples/port_features_template.csv`
- Example port geofences: `examples/ports.geojson` (approximate polygons for LA/LB, NY/NJ, Shanghai)

### Live AIS (Datalastic) Integration

Fetch recent AIS positions and build daily features for priority ports.

Setup:
- Get a Datalastic API key and set it as an environment variable:
  - macOS/Linux: `export DATALASTIC_API_KEY=YOUR_KEY`
  - Windows (PowerShell): `$env:DATALASTIC_API_KEY="YOUR_KEY"`

Fetch last 24h and append to DuckDB:
```
python -m scripts.fetch_ais_datalastic \
  --geojson examples/ports.geojson \
  --ports "Los Angeles / Long Beach,New York / New Jersey" \
  --hours 24 \
  --db data/ports.duckdb --table port_features
```

Then use Live Mode in the app (DuckDB) to backtest and predict.

Tip: Schedule the command hourly/daily via cron or Task Scheduler to keep the DB fresh.

Backfill multiple days (overseas ports):
```
python -m scripts.backfill_datalastic \
  --geojson examples/ports.geojson \
  --ports "Shenzhen/Yantian,Ningbo-Zhoushan,Singapore" \
  --days 60 \
  --db data/ports.duckdb --table port_features
```

US historical backfill (MarineCadastre):
1. Download monthly AIS CSVs from marinecadastre.gov/ais and put them in a folder.
2. Run:
```
python -m scripts.ingest_marinecadastre \
  --data-dir /path/to/ais_csvs \
  --geojson examples/ports.geojson \
  --ports "Los Angeles / Long Beach,New York / New Jersey" \
  --db data/ports.duckdb --table port_features
```

### Daily Scheduler (cron)

Example: fetch Shenzhen/Yantian daily at 02:00 UTC and append to DuckDB.

1. Ensure `DATALASTIC_API_KEY` is set for the user account running cron.
2. Edit crontab: `crontab -e` and add:

```
0 2 * * * cd /Users/fabes/Desktop/signal_discovery && \
  /Users/fabes/Desktop/signal_discovery/venv/bin/python -m scripts.fetch_ais_datalastic \
  --geojson examples/ports.geojson \
  --ports "Shenzhen/Yantian" \
  --hours 24 \
  --db data/ports.duckdb --table port_features >> outputs/cron.log 2>&1
```

Open the app in Live Mode (DuckDB) to analyze and predict using the newest data.

Multi-port daily fetch (rotate ports):
```
0 2 * * * cd /Users/fabes/Desktop/signal_discovery && \
  for P in "Shenzhen/Yantian" "Ningbo-Zhoushan" "Singapore"; do \
    /Users/fabes/Desktop/signal_discovery/venv/bin/python -m scripts.fetch_ais_datalastic \
      --geojson examples/ports.geojson --ports "$P" --hours 24 \
      --db data/ports.duckdb --table port_features >> outputs/cron.log 2>&1; \
  done
```

Refined polygons are available in `examples/ports_refined.geojson` for more precise terminal/anchorage targeting.

### AIS Stub Usage (Optional)

You can derive daily port features from AIS points using a lightweight, local stub (no external GIS dependencies):

```python
from src.ais import load_port_geofences, assign_port_to_points, derive_daily_port_features
import pandas as pd

# Load geofences (replace with your GeoJSON)
fences = load_port_geofences('examples/ports.geojson')

# Load AIS points (CSV must include: timestamp, mmsi, lat, lon, speed_knots, status)
ais = pd.read_csv('path/to/ais_points.csv')

# Tag points with port names
tagged = assign_port_to_points(ais, fences)

# Build daily features per port
features = derive_daily_port_features(tagged)
features.to_csv('daily_port_features.csv', index=False)
```

For production-grade spatial joins and higher performance, use GeoPandas/Shapely or your data provider’s geofencing.

### Priority Ports (Starter List)

Focus on these ports to cover most US import flows and key origins:

- US gateways: Los Angeles / Long Beach, New York / New Jersey, Savannah, Houston, Charleston, Norfolk (Hampton Roads), Seattle/Tacoma (NWSA), Oakland
- Canada/Mexico (US supply chain): Vancouver, Prince Rupert, Lázaro Cárdenas, Manzanillo
- Asia origins: Shanghai, Ningbo-Zhoushan, Shenzhen/Yantian, Qingdao, Tianjin, Busan, Singapore, Kaohsiung, Laem Chabang, Tanjung Pelepas, Port Klang, Ho Chi Minh (Cat Lai)
- Europe: Rotterdam, Antwerp-Bruges, Hamburg

### One-Click Start (macOS)

Option A — Double-clickable launcher:

1. Ensure the virtualenv is set up (`bash init_project.sh`).
2. Make the launcher executable:

```
chmod +x Start\ Signal\ Discovery.command
```

3. Drag `Start Signal Discovery.command` to your Desktop.
4. Double-click it. It opens the app at `http://localhost:8501` and logs to `outputs/streamlit.log`.

Option B — Automator application (no Terminal window):

1. Open Automator → New → Application.
2. Add “Run Shell Script”; set Shell `/bin/zsh`, Pass input “to stdin”.
3. Script:

```
/Users/fabes/Desktop/signal_discovery/venv/bin/python3 -m streamlit run /Users/fabes/Desktop/signal_discovery/app.py --server.port 8501 --server.headless false >/tmp/signal_discovery.log 2>&1 &
sleep 2
open http://localhost:8501
```

4. Save as `Signal Discovery.app` to Desktop. Double-click to launch.

### One-Click Start (Windows)

Double-click `Start_Signal_Discovery.bat`. If the venv isn’t set up yet, run `bash init_project.sh` in Git Bash or WSL first.

### Command-Line Pipeline (Advanced)

You can also run the pipeline headlessly for scripting and batch usage:

```
python -m src.pipeline \
  --op-files examples/sample_operational.csv \
  --date-col date \
  --value-cols value \
  --tickers SPY \
  --start-date 2025-09-01 \
  --end-date 2025-09-11 \
  --lags -5 0 5
```

This prints per-pair lag correlations and includes FDR-adjusted significance and stability in the app’s executive summary.

### Demo DuckDB (Live Mode Quickstart)

Build a demo database from the provided template and try Live Mode:

Option A — run as a module (preferred):
```
python -m scripts.build_demo_duckdb --db data/ports.duckdb --table port_features --csv examples/port_features_template.csv --mode replace
```

Option B — run directly (adds project root to sys.path automatically):
```
python scripts/build_demo_duckdb.py --db data/ports.duckdb --table port_features --csv examples/port_features_template.csv --mode replace
```

Then in the app, choose Port Signals Predictor → DuckDB (Live Mode) → `data/ports.duckdb`, table `port_features`, Load, and run the backtest.

### Docker

Build and run the app in a container:

```
docker build -t signal-discovery .
docker run --rm -p 8501:8501 signal-discovery
```

### Prefect Flow (Scheduled Fetch)

Use the provided Prefect flow to fetch AIS daily and append features to DuckDB.

```
export DATALASTIC_API_KEY=YOUR_KEY
python -m scripts.prefect_flow \
  --geojson examples/ports.geojson \
  --ports "Shenzhen/Yantian,Ningbo-Zhoushan,Singapore" \
  --hours 24 \
  --db data/ports.duckdb --table port_features
```

The flow writes a status file to `outputs/flow_status.json` that the app can read. You can run this via cron or a Prefect agent/work queue.

## Repository Structure

```
signal_discovery/
├── init_project.sh       # One‑command setup script for environment and diagnostics
├── requirements.txt      # Python dependencies
├── diagnostics.py        # Preliminary diagnostics to verify environment
├── src/                  # Source code for the pipeline
│   ├── __init__.py
│   ├── data_ingest.py    # Functions for loading and aligning data (to be implemented)
│   ├── transforms.py     # Rolling stats, anomaly scores, etc. (to be implemented)
│   ├── analysis.py       # Correlation and statistical testing (to be implemented)
│   └── reporting.py      # Streamlit UI and PDF generation (to be implemented)
└── README.md             # This file
```

## Next Steps

After running the setup script, the next step will be to implement the data ingestion functions in `src/data_ingest.py`.  When you are ready, return to the chat for a single, specific command or file edit to continue building the pipeline.

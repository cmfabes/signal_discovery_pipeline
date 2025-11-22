# Signal Discovery Pipeline

This project started as a personal workflow to make sense of operational and market data, especially around maritime logistics. Most of the raw inputs come from AIS (Automatic Identification System) vessel data — movement logs, port arrivals, queue times, and related operational patterns. The goal was simple: build a repeatable way to spot meaningful changes in behavior before they show up in headlines or market reports.

Over time, this evolved into a modular pipeline that cleans the data, builds rolling metrics, and highlights events that deserve a closer look.

---

## What the Pipeline Does

- Converts raw AIS data into standardized, usable tables  
- Cleans and aligns timestamps, vessel movements, and port activity  
- Computes rolling averages, rate-of-change, volatility, and congestion metrics  
- Flags unusual movements or operational shifts based on configurable thresholds  
- Generates summaries that make trend changes easy to interpret  

This pipeline isn’t built for automation or trading. It’s designed to provide structure when analyzing how operational signals might connect to broader market behavior.

---

## How It Helps With Analysis

This workflow makes it easier to:

- Turn messy AIS logs into something interpretable  
- Trace how vessel behavior affects timing, congestion, or market conditions  
- Identify trend breaks early rather than reacting after the fact  
- Combine operational indicators with financial or commodity data  
- Review potential issues with enough context to make an informed call  
- Keep each run consistent through clear, documented steps  

It reflects how analysts work when the information isn’t perfect: you organize what you have, clean it, quantify it, and look for movement that stands out.

```
signal_discovery/
│
├── app.py               # Main runner for the pipeline
├── diagnostics.py       # Produces readable summaries
├── requirements.txt
│
├── src/                 # Core logic for AIS + market data processing
│   ├── ingest.py
│   ├── transforms.py
│   ├── signals.py
│   └── utils.py
│
├── scripts/             # Helper scripts for data checks and conversions
│
├── examples/            # Sample datasets and demo notebooks
│
└── outputs/             # Generated results
```
```

## Running the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
Run the pipeline:

python app.py


Outputs are written to the outputs/ folder.

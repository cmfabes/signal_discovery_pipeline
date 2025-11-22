# Signal Discovery Pipeline

This project analyzes operational and market data to identify emerging patterns, anomalies, and behavior shifts that may signal future changes in logistics or commodity markets. The pipeline combines structured data processing, time-series analysis, and event-based logic to flag movements that merit closer review. It is designed to support analytical decision-making while remaining modular and easy to extend.

## Overview

The pipeline ingests external datasets, applies a series of transformations, computes rolling metrics, and evaluates multiple signal conditions. Outputs include diagnostic summaries, anomaly flags, and structured intermediate files that can be used for deeper analysis. The project is built to be accessible for both technical and non-technical users, with clear folder organization and documented workflows.

## Core Features

- **Data Ingestion:** Loads raw data from CSV, APIs, or exported logs into a consistent tabular format.
- **Cleaning & Preparation:** Standardizes timestamps, handles missing values, and normalizes numerical fields.
- **Rolling Metrics:** Computes moving averages, rate-of-change measures, and volatility windows.
- **Anomaly Detection:** Flags sudden changes or deviations from expected patterns based on configurable thresholds.
- **Diagnostics:** Produces human-readable summaries to highlight trends, outliers, and possible areas of concern.
- **Modular Structure:** Functions and scripts are organized to allow quick revisions, additions, or new data sources.

## Folder Structure


from __future__ import annotations
import os
import tempfile
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib

# Use Agg backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from fpdf import FPDF
import unicodedata
import plotly.express as px
import plotly.graph_objects as go

from .pipeline import run_pipeline
from .backtest import compute_forward_returns, walk_forward_regression
from .ports import load_port_features_csv, filter_port, create_lagged_features
from .config import load_priority_ports
from .duckdb_io import read_port_features as duckdb_read


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0) or 1.0
    return (s - mu) / sd


def _load_intraday_prices(tickers: list[str]) -> pd.DataFrame:
    """Best-effort intraday (1d, 5m) price load; falls back to last 60d daily if needed.
    Returns wide DataFrame of close prices indexed by timestamp.
    """
    try:
        import yfinance as yf
        data = {}
        for t in tickers:
            try:
                df = yf.download(t, period="1d", interval="5m", progress=False, auto_adjust=False)
                if df.empty:
                    # Fallback: 60d daily
                    df = yf.download(t, period="60d", interval="1d", progress=False, auto_adjust=False)
                # prefer Adj Close
                if "Adj Close" in df.columns:
                    s = df["Adj Close"].copy()
                elif "Close" in df.columns:
                    s = df["Close"].copy()
                else:
                    s = df.squeeze()
                s.name = t
                data[t] = s
            except Exception:
                continue
        if not data:
            raise RuntimeError("no data")
        out = pd.concat(data.values(), axis=1)
        return out
    except Exception:
        # Synthetic fallback for UI demo
        idx = pd.date_range(pd.Timestamp.now().floor("D"), periods=78, freq="5min")
        rng = np.random.default_rng(0)
        base = np.cumsum(rng.normal(0, 0.001, len(idx)))
        data = {t: 100 * (1 + base + 0.01 * (i / len(idx))) for i, t in enumerate(tickers)}
        return pd.DataFrame(data, index=idx)


def _draw_market_heatmap(prices: pd.DataFrame) -> plt.Figure:
    """Return a small heatmap figure of percent changes for overview."""
    # Compute pct change over the available window
    chg = prices.pct_change().fillna(0)
    # Collapse intraday to last value change
    perf = (prices.iloc[-1] / prices.iloc[0] - 1.0).to_frame(name="change")
    # Create a simple horizontal heatmap
    vals = perf["change"].values.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(6.5, 1.2), dpi=150)
    im = ax.imshow(vals, aspect="auto", cmap=cm.coolwarm, vmin=-0.05, vmax=0.05)
    ax.set_yticks([])
    ax.set_xticks(range(len(perf.index)))
    ax.set_xticklabels(perf.index.tolist(), fontsize=8, rotation=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ")
    fig.tight_layout()
    return fig


def _market_heatmap_plotly(prices: pd.DataFrame) -> go.Figure:
    perf = (prices.iloc[-1] / prices.iloc[0] - 1.0)
    z = [perf.values.tolist()]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=perf.index.tolist(),
            colorscale="RdBu",
            zmin=-0.05,
            zmax=0.05,
            showscale=True,
            colorbar=dict(title="Δ"),
        )
    )
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def compute_event_study(
    feature: pd.Series,
    market_price: pd.Series,
    *,
    z_window: int = 30,
    threshold: float = 2.0,
    direction: str = "above",  # 'above' | 'below' | 'both'
    min_spacing: int = 5,
    pre: int = 10,
    post: int = 20,
) -> Dict[str, Any]:
    f = feature.dropna().astype(float)
    z = _zscore(f.rolling(z_window, min_periods=z_window).mean().fillna(method="bfill"))
    # Simpler: z-score on level; could also zscore on returns
    z = _zscore(f)
    # Find events
    if direction == "above":
        ev_idx = z[z >= threshold].index
    elif direction == "below":
        ev_idx = z[z <= -threshold].index
    else:
        ev_idx = z[(z >= threshold) | (z <= -threshold)].index
    # Enforce spacing
    ev_idx = sorted(ev_idx)
    filtered = []
    last = None
    for dt in ev_idx:
        if last is None or (dt - last).days >= min_spacing:
            filtered.append(dt)
            last = dt
    ev_idx = filtered

    # Market returns
    mp = market_price.dropna().astype(float)
    rets = np.log(mp).diff()
    # Build event window matrix
    rel_days = np.arange(-pre, post + 1)
    mat = []
    for dt in ev_idx:
        if dt not in rets.index:
            continue
        # cumulative returns relative to event day 0
        window = []
        for k in rel_days:
            # cumulative from 0 to k for k>=0; for k<0, cumulative from k to -1
            if k >= 0:
                rng = rets.loc[dt : dt + pd.Timedelta(days=k)]
            else:
                rng = rets.loc[dt + pd.Timedelta(days=k) : dt + pd.Timedelta(days=-1)]
            window.append(rng.sum() if not rng.empty else np.nan)
        if not all(np.isnan(window)):
            mat.append(window)
    if not mat:
        return {"n_events": 0, "avg": pd.Series(dtype=float), "lower": None, "upper": None, "rel_days": rel_days}
    M = np.array(mat, dtype=float)
    avg = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), axis=0))
    # 95% CI ~ 1.96 * se
    lower = avg - 1.96 * se
    upper = avg + 1.96 * se
    return {
        "n_events": len(mat),
        "avg": pd.Series(avg, index=rel_days),
        "lower": pd.Series(lower, index=rel_days),
        "upper": pd.Series(upper, index=rel_days),
        "rel_days": rel_days,
    }


def _recommend_action(pk_coef: float, p_perm: float | None, p_nw: float | None, pred_log_ret: float | None, horizon: int) -> tuple[str, str]:
    """Return (headline, detail) recommendation based on significance and forecast.

    Simple rule: if significance is strong and forecast sign is positive → risk-on; negative → risk-off.
    """
    sig = (p_perm is not None and p_perm < 0.05) or (p_nw is not None and p_nw < 0.05)
    if pred_log_ret is None:
        pred_log_ret = float('nan')
    direction = "positive" if pred_log_ret == pred_log_ret and pred_log_ret > 0 else "negative"
    strength = abs(pk_coef)
    if sig and strength >= 0.2 and pred_log_ret == pred_log_ret:
        if pred_log_ret > 0:
            return (
                "Risk‑on bias",
                f"Signal shows statistically significant positive relationship and {horizon}d forecast is > 0. Consider increasing exposure or reducing hedges over next {horizon} days."
            )
        else:
            return (
                "Risk‑off bias",
                f"Signal shows statistically significant negative relationship and {horizon}d forecast is < 0. Consider adding hedges or reducing exposure over next {horizon} days."
            )
    elif strength >= 0.2 and pred_log_ret == pred_log_ret:
        if pred_log_ret > 0:
            return (
                "Constructive setup (needs confirmation)",
                f"Correlation is meaningful but not fully significant yet; forecast is > 0. Consider a smaller position with tighter risk controls for {horizon} days."
            )
        else:
            return (
                "Cautious setup (needs confirmation)",
                f"Correlation is meaningful but not fully significant; forecast is < 0. Consider defensive positioning with tighter stops for {horizon} days."
            )
    else:
        return (
            "No strong signal",
            "Evidence is insufficient (weak correlation or inconclusive p‑values). Monitor and revisit when more data accrues."
        )


def _infer_industries(feature_name: str) -> list[str]:
    name = feature_name.lower()
    if any(k in name for k in ["anchor", "queue", "dwell", "waiting"]):
        return ["Transports", "Retail", "Industrials"]
    if any(k in name for k in ["arrival", "throughput", "volume"]):
        return ["Retail", "Industrials", "Materials"]
    if any(k in name for k in ["oil", "fuel", "bunker"]):
        return ["Energy"]
    return ["Broad Market"]


def _suggest_tickers(industries: list[str]) -> list[str]:
    mapping = {
        "Transports": ["IYT"],
        "Retail": ["XRT"],
        "Industrials": ["XLI"],
        "Materials": ["XLB"],
        "Energy": ["XLE"],
        "Broad Market": ["SPY"],
    }
    out: list[str] = []
    for ind in industries:
        out.extend(mapping.get(ind, []))
    # Deduplicate, keep order
    seen = set(); uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq[:3]


def _load_default_features() -> pd.DataFrame | None:
    # Try DuckDB, then fallback to example CSV
    try:
        df = duckdb_read("data/ports.duckdb", "port_features")
        return df
    except Exception:
        pass
    try:
        demo_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "port_features_template.csv"))
        return load_port_features_csv(demo_csv, date_col="Date", port_col="port")
    except Exception:
        return None


def detect_today_events(df_all: pd.DataFrame, z_window: int = 30, z_thresh: float = 2.0) -> pd.DataFrame:
    """Detect today's port feature shocks (|z| >= threshold).

    Returns DataFrame with columns: Date, port, feature, zscore, industries.
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["Date","port","feature","zscore","industries"])
    df = df_all.copy()
    df["Date"] = pd.to_datetime(df["Date"])  
    last_date = df["Date"].max()
    numeric_cols = [c for c in df.columns if c not in ("Date","port") and pd.api.types.is_numeric_dtype(df[c])]
    out_rows = []
    for port, grp in df.groupby("port"):
        g = grp.sort_values("Date").set_index("Date")
        for feat in numeric_cols:
            if feat not in g.columns:
                continue
            s = g[feat].dropna()
            if s.empty or last_date not in s.index:
                continue
            roll = s.rolling(z_window, min_periods=max(5, z_window//2))
            mu = roll.mean(); sd = roll.std(ddof=0)
            z = (s - mu) / (sd.replace(0, np.nan))
            z_today = float(z.loc[last_date]) if last_date in z.index else float('nan')
            if z_today == z_today and abs(z_today) >= z_thresh:
                out_rows.append({
                    "Date": last_date,
                    "port": port,
                    "feature": feat,
                    "zscore": z_today,
                    "industries": ", ".join(_infer_industries(feat)),
                })
    return pd.DataFrame(out_rows).sort_values("zscore", key=lambda s: s.abs(), ascending=False)


def generate_demo_signals() -> list[dict]:
    """Generate synthetic signals with clear correlations for demo purposes."""
    idx = pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=180), periods=181, freq="D")
    rng = np.random.default_rng(42)
    signals: list[dict] = []
    for name, lag, ticker in [("anchored_count", 7, "IYT"), ("arrivals_count", 10, "XRT"), ("dwell_time", 5, "XLI")]:
        base = np.cumsum(rng.normal(0, 0.3, len(idx))) + np.linspace(0, 5, len(idx))
        feat = pd.Series(base + rng.normal(0, 0.2, len(idx)), index=idx)
        mkt = pd.Series(np.roll(base, lag) + rng.normal(0, 0.3, len(idx)), index=idx)
        feat.name = name; mkt.name = ticker
        from .analysis import to_log_returns, lagged_correlation
        sa = to_log_returns(feat); sb = to_log_returns(mkt)
        lags = [-15,-10,-5,0,5,10,15]
        cdf = lagged_correlation(sa, sb, lags=lags, method="spearman")
        pk_lag = int(cdf["coef"].abs().idxmax()); pk_coef = float(cdf.loc[pk_lag, "coef"])
        signals.append({
            "port": "Demo Port",
            "feature": name,
            "ticker": ticker,
            "lag": pk_lag,
            "coef": pk_coef,
            "p_perm": 0.001,
            "p_newey": 0.001,
            "expected": lag,
            "industries": _infer_industries(name),
            "start": str(idx.min().date()),
            "end": str(idx.max().date()),
            "score": abs(pk_coef),
            "series_feat": feat,
            "series_mkt": mkt,
            "cdf": cdf,
            "y_hat": 0.02,
            "action": "BUY",
            "etfs": _suggest_tickers(_infer_industries(name)),
            "confidence": 90,
        })
    return signals


def auto_scan_signals(
    df_all: pd.DataFrame,
    tickers: list[str],
    window_days: int = 365,
    lags: list[int] | None = None,
    max_signals: int = 5,
) -> list[dict]:
    if lags is None:
        lags = [-15, -10, -5, 0, 5, 10, 15]
    from .data_ingest import load_market_data
    from .analysis import to_log_returns, lagged_correlation, circular_shift_permutation_pvalue, newey_west_pvalue

    out: list[dict] = []
    if df_all.empty:
        return out
    df_all = df_all.copy()
    df_all["Date"] = pd.to_datetime(df_all["Date"])  
    last_date = df_all["Date"].max()
    start_cut = last_date - pd.Timedelta(days=window_days)
    # numeric columns
    ncols = [c for c in df_all.columns if c not in ("Date", "port") and pd.api.types.is_numeric_dtype(df_all[c])]
    for port, dfp in df_all.groupby("port"):
        dfp = dfp.sort_values("Date").set_index("Date")
        dfp = dfp.loc[dfp.index >= start_cut]
        if dfp.empty:
            continue
        # market data once per port window
        try:
            mkt = load_market_data(tickers, start_date=str(dfp.index.min().date()), end_date=str(dfp.index.max().date()))
        except Exception:
            continue
        for feat in ncols:
            if feat not in dfp.columns:
                continue
            s_feat = dfp[feat].dropna()
            if s_feat.empty:
                continue
            for ticker in tickers:
                if ticker not in mkt.columns:
                    continue
                s_mkt = mkt[ticker].dropna()
                # returns
                sa = to_log_returns(s_feat)
                sb = to_log_returns(s_mkt)
                idx = sa.index.intersection(sb.index)
                if len(idx) < 60:
                    continue
                sa = sa.loc[idx]
                sb = sb.loc[idx]
                cdf = lagged_correlation(sa, sb, lags=lags, method="spearman")
                if cdf.empty:
                    continue
                pk_lag = int(cdf["coef"].abs().idxmax())
                pk_coef = float(cdf.loc[pk_lag, "coef"])
                try:
                    p_perm = circular_shift_permutation_pvalue(sa, sb, lag=pk_lag, method="spearman", n_perm=200)
                except Exception:
                    p_perm = float("nan")
                try:
                    _, p_nw = newey_west_pvalue(s_feat, s_mkt, lag=pk_lag, use_returns=True, nw_lags=5)
                except Exception:
                    p_nw = float("nan")
                # Timing prior: prefer lags near expected lead days for this port
                def _expected_lead_days(port_name: str) -> int:
                    name = port_name.lower()
                    if any(k in name for k in ["yantian", "shanghai", "ningbo", "busan", "singapore", "klang", "pelepas"]):
                        return 10  # Asia → USWC approx
                    if any(k in name for k in ["new york", "savannah", "charleston", "norfolk"]):
                        return 20  # US gateways (for inbound from Asia) longer effect
                    if "los angeles" in name or "long beach" in name:
                        return 7
                    return 10
                expected = _expected_lead_days(port)
                timing_weight = 1.0 / (1.0 + abs(pk_lag - expected))
                score = abs(pk_coef) * (1 - (p_perm if p_perm == p_perm else 0.5)) * timing_weight
                out.append({
                    "port": port,
                    "feature": feat,
                    "ticker": ticker,
                    "lag": pk_lag,
                    "coef": pk_coef,
                    "p_perm": p_perm,
                    "p_newey": p_nw,
                    "expected": expected,
                    "industries": _infer_industries(feat),
                    "start": str(dfp.index.min().date()),
                    "end": str(dfp.index.max().date()),
                    "score": score,
                    "series_feat": s_feat,  # raw level series for overlay
                    "series_mkt": s_mkt,
                    "cdf": cdf,
                })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:max_signals]


def create_personalized_report(
    *,
    port: str,
    features_df: pd.DataFrame,
    ticker: str,
    market_series: pd.Series,
    lags: list[int],
    horizon: int,
    returns_mode: bool,
    event_options: Dict[str, Any],
    output_path: str,
) -> None:
    """Build a multi-section PDF tailored to user inputs.

    Sections: Summary of selections, Top Signals table, Backtest metrics/chart,
    Event study (if applicable) with chart.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _sanitize_text("Personalized Signal Report"), ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(2)
    feat_list = ", ".join([c for c in features_df.columns])
    pdf.multi_cell(0, 7, _sanitize_text(f"Port: {port}\nFeatures: {feat_list}\nTicker: {ticker}\nHorizon: {horizon} days\nLags: {', '.join(map(str, lags))}\nReturns-mode: {returns_mode}"))

    # Top signals (correlations across features)
    from .analysis import lagged_correlation, to_log_returns, circular_shift_permutation_pvalue, newey_west_pvalue
    rows = []
    for feat in features_df.columns:
        sa = features_df[feat].dropna().astype(float)
        sb = market_series.dropna().astype(float)
        if returns_mode:
            sa = to_log_returns(sa)
            sb = to_log_returns(sb)
        idx = sa.index.intersection(sb.index)
        if len(idx) < 30:
            continue
        sa = sa.loc[idx]
        sb = sb.loc[idx]
        cdf = lagged_correlation(sa, sb, lags=lags, method="spearman")
        if cdf.empty:
            continue
        pk_lag = int(cdf["coef"].abs().idxmax())
        pk_coef = float(cdf.loc[pk_lag, "coef"])
        p_perm = circular_shift_permutation_pvalue(sa, sb, lag=pk_lag, method="spearman", n_perm=300)
        _, p_nw = newey_west_pvalue(features_df[feat], market_series, lag=pk_lag, use_returns=True, nw_lags=5)
        rows.append({"feature": feat, "lag": pk_lag, "r": pk_coef, "perm_p": p_perm, "p_newey_west": p_nw})

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _sanitize_text("Top Signals (by |r| at best lag)"), ln=True)
    pdf.set_font("Arial", size=11)
    if rows:
        rows_sorted = sorted(rows, key=lambda d: abs(d["r"]), reverse=True)
        for r in rows_sorted[:10]:
            pdf.cell(0, 7, _sanitize_text(f"{r['feature']}: |r|={abs(r['r']):.2f} @ lag {r['lag']} | perm p={r['perm_p']:.3f} | NW p={(r['p_newey_west'] if r['p_newey_west']==r['p_newey_west'] else float('nan')):.3f}"), ln=True)
    else:
        pdf.cell(0, 7, _sanitize_text("Insufficient overlapping data to compute signals."), ln=True)

    # Backtest on selected features (quick Ridge)
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _sanitize_text("Backtest (Ridge, OOS)"), ln=True)
    try:
        from .ports import create_lagged_features
        from .backtest import compute_forward_returns, walk_forward_regression
        # Use a default lag set to build X
        bt_lags = [0, 1, 2, 3, 5, 10]
        X = create_lagged_features(features_df, list(features_df.columns), bt_lags)
        y = compute_forward_returns(market_series, horizon=horizon, log=True)
        joined = X.join(y, how="inner").dropna()
        if not joined.empty:
            X_al = joined.drop(columns=[y.name])
            y_al = joined[y.name]
            bt = walk_forward_regression(X_al, y_al, n_splits=5, alpha=1.0)
            for k, v in bt.metrics.items():
                pdf.cell(0, 7, _sanitize_text(f"{k}: {v}"), ln=True)
            # Cumulative chart
            fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=150)
            cum_true = bt.predictions["y_true"].cumsum()
            cum_pred = bt.predictions["y_pred"].cumsum()
            ax.plot(cum_true.index, cum_true.values, label="Cumulative True", linewidth=1.5)
            ax.plot(cum_pred.index, cum_pred.values, label="Cumulative Pred", linewidth=1.3)
            ax.legend(); ax.set_title("Cumulative Forward Returns"); fig.tight_layout()
            tmp_img = os.path.join(os.path.dirname(output_path), "personal_bt.png")
            fig.savefig(tmp_img); plt.close(fig)
            pdf.image(tmp_img, w=180)
        else:
            pdf.cell(0, 7, _sanitize_text("Not enough data to backtest."), ln=True)
    except Exception as exc:
        pdf.cell(0, 7, _sanitize_text(f"Backtest failed: {exc}"), ln=True)

    # Event study (optional)
    try:
        feat_name = features_df.columns[0] if len(features_df.columns) else None
        if feat_name:
            es = compute_event_study(
                features_df[feat_name],
                market_series,
                z_window=int(event_options.get("z_window", 30)),
                threshold=float(event_options.get("threshold", 2.0)),
                direction=event_options.get("direction", "above"),
                min_spacing=int(event_options.get("min_spacing", 5)),
                pre=int(event_options.get("pre", 10)),
                post=int(event_options.get("post", 20)),
            )
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, _sanitize_text(f"Event Study – {feat_name}"), ln=True)
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 7, _sanitize_text(f"n_events: {es['n_events']}"), ln=True)
            if es["n_events"] > 0:
                # Plot avg with CI
                fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=150)
                ax.plot(es["avg"].index, es["avg"].values, label="Average cumulative return")
                if es["lower"] is not None and es["upper"] is not None:
                    ax.fill_between(es["avg"].index, es["lower"].values, es["upper"].values, alpha=0.2)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_xlabel("Days from event"); ax.set_ylabel("Cumulative log return"); ax.legend(); fig.tight_layout()
                tmp_es = os.path.join(os.path.dirname(output_path), "personal_es.png")
                fig.savefig(tmp_es); plt.close(fig)
                pdf.image(tmp_es, w=180)
    except Exception as exc:
        pdf.add_page(); pdf.cell(0, 7, _sanitize_text(f"Event study failed: {exc}"), ln=True)

    pdf.output(output_path)


def generate_plots(
    data: pd.DataFrame,
    pairs: List[Dict[str, Any]],
    output_dir: str,
    normalize: bool = True,
) -> Dict[str, List[str]]:
    """Generate overlay and heatmap plots per (metric, ticker) pair.

    Returns mapping metric -> list of file paths for its plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths: Dict[str, List[str]] = {}
    for entry in pairs:
        metric = entry["metric"]
        ticker = entry["ticker"]
        peak = entry.get("peak", {})
        peak_lag = peak.get("lag")
        coef = peak.get("coef")
        # Overlay plot
        fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=150)
        series_a = data[metric]
        series_b = data[ticker]
        if normalize:
            series_a = _zscore(series_a)
            series_b = _zscore(series_b)
        ax.plot(data.index, series_a, label=metric, linewidth=1.6)
        ax.plot(data.index, series_b, label=ticker, linewidth=1.2, alpha=0.8)
        ttl = f"{metric} vs {ticker}"
        if peak_lag is not None and coef is not None and not pd.isna(coef):
            ttl += f"  (peak lag {peak_lag}, r={coef:.2f})"
        ax.set_title(ttl)
        ax.set_xlabel("Date")
        ax.set_ylabel("Z-score" if normalize else "Value")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fname = f"{metric}_vs_{ticker}_overlay.png".replace("/", "_")
        p1 = os.path.join(output_dir, fname)
        fig.savefig(p1)
        plt.close(fig)

        # Heatmap of correlations by lag (single-row heatmap)
        corr_by_lag = entry.get("correlations", {})
        if corr_by_lag:
            lags = sorted(int(l) for l in corr_by_lag.keys())
            vals = [corr_by_lag[l] for l in lags]
            hm = np.array([vals])
            fig2, ax2 = plt.subplots(figsize=(6.5, 1.6), dpi=150)
            im = ax2.imshow(hm, aspect="auto", cmap=cm.coolwarm, vmin=-1, vmax=1)
            ax2.set_yticks([])
            ax2.set_xticks(range(len(lags)))
            ax2.set_xticklabels(lags, rotation=0, fontsize=8)
            ax2.set_title(f"Correlation by lag: {metric} vs {ticker}")
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="r")
            fig2.tight_layout()
            fname2 = f"{metric}_vs_{ticker}_lags.png".replace("/", "_")
            p2 = os.path.join(output_dir, fname2)
            fig2.savefig(p2)
            plt.close(fig2)
        else:
            p2 = None

        plot_paths.setdefault(metric, []).extend([p for p in [p1, p2] if p])

    return plot_paths


def _sanitize_text(text: str) -> str:
    if text is None:
        return ""
    # Replace common Unicode punctuation with ASCII equivalents
    replacements = {
        "\u2011": "-",  # non-breaking hyphen
        "\u2010": "-",  # hyphen
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...",  # ellipsis
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Remove/flatten any remaining non-latin1 chars
    text_norm = unicodedata.normalize("NFKD", text)
    text_ascii = text_norm.encode("latin-1", "ignore").decode("latin-1", "ignore")
    return text_ascii


def _summarize_top_results(results: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    df_rows = []
    for r in results:
        peak = r.get("peak", {})
        df_rows.append(
            {
                "metric": r["metric"],
                "ticker": r["ticker"],
                "lag": peak.get("lag"),
                "coef": peak.get("coef"),
                "p_adj": peak.get("p_adj"),
                "perm_p": peak.get("perm_p"),
                "p_newey_west": peak.get("p_newey_west"),
                "ci_lower": peak.get("ci_lower"),
                "ci_upper": peak.get("ci_upper"),
                "significant": bool(peak.get("significant", False)),
                "stability": peak.get("stability_share"),
            }
        )
    df = pd.DataFrame(df_rows)
    # Sort: significant first, then by abs(coef) desc, then stability desc
    df["abs_coef"] = df["coef"].abs()
    df_sorted = df.sort_values(by=["significant", "abs_coef", "stability"], ascending=[False, False, False])
    return df_sorted.head(top_n).to_dict(orient="records")


def create_backtest_report(
    *,
    port: str,
    ticker: str,
    horizon: int,
    metrics: Dict[str, Any],
    pred_df: pd.DataFrame,
    features_used: List[str],
    lags_used: List[int],
    output_path: str,
) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _sanitize_text("Port Signals Backtest"), ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(2)
    pdf.multi_cell(0, 7, _sanitize_text(f"Port: {port}\nTicker: {ticker}\nHorizon: {horizon} days"))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _sanitize_text("Metrics"), ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in metrics.items():
        pdf.cell(0, 7, _sanitize_text(f"{k}: {v}"), ln=True)

    # Plot cumulative returns
    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=150)
    cum_true = pred_df["y_true"].cumsum()
    cum_pred = pred_df["y_pred"].cumsum()
    ax.plot(cum_true.index, cum_true.values, label="Cumulative True", linewidth=1.6)
    ax.plot(cum_pred.index, cum_pred.values, label="Cumulative Pred", linewidth=1.4)
    ax.set_title("Cumulative Forward Returns")
    ax.legend()
    fig.tight_layout()
    tmp_img = os.path.join(os.path.dirname(output_path), "bt_cum.png")
    fig.savefig(tmp_img)
    plt.close(fig)
    pdf.ln(3)
    pdf.image(tmp_img, w=180)
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _sanitize_text("Features & Lags"), ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, _sanitize_text(f"Features: {', '.join(features_used)}\nLags: {', '.join(map(str, lags_used))}"))
    pdf.output(output_path)


def create_pdf_report(
    results: List[Dict[str, Any]], plot_paths: Dict[str, List[str]], output_path: str
) -> None:
    """Create a PDF report with executive summary and plots."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title + Executive Summary
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, _sanitize_text("Signal Discovery Report"), ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0,
        7,
        _sanitize_text(
            "Executive Summary: Top signals with lead/lag correlations. "
            "Significance is FDR-adjusted. Stability is the share of windows with significant correlation."
        ),
    )

    top = _summarize_top_results(results, top_n=5)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _sanitize_text("Top Signals"), ln=True)
    pdf.set_font("Arial", size=11)
    for row in top:
        line = (
            f"{row['metric']} vs {row['ticker']}  |  lag {row['lag']}, r={row['coef']:.2f}  "
            f"|  p_adj={(row['p_adj'] if row['p_adj'] is not None else float('nan')):.3f}  "
            f"|  stability={(row['stability'] if row['stability'] is not None else float('nan')):.2f}"
        )
        pdf.cell(0, 7, _sanitize_text(line), ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _sanitize_text("Recommendation"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(
        0,
        6,
        _sanitize_text(
            "Pilot decision rule using top signal(s) with clear guardrails. "
            "Track impact and promote to production if KPIs improve."
        ),
    )

    # Detail pages per pair
    for entry in results:
        metric = entry["metric"]
        ticker = entry["ticker"]
        correlations = entry.get("correlations", {})
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, _sanitize_text(f"{metric} vs {ticker}"), ln=True)
        pdf.set_font("Arial", size=11)
        peak = entry.get("peak", {})
        pdf.cell(
            0,
            8,
            _sanitize_text(
                f"Peak lag {peak.get('lag')} | r={peak.get('coef', float('nan')):.3f} | p_adj={(peak.get('p_adj') if peak.get('p_adj') is not None else float('nan')):.3f} | stability={(peak.get('stability_share') if peak.get('stability_share') is not None else float('nan')):.2f}"
            ),
            ln=True,
        )
        # Additional significance details
        p_perm = peak.get("perm_p")
        p_nw = peak.get("p_newey_west")
        ci_lo = peak.get("ci_lower")
        ci_hi = peak.get("ci_upper")
        if p_perm is not None or p_nw is not None or (ci_lo is not None and ci_hi is not None):
            pdf.cell(0, 7, _sanitize_text(f"Permutation p={p_perm if p_perm is not None else float('nan'):.3f} | Newey-West p={p_nw if p_nw is not None else float('nan'):.3f}"), ln=True)
            if ci_lo is not None and ci_hi is not None:
                pdf.cell(0, 7, _sanitize_text(f"Block-Bootstrap 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"), ln=True)
        for lag, coef in correlations.items():
            pdf.cell(0, 7, _sanitize_text(f"Lag {lag}: r={coef:.3f}"), ln=True)
        pdf.ln(3)
        # Include overlay then heatmap if available
        if metric in plot_paths:
            # overlay first
            overlay = next((p for p in plot_paths[metric] if f"{metric}_vs_{ticker}_overlay" in os.path.basename(p)), None)
            if overlay and os.path.exists(overlay):
                pdf.image(overlay, w=180)
            heat = next((p for p in plot_paths[metric] if f"{metric}_vs_{ticker}_lags" in os.path.basename(p)), None)
            if heat and os.path.exists(heat):
                pdf.ln(2)
                pdf.image(heat, w=180)

    pdf.output(output_path)


def build_streamlit_app() -> None:
    """Launch a Streamlit app for interactive exploration."""
    import streamlit as st
    from datetime import date
    from .data_ingest import load_operational_data, load_market_data
    from .imagery import load_image, detect_vessels, annotate_detections
    try:
        st.set_page_config(page_title="Signal Discovery", page_icon="📈", layout="wide")
    except Exception:
        pass
    # Lightweight pro styling
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {background-color:#2F80ED;color:white;border-radius:8px;padding:0.6rem 1rem;font-weight:600}
        .small-caption {color:#9aa4b2;font-size:0.85rem}
        div[data-testid="stMetricValue"] {font-size:1.1rem}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Signal Discovery")
    # Additional UI styles
    st.markdown(
        """
        <style>
        .card {background:#1F2630;border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.5rem;border:1px solid #2a3340}
        .badge {display:inline-block;padding:0.15rem 0.5rem;border-radius:999px;font-size:0.75rem;font-weight:700;letter-spacing:0.02em}
        .buy {background:#1b5e20;color:#bbf7d0}
        .sell {background:#7f1d1d;color:#fecaca}
        .bar {height:6px;background:#2a3340;border-radius:999px;margin-top:8px}
        .bar > div {height:6px;background:#2F80ED;border-radius:999px}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("A timing‑aware scanner for port activity → market impact. Click once to see today’s signals; open Advanced only when needed.")

    # First-Time Checklist + One-click Demo Setup
    with st.expander("First-Time Checklist", expanded=False):
        st.markdown(
            "- [1] Install deps: `pip install -r requirements.txt` (inside venv)\n"
            "- [2] Launch app: `streamlit run app.py`\n"
            "- [3] (Optional) Set `DATALASTIC_API_KEY` for live AIS.\n"
            "- [4] Build demo DB and try Live Mode."
        )
        c1, c2 = st.columns(2)
        demo_db_path = c1.text_input("Demo DB path", value="data/ports.duckdb", key="check_db")
        demo_table = c2.text_input("Demo table", value="port_features", key="check_tbl")
        if st.button("Run Demo Setup", key="check_run"):
            try:
                from .duckdb_io import write_port_features_df
                demo_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "port_features_template.csv"))
                df_demo = pd.read_csv(demo_csv)
                total = write_port_features_df(demo_db_path, df_demo, table=demo_table, mode="replace")
                st.success(f"Demo data written to {demo_db_path}:{demo_table}. Row count now ~{total}.")
            except Exception as exc:
                st.error(f"Demo setup failed: {exc}")

    # Optional Port Map
    with st.expander("Port Map", expanded=False):
        try:
            import pydeck as pdk  # type: ignore
            from .ais import load_port_geofences, polygon_bounds
            geojson_default = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "ports.geojson"))
            geo_file = st.file_uploader("GeoJSON (optional)", type=["geojson"], key="map_geojson")
            if geo_file is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tf:
                    tf.write(geo_file.getvalue())
                    gpath = tf.name
            else:
                gpath = geojson_default
            fences = load_port_geofences(gpath)
            data = []
            for f in fences:
                mnx, mny, mxx, mxy = polygon_bounds(f.polygons)
                lat = (mny + mxy) / 2.0
                lon = (mnx + mxx) / 2.0
                data.append({"name": f.name, "lat": lat, "lon": lon})
            if data:
                df_pts = pd.DataFrame(data)
                view_state = pdk.ViewState(latitude=float(df_pts["lat"].mean()), longitude=float(df_pts["lon"].mean()), zoom=2)
                layer = pdk.Layer("ScatterplotLayer", data=df_pts, get_position="[lon, lat]", get_radius=60000, get_fill_color=[47,128,237], pickable=True)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"}))
            else:
                st.info("No ports to display.")
        except Exception as exc:
            st.info(f"Map preview unavailable: {exc}")

    # Satellite Imagery (beta)
    with st.expander("Satellite Imagery (beta)", expanded=False):
        st.caption("Upload SAR/optical snapshots to estimate vessel counts quickly. Counts can be appended to DuckDB as today's anchored_count for a selected port.")
        img_files = st.file_uploader("Upload image(s)", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True, key="upl_sat")
        sel_port = None
        try:
            # Try to offer ports from DuckDB features
            df_ports = duckdb_read("data/ports.duckdb", "port_features")
            ports_list = sorted(df_ports["port"].dropna().unique().tolist()) if not df_ports.empty else []
        except Exception:
            ports_list = []
        if ports_list:
            sel_port = st.selectbox("Assign to port (optional)", ports_list)
        thresh = st.slider("Detection sensitivity (higher = fewer picks)", min_value=0.5, max_value=0.98, value=0.88, step=0.01)
        min_area = st.slider("Min area (px)", min_value=2, max_value=50, value=4)
        max_area = st.slider("Max area (px)", min_value=50, max_value=2000, value=400)
        append = st.checkbox("Append counts to DuckDB as anchored_count (today)", value=False)
        if st.button("Run detection", key="run_sat"):
            total_added = 0
            for f in (img_files or []):
                try:
                    img = load_image(f.getvalue())
                    cnt, pts = detect_vessels(img, thresh_rel=float(thresh), min_area_px=int(min_area), max_area_px=int(max_area))
                    st.write({"file": f.name, "vessel_like_targets": cnt})
                    ann = annotate_detections(img, pts)
                    st.image(ann, caption=f"Detections – {f.name}")
                    if append and sel_port:
                        # Append as today's anchored_count to DuckDB
                        import pandas as _pd
                        from datetime import datetime as _dt
                        from .duckdb_io import write_port_features_df, ensure_port_features_table
                        ensure_port_features_table("data/ports.duckdb", "port_features")
                        row = _pd.DataFrame([
                            {"Date": _dt.utcnow().date(), "port": sel_port, "anchored_count": int(cnt)}
                        ])
                        total = write_port_features_df("data/ports.duckdb", row, table="port_features", mode="append")
                        total_added += 1
                except Exception as exc:
                    st.error(f"Failed on {getattr(f,'name','image')}: {exc}")
            if append and sel_port:
                st.success(f"Appended {total_added} row(s) to DuckDB for port {sel_port}.")

    # Live Monitor (persistent KPI section)
    st.markdown("## Live Monitor")
    cpath, ctbl = st.columns(2)
    db_path_kpi = cpath.text_input("DuckDB path", value="data/ports.duckdb", key="kpi_db")
    table_kpi = ctbl.text_input("Table", value="port_features", key="kpi_tbl")

    # Load features table (if available)
    df_kpi = None
    try:
        df_kpi = duckdb_read(db_path_kpi, table_kpi)
    except Exception:
        df_kpi = None

    # Data Health metrics (auto if data available)
    if df_kpi is not None and not df_kpi.empty:
        df_kpi["Date"] = pd.to_datetime(df_kpi["Date"])  # ensure datetime
        last_date = df_kpi["Date"].max()
        n_ports = int(df_kpi["port"].nunique())
        n_rows = int(len(df_kpi))
        d1, d2, d3 = st.columns(3)
        d1.metric("Last Date", str(getattr(last_date, 'date', lambda: last_date)()))
        d2.metric("Ports", n_ports)
        d3.metric("Rows", n_rows)
    else:
        st.info("No DuckDB data loaded yet. Use Live Mode to load, or refresh below.")

    # Refresh Data (Datalastic)
    with st.expander("Refresh Data (Datalastic)"):
        try:
            import os
            from .connectors.ais_datalastic import DatalasticClient
            from .ais import load_port_geofences, assign_port_to_points, derive_daily_port_features, polygon_bounds
            from .duckdb_io import write_port_features_df
            from datetime import datetime, timedelta, timezone
            geo_col, hr_col, key_col = st.columns(3)
            geojson_file = geo_col.file_uploader("GeoJSON (ports)", type=["geojson"], key="kpi_geojson")
            hours = int(hr_col.number_input("Lookback hours", min_value=1, max_value=168, value=24, key="kpi_hours"))
            api_key = key_col.text_input("Datalastic API Key", type="password", key="kpi_api")
            cbtn1, cbf, _ = st.columns(3)
            do_refresh = cbtn1.button("Refresh Now", key="kpi_refresh")
            backfill_days = int(cbf.number_input("Backfill days", min_value=0, max_value=180, value=0, key="kpi_bdays"))
            do_backfill = st.button("Run Backfill", key="kpi_backfill")
            # Resolve geojson path
            import tempfile
            if geojson_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tf:
                    tf.write(geojson_file.getvalue())
                    geojson_path = tf.name
            else:
                geojson_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "ports.geojson"))
            if do_refresh or (do_backfill and backfill_days > 0):
                fences = load_port_geofences(geojson_path)
                if not fences:
                    st.error("No ports found in GeoJSON")
                else:
                    client = DatalasticClient(api_key=api_key or os.environ.get("DATALASTIC_API_KEY"))
                    appended = 0
                    if do_refresh:
                        t_end = datetime.now(timezone.utc)
                        t_start = t_end - timedelta(hours=hours)
                        def _iso(dt):
                            return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                        for fence in fences:
                            mnx, mny, mxx, mxy = polygon_bounds(fence.polygons)
                            try:
                                df = client.fetch_positions_bbox(mnx, mny, mxx, mxy, _iso(t_start), _iso(t_end))
                                if df.empty:
                                    continue
                                df_tag = assign_port_to_points(df, [fence])
                                df_tag = df_tag[df_tag["port"].notna()]
                                if df_tag.empty:
                                    continue
                                feats = derive_daily_port_features(df_tag)
                                total = write_port_features_df(db_path_kpi, feats, table=table_kpi, mode="append")
                                appended += len(feats)
                            except Exception as exc:
                                st.warning(f"Fetch failed for {fence.name}: {exc}")
                                continue
                    if do_backfill and backfill_days > 0:
                        for d in range(backfill_days, 0, -1):
                            t0 = (datetime.now(timezone.utc) - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
                            t1 = t0 + timedelta(days=1)
                            def _iso(dt):
                                return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                            day_parts = []
                            for fence in fences:
                                mnx, mny, mxx, mxy = polygon_bounds(fence.polygons)
                                try:
                                    df = client.fetch_positions_bbox(mnx, mny, mxx, mxy, _iso(t0), _iso(t1))
                                except Exception as exc:
                                    st.warning(f"Backfill failed {fence.name} {t0.date()}: {exc}")
                                    continue
                                if df.empty:
                                    continue
                                df_tag = assign_port_to_points(df, [fence])
                                df_tag = df_tag[df_tag["port"].notna()]
                                if df_tag.empty:
                                    continue
                                feats = derive_daily_port_features(df_tag)
                                day_parts.append(feats)
                            if day_parts:
                                out = pd.concat(day_parts, ignore_index=True)
                                total = write_port_features_df(db_path_kpi, out, table=table_kpi, mode="append")
                                appended += len(out)
                    st.success(f"Appended {appended} rows. Table now ~{total if 'total' in locals() else 'unknown'} rows.")
        except Exception as exc:
            st.info(f"Datalastic refresh not available: {exc}")

    # Port KPIs
    st.markdown("### Port KPIs")
    if df_kpi is not None and not df_kpi.empty:
        ports_kpi = sorted(df_kpi["port"].dropna().unique().tolist())
        cport, cfeat = st.columns(2)
        sel_port_kpi = cport.selectbox("Port", ports_kpi, key="kpi_port")
        num_cols = [c for c in df_kpi.columns if c not in ("Date", "port") and pd.api.types.is_numeric_dtype(df_kpi[c])]
        sel_feat_kpi = cfeat.selectbox("Feature", num_cols, index=(num_cols.index("anchored_count") if "anchored_count" in num_cols else 0), key="kpi_feat")
        ctick, chz, clags = st.columns(3)
        kpi_ticker = ctick.text_input("Market ticker", value="SPY", key="kpi_ticker")
        kpi_horizon = int(chz.number_input("Pred horizon (days)", min_value=1, max_value=90, value=10, key="kpi_hz"))
        kpi_lags = clags.text_input("Test lags", value="-10,-5,0,5,10", key="kpi_lags")
        returns_mode_kpi = st.checkbox("Compute correlations on log returns", value=True, key="kpi_returns")
        if st.button("Compute KPIs", key="kpi_run"):
            try:
                dfp = df_kpi[df_kpi["port"] == sel_port_kpi].copy().set_index("Date").sort_index()
                s_feat = dfp[sel_feat_kpi].dropna()
                start_d = str(s_feat.index.min().date())
                end_d = str(s_feat.index.max().date())
                mkt = load_market_data([kpi_ticker], start_date=start_d, end_date=end_d)
                s_mkt = mkt[kpi_ticker]
                from .analysis import lagged_correlation, to_log_returns, circular_shift_permutation_pvalue, newey_west_pvalue
                if returns_mode_kpi:
                    sa = to_log_returns(s_feat)
                    sb = to_log_returns(s_mkt)
                else:
                    sa, sb = s_feat, s_mkt
                lags_list = [int(x.strip()) for x in kpi_lags.split(",") if x.strip()]
                corr_df = lagged_correlation(sa, sb, lags=lags_list, method="spearman")
                if not corr_df.empty:
                    pk_lag = int(corr_df["coef"].abs().idxmax())
                    pk_coef = float(corr_df.loc[pk_lag, "coef"])
                    p_perm = circular_shift_permutation_pvalue(sa, sb, lag=pk_lag, method="spearman", n_perm=300)
                    _, p_nw = newey_west_pvalue(s_feat, s_mkt, lag=pk_lag, use_returns=True, nw_lags=5)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Top |r| @ lag", f"{abs(pk_coef):.2f} @ {pk_lag}d")
                    m2.metric("Permutation p", f"{p_perm:.3f}")
                    m3.metric("Newey–West p", f"{(p_nw if p_nw==p_nw else float('nan')):.3f}")
                # Quick prediction
                from .ports import create_lagged_features
                from .backtest import compute_forward_returns
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                pred_lags = [int(x.strip()) for x in "0,1,2,3,5,10".split(",")]
                X = create_lagged_features(dfp, [sel_feat_kpi], pred_lags)
                target = compute_forward_returns(s_mkt, horizon=kpi_horizon, log=True)
                joined = X.join(target, how="inner").dropna()
                if not joined.empty:
                    X_al = joined.drop(columns=[target.name])
                    y_al = joined[target.name]
                    model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=0))])
                    model.fit(X_al, y_al)
                    latest_dt = X.index.max()
                    x_last = X.loc[[latest_dt]].reindex(columns=X_al.columns)
                    if not x_last.isna().any().any():
                        y_hat = float(model.predict(x_last)[0])
                        st.metric(f"Next {kpi_horizon}d predicted log return", f"{y_hat:.4f}")
            except Exception as exc:
                st.info(f"KPI computation failed: {exc}")
    else:
        st.info("Load DuckDB to compute KPIs.")

    # Top Signals Today (scan)
    st.markdown("### Top Signals Today")
    if df_kpi is not None and not df_kpi.empty:
        ctick2, cwin2, clags2, calpha2, ctopn2 = st.columns(5)
        scan_ticker = ctick2.text_input("Ticker", value="SPY", key="scan_tick")
        scan_window = int(cwin2.number_input("Window days", min_value=30, max_value=3650, value=180, key="scan_win"))
        scan_lags = clags2.text_input("Lags", value="-15,-10,-5,0,5,10,15", key="scan_lags")
        scan_alpha = float(calpha2.number_input("p-threshold", min_value=0.001, max_value=0.25, value=0.05, step=0.001, key="scan_alpha"))
        scan_topn = int(ctopn2.number_input("Show top N", min_value=1, max_value=50, value=5, key="scan_topn"))
        scan_returns = st.checkbox("Use log returns", value=True, key="scan_returns")
        if st.button("Compute Top Signals", key="scan_run"):
            try:
                end_date = pd.to_datetime(df_kpi["Date"]).max().date()
                start_date = (pd.to_datetime(df_kpi["Date"]).max() - pd.Timedelta(days=scan_window)).date()
                mkt = load_market_data([scan_ticker], start_date=str(start_date), end_date=str(end_date))
                s_mkt_full = mkt[scan_ticker]
                # Prepare features
                df_kpi_idx = df_kpi.set_index(pd.to_datetime(df_kpi["Date"])).sort_index()
                df_kpi_idx = df_kpi_idx.loc[str(start_date): str(end_date)]
                num_cols = [c for c in df_kpi_idx.columns if c not in ("Date", "port") and pd.api.types.is_numeric_dtype(df_kpi_idx[c])]
                lags_list = [int(x.strip()) for x in scan_lags.split(",") if x.strip()]
                from .analysis import lagged_correlation, to_log_returns, circular_shift_permutation_pvalue
                rows = []
                ports = sorted(df_kpi_idx["port"].dropna().unique().tolist())
                for port in ports:
                    dfp = df_kpi_idx[df_kpi_idx["port"] == port]
                    dfp = dfp.drop(columns=["port"], errors="ignore").copy()
                    dfp = dfp.drop(columns=["Date"], errors="ignore")
                    dfp = dfp.sort_index()
                    for feat in num_cols:
                        if feat not in dfp.columns:
                            continue
                        s_feat = dfp[feat].dropna()
                        s_mkt = s_mkt_full
                        if scan_returns:
                            sa = to_log_returns(s_feat)
                            sb = to_log_returns(s_mkt)
                        else:
                            sa, sb = s_feat, s_mkt
                        # Align and skip if tiny sample
                        idx = sa.index.intersection(sb.index)
                        if len(idx) < 50:
                            continue
                        sa, sb = sa.loc[idx], sb.loc[idx]
                        cd = lagged_correlation(sa, sb, lags=lags_list, method="spearman")
                        if cd.empty:
                            continue
                        pk_lag = int(cd["coef"].abs().idxmax())
                        pk_coef = float(cd.loc[pk_lag, "coef"])
                        p_perm = circular_shift_permutation_pvalue(sa, sb, lag=pk_lag, method="spearman", n_perm=200)
                        if p_perm < scan_alpha:
                            rows.append({
                                "port": port,
                                "feature": feat,
                                "lag": pk_lag,
                                "r": pk_coef,
                                "perm_p": p_perm,
                            })
                if rows:
                    df_top = pd.DataFrame(rows)
                    df_top = df_top.reindex(df_top["r"].abs().sort_values(ascending=False).index)
                    top_show = df_top.head(scan_topn).reset_index(drop=True)
                    st.dataframe(top_show)
                    # One-click personalized report per row
                    st.markdown("#### Generate Report for a Top Signal")
                    for i, r in top_show.iterrows():
                        col_btn, col_lbl = st.columns([1,4])
                        if col_btn.button("Generate Report", key=f"scan_rep_{i}"):
                            try:
                                # Build features df (single feature) for that port
                                dfp_port = df_kpi[df_kpi["port"] == r["port"]].copy()
                                dfp_port["Date"] = pd.to_datetime(dfp_port["Date"]) 
                                dfp_port = dfp_port.set_index("Date").sort_index()
                                feats_df = dfp_port[[r["feature"]]].dropna()
                                # Market series over the scan window
                                ms = s_mkt_full
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    pdf_out = os.path.join(tmpdir, f"report_{i}.pdf")
                                    create_personalized_report(
                                        port=r["port"],
                                        features_df=feats_df,
                                        ticker=scan_ticker,
                                        market_series=ms,
                                        lags=[int(x.strip()) for x in scan_lags.split(",") if x.strip()],
                                        horizon=10,
                                        returns_mode=scan_returns,
                                        event_options={"threshold": 2.0, "direction": "above", "pre": 10, "post": 20},
                                        output_path=pdf_out,
                                    )
                                    with open(pdf_out, "rb") as f:
                                        st.download_button(
                                            f"Download {r['port']} – {r['feature']} report",
                                            data=f,
                                            file_name=f"report_{r['port'].replace(' ','_')}_{r['feature']}.pdf",
                                            mime="application/pdf",
                                            key=f"scan_rep_dl_{i}",
                                        )
                            except Exception as exc:
                                st.error(f"Report generation failed: {exc}")
                        col_lbl.write(f"{r['port']} – {r['feature']} (|r|={abs(r['r']):.2f} @ lag {int(r['lag'])}, p={r['perm_p']:.3f})")
                else:
                    st.info("No significant signals under current settings.")
            except Exception as exc:
                st.error(f"Top signals scan failed: {exc}")

    # Personalized Report Builder
    st.markdown("## Personalized Report")
    if df_kpi is not None and not df_kpi.empty:
        ports_avail = sorted(df_kpi["port"].dropna().unique().tolist())
        r1, r2 = st.columns(2)
        sel_port_rep = r1.selectbox("Port", ports_avail, key="rep_port")
        dfp_all = df_kpi[df_kpi["port"] == sel_port_rep].copy()
        dfp_all["Date"] = pd.to_datetime(dfp_all["Date"]) 
        dfp_all = dfp_all.set_index("Date").sort_index()
        numeric_cols = [c for c in dfp_all.columns if pd.api.types.is_numeric_dtype(dfp_all[c])]
        sel_feats_rep = r2.multiselect("Features", numeric_cols, default=[c for c in ["anchored_count","arrivals_count"] if c in numeric_cols], key="rep_feats")
        t1, h1, l1 = st.columns(3)
        rep_ticker = t1.text_input("Ticker", value="SPY", key="rep_tick")
        rep_horizon = int(h1.number_input("Horizon (days)", min_value=1, max_value=90, value=10, key="rep_hz"))
        rep_lags = [int(x.strip()) for x in l1.text_input("Lags", value="-10,-5,0,5,10", key="rep_lags").split(",") if x.strip()]
        rep_returns = st.checkbox("Use log returns", value=True, key="rep_returns")
        # Event study options
        e1, e2, e3, e4 = st.columns(4)
        es_thr = float(e1.number_input("Event threshold (z)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key="rep_es_thr"))
        es_dir = e2.selectbox("Direction", ["above","below","both"], key="rep_es_dir")
        es_pre = int(e3.number_input("Pre days", min_value=0, max_value=60, value=10, key="rep_es_pre"))
        es_post = int(e4.number_input("Post days", min_value=1, max_value=90, value=20, key="rep_es_post"))
        if st.button("Generate Personalized Report", key="rep_go"):
            try:
                # Build features df and market series
                feats_df = dfp_all[sel_feats_rep].dropna(how="all")
                from .data_ingest import load_market_data
                start_d = str(feats_df.index.min().date())
                end_d = str(feats_df.index.max().date())
                mkt = load_market_data([rep_ticker], start_date=start_d, end_date=end_d)
                ms = mkt[rep_ticker]
                if feats_df.empty or ms.empty:
                    st.error("Insufficient data for the selected inputs.")
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pdf_out = os.path.join(tmpdir, "personalized_report.pdf")
                        create_personalized_report(
                            port=sel_port_rep,
                            features_df=feats_df,
                            ticker=rep_ticker,
                            market_series=ms,
                            lags=rep_lags,
                            horizon=rep_horizon,
                            returns_mode=rep_returns,
                            event_options={"threshold": es_thr, "direction": es_dir, "pre": es_pre, "post": es_post},
                            output_path=pdf_out,
                        )
                        with open(pdf_out, "rb") as f:
                            st.download_button("Download Personalized Report (PDF)", data=f, file_name="personalized_signal_report.pdf", mime="application/pdf")
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

    # Sidebar status panel (DuckDB Live Mode)
    with st.sidebar:
        # Optional logo slot
        try:
            logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png"))
            if os.path.exists(logo_path):
                st.image(logo_path, width=160)
        except Exception:
            pass
        # Advanced tools toggle (sidebar)
        st.checkbox("Show Advanced Tools", value=False, key="show_adv")
        st.markdown("### Data Health")
        db_path_sb = st.text_input("DuckDB path", value="data/ports.duckdb", key="sb_db")
        table_sb = st.text_input("Table", value="port_features", key="sb_tbl")
        try:
            df_sb = duckdb_read(db_path_sb, table_sb)
            last_date = pd.to_datetime(df_sb["Date"]).max()
            ports_sb = df_sb["port"].nunique()
            rows_sb = len(df_sb)
            c1, c2, c3 = st.columns(3)
            c1.metric("Last Date", str(getattr(last_date, 'date', lambda: last_date)()))
            c2.metric("Ports", ports_sb)
            c3.metric("Rows", rows_sb)
        except Exception as exc:
            st.info(f"Status unavailable: {exc}")
        # Flow status
        try:
            status_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "flow_status.json"))
            if os.path.exists(status_path):
                import json as _json
                with open(status_path, "r", encoding="utf-8") as f:
                    st.caption("Last Flow Run:")
                    st.code(f.read(), language="json")
        except Exception:
            pass

    # Simplicity first: hide advanced by default (toggle in sidebar)
    show_adv = st.session_state.get("show_adv", False)
    if show_adv:
        mode = st.radio(
            "Workflow",
            ["Signal Feed", "Quick Signal", "Correlation Explorer", "Port Signals Predictor", "Event Study"],
            index=0,
            horizontal=True,
            help="Auto‑curated feed of top signals, or switch to manual workflows.",
        )
    else:
        mode = "Signal Feed"

    if mode == "Signal Feed":
        # Market overview first (tiles/heatmap)
        try:
            tape = ["SPY","QQQ","IWM","IYT","XRT","XLI","XLB","XLE"]
            prices_intraday = _load_intraday_prices(tape)
            # Ticker tape style metrics (last price and % change)
            tape_cols = st.columns(len(tape))
            for i, t in enumerate(tape):
                try:
                    s = prices_intraday[t].dropna()
                    last = float(s.iloc[-1])
                    pct = float((s.iloc[-1] / s.iloc[0] - 1.0) * 100)
                    tape_cols[i].metric(t, f"{last:.2f}", delta=f"{pct:.2f}%")
                except Exception:
                    pass
            mfig = _draw_market_heatmap(prices_intraday)
            st.pyplot(mfig); plt.close(mfig)
            # Mini intraday charts for SPY and IYT
            cspy, city = st.columns(2)
            try:
                cspy.line_chart(prices_intraday[["SPY"]].dropna())
                city.line_chart(prices_intraday[["IYT"]].dropna())
            except Exception:
                pass
        except Exception:
            pass

        # One-button feed with Settings gear
        # Defaults
        min_conf = int(st.session_state.get("min_conf", 60))
        window_days = int(st.session_state.get("window_days", 365))
        # Controls row
        cgear, cbtn1, cbtn2, _sp = st.columns([0.6, 1.2, 1.0, 6])
        try:
            # Streamlit >= 1.31 has popover
            pop = getattr(st, "popover", None)
            if pop is not None:
                with cgear.popover("⚙ Settings"):
                    min_conf = int(st.slider("Minimum confidence", min_value=50, max_value=95, value=min_conf, step=5))
                    window_days = int(st.slider("Scan window (days)", min_value=60, max_value=730, value=window_days, step=30))
                    st.session_state["min_conf"] = min_conf
                    st.session_state["window_days"] = window_days
            else:
                with cgear.expander("⚙ Settings", expanded=False):
                    min_conf = int(st.slider("Minimum confidence", min_value=50, max_value=95, value=min_conf, step=5))
                    window_days = int(st.slider("Scan window (days)", min_value=60, max_value=730, value=window_days, step=30))
                    st.session_state["min_conf"] = min_conf
                    st.session_state["window_days"] = window_days
        except Exception:
            pass
        run = cbtn1.button("Show Top 3 Signals Today", type="primary")
        refresh = cbtn2.button("Refresh Signals", help="Rescan ports and markets now")
        # Auto-run on first load for instant visuals
        if "__feed_ran__" not in st.session_state:
            st.session_state["__feed_ran__"] = True
            run = True
        if run or refresh:
            df_all = _load_default_features()
            if df_all is None or df_all.empty:
                st.warning("No features found. Using demo signals to illustrate.")
                signals = generate_demo_signals()
            else:
                tickers_default = ["SPY", "IYT", "XRT", "XLI", "XLB", "XLE"]
                with st.spinner("Scanning signals..."):
                    signals = auto_scan_signals(df_all, tickers=tickers_default, window_days=window_days, lags=[-15,-10,-5,0,5,10,15], max_signals=3)
                if not signals:
                    st.info("No strong candidates right now. Showing demo signals for illustration.")
                    signals = generate_demo_signals()
                # Events panel
                try:
                    events = detect_today_events(df_all)
                    if not events.empty:
                        st.markdown("### Today’s Port Events")
                        st.dataframe(events.head(5), use_container_width=True)
                except Exception:
                    pass
                # Precompute quick forecasts, actions, and confidence for Trade Board
                max_score = max((s.get("score", 0.0) for s in signals), default=1.0) or 1.0
                for s in signals:
                    # Quick 10d forecast on selected feature for the port
                    try:
                        dfp_port = df_all[df_all["port"] == s["port"]].copy()
                        dfp_port["Date"] = pd.to_datetime(dfp_port["Date"]) 
                        dfp_port = dfp_port.set_index("Date").sort_index()
                        from .ports import create_lagged_features
                        from .backtest import compute_forward_returns
                        from sklearn.linear_model import Ridge
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.pipeline import Pipeline
                        Xb = create_lagged_features(dfp_port, [s["feature"]], [0,1,2,3,5,10])
                        yb = compute_forward_returns(s["series_mkt"], horizon=10, log=True)
                        jj = Xb.join(yb, how="inner").dropna()
                        y_hat = float('nan')
                        if not jj.empty:
                            X_al = jj.drop(columns=[yb.name]); y_al = jj[yb.name]
                            model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=0))])
                            model.fit(X_al, y_al)
                            latest_dt = Xb.index.max()
                            x_last = Xb.loc[[latest_dt]].reindex(columns=X_al.columns)
                            if not x_last.isna().any().any():
                                y_hat = float(model.predict(x_last)[0])
                    except Exception:
                        y_hat = float('nan')
                    s["y_hat"] = y_hat
                    s["action"] = "BUY" if (y_hat == y_hat and y_hat > 0) else "SELL"
                    s["etfs"] = _suggest_tickers(s["industries"]) or [s["ticker"]]
                    # Confidence scaled 0–100 using score relative to best + p-values
                    conf_base = (s.get("score", 0.0) / max_score)
                    s["confidence"] = int(60 + 40 * max(0.0, min(1.0, conf_base)))

                # Trade Board top strip
                st.markdown("## Trade Board")
                # Last updated stamp (intraday prices timestamp if available)
                try:
                    last_ts = prices_intraday.index[-1]
                    ts_txt = str(getattr(last_ts, 'tz_convert', lambda tz: last_ts)())
                    st.caption(f"Last updated: {last_ts}")
                except Exception:
                    pass
                cols = st.columns(3)
                # filter by confidence preference if available
                filtered = [s for s in signals if s.get("confidence", 0) >= min_conf] or signals
                top3 = filtered[:3]
                for c, s in zip(cols, top3):
                    with c:
                        # Card-style tile
                        pct = (np.exp(s["y_hat"]) - 1.0) * 100 if (s["y_hat"] == s["y_hat"]) else float('nan')
                        badge = 'buy' if s['action'] == 'BUY' else 'sell'
                        etfs = ', '.join(s['etfs'])
                        pct_txt = f"{pct:.2f}%" if pct == pct else "NA"
                        html = f"""
                        <div class='card'>
                          <div class='badge {badge}'>{s['action']}</div>
                          <div style='font-weight:700;margin-top:6px'>{etfs}</div>
                          <div style='font-size:0.9rem;color:#cbd5e1'>Forecast ~10d: {pct_txt}</div>
                          <div class='bar'><div style='width:{s['confidence']}%'></div></div>
                          <div class='small-caption'>{s['port']} → {s['ticker']} | best lag {s['lag']}d | exp {s.get('expected','?')}d</div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)

                # Detailed cards with visuals and reports
                for i, sig in enumerate(top3, 1):
                    st.markdown("")
                    st.markdown(f"### {i}. {sig['port']} – {sig['feature']} → {sig['ticker']}")
                    st.caption(f"Expected impact window ~{sig.get('expected', 'N/A')}d | best lag {sig['lag']}d")
                    # Quick forecast & recommendation
                    y_hat = sig.get("y_hat", float('nan'))
                    headline, detail = _recommend_action(sig["coef"], sig.get("p_perm"), sig.get("p_newey"), y_hat, 10)
                    st.markdown(f"**Recommendation:** {headline}")
                    st.caption(detail)
                    tick_list = sig.get("etfs") or [sig["ticker"]]
                    action = sig.get("action", "BUY")
                    st.markdown(f"**Trade idea:** {action} {', '.join(tick_list)} (horizon ~10d)")
                    # Affected industries summary (aggregated across top 3)
                    try:
                        if i == 1:
                            counts = {}
                            for s in top3:
                                for ind in s.get("industries", []):
                                    counts[ind] = counts.get(ind, 0) + 1
                            if counts:
                                st.markdown("#### Most Impacted Industries")
                                fig_inds = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={"x":"Industry","y":"Signals"})
                                fig_inds.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
                                st.plotly_chart(fig_inds, use_container_width=True)
                    except Exception:
                        pass

                    # Three visuals: Today vs 30d, Overlay, Heatmap
                    col1, col2, col3 = st.columns([1,1,1])
                    # Today vs 30d histogram
                    with col1:
                        try:
                            s_feat = sig["series_feat"].dropna()
                            last30 = s_feat.iloc[-30:]
                            fig = px.histogram(last30, nbins=15, title="Feature Today vs 30d range")
                            fig.add_vline(x=float(s_feat.iloc[-1]), line_color="red")
                            fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    # Overlay
                    with col2:
                        try:
                            zf = (sig["series_feat"] - sig["series_feat"].mean()) / (sig["series_feat"].std(ddof=0) or 1.0)
                            zm = (sig["series_mkt"] - sig["series_mkt"].mean()) / (sig["series_mkt"].std(ddof=0) or 1.0)
                            ov = pd.concat({sig["feature"]: zf, sig["ticker"]: zm}, axis=1).dropna()
                            fig = px.line(ov, labels={"value":"z-score","index":"Date","variable":"Series"})
                            fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    # Heatmap
                    with col3:
                        try:
                            lags_sorted = sorted([int(x) for x in sig["cdf"].index.tolist()])
                            vals = [float(sig["cdf"].loc[l, "coef"]) for l in lags_sorted]
                            fig = go.Figure(data=go.Heatmap(z=[vals], x=lags_sorted, colorscale="RdBu", zmin=-1, zmax=1, showscale=True))
                            fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10), title="r vs lag")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

                    # Report button
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pdf_out = os.path.join(tmpdir, f"signal_{i}.pdf")
                        create_personalized_report(
                            port=sig["port"],
                            features_df=sig["series_feat"].to_frame(name=sig["feature"]).dropna(),
                            ticker=sig["ticker"],
                            market_series=sig["series_mkt"],
                            lags=[-15,-10,-5,0,5,10,15],
                            horizon=10,
                            returns_mode=True,
                            event_options={"threshold": 2.0, "direction": "above", "pre": 10, "post": 20},
                            output_path=pdf_out,
                        )
                        with open(pdf_out, "rb") as f:
                            st.download_button(
                                f"Download Report – {sig['port']} → {sig['ticker']}",
                                data=f,
                                file_name=f"signal_{sig['port'].replace(' ','_')}_{sig['ticker']}.pdf",
                                mime="application/pdf",
                                key=f"sig_dl_{i}",
                            )
                    with st.expander("Advanced statistics"):
                        try:
                            st.write({
                                "r_best": round(sig["coef"], 3),
                                "lag_best": int(sig["lag"]),
                                "perm_p": (round(sig["p_perm"], 4) if sig["p_perm"] == sig["p_perm"] else None),
                                "newey_west_p": (round(sig["p_newey"], 4) if sig["p_newey"] == sig["p_newey"] else None),
                                "expected_lead_days": sig.get("expected"),
                            })
                            st.dataframe(sig["cdf"].rename(columns={"coef":"r","p_value":"p"}))
                        except Exception:
                            st.info("Details unavailable.")

    elif mode == "Quick Signal":
        # Minimal, fast view: pick a port + feature and a ticker, get key KPIs
        st.subheader("Quick Signal")
        st.caption("Pick a port feature and a market ticker to see lead/lag, significance, and a quick forecast.")
        # Behind-the-scenes: try to auto-load features (DuckDB → demo CSV fallback)
        df_feat_all = None
        try:
            df_feat_all = duckdb_read("data/ports.duckdb", "port_features")
        except Exception:
            pass
        if df_feat_all is None or df_feat_all.empty:
            try:
                demo_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "port_features_template.csv"))
                df_feat_all = load_port_features_csv(demo_csv, date_col="Date", port_col="port")
            except Exception:
                df_feat_all = None

        if df_feat_all is not None and not df_feat_all.empty:
            ports = sorted(df_feat_all["port"].dropna().unique().tolist())
            pcol, fcol, tcol = st.columns([1.2, 1.2, 1])
            qs_port = pcol.selectbox("Port", ports, key="qs_port", help="Select the port to analyze.")
            dfp = filter_port(df_feat_all, qs_port, date_col="Date", port_col="port")
            num_cols = [c for c in dfp.columns if pd.api.types.is_numeric_dtype(dfp[c])]
            # Sensible defaults: prefer anchored_count, else first numeric column
            default_feat = "anchored_count" if "anchored_count" in num_cols else (num_cols[0] if num_cols else None)
            qs_feat = fcol.selectbox("Feature", num_cols, index=(num_cols.index(default_feat) if default_feat in num_cols else 0), key="qs_feat", help="Port feature to use.")
            qs_ticker = tcol.selectbox("Ticker", ["SPY", "IYT", "DIA", "QQQ", "XRT"], index=0, key="qs_ticker", help="Market ticker to compare/predict.")

            # Hidden defaults for simplicity
            qs_lags = "-15,-10,-5,0,5,10,15"
            qs_hz = 10

            if st.button("Analyze", key="qs_run", help="Compute signal KPIs and a quick forecast"):
                try:
                    # Build series
                    feat = dfp[qs_feat].dropna()
                    start_d = str(feat.index.min().date())
                    end_d = str(feat.index.max().date())
                    mkt_df = load_market_data([qs_ticker], start_date=start_d, end_date=end_d)
                    mkt = mkt_df[qs_ticker]

                    from .analysis import to_log_returns, lagged_correlation, circular_shift_permutation_pvalue, newey_west_pvalue

                    sa = to_log_returns(feat)
                    sb = to_log_returns(mkt)
                    idx = sa.index.intersection(sb.index)
                    if len(idx) < 50:
                        st.error("Not enough overlapping data (need ~50+ points). Try a broader date range.")
                        st.stop()
                    sa = sa.loc[idx]
                    sb = sb.loc[idx]
                    lags = [int(x.strip()) for x in qs_lags.split(",") if x.strip()]
                    cdf = lagged_correlation(sa, sb, lags=lags, method="spearman")
                    if cdf.empty:
                        st.error("Correlation could not be computed. Try different lags.")
                        st.stop()
                    pk_lag = int(cdf["coef"].abs().idxmax())
                    pk_coef = float(cdf.loc[pk_lag, "coef"])
                    p_perm = circular_shift_permutation_pvalue(sa, sb, lag=pk_lag, method="spearman", n_perm=300)
                    _, p_nw = newey_west_pvalue(feat, mkt, lag=pk_lag, use_returns=True, nw_lags=5)

                    # KPIs
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Lead/Lag (days)", f"{pk_lag}")
                    k2.metric("r at best lag", f"{pk_coef:.2f}")
                    k3.metric("Permutation p", f"{p_perm:.3f}")
                    k4.metric("Newey–West p", f"{(p_nw if p_nw==p_nw else float('nan')):.3f}")

                    # Quick prediction using lagged feature(s)
                    from .ports import create_lagged_features
                    from .backtest import compute_forward_returns
                    from sklearn.linear_model import Ridge
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline

                    X = create_lagged_features(dfp, [qs_feat], [0, 1, 2, 3, 5, 10])
                    y = compute_forward_returns(mkt, horizon=qs_hz, log=True)
                    jj = X.join(y, how="inner").dropna()
                    if not jj.empty:
                        X_al = jj.drop(columns=[y.name])
                        y_al = jj[y.name]
                        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=0))])
                        model.fit(X_al, y_al)
                        latest_dt = X.index.max()
                        x_last = X.loc[[latest_dt]].reindex(columns=X_al.columns)
                        if not x_last.isna().any().any():
                            y_hat = float(model.predict(x_last)[0])
                            st.metric(f"Next {qs_hz}d predicted log return", f"{y_hat:.4f}")
                        else:
                            y_hat = float('nan')
                    else:
                        y_hat = float('nan')

                    # Recommendation
                    headline, detail = _recommend_action(pk_coef, p_perm, p_nw, y_hat, qs_hz)
                    st.subheader(headline)
                    st.write(detail)

                    # Simple overlay plot (z‑scored levels)
                    z_a = (dfp[qs_feat] - dfp[qs_feat].mean()) / (dfp[qs_feat].std(ddof=0) or 1.0)
                    z_b = (mkt - mkt.mean()) / (mkt.std(ddof=0) or 1.0)
                    ov = pd.concat({qs_feat: z_a, qs_ticker: z_b}, axis=1).dropna()
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.line_chart(ov)
                    with c_right:
                        try:
                            # Lag heatmap
                            lags_sorted = [int(x.strip()) for x in qs_lags.split(',') if x.strip()]
                            vals = [float(cdf.loc[l, 'coef']) if l in cdf.index else float('nan') for l in lags_sorted]
                            hm = np.array([vals])
                            fig, ax = plt.subplots(figsize=(4.2, 1.6), dpi=150)
                            im = ax.imshow(hm, aspect="auto", cmap=cm.coolwarm, vmin=-1, vmax=1)
                            ax.set_yticks([]); ax.set_xticks(range(len(lags_sorted)))
                            ax.set_xticklabels(lags_sorted, fontsize=8)
                            ax.set_title("r vs lag")
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            pass

                    # Quick report
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pdf_out = os.path.join(tmpdir, "quick_signal_report.pdf")
                        create_personalized_report(
                            port=qs_port,
                            features_df=dfp[[qs_feat]].dropna(),
                            ticker=qs_ticker,
                            market_series=mkt,
                            lags=[int(x.strip()) for x in qs_lags.split(',') if x.strip()],
                            horizon=qs_hz,
                            returns_mode=True,
                            event_options={"threshold": 2.0, "direction": "above", "pre": 10, "post": 20},
                            output_path=pdf_out,
                        )
                        with open(pdf_out, "rb") as f:
                            st.download_button("Download Quick Report (PDF)", data=f, file_name="quick_signal_report.pdf", mime="application/pdf")
                except Exception as exc:
                    st.error(f"Quick Signal failed: {exc}")

    elif mode == "Correlation Explorer":
        uploaded_files = st.file_uploader(
            "Operational CSV files",
            type="csv",
            accept_multiple_files=True,
            help="Upload one or more CSVs with a date column and numeric metrics.",
        )
        date_col = st.text_input(
            "Date column name", value="date", help="Name of the date/timestamp column in your CSV(s)."
        )
        value_cols_input = st.text_input(
            "Value column names (comma separated)",
            value="value",
            help="Comma‑separated numeric columns to analyze (e.g., value,metric1).",
        )
        with st.expander("Advanced options", expanded=False):
            returns_mode = st.checkbox("Compute on log returns (recommended)", value=True, key="adv_ret_corr")
            adv_granger = st.checkbox("Run Granger causality test", value=True, key="adv_gr_corr")
            adv_coint = st.checkbox("Run cointegration test (levels)", value=False, key="adv_ci_corr")
            adv_neutral = st.checkbox("Neutralize vs benchmark returns", value=False, key="adv_neut_corr")
        source = st.radio(
            "Market data source",
            ["Yahoo Finance", "Upload CSV"],
            horizontal=True,
            help="Download prices from Yahoo or upload a custom market CSV.",
        )
        if source == "Yahoo Finance":
            tickers_input = st.text_input(
                "Market tickers (comma separated)", value="SPY", help="One or more tickers, comma‑separated."
            )
            bench_ticker = None
            if adv_neutral:
                bench_ticker = st.text_input(
                    "Benchmark ticker for neutralization", value="SPY", help="Ticker to regress out from both series."
                )
            start_date = st.date_input("Start date", value=date(2025, 9, 1), min_value=date(2000, 1, 1))
            end_date = st.date_input("End date", value=date(2025, 9, 10), min_value=date(2000, 1, 2))
            market_upload = None
        else:
            market_upload = st.file_uploader("Market CSV (Date + one or more ticker columns)", type="csv")
            tickers_input = ""
            start_date = None
            end_date = None
        lags_input = st.text_input(
            "Lags (comma separated integers)", value="0", help="Lag values (in days) to evaluate."
        )

        if st.button("Run Analysis", help="Compute correlations, significance and plots for these inputs."):
            if not uploaded_files:
                st.error("Please upload at least one operational CSV file.")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    op_paths = []
                    for file in uploaded_files:
                        file_path = os.path.join(tmpdir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        op_paths.append(file_path)

                    value_cols = [c.strip() for c in value_cols_input.split(",") if c.strip()]
                    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()] if tickers_input else []
                    lags = [int(l.strip()) for l in lags_input.split(",") if l.strip()]

                    # Prepare market data either from Yahoo or uploaded CSV
                    try:
                        if source == "Yahoo Finance":
                            if not tickers:
                                st.error("Enter at least one ticker or switch to CSV upload.")
                                return
                            results = run_pipeline(
                                op_files=op_paths,
                                date_col=date_col,
                                value_cols=value_cols,
                                tickers=tickers,
                                start_date=start_date.strftime("%Y-%m-%d"),
                                end_date=end_date.strftime("%Y-%m-%d"),
                                lags=lags,
                                returns_mode=returns_mode,
                                do_granger=adv_granger,
                                do_coint=adv_coint,
                                neutralize_benchmark_ticker=(bench_ticker if 'bench_ticker' in locals() and bench_ticker else None),
                            )
                            # Regenerate joined data for plotting
                            from .data_ingest import load_market_data
                            op_df = load_operational_data(op_paths, date_col=date_col, value_cols=value_cols)
                            market_df = load_market_data(tickers, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
                        else:
                            if market_upload is None:
                                st.error("Please upload a market CSV.")
                                return
                            # Save uploaded market CSV
                            mkt_path = os.path.join(tmpdir, market_upload.name)
                            with open(mkt_path, "wb") as f:
                                f.write(market_upload.getvalue())
                            # Load operational and market
                            op_df = load_operational_data(op_paths, date_col=date_col, value_cols=value_cols)
                            mdf = pd.read_csv(mkt_path)
                            # Expect 'Date' column
                            if "Date" not in mdf.columns:
                                st.error("Market CSV must contain a 'Date' column.")
                                return
                            mdf["Date"] = pd.to_datetime(mdf["Date"]) 
                            mdf = mdf.set_index("Date").sort_index()
                            market_df = mdf
                            tickers = [c for c in market_df.columns]
                            # Build results using pipeline logic but bypassing download
                            from .analysis import lagged_correlation, fdr_adjust, rolling_stability
                            from .analysis import to_log_returns, run_granger, run_cointegration, neutralize_against
                            joined = op_df.join(market_df, how="inner")
                            corr_frame = joined.apply(to_log_returns) if returns_mode else joined
                            if adv_neutral:
                                # User selects benchmark column for neutralization
                                bench_col = st.selectbox("Benchmark column (neutralization)", market_df.columns.tolist(), key="bench_csv_corr")
                                bench = corr_frame[bench_col] if bench_col in corr_frame.columns else market_df[bench_col].pct_change()
                                bench = bench.rename("bench")
                                # Neutralize op and each ticker vs bench
                                neuts = {}
                                for c in corr_frame.columns:
                                    if c == bench_col:
                                        continue
                                    try:
                                        neuts[c] = neutralize_against(corr_frame[c], bench.to_frame())
                                    except Exception:
                                        neuts[c] = corr_frame[c]
                                corr_frame = pd.DataFrame(neuts)
                            results = []
                            pidx = []
                            pvals = []
                            cache = {}
                            for op_col in op_df.columns:
                                for t in tickers:
                                    df = lagged_correlation(corr_frame[op_col], corr_frame[t], lags=lags, method="spearman")
                                    cache[(op_col, t)] = df
                                    for lag, row in df.iterrows():
                                        pidx.append((op_col, t, int(lag)))
                                        pvals.append(float(row.get("p_value", float("nan"))))
                            pseries = pd.Series(pvals, index=pd.MultiIndex.from_tuples(pidx, names=["metric","ticker","lag"]))
                            fdr = fdr_adjust(pseries)
                            for (op_col, t), df in cache.items():
                                sub = fdr.loc[(op_col, t)] if (op_col, t) in fdr.index else None
                                if sub is not None:
                                    df = df.copy()
                                    df["p_adj"] = sub.loc[df.index, "p_adj"].values
                                    df["rejected"] = sub.loc[df.index, "rejected"].values
                                if "rejected" in df.columns and df["rejected"].any():
                                    peak_idx = df.loc[df["rejected"]]["coef"].abs().idxmax()
                                else:
                                    peak_idx = df["coef"].abs().idxmax()
                                peak_row = df.loc[peak_idx]
                                stab = rolling_stability(corr_frame[op_col], corr_frame[t], lead_lag=int(peak_idx), window=90, step=14, method="spearman", alpha=0.05)
                                results.append(
                                    {
                                        "metric": op_col,
                                        "ticker": t,
                                        "correlations": df["coef"].to_dict(),
                                        "p_values": df["p_value"].to_dict(),
                                        "p_adj": (df["p_adj"].to_dict() if "p_adj" in df.columns else {}),
                                        "significant_lags": (
                                            [int(l) for l, v in df["rejected"].items() if bool(v)] if "rejected" in df.columns else []
                                        ),
                                        "peak": {
                                            "lag": int(peak_idx),
                                            "coef": float(peak_row["coef"]),
                                            "p_value": float(peak_row.get("p_value", float("nan"))),
                                            "p_adj": float(peak_row.get("p_adj", float("nan"))) if not pd.isna(peak_row.get("p_adj", float("nan"))) else None,
                                            "significant": bool(peak_row.get("rejected", False)) if "rejected" in df.columns else False,
                                            "stability_share": float(stab.get("stability_share", float("nan"))),
                                            "stability_windows": int(stab.get("n_windows", 0)),
                                        },
                                        **({"granger": run_granger(joined[op_col], joined[t], maxlag=max(lags) if hasattr(lags,'__iter__') else 10, use_returns=True)} if adv_granger else {}),
                                        **({"cointegration": run_cointegration(joined[op_col], joined[t])} if adv_coint else {}),
                                    }
                                )
                    except Exception as exc:
                        st.error(f"Error running pipeline: {exc}")
                        return

                    # Compose joined data for plotting
                    joined = op_df.join(market_df, how="inner")
                    plot_paths = generate_plots(joined, results, output_dir=tmpdir, normalize=True)

                    tabs = st.tabs(["Summary", "Plots", "Download", "Raw"])
                    with tabs[0]:
                        top = _summarize_top_results(results, top_n=5)
                        st.dataframe(pd.DataFrame(top))
                    with tabs[1]:
                        for metric, paths in plot_paths.items():
                            for p in paths:
                                st.image(p, caption=os.path.basename(p))
                    with tabs[2]:
                        pdf_path = os.path.join(tmpdir, "report.pdf")
                        create_pdf_report(results, plot_paths, pdf_path)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Download PDF Report",
                                data=f,
                                file_name="signal_discovery_report.pdf",
                                mime="application/pdf",
                            )
                    with tabs[3]:
                        st.write(results)
    elif mode == "Port Signals Predictor":
        st.subheader("Port Signals Predictor")
        st.markdown("Analyze port activity features vs market returns. Use CSV or Live Mode (DuckDB).")

        data_source = st.radio(
            "Port features source",
            ["Upload CSV", "DuckDB (Live Mode)"],
            horizontal=True,
            help="Provide features via file upload or read from a live DuckDB table.",
        )
        date_col_p = st.text_input("Port Date column", value="Date", help="Date column in your features data.")
        port_col_p = st.text_input("Port name column", value="port", help="Port name column in your features data.")

        if data_source == "Upload CSV":
            port_csv = st.file_uploader(
                "Port features CSV", type="csv", help="Upload daily port features (Date, port, and numeric columns)."
            )
            df_port_all = None
            if port_csv is not None:
                try:
                    df_port_all = load_port_features_csv(port_csv, date_col=date_col_p, port_col=port_col_p)
                except Exception as exc:
                    st.error(f"Error reading port CSV: {exc}")
                    return
        else:
            db_path = st.text_input("DuckDB path", value="data/ports.duckdb", help="Path to DuckDB file.")
            table = st.text_input("DuckDB table", value="port_features", help="Features table name in DuckDB.")
            df_port_all = None
            if st.button("Load from DuckDB", help="Load features into memory from DuckDB for analysis."):
                try:
                    df_port_all = duckdb_read(db_path, table)
                    st.success(f"Loaded {len(df_port_all)} rows from {db_path}:{table}")
                except Exception as exc:
                    st.error(f"Error loading DuckDB: {exc}")
                    return
            with st.expander("Fetch latest from Datalastic (optional)"):
                try:
                    import os
                    from .connectors.ais_datalastic import DatalasticClient
                    from .ais import load_port_geofences, assign_port_to_points, derive_daily_port_features, polygon_bounds
                    from .duckdb_io import write_port_features_df
                    from datetime import datetime, timedelta, timezone
                    geojson_bytes = None
                    geojson_file = st.file_uploader(
                        "GeoJSON with port polygons",
                        type=["geojson"],
                        key="dgeojson",
                        help="Upload port polygons; defaults to examples/ports.geojson if omitted.",
                    )
                    if geojson_file is None:
                        st.caption("Using default examples/ports.geojson if not provided.")
                    hours = st.number_input(
                        "Lookback window (hours)", min_value=1, max_value=168, value=24, key="d_hours", help="How many hours back to fetch."
                    )
                    api_key = st.text_input(
                        "Datalastic API Key (or set env DATALASTIC_API_KEY)", type="password", key="d_api", help="Your Datalastic API key."
                    )
                    do_fetch = st.button("Fetch & Append to DuckDB", key="d_fetch", help="Fetch recent AIS and append features.")
                    backfill_days = st.number_input(
                        "Backfill days (historical)", min_value=0, max_value=120, value=0, key="d_backfill_days", help="Fetch historical days and append."
                    )
                    do_backfill = st.button("Backfill & Append", key="d_backfill_btn", help="Run the backfill now.")
                    if do_fetch:
                        # Load geofences
                        import tempfile, json
                        if geojson_file is not None:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tf:
                                tf.write(geojson_file.getvalue())
                                geojson_path = tf.name
                        else:
                            geojson_path = os.path.join(os.path.dirname(__file__), "..", "examples", "ports.geojson")
                            geojson_path = os.path.abspath(geojson_path)
                        fences = load_port_geofences(geojson_path)
                        if not fences:
                            st.error("No ports found in GeoJSON")
                            st.stop()
                        # Time window
                        t_end = datetime.now(timezone.utc)
                        t_start = t_end - timedelta(hours=int(hours))
                        client = DatalasticClient(api_key=api_key or os.environ.get("DATALASTIC_API_KEY"))
                        from src.connectors.ais_datalastic import DatalasticClient as _DC  # noqa
                        import pandas as _pd  # noqa
                        all_features = []
                        for fence in fences:
                            mnx, mny, mxx, mxy = polygon_bounds(fence.polygons)
                            def _iso(dt):
                                return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                            try:
                                df = client.fetch_positions_bbox(mnx, mny, mxx, mxy, _iso(t_start), _iso(t_end))
                            except Exception as exc:
                                st.warning(f"Fetch failed for {fence.name}: {exc}")
                                continue
                            if df.empty:
                                continue
                            df_tag = assign_port_to_points(df, [fence])
                            df_tag = df_tag[df_tag["port"].notna()]
                            if df_tag.empty:
                                continue
                            feats = derive_daily_port_features(df_tag)
                            all_features.append(feats)
                        if not all_features:
                            st.info("No features to append from this fetch window.")
                        else:
                            out = _pd.concat(all_features, ignore_index=True)
                            total = write_port_features_df(db_path, out, table=table, mode="append")
                            st.success(f"Appended {len(out)} rows. Table now has ~{total} rows.")
                    if do_backfill and backfill_days > 0:
                        fences = load_port_geofences(geojson_path)
                        if not fences:
                            st.error("No ports found in GeoJSON")
                            st.stop()
                        appended = 0
                        for d in range(int(backfill_days), 0, -1):
                            t0 = (datetime.now(timezone.utc) - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
                            t1 = t0 + timedelta(days=1)
                            day_feats = []
                            for fence in fences:
                                mnx, mny, mxx, mxy = polygon_bounds(fence.polygons)
                                def _iso(dt):
                                    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                                try:
                                    df = client.fetch_positions_bbox(mnx, mny, mxx, mxy, _iso(t0), _iso(t1))
                                except Exception as exc:
                                    st.warning(f"Backfill failed {fence.name} {t0.date()}: {exc}")
                                    continue
                                if df.empty:
                                    continue
                                df_tag = assign_port_to_points(df, [fence])
                                df_tag = df_tag[df_tag["port"].notna()]
                                if df_tag.empty:
                                    continue
                                feats = derive_daily_port_features(df_tag)
                                day_feats.append(feats)
                            if day_feats:
                                out = _pd.concat(day_feats, ignore_index=True)
                                total = write_port_features_df(db_path, out, table=table, mode="append")
                                appended += len(out)
                        st.success(f"Backfill appended {appended} rows. Table now has ~{total} rows.")
                except Exception as exc:
                    st.info(f"Datalastic fetch not available: {exc}")

        source = st.radio(
            "Market data source",
            ["Yahoo Finance", "Upload CSV"],
            horizontal=True,
            key="ports_source",
            help="Download prices from Yahoo or upload a market CSV (Date + ticker).",
        )
        if source == "Yahoo Finance":
            ticker = st.text_input("Market ticker", value="SPY")
            start_date = st.date_input("Start date", value=date(2025, 9, 1), min_value=date(2000, 1, 1), key="ports_start")
            end_date = st.date_input("End date", value=date(2025, 9, 10), min_value=date(2000, 1, 2), key="ports_end")
            market_upload = None
        else:
            market_upload = st.file_uploader(
                "Market CSV (Date + ticker price column)", type="csv", key="ports_market_csv", help="Upload a price series to predict."
            )
            ticker = st.text_input(
                "Ticker column name (in Market CSV)", value="SPY", help="Column name in the uploaded CSV to use as target."
            )

        horizon = st.number_input(
            "Forward return horizon (days)", min_value=1, max_value=90, value=10, help="Forward return horizon for the prediction target."
        )
        feature_lags = st.text_input(
            "Feature lags (comma separated)", value="0,1,2,3,5,10", help="Lagged feature offsets to include in the model."
        )

        if st.button("Run Port Backtest"):
            if df_port_all is None or df_port_all.empty:
                st.error("No port features loaded. Upload a CSV or load from DuckDB.")
                return
            ports = sorted(df_port_all[port_col_p].dropna().unique().tolist())
            if not ports:
                st.error("No ports found in the CSV. Check your column mappings.")
                return
            # Priority ports hint (sidebar note)
            try:
                pr = load_priority_ports()
                recommended = sum(pr.values(), []) if pr else []
                st.caption(f"Priority ports: {', '.join(recommended[:8])}...")
            except Exception:
                pass
            sel_port = st.selectbox("Select port", ports, index=(ports.index(recommended[0]) if ports and recommended and recommended[0] in ports else 0))
            df_port = filter_port(df_port_all, sel_port, date_col=date_col_p, port_col=port_col_p)
            # Choose features
            numeric_cols = [c for c in df_port.columns if pd.api.types.is_numeric_dtype(df_port[c])]
            feat_cols = st.multiselect("Features", numeric_cols, default=numeric_cols[: min(5, len(numeric_cols))])
            if not feat_cols:
                st.error("Select at least one feature.")
                return

            lags = [int(x.strip()) for x in feature_lags.split(",") if x.strip()]
            X = create_lagged_features(df_port, feat_cols, lags)

            # Market data
            if source == "Yahoo Finance":
                from .data_ingest import load_market_data
                try:
                    mkt = load_market_data([ticker], start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
                except Exception as exc:
                    st.error(f"Error loading market data: {exc}")
                    return
            else:
                if market_upload is None:
                    st.error("Please upload a Market CSV.")
                    return
                mdf = pd.read_csv(market_upload)
                if "Date" not in mdf.columns or ticker not in mdf.columns:
                    st.error("Market CSV must contain 'Date' and selected ticker column.")
                    return
                mdf["Date"] = pd.to_datetime(mdf["Date"]) 
                mkt = mdf.set_index("Date").sort_index()[[ticker]]

            target = compute_forward_returns(mkt[ticker], horizon=horizon, log=True)
            # Align features and target
            joined = X.join(target, how="inner").dropna()
            if joined.empty:
                st.error("No overlapping data between features and market target after alignment.")
                return
            X_aligned = joined.drop(columns=[target.name])
            y_aligned = joined[target.name]

            # Backtest
            try:
                bt = walk_forward_regression(X_aligned, y_aligned, n_splits=5, alpha=1.0)
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                return

            pred_df = bt.predictions.copy()
            pred_df["cum_true"] = pred_df["y_true"].cumsum()
            pred_df["cum_pred"] = pred_df["y_pred"].cumsum()

            tabs = st.tabs(["Summary", "Chart", "Downloads"])
            with tabs[0]:
                c1, c2, c3 = st.columns(3)
                c1.metric("R² (OOS)", f"{bt.metrics.get('r2', float('nan')):.3f}")
                c2.metric("MAE", f"{bt.metrics.get('mae', float('nan')):.4f}")
                da = bt.metrics.get('directional_accuracy', float('nan'))
                c3.metric("Directional Acc.", f"{da:.2%}" if da == da else "NA")
            with tabs[1]:
                st.line_chart(pred_df[["cum_true", "cum_pred"]])
            with tabs[2]:
                csv_bytes = pred_df.to_csv().encode("utf-8")
                st.download_button("Download Predictions CSV", data=csv_bytes, file_name="port_backtest_predictions.csv", mime="text/csv")
                with tempfile.TemporaryDirectory() as tmpout:
                    pdf_out = os.path.join(tmpout, "backtest_report.pdf")
                    create_backtest_report(
                        port=sel_port,
                        ticker=ticker,
                        horizon=int(horizon),
                        metrics=bt.metrics,
                        pred_df=pred_df,
                        features_used=feat_cols,
                        lags_used=lags,
                        output_path=pdf_out,
                    )
                    with open(pdf_out, "rb") as f:
                        st.download_button("Download Backtest PDF", data=f, file_name="port_backtest_report.pdf", mime="application/pdf")
                    model_card = {
                        "port": sel_port,
                        "ticker": ticker,
                        "horizon_days": int(horizon),
                        "features": feat_cols,
                        "lags": lags,
                        "metrics": bt.metrics,
                        "n_obs": int(len(pred_df)),
                        "date_range": {
                            "start": str(pred_df.index.min().date()),
                            "end": str(pred_df.index.max().date()),
                        },
                    }
                    import json
                    st.download_button(
                        "Download Model Card (JSON)",
                        data=json.dumps(model_card, indent=2).encode("utf-8"),
                        file_name="port_model_card.json",
                        mime="application/json",
                    )

            # Optional: Train-final and predict next horizon using latest features row
            st.subheader("Live Prediction (using latest features row)")
            try:
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                model = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("reg", Ridge(alpha=1.0, random_state=0)),
                ])
                model.fit(X_aligned, y_aligned)
                # Construct latest feature row
                latest_date = X.index.max()
                x_last = X.loc[[latest_date]].reindex(columns=X_aligned.columns)
                if x_last.isna().any().any():
                    st.warning("Latest feature row has missing lag values; prediction may be unavailable.")
                else:
                    y_hat = float(model.predict(x_last)[0])
                    st.write({"port": sel_port, "ticker": ticker, "horizon_days": int(horizon), "predicted_log_return": y_hat})
            except Exception as exc:
                st.info(f"Could not compute live prediction: {exc}")
    else:
        st.subheader("Event Study")
        st.markdown("Identify port feature shocks and measure average market impact in a window around events.")
        src_feat = st.radio(
            "Feature source",
            ["Upload CSV", "DuckDB (Live Mode)"],
            horizontal=True,
            key="es_src",
            help="Load port features from a file or DuckDB table.",
        )
        date_col_e = st.text_input("Feature Date column", value="Date", key="es_date")
        port_col_e = st.text_input("Port column", value="port", key="es_port_col")
        if src_feat == "Upload CSV":
            fcsv = st.file_uploader(
                "Port features CSV", type="csv", key="es_csv", help="Upload daily port features (Date, port, metrics)."
            )
            df_all = None
            if fcsv is not None:
                df_all = load_port_features_csv(fcsv, date_col=date_col_e, port_col=port_col_e)
        else:
            db_path = st.text_input("DuckDB path", value="data/ports.duckdb", key="es_db", help="Path to DuckDB file.")
            table = st.text_input("DuckDB table", value="port_features", key="es_table", help="Features table name.")
            df_all = None
            if st.button("Load features", key="es_load", help="Load features from DuckDB for the event study."):
                try:
                    df_all = duckdb_read(db_path, table)
                    st.success(f"Loaded {len(df_all)} rows")
                except Exception as exc:
                    st.error(f"DuckDB load failed: {exc}")
                    return
        if df_all is not None and not df_all.empty:
            ports = sorted(df_all[port_col_e].dropna().unique().tolist())
            sel_port = st.selectbox("Port", ports, key="es_port")
            dfp = filter_port(df_all, sel_port, date_col=date_col_e, port_col=port_col_e)
            num_cols = [c for c in dfp.columns if pd.api.types.is_numeric_dtype(dfp[c])]
            feat = st.selectbox("Feature", num_cols, key="es_feat", help="Which feature to detect shock events on.")
            zwin = st.number_input("Z-score window (days)", 5, 365, 30, key="es_zwin", help="Window for computing feature z-scores.")
            thr = st.number_input("Event threshold (z-score)", 0.5, 5.0, 2.0, step=0.1, key="es_thr", help="z-score threshold to flag events.")
            direction = st.selectbox("Direction", ["above", "below", "both"], key="es_dir", help="Event direction: positive, negative, or both.")
            spacing = st.number_input("Min spacing between events (days)", 0, 30, 5, key="es_space", help="Minimum days between detected events.")
            pre = st.number_input("Pre-window days", 0, 60, 10, key="es_pre", help="Days before the event in the window.")
            post = st.number_input("Post-window days", 1, 90, 20, key="es_post", help="Days after the event in the window.")

            src_mkt = st.radio(
                "Market data source",
                ["Yahoo Finance", "Upload CSV"],
                horizontal=True,
                key="es_mkt_src",
                help="Pull market prices from Yahoo or upload a CSV.",
            )
            if src_mkt == "Yahoo Finance":
                tick = st.text_input("Market ticker", value="SPY", key="es_tick", help="Target ticker for event impact.")
                sd = st.date_input("Start date", value=dfp.index.min().date() if hasattr(dfp.index, 'min') else date(2024,1,1), key="es_sd", help="Start date for market data.")
                ed = st.date_input("End date", value=dfp.index.max().date() if hasattr(dfp.index, 'max') else date(2025,1,1), key="es_ed", help="End date for market data.")
                if st.button("Run Event Study", key="es_run_y", help="Compute average market impact around feature events."):
                    try:
                        from .data_ingest import load_market_data
                        mkt = load_market_data([tick], start_date=str(sd), end_date=str(ed))
                        series = dfp[feat]
                        out = compute_event_study(series, mkt[tick], z_window=zwin, threshold=thr, direction=direction, min_spacing=spacing, pre=pre, post=post)
                    except Exception as exc:
                        st.error(f"Event study failed: {exc}")
                        return
                    if out["n_events"] == 0:
                        st.info("No events detected with the current settings.")
                    else:
                        st.write({"n_events": out["n_events"]})
                        # Plot
                        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=150)
                        ax.plot(out["avg"].index, out["avg"].values, label="Average cumulative return", color="#1f77b4")
                        if out["lower"] is not None and out["upper"] is not None:
                            ax.fill_between(out["avg"].index, out["lower"].values, out["upper"].values, color="#1f77b4", alpha=0.15, label="95% CI")
                        ax.axvline(0, color="black", linewidth=1)
                        ax.set_xlabel("Days from event")
                        ax.set_ylabel("Cumulative log return")
                        ax.legend()
                        fig.tight_layout()
                        st.pyplot(fig)
            else:
                mkt_csv = st.file_uploader("Market CSV (Date + ticker column)", type="csv", key="es_mkt_csv")
                mcol = st.text_input("Ticker column name", value="SPY", key="es_mcol")
                if st.button("Run Event Study", key="es_run_u", help="Compute event impact using uploaded market CSV.") and mkt_csv is not None:
                    try:
                        mdf = pd.read_csv(mkt_csv)
                        mdf["Date"] = pd.to_datetime(mdf["Date"]).dt.tz_localize(None)
                        mdf = mdf.set_index("Date").sort_index()
                        mkt = mdf[mcol]
                        series = dfp[feat]
                        out = compute_event_study(series, mkt, z_window=zwin, threshold=thr, direction=direction, min_spacing=spacing, pre=pre, post=post)
                    except Exception as exc:
                        st.error(f"Event study failed: {exc}")
                        return
                    if out["n_events"] == 0:
                        st.info("No events detected with the current settings.")
                    else:
                        st.write({"n_events": out["n_events"]})
                        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=150)
                        ax.plot(out["avg"].index, out["avg"].values, label="Average cumulative return", color="#1f77b4")
                        if out["lower"] is not None and out["upper"] is not None:
                            ax.fill_between(out["avg"].index, out["lower"].values, out["upper"].values, color="#1f77b4", alpha=0.15, label="95% CI")
                        ax.axvline(0, color="black", linewidth=1)
                        ax.set_xlabel("Days from event")
                        ax.set_ylabel("Cumulative log return")
                        ax.legend()
                        fig.tight_layout()
                        st.pyplot(fig)

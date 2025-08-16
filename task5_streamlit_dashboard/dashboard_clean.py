# streamlit_superstore_dashboard.py
# Run with:  streamlit run streamlit_superstore_dashboard.py

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# Page & Global Config
# =========================
st.set_page_config(page_title="Global Superstore BI Dashboard", layout="wide")

# =========================
# Helpers
# =========================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_dates_best(series: pd.Series) -> pd.Series:
    s1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return s1 if s1.notna().sum() >= s2.notna().sum() else s2

# =========================
# Robust CSV Loader (cached)
# =========================
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    # Read bytes (works for UploadedFile or local path)
    if hasattr(path_or_file, "read"):  # Streamlit UploadedFile
        data_bytes = path_or_file.getvalue()
    elif isinstance(path_or_file, (str, bytes, bytearray)):
        with open(path_or_file, "rb") as f:
            data_bytes = f.read()
    else:
        raise ValueError("Provide a CSV path or upload a CSV file.")

    # Try multiple encodings; prefer Python engine (sep sniffing), then C engine fallback
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]
    df = None
    last_err = None

    for enc in encodings:
        # Try Python engine (no low_memory here!)
        try:
            bio = io.BytesIO(data_bytes)
            df = pd.read_csv(
                bio,
                encoding=enc,
                engine="python",   # allows sep sniffing
                sep=None,          # let pandas sniff delimiter
                on_bad_lines="skip"
            )
            break
        except Exception as e_py:
            last_err = e_py
            # Fallback: C engine (assumes comma delimiter)
            try:
                bio2 = io.BytesIO(data_bytes)
                df = pd.read_csv(
                    bio2,
                    encoding=enc,    # C engine tolerates low_memory, but we don't need it
                    engine="c"
                )
                break
            except Exception as e_c:
                last_err = e_c
                continue

    if df is None:
        raise RuntimeError(f"Failed to read CSV with common encodings. Last error: {last_err}")

    # Normalize headers
    df.columns = [_norm(c) for c in df.columns]

    # Map common headers to canonical names
    def first_present(cands):
        for cand in cands:
            if _norm(cand) in df.columns:
                return _norm(cand)
        return None

    colmap = {}
    od = first_present(["order date", "order_date", "orderdate", "date"])
    if od: colmap[od] = "order_date"
    rg = first_present(["region"])
    if rg: colmap[rg] = "region"
    cat = first_present(["category"])
    if cat: colmap[cat] = "category"
    scat = first_present(["sub-category", "subcategory", "sub_category", "sub category"])
    if scat: colmap[scat] = "sub_category"
    seg = first_present(["segment"])
    if seg: colmap[seg] = "segment"
    cust = first_present(["customer name", "customer_name", "customername"])
    if cust: colmap[cust] = "customer_name"
    sales = first_present(["sales", "sales amount", "revenue"])
    if sales: colmap[sales] = "sales"
    profit = first_present(["profit", "margin"])
    if profit: colmap[profit] = "profit"

    df = df.rename(columns=colmap)

    # Required columns
    required = ["order_date", "region", "category", "sub_category", "sales", "profit", "customer_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\nFound: {list(df.columns)}"
        )

    # Parse dates
    df["order_date"] = _parse_dates_best(df["order_date"])
    df = df.dropna(subset=["order_date"])

    # Ensure numeric metrics
    for c in ("sales", "profit"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["sales", "profit"])

    # Cast dims to string
    for c in ["region", "category", "sub_category", "segment", "customer_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

# =========================
# Sidebar: Source & Filters
# =========================
st.sidebar.title("🔧 Controls")

default_path = "Global_Superstore2.csv"  # put the CSV next to this script or upload via UI
uploaded = st.sidebar.file_uploader("Upload Global Superstore CSV", type=["csv"])
source = uploaded if uploaded else default_path

# Load data
try:
    df = load_data(source)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Debug panel (prevents blank page confusion)
st.sidebar.success(f"Loaded {'uploaded file' if uploaded else default_path} | {df.shape[0]:,} rows × {df.shape[1]} cols")
with st.expander("🔎 Data preview & columns"):
    st.write("Columns:", list(df.columns))
    st.write("Order date range:", df["order_date"].min(), "→", df["order_date"].max())
    st.dataframe(df.head(25))

# Date selector (robust)
min_date, max_date = df["order_date"].min(), df["order_date"].max()
if pd.isna(min_date) or pd.isna(max_date):
    st.error("No valid 'order_date' values after parsing. Check the CSV column/header.")
    st.stop()

date_range = st.sidebar.date_input(
    "Order Date Range",
    [min_date.date(), max_date.date()],
    min_value=min_date.date(),
    max_value=max_date.date(),
)

regions = sorted(df["region"].dropna().unique())
cats = sorted(df["category"].dropna().unique())
sel_regions = st.sidebar.multiselect("Region", regions, default=regions or None)
sel_cats = st.sidebar.multiselect("Category", cats, default=cats or None)
valid_subcats = sorted(df[df["category"].isin(sel_cats if sel_cats else cats)]["sub_category"].dropna().unique())
sel_subcats = st.sidebar.multiselect("Sub-Category", valid_subcats, default=valid_subcats or None)

# Optional segment filter
segments = sorted(df.get("segment", pd.Series([], dtype=str)).dropna().unique())
sel_segments = st.sidebar.multiselect("Segment (optional)", segments, default=segments or None) if segments else []

# Apply filters; fall back to full data if empty
mask = (
    (df["order_date"].dt.date >= pd.to_datetime(date_range[0]).date()) &
    (df["order_date"].dt.date <= pd.to_datetime(date_range[1]).date()) &
    (df["region"].isin(sel_regions if sel_regions else regions)) &
    (df["category"].isin(sel_cats if sel_cats else cats)) &
    (df["sub_category"].isin(sel_subcats if sel_subcats else valid_subcats))
)
if segments:
    mask &= df["segment"].isin(sel_segments if sel_segments else segments)

fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("⚠️ No data match your filters. Showing full dataset so you can adjust filters.")
    fdf = df.copy()

# =========================
# Main Dashboard
# =========================
st.title("📊 Global Superstore — Interactive BI Dashboard")
st.caption("Filters update all KPIs and charts in real time.")

# KPIs
total_sales = float(fdf["sales"].sum())
total_profit = float(fdf["profit"].sum())
margin = (total_profit / total_sales * 100) if total_sales != 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total Sales", f"${total_sales:,.0f}")
c2.metric("Total Profit", f"${total_profit:,.0f}")
c3.metric("Profit Margin", f"{margin:.1f}%")

st.markdown("---")

# Time series (monthly)
ts = fdf.set_index("order_date").sort_index()
ts_month = ts[["sales", "profit"]].resample("MS").sum().reset_index()

colA, colB = st.columns(2)
with colA:
    fig1 = px.line(ts_month, x="order_date", y="sales", title="Sales Over Time (Monthly)",
                   labels={"order_date": "Month", "sales": "Sales"})
    st.plotly_chart(fig1, use_container_width=True)
with colB:
    fig2 = px.line(ts_month, x="order_date", y="profit", title="Profit Over Time (Monthly)",
                   labels={"order_date": "Month", "profit": "Profit"})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Performance by Category / Sub-Category")

# Category breakdown
cat_agg = fdf.groupby("category", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
fig3 = px.bar(cat_agg, x="category", y="sales", color="profit",
              title="Sales by Category (Color = Profit)", text_auto=True)
st.plotly_chart(fig3, use_container_width=True)

# Sub-Category breakdown (top 15 by sales)
sub_agg = fdf.groupby("sub_category", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
sub_agg = sub_agg.sort_values("sales", ascending=False).head(15)
fig4 = px.bar(sub_agg, x="sub_category", y="sales", color="profit",
              title="Top Sub-Categories by Sales (Top 15)", text_auto=True)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("### Top 5 Customers by Sales")
top_customers = fdf.groupby("customer_name", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
top_customers = top_customers.sort_values("sales", ascending=False).head(5)
fig5 = px.bar(top_customers, x="customer_name", y="sales", color="profit",
              title="Top 5 Customers by Sales", text_auto=True)
st.plotly_chart(fig5, use_container_width=True)

# Segment-wise (optional)
if "segment" in fdf.columns and fdf["segment"].notna().any():
    st.markdown("### Segment-wise Performance")
    seg_agg = fdf.groupby("segment", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
    seg_agg["margin_%"] = np.where(seg_agg["sales"] > 0, seg_agg["profit"] / seg_agg["sales"] * 100, 0)
    fig6 = px.bar(seg_agg, x="segment", y="sales", color="margin_%",
                  title="Sales by Segment (Color = Profit Margin %)", text_auto=True)
    st.plotly_chart(fig6, use_container_width=True)

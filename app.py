import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import csv
from io import BytesIO

# ---------------- Page / Theme ----------------
st.set_page_config(page_title="Bio Companies Explorer", layout="wide")
st.title("🧪 Bio Companies Explorer")

st.caption("Upload your CSV to explore companies, regions, categories, products, and more.")

# ---------------- Helpers ----------------
def try_parse_dates(df: pd.DataFrame):
    for c in df.columns:
        lower = c.lower()
        if any(tok in lower for tok in ["date", "time", "year", "month"]):
            with pd.option_context("mode.chained_assignment", None):
                try:
                    df[c] = pd.to_datetime(df[c], errors="ignore", infer_datetime_format=True)
                except Exception:
                    pass
    return df

def first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_csv_resilient(uploaded_file) -> pd.DataFrame:
    # Detect delimiter using a sample, try utf-8 then latin1
    pos = uploaded_file.tell()
    sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
    uploaded_file.seek(pos)
    try:
        dialect = csv.Sniffer().sniff(sample)
        delim = dialect.delimiter
    except Exception:
        delim = ","
    # Try utf-8
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="utf-8", sep=delim, engine="python")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1", sep=delim, engine="python")

def metric(label, value):
    st.metric(label, value if value is not None else "—")

def filter_multiselect(df, col_name, key=None):
    if col_name and col_name in df.columns:
        opts = sorted(df[col_name].dropna().astype(str).unique().tolist())
        default = opts
        sel = st.sidebar.multiselect(col_name, options=opts, default=default, key=key or f"ms_{col_name}")
        return sel
    return None

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Persist across pages
if uploaded_file is not None:
    st.session_state["_uploaded_name"] = uploaded_file.name
    st.session_state["_uploaded_bytes"] = uploaded_file.getvalue()

def get_df():
    if "_uploaded_bytes" in st.session_state:
        buf = BytesIO(st.session_state["_uploaded_bytes"])
        df = load_csv_resilient(buf)
        return try_parse_dates(df)
    return None

# ---------------- Navigation ----------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Company Deep Dive", "Segments & Regions", "Data Explorer", "About"],
    index=0
)

df = get_df()

if df is None:
    st.info("👆 Upload a CSV to begin. (UTF-8 comma CSV works best.)")
else:
    # Identify semantic columns
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols  = df.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    company_col   = first_present(df, ["Company","CompanyName","Firm","Name"])
    region_col    = first_present(df, ["Region","Geography","Country","Market"])
    category_col  = first_present(df, ["Category","Segment","Type","Class"])
    subcat_col    = first_present(df, ["Subcategory","SubCategory","Subsegment","Sub-Segment"])
    product_col   = first_present(df, ["Product","ProductName","Offering"])
    status_col    = first_present(df, ["Status","Stage","Lifecycle"])
    year_col      = first_present(df, ["Year"]) or (date_cols[0] if date_cols else None)

    # Sidebar filters (global for all pages)
    st.sidebar.header("Filters")
    sel_company  = filter_multiselect(df, company_col, "f_company")
    sel_region   = filter_multiselect(df, region_col, "f_region")
    sel_category = filter_multiselect(df, category_col, "f_category")
    sel_subcat   = filter_multiselect(df, subcat_col, "f_subcat")
    sel_status   = filter_multiselect(df, status_col, "f_status")

    # Date/year
    start_dt = end_dt = None
    if year_col is not None:
        if np.issubdtype(df[year_col].dtype, np.datetime64):
            min_d, max_d = pd.to_datetime(df[year_col]).min(), pd.to_datetime(df[year_col]).max()
            start, end = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
            start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        else:
            min_y, max_y = int(df[year_col].min()), int(df[year_col].max())
            start_y, end_y = st.sidebar.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y))

    # Apply filters
    fdf = df.copy()
    if sel_company is not None:  fdf = fdf[fdf[company_col].astype(str).isin(sel_company)]
    if sel_region is not None:   fdf = fdf[fdf[region_col].astype(str).isin(sel_region)]
    if sel_category is not None: fdf = fdf[fdf[category_col].astype(str).isin(sel_category)]
    if sel_subcat is not None:   fdf = fdf[fdf[subcat_col].astype(str).isin(sel_subcat)]
    if sel_status is not None:   fdf = fdf[fdf[status_col].astype(str).isin(sel_status)]
    if year_col is not None:
        if np.issubdtype(df[year_col].dtype, np.datetime64) and start_dt is not None:
            fdf = fdf[(pd.to_datetime(fdf[year_col]) >= start_dt) & (pd.to_datetime(fdf[year_col]) <= end_dt)]
        elif not np.issubdtype(df[year_col].dtype, np.datetime64):
            fdf = fdf[(fdf[year_col] >= start_y) & (fdf[year_col] <= end_y)]

    # ---------------- Pages ----------------
    if page == "Overview":
        st.subheader("Key Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: metric("Records", len(fdf))
        with k2: metric("Companies", fdf[company_col].nunique() if company_col else None)
        with k3: metric("Regions", fdf[region_col].nunique() if region_col else None)
        with k4: metric("Categories", fdf[category_col].nunique() if category_col else None)
        with k5: metric("Products", fdf[product_col].nunique() if product_col else None)

        st.markdown("---")
        top = st.columns(2)

        with top[0]:
            gcol = category_col or subcat_col or company_col
            if gcol:
                tb = (fdf.groupby(gcol).size().reset_index(name="Count").sort_values("Count", ascending=False).head(20))
                st.plotly_chart(px.bar(tb, x=gcol, y="Count", title=f"Top {gcol}"), use_container_width=True)
            else:
                st.info("No category-like column found.")

        with top[1]:
            if region_col:
                rb = (fdf.groupby(region_col).size().reset_index(name="Count").sort_values("Count", ascending=False))
                st.plotly_chart(px.bar(rb, x=region_col, y="Count", title=f"Records by {region_col}"), use_container_width=True)
            else:
                st.info("No region column found.")

        st.subheader("Hierarchy View")
        hierarchy = [c for c in [region_col, category_col, subcat_col, company_col] if c]
        if len(hierarchy) >= 2:
            sb = (fdf.groupby(hierarchy).size().reset_index(name="Count"))
            st.plotly_chart(px.sunburst(sb, path=hierarchy, values="Count", title="Region → Category → Subcategory → Company"),
                            use_container_width=True)
        else:
            st.info("Need at least two categorical columns for a hierarchy.")

        st.subheader("Trend")
        if year_col is not None:
            if np.issubdtype(df[year_col].dtype, np.datetime64):
                ts = (fdf.assign(_date=pd.to_datetime(fdf[year_col]).dt.to_period("M").dt.to_timestamp())
                        .groupby("_date").size().reset_index(name="Count"))
                st.plotly_chart(px.line(ts, x="_date", y="Count", markers=True, title="Monthly Records"),
                                use_container_width=True)
            else:
                ts = fdf.groupby(year_col).size().reset_index(name="Count")
                st.plotly_chart(px.line(ts, x=year_col, y="Count", markers=True, title="Records by Year"),
                                use_container_width=True)
        else:
            st.info("No date/year column detected.")

    elif page == "Company Deep Dive":
        st.subheader("Company Profile & Comparisons")
        if company_col:
            companies = sorted(fdf[company_col].dropna().unique().tolist())
            target = st.selectbox("Choose a company", companies)
            cdf = fdf[fdf[company_col] == target]
            st.write(f"Records for **{target}**: {len(cdf)}")

            # Show categorical distribution for this company
            two = st.columns(2)
            with two[0]:
                if category_col:
                    ct = cdf[category_col].value_counts().reset_index()
                    ct.columns = [category_col, "Count"]
                    st.plotly_chart(px.bar(ct, x=category_col, y="Count",
                                           title=f"{target}: {category_col} distribution"),
                                    use_container_width=True)
            with two[1]:
                if region_col:
                    rt = cdf[region_col].value_counts().reset_index()
                    rt.columns = [region_col, "Count"]
                    st.plotly_chart(px.bar(rt, x=region_col, y="Count",
                                           title=f"{target}: {region_col} distribution"),
                                    use_container_width=True)

            st.write("Raw records:")
            st.dataframe(cdf, use_container_width=True)
        else:
            st.info("No company column detected.")

    elif page == "Segments & Regions":
        st.subheader("Segments & Regions")
        if region_col and category_col:
            piv = (fdf.pivot_table(index=region_col, columns=category_col, values=fdf.columns[0], aggfunc="count")
                     .fillna(0).astype(int))
            st.dataframe(piv, use_container_width=True)
            heat = piv.reset_index().melt(id_vars=[region_col], var_name=category_col, value_name="Count")
            st.plotly_chart(px.density_heatmap(heat, x=category_col, y=region_col, z="Count",
                                               title="Heatmap: Region vs Category"), use_container_width=True)
        else:
            st.info("Need both a region-like and a category-like column.")

    elif page == "Data Explorer":
        st.subheader("🔎 Query the Dataset (full-text)")
        query = st.text_input("Type keywords (e.g., 'biostimulant Europe R&D')")
        if query:
            keywords = [k.strip().lower() for k in query.split() if k.strip()]
            mask = fdf.apply(lambda row: all(any(k in str(v).lower() for v in row.values) for k in keywords), axis=1)
            results = fdf[mask]
            st.write(f"Found {len(results)} matching records:")
            st.dataframe(results, use_container_width=True)
        else:
            st.info("Type some words to search across all columns.")

        st.markdown("---")
        st.subheader("Ad‑hoc Scatter")
        num_cols = fdf.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns([2,2,2])
            with c1:
                x_sel = st.selectbox("X (numeric)", num_cols, index=0)
            with c2:
                y_sel = st.selectbox("Y (numeric)", num_cols, index=min(1, len(num_cols)-1))
            with c3:
                color_sel = st.selectbox("Color (categorical)", [c for c in [category_col, region_col, company_col, status_col] if c] or [None])
            st.plotly_chart(px.scatter(fdf, x=x_sel, y=y_sel, color=color_sel, hover_data=[company_col] if company_col else None,
                                       title=f"{y_sel} vs {x_sel}"), use_container_width=True)
        else:
            st.info("Need at least two numeric columns for a scatter.")

        st.markdown("---")
        st.subheader("Pivot Table")
        text_cols = fdf.select_dtypes(include="object").columns.tolist()
        left, right = st.columns(2)
        with left:
            rows_col = st.selectbox("Rows", text_cols or fdf.columns)
        with right:
            vals_col = st.selectbox("Values (numeric)", fdf.select_dtypes(include='number').columns.tolist())
        if rows_col and vals_col:
            pv = (fdf.pivot_table(index=rows_col, values=vals_col, aggfunc=["count","mean","sum"])
                    .sort_values(("count", vals_col), ascending=False))
            st.dataframe(pv, use_container_width=True)

        st.markdown("---")
        st.subheader("Download Filtered Data")
        csv_bytes = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")

    elif page == "About":
        st.markdown("""
        **Bio Companies Explorer**  
        - Upload your UTF‑8 CSV and navigate with the left sidebar.  
        - Global filters apply across all pages.  
        - Use the **Data Explorer** page for full‑text search and pivots.  
        """)
        st.write("Detected columns:")
        st.json({
            "company_col": company_col,
            "region_col": region_col,
            "category_col": category_col,
            "subcat_col": subcat_col,
            "product_col": product_col,
            "status_col": status_col,
            "year_col": str(year_col) if year_col is not None else None
        })
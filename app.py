import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Biologicals AI Training Companies", layout="wide")
st.title("🧬 Biologicals AI Training Companies Explorer")

st.caption("Upload an Excel (.xlsx) or CSV file to explore your dataset interactively.")

# ---------------- Helper Functions ----------------
def try_parse_dates(df: pd.DataFrame):
    """Try to convert any date-like columns to datetime."""
    for c in df.columns:
        if any(token in c.lower() for token in ["date", "time", "year", "month"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore", infer_datetime_format=True)
            except Exception:
                pass
    return df

def first_present(df: pd.DataFrame, candidates):
    """Find the first matching column name from a list of possible options."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_file(uploaded_file):
    """Load Excel or CSV and let user pick a sheet if needed."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Select Excel Sheet", xls.sheet_names, index=0)
        df = pd.read_excel(xls, sheet_name=sheet)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    else:
        st.error("Unsupported file type. Please upload a .xlsx or .csv file.")
        return None
    return try_parse_dates(df)

def metric(label, value):
    st.metric(label, value if value is not None else "—")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    df = load_file(uploaded_file)

    if df is not None:
        # Identify key columns
        company_col  = first_present(df, ["Company", "Company Name", "Firm", "Organization"])
        region_col   = first_present(df, ["Region", "Country", "Market"])
        category_col = first_present(df, ["Category", "Segment", "Type", "Class"])
        product_col  = first_present(df, ["Product", "Technology", "Offering"])
        year_col     = first_present(df, ["Year", "Date", "Timestamp"])
        status_col   = first_present(df, ["Status", "Stage", "Lifecycle"])

        # ---------------- Filters ----------------
        st.sidebar.header("Filters")

        def multiselect_filter(col):
            if col and col in df.columns:
                options = sorted(df[col].dropna().unique().tolist())
                return st.sidebar.multiselect(f"{col}", options, default=options)
            return None

        sel_company  = multiselect_filter(company_col)
        sel_region   = multiselect_filter(region_col)
        sel_category = multiselect_filter(category_col)
        sel_status   = multiselect_filter(status_col)

        # Apply filters
        fdf = df.copy()
        if sel_company:  fdf = fdf[fdf[company_col].isin(sel_company)]
        if sel_region:   fdf = fdf[fdf[region_col].isin(sel_region)]
        if sel_category: fdf = fdf[fdf[category_col].isin(sel_category)]
        if sel_status:   fdf = fdf[fdf[status_col].isin(sel_status)]

        # ---------------- KPIs ----------------
        st.subheader("Key Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric("Records", len(fdf))
        with c2: metric("Companies", fdf[company_col].nunique() if company_col else None)
        with c3: metric("Regions", fdf[region_col].nunique() if region_col else None)
        with c4: metric("Categories", fdf[category_col].nunique() if category_col else None)
        with c5: metric("Products", fdf[product_col].nunique() if product_col else None)

        st.markdown("---")

        # ---------------- Charts ----------------
        left, right = st.columns(2)

        with left:
            if category_col:
                cat_table = fdf[category_col].value_counts().reset_index()
                cat_table.columns = [category_col, "Count"]
                fig = px.bar(cat_table, x=category_col, y="Count", title=f"Top {category_col}")
                st.plotly_chart(fig, use_container_width=True)

        with right:
            if region_col:
                reg_table = fdf[region_col].value_counts().reset_index()
                reg_table.columns = [region_col, "Count"]
                fig2 = px.bar(reg_table, x=region_col, y="Count", title=f"Records by {region_col}")
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # ---------------- Trend ----------------
        st.subheader("Trend Over Time")
        if year_col is not None:
            if np.issubdtype(df[year_col].dtype, np.datetime64):
                fdf["_date"] = pd.to_datetime(fdf[year_col])
                ts = fdf.groupby(fdf["_date"].dt.to_period("M")).size().reset_index(name="Count")
                ts["_date"] = ts["_date"].dt.to_timestamp()
                fig3 = px.line(ts, x="_date", y="Count", markers=True, title="Monthly Records")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                ts = fdf.groupby(year_col).size().reset_index(name="Count")
                fig3 = px.line(ts, x=year_col, y="Count", markers=True, title="Records by Year")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No Year/Date column detected.")

        st.markdown("---")

        # ---------------- Full Data Table ----------------
        st.subheader("Filtered Data")
        st.dataframe(fdf, use_container_width=True)
        st.download_button(
            "⬇️ Download Filtered Data",
            fdf.to_csv(index=False).encode("utf-8"),
            "filtered_data.csv",
            "text/csv"
        )

    else:
        st.warning("Could not read the uploaded file.")
else:
    st.info("👆 Upload an Excel (.xlsx) or CSV file to begin.")

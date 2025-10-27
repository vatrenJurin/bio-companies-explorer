import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Bio Companies Explorer", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem;}
    .metric-card {
        background-color: #f9f7f4;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- APP HEADER --------------------
st.title("🧬 Bio Companies Explorer")
st.caption("Explore biological companies by region, business type, technology sector, and text analytics from descriptions and overviews.")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Select Excel Sheet", xls.sheet_names, index=0)
        df = pd.read_excel(xls, sheet_name=sheet)
    else:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    return df

# -------------------- SIDEBAR NAVIGATION --------------------
page = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Business & Tech",
        "Data Explorer",
        "About"
    ],
    index=0
)

# -------------------- MAIN LOGIC --------------------
if uploaded_file is not None:
    df = load_file(uploaded_file)
    fdf = df.copy()

    # Identify useful columns
    company_col = next((c for c in df.columns if "company" in c.lower()), None)
    region_col = next((c for c in df.columns if "region" in c.lower() or "country" in c.lower()), None)
    category_col = next((c for c in df.columns if "category" in c.lower() or "segment" in c.lower()), None)
    product_col = next((c for c in df.columns if "product" in c.lower() or "technology" in c.lower()), None)
    date_col = next((c for c in df.columns if "date" in c.lower() or "year" in c.lower()), None)

    # Sidebar filters
    st.sidebar.header("Filters")
    def multiselect(col):
        if col and col in df.columns:
            opts = sorted(df[col].dropna().unique())
            return st.sidebar.multiselect(col, opts, default=opts)
        return None

    sel_region = multiselect(region_col)
    sel_category = multiselect(category_col)
    if sel_region: fdf = fdf[fdf[region_col].isin(sel_region)]
    if sel_category: fdf = fdf[fdf[category_col].isin(sel_category)]

    # -------------------- OVERVIEW PAGE --------------------
    if page == "Overview":
        st.markdown("### Key Metrics")
        cols = st.columns(5)
        metrics = [
            ("Records", len(fdf)),
            ("Companies", fdf[company_col].nunique() if company_col else 0),
            ("Regions", fdf[region_col].nunique() if region_col else 0),
            ("Categories", fdf[category_col].nunique() if category_col else 0),
            ("Products", fdf[product_col].nunique() if product_col else 0),
        ]
        for c, (label, val) in zip(cols, metrics):
            c.markdown(f"<div class='metric-card'><h4>{label}</h4><h2>{val}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Distribution by Category, Region, and Product")

        c1, c2, c3 = st.columns(3)
        def donut_chart(df, column, title):
            if column and column in df.columns:
                counts = df[column].value_counts().reset_index()
                counts.columns = [column, "Count"]
                fig = px.pie(counts, names=column, values="Count", hole=0.6,
                             color_discrete_sequence=px.colors.sequential.Tealgrn)
                fig.update_layout(title_text=title, showlegend=False, height=250, margin=dict(t=40,b=0,l=0,r=0))
                return fig
            return None

        with c1:
            fig = donut_chart(fdf, category_col, "By Category")
            if fig: st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = donut_chart(fdf, region_col, "By Region")
            if fig: st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = donut_chart(fdf, product_col, "By Product")
            if fig: st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Data Preview")
        st.dataframe(fdf, use_container_width=True)
        st.download_button("⬇️ Download Filtered Data", fdf.to_csv(index=False).encode("utf-8"), "filtered_data.csv", "text/csv")

    # -------------------- BUSINESS & TECH PAGE --------------------
    elif page == "Business & Tech":
        st.subheader("Business & Technology Insights")

        BUSINESS_COL = "Business"  # change this if needed
        TECH_COL = "Sector (Practice Are & Feed)"  # your exact column
        DESC_COLS = [c for c in df.columns if c.lower() in {"description", "overview"}]

        if BUSINESS_COL not in df.columns:
            st.warning(f"Business column '{BUSINESS_COL}' not found.")
            st.stop()
        if TECH_COL not in df.columns:
            st.warning(f"Tech column '{TECH_COL}' not found.")
            st.stop()

        # --- TEXT SEARCH ---
        st.markdown("### 🔎 Text Search (Description & Overview)")
        q = st.text_input("Enter keywords to search across Description & Overview:")
        tdf = fdf.copy()
        if q and DESC_COLS:
            keys = [k.strip().lower() for k in q.split() if k.strip()]
            def match(row):
                blob = " ".join(str(row[c]) for c in DESC_COLS if c in row and pd.notna(row[c])).lower()
                return all(k in blob for k in keys)
            tdf = tdf[tdf.apply(match, axis=1)]
            st.caption(f"Matches: {len(tdf)} rows")

        # --- Top combinations ---
        st.markdown("### Top Business × Technology Combinations")
        topN = st.slider("Show top N", 5, 50, 15)
        bt = (tdf.groupby([BUSINESS_COL, TECH_COL]).size()
              .reset_index(name="Count")
              .sort_values("Count", ascending=False)
              .head(topN))
        st.plotly_chart(px.bar(bt, x="Count", y=TECH_COL, color=BUSINESS_COL,
                               orientation="h", title="Top Business × Technology"), use_container_width=True)

        # --- Business share per Technology ---
        st.markdown("### Business Share within Each Technology")
        share = (tdf.groupby([TECH_COL, BUSINESS_COL]).size().reset_index(name="Count"))
        share["Share%"] = share.groupby(TECH_COL)["Count"].apply(lambda s: 100*s/s.sum()).values
        st.plotly_chart(px.bar(share, x=TECH_COL, y="Share%", color=BUSINESS_COL, barmode="stack",
                               title="Business Mix within Each Technology"), use_container_width=True)

        # --- Heatmap ---
        st.markdown("### Co-occurrence Heatmap (Business × Technology)")
        ct = pd.crosstab(tdf[BUSINESS_COL], tdf[TECH_COL])
        st.plotly_chart(px.imshow(ct, color_continuous_scale="Tealgrn",
                                  title="Counts by Business × Technology"), use_container_width=True)

        # --- Sunburst ---
        st.markdown("### Hierarchy: Region → Business → Technology → Company")
        path_cols = [c for c in [region_col, BUSINESS_COL, TECH_COL, company_col] if c]
        if len(path_cols) >= 3:
            sb = tdf.groupby(path_cols).size().reset_index(name="Count")
            st.plotly_chart(px.sunburst(sb, path=path_cols, values="Count",
                                        color=BUSINESS_COL), use_container_width=True)

        # --- Sankey Flow ---
        st.markdown("### Flow: Business → Technology")
        flows = tdf.groupby([BUSINESS_COL, TECH_COL]).size().reset_index(name="Count")
        sources = flows[BUSINESS_COL].astype(str).unique().tolist()
        targets = flows[TECH_COL].astype(str).unique().tolist()
        labels = sources + targets
        id_map = {lab:i for i, lab in enumerate(labels)}
        sankey = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=12, thickness=12),
            link=dict(
                source=[id_map[s] for s in flows[BUSINESS_COL].astype(str)],
                target=[id_map[t] for t in flows[TECH_COL].astype(str)],
                value=flows["Count"].tolist()
            )
        )])
        sankey.update_layout(height=400, title="Business → Technology Flow")
        st.plotly_chart(sankey, use_container_width=True)

        # --- Text analytics (most common words) ---
        if DESC_COLS:
            st.markdown("### Most Common Words (Description & Overview)")
            text = " ".join(
                str(x) for c in DESC_COLS if c in tdf.columns for x in tdf[c].dropna().tolist()
            ).lower()
            tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)
            stop = set("a an the and or for of in on to with from by at as is are be it its this that these".split())
            tokens = [t for t in tokens if t not in stop and len(t) > 2]
            counts = Counter(tokens).most_common(25)
            if counts:
                words = pd.DataFrame(counts, columns=["word", "count"])
                st.plotly_chart(px.bar(words, x="word", y="count", title="Top Terms"),
                                use_container_width=True)

    # -------------------- DATA EXPLORER --------------------
    elif page == "Data Explorer":
        st.subheader("Full Dataset")
        st.dataframe(fdf, use_container_width=True)

    # -------------------- ABOUT --------------------
    elif page == "About":
        st.subheader("About this App")
        st.markdown("""
        This Streamlit dashboard helps visualize biological company data.
        - Upload Excel or CSV files.
        - Explore by Business, Technology, Region, and Descriptive text.
        - Uses Plotly for interactive visuals.
        """)
else:
    st.info("👆 Upload your Excel (.xlsx) or CSV file to begin.")

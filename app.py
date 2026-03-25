import io
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="AI-Assisted Data Wrangler & Visualizer", layout="wide")


# ------------------------------
# Caching and loading utilities
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    lower = file_name.lower()
    buffer = io.BytesIO(file_bytes)
    if lower.endswith(".csv"):
        return pd.read_csv(buffer)
    if lower.endswith(".xlsx"):
        return pd.read_excel(buffer)
    if lower.endswith(".json"):
        return pd.read_json(buffer)
    raise ValueError("Unsupported format. Please upload CSV, XLSX, or JSON.")


@st.cache_data(show_spinner=False)
def profile_data(df: pd.DataFrame) -> dict:
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
    })
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str),
        "missing": missing,
        "duplicates": int(df.duplicated().sum()),
    }


# ------------------------------
# Session state helpers
# ------------------------------
def init_state() -> None:
    defaults = {
        "original_df": None,
        "working_df": None,
        "log": [],
        "last_df_snapshot": None,
        "violations": pd.DataFrame(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(operation: str, params: dict, affected_columns: list[str] | None = None) -> None:
    st.session_state.log.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "operation": operation,
            "params": params,
            "affected_columns": affected_columns or [],
        }
    )


def set_working_df(new_df: pd.DataFrame, operation: str, params: dict, affected_columns: list[str] | None = None) -> None:
    st.session_state.last_df_snapshot = st.session_state.working_df.copy()
    st.session_state.working_df = new_df
    add_log(operation, params, affected_columns)


def undo_last_step() -> None:
    if st.session_state.last_df_snapshot is not None and st.session_state.log:
        st.session_state.working_df = st.session_state.last_df_snapshot
        st.session_state.log.pop()
        st.session_state.last_df_snapshot = None


def reset_session() -> None:
    st.session_state.original_df = None
    st.session_state.working_df = None
    st.session_state.log = []
    st.session_state.last_df_snapshot = None
    st.session_state.violations = pd.DataFrame()


# ------------------------------
# Data ops helpers
# ------------------------------
def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def apply_formula(df: pd.DataFrame, new_col: str, formula: str) -> pd.DataFrame:
    local_env = {"np": np}
    for c in df.columns:
        local_env[c] = df[c]
    local_env["mean"] = lambda s: pd.to_numeric(s, errors="coerce").mean()
    df = df.copy()
    df[new_col] = eval(formula, {"__builtins__": {}}, local_env)
    return df


def iqr_outliers(s: pd.Series) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return (s < low) | (s > high)


# ------------------------------
# UI start
# ------------------------------
init_state()
st.title("AI-Assisted Data Wrangler & Visualizer")

page = st.sidebar.radio(
    "Navigate",
    ["Page A — Upload & Overview", "Page B — Cleaning & Preparation Studio", "Page C — Visualization Builder", "Page D — Export & Report"],
)

with st.sidebar:
    if st.button("Reset session", use_container_width=True):
        reset_session()
        st.success("Session reset complete.")


if page == "Page A — Upload & Overview":
    st.header("Upload & Overview")
    uploaded_file = st.file_uploader("Upload CSV, XLSX, or JSON", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file.getvalue(), uploaded_file.name)
            st.session_state.original_df = data.copy()
            st.session_state.working_df = data.copy()
            st.session_state.log = []
            st.success("File loaded successfully.")
        except Exception as exc:
            st.error(f"Failed to load file: {exc}")

    if st.session_state.working_df is not None:
        df = st.session_state.working_df
        p = profile_data(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", p["shape"][0])
        c2.metric("Columns", p["shape"][1])
        c3.metric("Duplicate rows", p["duplicates"])

        st.info(f"Number of columns: {df.shape[1]}")

        st.subheader("Column names and inferred dtypes")
        st.dataframe(p["dtypes"].rename("dtype").to_frame())

        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").transpose())

        st.subheader("Missing values")
        st.dataframe(p["missing"])

        st.subheader("Preview")
        st.dataframe(df.head(20))


if page == "Page B — Cleaning & Preparation Studio":
    st.header("Cleaning & Preparation Studio")
    if st.session_state.working_df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.working_df
    st.caption(f"Working shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 4.1 Missing values
    with st.expander("4.1 Missing Values", expanded=True):
        miss = pd.DataFrame({
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
        })
        st.dataframe(miss)

        action = st.selectbox("Null handling action", ["Drop rows by selected columns", "Drop columns above missing % threshold", "Fill selected columns"])
        selected_cols = st.multiselect("Columns", list(df.columns), default=list(df.columns[:1]))

        if action == "Drop rows by selected columns":
            if st.button("Apply drop rows"):
                before = len(df)
                new_df = df.dropna(subset=selected_cols)
                set_working_df(new_df, "drop_rows_missing", {"subset": selected_cols, "before_rows": before, "after_rows": len(new_df)}, selected_cols)
                st.success(f"Rows before: {before}, after: {len(new_df)}")

        elif action == "Drop columns above missing % threshold":
            th = st.slider("Threshold %", 0, 100, 40)
            if st.button("Apply drop columns"):
                keep_cols = [c for c in df.columns if (df[c].isna().mean() * 100) <= th]
                dropped = [c for c in df.columns if c not in keep_cols]
                new_df = df[keep_cols]
                set_working_df(new_df, "drop_cols_missing_threshold", {"threshold": th, "dropped": dropped}, dropped)
                st.success(f"Dropped {len(dropped)} columns")

        else:
            method = st.selectbox("Fill method", ["constant", "mean", "median", "mode", "most_frequent", "ffill", "bfill"])
            constant_val = st.text_input("Constant value", "0")
            if st.button("Apply fill"):
                new_df = df.copy()
                for col in selected_cols:
                    if method == "constant":
                        new_df[col] = new_df[col].fillna(constant_val)
                    elif method in ["mean", "median"]:
                        series = pd.to_numeric(new_df[col], errors="coerce")
                        val = series.mean() if method == "mean" else series.median()
                        new_df[col] = series.fillna(val)
                    elif method in ["mode", "most_frequent"]:
                        mode_val = new_df[col].mode(dropna=True)
                        if not mode_val.empty:
                            new_df[col] = new_df[col].fillna(mode_val.iloc[0])
                    elif method == "ffill":
                        new_df[col] = new_df[col].ffill()
                    elif method == "bfill":
                        new_df[col] = new_df[col].bfill()
                set_working_df(new_df, "fill_missing", {"method": method, "columns": selected_cols}, selected_cols)
                st.success("Missing value strategy applied.")

    # 4.2 Duplicates
    with st.expander("4.2 Duplicates", expanded=False):
        dup_mode = st.radio("Duplicate detection", ["Full-row duplicates", "Subset duplicates"])
        subset = st.multiselect("Subset keys", list(df.columns)) if dup_mode == "Subset duplicates" else None
        dup_mask = df.duplicated(subset=subset, keep=False) if subset else df.duplicated(keep=False)
        st.write(f"Duplicate rows found: {int(dup_mask.sum())}")
        st.dataframe(df[dup_mask].head(100))
        keep_opt = st.selectbox("Keep option", ["first", "last"])
        if st.button("Remove duplicates"):
            new_df = df.drop_duplicates(subset=subset, keep=keep_opt)
            set_working_df(new_df, "remove_duplicates", {"subset": subset, "keep": keep_opt}, subset or list(df.columns))
            st.success("Duplicates removed.")

    # 4.3 Data types
    with st.expander("4.3 Data Types & Parsing", expanded=False):
        target_col = st.selectbox("Column to convert", list(df.columns))
        target_type = st.selectbox("Target type", ["numeric", "category", "datetime"])
        dt_format = st.text_input("Datetime format (optional)", "")
        if st.button("Convert type"):
            new_df = df.copy()
            if target_type == "numeric":
                new_df[target_col] = clean_numeric_series(new_df[target_col])
            elif target_type == "category":
                new_df[target_col] = new_df[target_col].astype("category")
            else:
                new_df[target_col] = pd.to_datetime(new_df[target_col], format=dt_format or None, errors="coerce")
            set_working_df(new_df, "convert_type", {"column": target_col, "target_type": target_type}, [target_col])
            st.success("Type conversion complete.")

    # 4.4 Categorical
    with st.expander("4.4 Categorical Data Tools", expanded=False):
        cat_col = st.selectbox("Categorical column", list(df.columns), key="cat_col")
        case_opt = st.selectbox("Case standardization", ["none", "lower", "title"])
        map_input = st.text_area("Mapping dictionary as JSON (e.g., {\"ny\":\"New York\"})", "{}")
        set_other = st.checkbox("Set unmatched mapped values to 'Other'")
        rare_th = st.slider("Rare category threshold (%)", 0, 20, 3)
        do_ohe = st.checkbox("Apply one-hot encoding")

        if st.button("Apply categorical tools"):
            new_df = df.copy()
            s = new_df[cat_col].astype(str).str.strip()
            if case_opt == "lower":
                s = s.str.lower()
            elif case_opt == "title":
                s = s.str.title()

            mapping = json.loads(map_input or "{}")
            mapped = s.map(mapping)
            if set_other:
                s = mapped.fillna("Other")
            else:
                s = mapped.fillna(s)

            freq = s.value_counts(normalize=True) * 100
            rare_values = freq[freq < rare_th].index
            s = s.replace({rv: "Other" for rv in rare_values})
            new_df[cat_col] = s

            if do_ohe:
                new_df = pd.get_dummies(new_df, columns=[cat_col], prefix=cat_col, drop_first=False)
            set_working_df(new_df, "categorical_tools", {"column": cat_col, "case": case_opt, "rare_th": rare_th, "one_hot": do_ohe}, [cat_col])
            st.success("Categorical processing complete.")

    # 4.5 Numeric cleaning
    with st.expander("4.5 Numeric Cleaning (Outliers)", expanded=False):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            out_col = st.selectbox("Numeric column", num_cols)
            action = st.selectbox("Outlier action", ["do nothing", "cap/winsorize", "remove outlier rows"])
            lower_q, upper_q = st.slider("Winsorize quantiles", 0.0, 1.0, (0.05, 0.95), 0.01)
            mask = iqr_outliers(pd.to_numeric(df[out_col], errors="coerce").dropna())
            st.write(f"Approx outliers by IQR in {out_col}: {int(mask.sum())}")

            if st.button("Apply outlier action"):
                new_df = df.copy()
                s = pd.to_numeric(new_df[out_col], errors="coerce")
                if action == "cap/winsorize":
                    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
                    new_df[out_col] = s.clip(lo, hi)
                elif action == "remove outlier rows":
                    m = iqr_outliers(s.fillna(s.median()))
                    new_df = new_df.loc[~m].copy()
                set_working_df(new_df, "numeric_outlier_action", {"column": out_col, "action": action}, [out_col])
                st.success("Outlier action applied.")
        else:
            st.info("No numeric columns found.")

    # 4.6 Scaling
    with st.expander("4.6 Normalization / Scaling", expanded=False):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        sc_cols = st.multiselect("Columns to scale", num_cols)
        sc_method = st.selectbox("Scaling method", ["minmax", "zscore"])
        if st.button("Apply scaling"):
            if not sc_cols:
                st.error("Select at least one numeric column.")
            else:
                new_df = df.copy()
                before_stats = new_df[sc_cols].describe().transpose()
                for c in sc_cols:
                    s = pd.to_numeric(new_df[c], errors="coerce")
                    if sc_method == "minmax":
                        rng = s.max() - s.min()
                        new_df[c] = 0 if rng == 0 else (s - s.min()) / rng
                    else:
                        std = s.std()
                        new_df[c] = 0 if std == 0 else (s - s.mean()) / std
                set_working_df(new_df, "scale_columns", {"columns": sc_cols, "method": sc_method}, sc_cols)
                after_stats = new_df[sc_cols].describe().transpose()
                st.write("Before stats")
                st.dataframe(before_stats)
                st.write("After stats")
                st.dataframe(after_stats)

    # 4.7 Column operations
    with st.expander("4.7 Column Operations", expanded=False):
        st.markdown("**Rename / Drop**")
        old = st.selectbox("Rename column", list(df.columns), key="rename_old")
        new = st.text_input("New name", value=old)
        drops = st.multiselect("Drop columns", list(df.columns), key="drop_cols")
        if st.button("Apply rename/drop"):
            new_df = df.copy().rename(columns={old: new})
            if drops:
                new_df = new_df.drop(columns=drops, errors="ignore")
            set_working_df(new_df, "rename_drop_columns", {"rename": {old: new}, "drop": drops}, [old] + drops)
            st.success("Column operations applied.")

        st.markdown("**Create new column with formula**")
        new_col = st.text_input("New column name", "new_feature")
        formula = st.text_input("Formula (example: colA / colB or np.log(colA + 1))", "")
        if st.button("Apply formula"):
            try:
                new_df = apply_formula(df, new_col, formula)
                set_working_df(new_df, "create_formula_column", {"new_col": new_col, "formula": formula}, [new_col])
                st.success("New column created.")
            except Exception as exc:
                st.error(f"Formula failed: {exc}")

        st.markdown("**Binning**")
        bcol = st.selectbox("Numeric column to bin", df.select_dtypes(include=np.number).columns.tolist() or [""], key="bin_col")
        bmethod = st.selectbox("Binning type", ["equal_width", "quantile"])
        bins = st.slider("Number of bins", 2, 10, 4)
        if st.button("Apply binning") and bcol:
            new_df = df.copy()
            if bmethod == "equal_width":
                new_df[f"{bcol}_bin"] = pd.cut(pd.to_numeric(new_df[bcol], errors="coerce"), bins=bins)
            else:
                new_df[f"{bcol}_bin"] = pd.qcut(pd.to_numeric(new_df[bcol], errors="coerce"), q=bins, duplicates="drop")
            set_working_df(new_df, "bin_column", {"column": bcol, "method": bmethod, "bins": bins}, [bcol, f"{bcol}_bin"])
            st.success("Binning complete.")

    # 4.8 Validation rules
    with st.expander("4.8 Data Validation Rules", expanded=False):
        rule_col = st.selectbox("Column", list(df.columns), key="rule_col")
        rule = st.selectbox("Rule type", ["numeric_range", "allowed_categories", "non_null"])
        violations = pd.DataFrame()

        if rule == "numeric_range":
            min_v = st.number_input("Min", value=0.0)
            max_v = st.number_input("Max", value=100.0)
            if st.button("Run validation"):
                s = pd.to_numeric(df[rule_col], errors="coerce")
                violations = df[(s < min_v) | (s > max_v) | s.isna()].copy()
        elif rule == "allowed_categories":
            allowed = st.text_input("Allowed categories (comma separated)", "")
            allowed_set = {x.strip() for x in allowed.split(",") if x.strip()}
            if st.button("Run validation"):
                violations = df[~df[rule_col].astype(str).isin(allowed_set)].copy()
        else:
            if st.button("Run validation"):
                violations = df[df[rule_col].isna()].copy()

        if not violations.empty:
            st.session_state.violations = violations
            st.warning(f"Violations found: {len(violations)}")
            st.dataframe(violations.head(200))
            add_log("validation_rule", {"rule": rule, "column": rule_col, "violations": len(violations)}, [rule_col])

    st.subheader("Transformation log")
    st.dataframe(pd.DataFrame(st.session_state.log))
    c1, c2 = st.columns(2)
    if c1.button("Undo last step"):
        undo_last_step()
        st.info("Last step undone (single-level).")
    if c2.button("Reset all transformations"):
        st.session_state.working_df = st.session_state.original_df.copy() if st.session_state.original_df is not None else None
        st.session_state.log = []
        st.info("Transformations reset.")


if page == "Page C — Visualization Builder":
    st.header("Visualization Builder")
    if st.session_state.working_df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.working_df
    numeric = df.select_dtypes(include=np.number).columns.tolist()

    # filtering
    st.subheader("Filters")
    filter_cat_col = st.selectbox("Category filter column", ["None"] + list(df.columns))
    filtered_df = df.copy()
    if filter_cat_col != "None":
        opts = sorted(filtered_df[filter_cat_col].dropna().astype(str).unique().tolist())
        selected_opts = st.multiselect("Allowed values", opts, default=opts[: min(len(opts), 10)])
        filtered_df = filtered_df[filtered_df[filter_cat_col].astype(str).isin(selected_opts)]

    if numeric:
        filter_num_col = st.selectbox("Numeric range column", ["None"] + numeric)
        if filter_num_col != "None":
            s = pd.to_numeric(filtered_df[filter_num_col], errors="coerce")
            minv, maxv = float(np.nanmin(s)), float(np.nanmax(s))
            lo, hi = st.slider("Numeric range", minv, maxv, (minv, maxv))
            filtered_df = filtered_df[(s >= lo) & (s <= hi)]

    st.subheader("Build chart")
    chart_type = st.selectbox("Chart type", ["histogram", "box", "scatter", "line", "bar", "correlation_heatmap"])
    x = st.selectbox("X column", list(filtered_df.columns))
    y = st.selectbox("Y column", ["None"] + list(filtered_df.columns))
    group = st.selectbox("Color/Group column", ["None"] + list(filtered_df.columns))
    agg = st.selectbox("Aggregation", ["none", "sum", "mean", "count", "median"])
    top_n = st.slider("Top N categories (bar)", 3, 30, 10)

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        if chart_type == "histogram":
            sns.histplot(data=filtered_df, x=x, hue=None if group == "None" else group, kde=True, ax=ax)
        elif chart_type == "box":
            sns.boxplot(data=filtered_df, x=x, y=None if y == "None" else y, ax=ax)
        elif chart_type == "scatter":
            if y == "None":
                st.error("Scatter needs Y column.")
            else:
                sns.scatterplot(data=filtered_df, x=x, y=y, hue=None if group == "None" else group, ax=ax)
        elif chart_type == "line":
            if y == "None":
                st.error("Line needs Y column.")
            else:
                plot_df = filtered_df.copy().sort_values(by=x)
                sns.lineplot(data=plot_df, x=x, y=y, hue=None if group == "None" else group, ax=ax)
        elif chart_type == "bar":
            if y == "None":
                bar_df = filtered_df[x].value_counts().head(top_n).reset_index()
                bar_df.columns = [x, "count"]
                sns.barplot(data=bar_df, x=x, y="count", ax=ax)
            else:
                bar_df = filtered_df[[x, y]].copy()
                if agg != "none":
                    bar_df = getattr(bar_df.groupby(x)[y], agg)().reset_index()
                bar_df = bar_df.sort_values(by=y, ascending=False).head(top_n)
                sns.barplot(data=bar_df, x=x, y=y, ax=ax)
            plt.xticks(rotation=30)
        else:
            corr = filtered_df.select_dtypes(include=np.number).corr(numeric_only=True)
            sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)

        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Chart failed: {exc}")


if page == "Page D — Export & Report":
    st.header("Export & Report")
    if st.session_state.working_df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.working_df
    st.subheader("Transformation report")
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned CSV", csv_bytes, file_name=f"cleaned_{stamp}.csv", mime="text/csv")

    xls_buffer = io.BytesIO()
    with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned")
        log_df.to_excel(writer, index=False, sheet_name="transform_log")
    st.download_button(
        "Download cleaned Excel",
        data=xls_buffer.getvalue(),
        file_name=f"cleaned_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "steps": st.session_state.log,
    }
    st.download_button("Download transformation report (JSON)", json.dumps(report, indent=2), file_name=f"report_{stamp}.json", mime="application/json")
    st.download_button("Download recipe (JSON)", json.dumps(st.session_state.log, indent=2), file_name=f"recipe_{stamp}.json", mime="application/json")

    if not st.session_state.violations.empty:
        st.download_button(
            "Download validation violations CSV",
            st.session_state.violations.to_csv(index=False).encode("utf-8"),
            file_name=f"violations_{stamp}.csv",
            mime="text/csv",
        )

st.sidebar.markdown("---")
st.sidebar.caption("Optional AI assistant is intentionally omitted; app works fully without AI as required.")

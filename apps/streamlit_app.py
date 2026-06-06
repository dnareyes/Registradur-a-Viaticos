from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    APP_DIR.parent
    / "Estad\u00edstica para analitica de datos"
    / "data"
    / "processed"
    / "registraduria_database_clean.xlsx"
)

TARGET = "total_travel_allowance"
DATE_COL = "full_date"

NUM_COLS = ["commission_days", "distance_km", "destination_count", "position_level"]
CAT_COLS = [
    "purpose",
    "employee_position_norm",
    "month",
    "day_of_week",
    "city_origin",
    "city_destination_main",
]
FLAG_COLS = ["is_contractor"]

HGB_CAT_COLS = [
    "purpose",
    "employee_position_norm",
    "month",
    "day_of_week",
    "quarter",
    "city_origin",
    "city_destination_main",
]
HGB_MAX_CARDINALITY = 255

MODEL_LABELS = {
    "baseline": "Regresión lineal",
    "hgb": "HistGradientBoosting",
}

# Mejora visual: plantilla por defecto para plotly
px.defaults.template = "plotly_white"


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns:
        return df
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["month"] = df[DATE_COL].dt.month
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["quarter"] = df[DATE_COL].dt.quarter
    return df


def apply_cleaning(df: pd.DataFrame, remove_zero: bool, fill_level: bool) -> pd.DataFrame:
    df = df.copy()
    if remove_zero and TARGET in df.columns:
        df = df[df[TARGET] != 0]
    if fill_level and "position_level" in df.columns:
        df["position_level"] = df["position_level"].fillna(0)
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        min_date = df[DATE_COL].min()
        max_date = df[DATE_COL].max()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df[DATE_COL] >= pd.to_datetime(start_date)) & (df[DATE_COL] <= pd.to_datetime(end_date))]

    for col, label in [
        ("purpose", "Purpose"),
        ("employee_position_norm", "Employee position"),
        ("city_origin", "City origin"),
        ("city_destination_main", "City destination"),
    ]:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            chosen = st.sidebar.multiselect(label, options, default=options)
            if chosen:
                df = df[df[col].isin(chosen)]

    if "is_contractor" in df.columns:
        contractor_choice = st.sidebar.multiselect(
            "Is contractor",
            options=[0, 1],
            default=[0, 1],
        )
        if contractor_choice:
            df = df[df["is_contractor"].isin(contractor_choice)]

    return df


def summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if TARGET not in df.columns or df.empty:
        return {}
    return {
        "rows": float(len(df)),
        "total": float(df[TARGET].sum()),
        "mean": float(df[TARGET].mean()),
        "median": float(df[TARGET].median()),
        "std": float(df[TARGET].std()),
    }


def format_currency(value: float) -> str:
    try:
        return f"COP {value:,.0f}"
    except Exception:
        return str(value)


def categorical_plot(df: pd.DataFrame, col: str) -> Tuple[go.Figure, go.Figure]:
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    fig_bar = px.bar(counts, x=col, y="count", text="count", title=f"Distribution of {col}")
    fig_bar.update_layout(xaxis_title=col, yaxis_title="Count")

    fig_pie = px.pie(counts, names=col, values="count", title=f"Share of {col}")
    fig_pie.update_traces(textinfo="percent+label")
    return fig_bar, fig_pie


def numeric_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=30, marginal="box", opacity=0.85)
    fig.update_layout(title=f"Histogram of {col}", xaxis_title=col, yaxis_title="Count")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return go.Figure()
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(title="Correlation matrix")
    return fig


def build_pipeline(feature_cols: List[str]) -> Pipeline:
    numeric_features = [c for c in feature_cols if c in NUM_COLS]
    cat_features = [c for c in feature_cols if c in CAT_COLS]
    flag_features = [c for c in feature_cols if c in FLAG_COLS]

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if cat_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                cat_features,
            )
        )
    if flag_features:
        transformers.append(("flag", "passthrough", flag_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])


def split_hgb_categoricals(df: pd.DataFrame, candidates: List[str]) -> Tuple[List[str], List[str]]:
    low_card = []
    high_card = []
    for col in candidates:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= HGB_MAX_CARDINALITY:
            low_card.append(col)
        else:
            high_card.append(col)
    return low_card, high_card


def build_hgb_pipeline(
    numeric_features: List[str],
    flag_features: List[str],
    low_card_cat: List[str],
    high_card_cat: List[str],
) -> Pipeline:
    cat_features = low_card_cat + high_card_cat

    transformers = []
    if numeric_features:
        transformers.append(("num", "passthrough", numeric_features))
    if flag_features:
        transformers.append(("flag", "passthrough", flag_features))
    if cat_features:
        transformers.append(
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    cat_start = len(numeric_features) + len(flag_features)
    cat_idx = list(range(cat_start, cat_start + len(low_card_cat))) if low_card_cat else None

    regressor = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        categorical_features=cat_idx,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)
    y_pred_orig = np.exp(y_pred_log)
    y_test_orig = np.exp(y_test)

    metrics = {
        "r2_log": r2_score(y_test, y_pred_log),
        "mae_log": mean_absolute_error(y_test, y_pred_log),
        "mae": mean_absolute_error(y_test_orig, y_pred_orig),
        "rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        "mape": mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100,
    }

    residuals = y_test.values - y_pred_log

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "residuals": residuals,
    }


def train_models(df: pd.DataFrame) -> Dict[str, object]:
    df = df[df[TARGET] > 0].copy()
    df["log_target"] = np.log(df[TARGET])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_df["log_target"]
    y_test = test_df["log_target"]

    baseline_cols = [c for c in NUM_COLS + CAT_COLS + FLAG_COLS if c in df.columns]
    hgb_cat_low, hgb_cat_high = split_hgb_categoricals(
        df,
        [c for c in HGB_CAT_COLS if c in df.columns],
    )
    hgb_cols = [c for c in NUM_COLS + FLAG_COLS + hgb_cat_low + hgb_cat_high if c in df.columns]

    baseline_pipeline = build_pipeline(baseline_cols)
    hgb_pipeline = build_hgb_pipeline(
        numeric_features=[c for c in NUM_COLS if c in hgb_cols],
        flag_features=[c for c in FLAG_COLS if c in hgb_cols],
        low_card_cat=hgb_cat_low,
        high_card_cat=hgb_cat_high,
    )

    baseline = evaluate_model(
        baseline_pipeline,
        train_df[baseline_cols],
        y_train,
        test_df[baseline_cols],
        y_test,
    )
    baseline["feature_cols"] = baseline_cols

    hgb = evaluate_model(
        hgb_pipeline,
        train_df[hgb_cols],
        y_train,
        test_df[hgb_cols],
        y_test,
    )
    hgb["feature_cols"] = hgb_cols
    hgb["cat_low"] = hgb_cat_low
    hgb["cat_high"] = hgb_cat_high

    return {"baseline": baseline, "hgb": hgb}


def predict_single(pipeline: Pipeline, feature_cols: List[str], payload: Dict[str, object]) -> float:
    row = pd.DataFrame([{col: payload.get(col) for col in feature_cols}])
    return float(pipeline.predict(row)[0])


def run_kruskal(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[float, float]:
    groups = [
        df[df[group_col] == cat][value_col].dropna()
        for cat in df[group_col].dropna().unique()
    ]
    return stats.kruskal(*groups)


def run_mann_whitney(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[float, float]:
    groups = df[group_col].dropna().unique()
    g1, g2 = groups[0], groups[1]
    data1 = df[df[group_col] == g1][value_col].dropna()
    data2 = df[df[group_col] == g2][value_col].dropna()
    return stats.mannwhitneyu(data1, data2, alternative="two-sided")


def main() -> None:
    st.set_page_config(page_title="Registraduría - Análisis", layout="wide")
    st.title("Registraduría: análisis de viáticos")

    st.sidebar.header("Datos")
    data_path_input = st.sidebar.text_input("Ruta de datos", value=str(DEFAULT_DATA_PATH))
    data_path = Path(data_path_input)

    if not data_path.exists():
        st.error("Data file not found. Update the path in the sidebar.")
        st.stop()

    raw_df = load_data(data_path)

    st.sidebar.header("Limpieza")
    remove_zero = st.sidebar.checkbox("Eliminar filas donde el objetivo = 0", value=True)
    fill_level = st.sidebar.checkbox("Rellenar 'position_level' faltante con 0", value=True)

    df = apply_cleaning(raw_df, remove_zero, fill_level)
    df = add_date_features(df)
    df = filter_data(df)

    if df.empty:
        st.warning("No hay filas después de aplicar los filtros.")
        st.stop()

    tabs = st.tabs(["Resumen", "EDA", "Hipótesis", "Modelado", "Conclusiones"])

    with tabs[0]:
        st.subheader("Resumen del conjunto de datos")
        metrics = summary_metrics(df)
        cols = st.columns(5)
        if metrics:
            cols[0].metric("Filas", f"{int(metrics['rows']):,}")
            cols[1].metric("Total (COP)", format_currency(metrics["total"]))
            cols[2].metric("Media (COP)", format_currency(metrics["mean"]))
            cols[3].metric("Mediana (COP)", format_currency(metrics["median"]))
            cols[4].metric("Desv. estándar", format_currency(metrics["std"]))

        st.write("Preview")
        st.dataframe(df.head(20), use_container_width=True)

    with tabs[1]:
        st.subheader("Distribuciones categóricas")
        cat_choices = [c for c in ["purpose", "employee_position_norm", "city_origin", "city_destination_main"] if c in df.columns]
        if cat_choices:
            cat_col = st.selectbox("Variable categórica", cat_choices)
            fig_bar, fig_pie = categorical_plot(df, cat_col)
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_bar, use_container_width=True)
            c2.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No hay columnas categóricas disponibles para graficar.")

        st.subheader("Distribuciones numéricas")
        num_choices = [c for c in [TARGET, "commission_days", "distance_km", "destination_count"] if c in df.columns]
        if num_choices:
            num_col = st.selectbox("Variable numérica", num_choices)
            st.plotly_chart(numeric_histogram(df, num_col), use_container_width=True)
        else:
            st.info("No hay columnas numéricas disponibles para graficar.")

        st.subheader("Correlación")
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    with tabs[2]:
        st.subheader("Pruebas no paramétricas")
        value_choices = [c for c in [TARGET, "commission_days", "distance_km", "destination_count"] if c in df.columns]
        group_choices = [c for c in ["purpose", "employee_position_norm", "position_level", "is_contractor"] if c in df.columns]

        if value_choices and group_choices:
            value_col = st.selectbox("Value column", value_choices)
            group_col = st.selectbox("Group column", group_choices)

            if st.button("Ejecutar Kruskal-Wallis"):
                h_stat, p_val = run_kruskal(df, group_col, value_col)
                st.write(f"H = {h_stat:.4f}")
                st.write(f"p-valor = {p_val:.6f}")

            if "is_contractor" in df.columns and df["is_contractor"].nunique() == 2:
                if st.button("Ejecutar Mann-Whitney"):
                    u_stat, p_val = run_mann_whitney(df, "is_contractor", value_col)
                    st.write(f"U = {u_stat:.4f}")
                    st.write(f"p-valor = {p_val:.6f}")
        else:
            st.info("Missing columns for hypothesis testing.")

        st.subheader("Pruebas de correlación")
        numeric_candidates = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]
        if numeric_candidates:
            x_col = st.selectbox("X", numeric_candidates)
            y_col = st.selectbox("Y", [c for c in numeric_candidates if c != x_col])
            if st.button("Ejecutar Pearson/Spearman"):
                pearson_r, pearson_p = stats.pearsonr(df[x_col], df[y_col])
                spearman_r, spearman_p = stats.spearmanr(df[x_col], df[y_col])
                st.write(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
                st.write(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.6f}")
        else:
            st.info("No hay suficientes columnas numéricas para pruebas de correlación.")

    with tabs[3]:
        st.subheader("Modelado")
        st.caption("Entrena dos modelos (regresión lineal y HistGradientBoosting) y compara su desempeño.")

        if st.button("Entrenar modelos"):
            st.session_state["model_result"] = train_models(df)

        model_result = st.session_state.get("model_result")
        if model_result:
            st.write("Métricas (split de test)")
            rows = []
            for key, label in MODEL_LABELS.items():
                result = model_result.get(key)
                if not result:
                    continue
                metrics = result["metrics"]
                rows.append(
                    {
                        "Modelo": label,
                        "R2 (log)": round(metrics["r2_log"], 4),
                        "MAE (log)": round(metrics["mae_log"], 4),
                        "MAE (COP)": format_currency(metrics["mae"]),
                        "RMSE (COP)": format_currency(metrics["rmse"]),
                        "MAPE (%)": round(metrics["mape"], 2),
                    }
                )

            if rows:
                metrics_df = pd.DataFrame(rows).set_index("Modelo")
                st.dataframe(metrics_df, use_container_width=True)

            hgb_result = model_result.get("hgb")
            if hgb_result:
                with st.expander("HGB categorical handling"):
                    st.write("Low-cardinality (treated as categorical):")
                    st.write(hgb_result.get("cat_low", []))
                    st.write("High-cardinality (treated as numeric-encoded):")
                    st.write(hgb_result.get("cat_high", []))

            st.subheader("Predicción individual")
            model_choice = st.selectbox("Model", [MODEL_LABELS[k] for k in model_result.keys()])
            model_key = next(k for k, v in MODEL_LABELS.items() if v == model_choice)
            chosen = model_result[model_key]

            feature_cols = chosen["feature_cols"]
            payload = {}
            for col in feature_cols:
                if col in ["purpose", "employee_position_norm", "city_origin", "city_destination_main"]:
                    options = df[col].dropna().unique().tolist()
                    payload[col] = st.selectbox(col, options)
                elif col in ["month", "day_of_week", "quarter"]:
                    min_val = int(df[col].min())
                    max_val = int(df[col].max())
                    payload[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=min_val)
                elif col == "is_contractor":
                    payload[col] = st.selectbox(col, [0, 1])
                else:
                    payload[col] = st.number_input(col, value=float(df[col].median()))

            if st.button("Predecir"):
                pipeline = chosen["pipeline"]
                residuals = chosen["residuals"]
                log_pred = predict_single(pipeline, feature_cols, payload)
                sigma_log = float(np.std(residuals))

                pred = np.exp(log_pred)
                ci_low = np.exp(log_pred - 1.96 * sigma_log)
                ci_high = np.exp(log_pred + 1.96 * sigma_log)

                st.write(f"Predicción (COP): {format_currency(pred)}")
                st.write(f"IC 95%: [{format_currency(ci_low)}, {format_currency(ci_high)}]")
        else:
            st.info("Entrena los modelos para habilitar métricas y predicciones.")

    with tabs[4]:
        st.subheader("Conclusiones")
        model_result = st.session_state.get("model_result")
        if model_result:
            baseline = model_result.get("baseline")
            hgb = model_result.get("hgb")
            if baseline and hgb:
                b_metrics = baseline["metrics"]
                h_metrics = hgb["metrics"]
                best_key = "baseline" if b_metrics["r2_log"] >= h_metrics["r2_log"] else "hgb"
                best_label = MODEL_LABELS[best_key]
                best_r2 = model_result[best_key]["metrics"]["r2_log"]
                st.markdown(
                    "\n".join(
                        [
                            f"- Mejor R2 (log): {best_label} ({best_r2:.3f}).",
                            f"- MAE (COP): Regresión lineal {format_currency(b_metrics['mae'])} | HGB {format_currency(h_metrics['mae'])}.",
                            "- Los resultados dependen de los filtros y opciones de limpieza.",
                            "- Usar el modelo con mejor desempeño para análisis y planificación.",
                        ]
                    )
                )
            else:
                st.info("Entrena los modelos para generar conclusiones automáticas.")
        else:
            st.info("Entrena los modelos para generar conclusiones automáticas.")

        st.text_area("Notas narrativas", placeholder="Agrega aquí tus conclusiones...")


if __name__ == "__main__":
    main()

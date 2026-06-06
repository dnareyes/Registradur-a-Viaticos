"""
Registraduría – Análisis de Viáticos
=====================================
App Streamlit mejorada: mejor UX, carga por upload, exportación de resultados,
gráficas adicionales (serie de tiempo, scatter, residuos), importancia de
características y manejo de errores robusto.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR.parent / "data" / "registraduria_database_clean.xlsx"

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
    "baseline": "Regresión Lineal",
    "hgb": "HistGradientBoosting",
}

PALETTE = px.colors.qualitative.Plotly
px.defaults.template = "plotly_white"


# ─────────────────────────────────────────────────────────────
# Carga y preprocesamiento
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: Optional[Path] = None, file_bytes: Optional[bytes] = None, file_name: str = "") -> pd.DataFrame:
    """Carga datos desde ruta de disco o bytes de archivo subido."""
    if file_bytes is not None:
        ext = Path(file_name).suffix.lower()
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(io.BytesIO(file_bytes))
        return pd.read_csv(io.BytesIO(file_bytes))
    if path is not None and path.exists():
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path)
    raise FileNotFoundError("No se encontró el archivo de datos.")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns:
        return df
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["month"] = df[DATE_COL].dt.month
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["quarter"] = df[DATE_COL].dt.quarter
    df["year"] = df[DATE_COL].dt.year
    df["year_month"] = df[DATE_COL].dt.to_period("M").astype(str)
    return df


def apply_cleaning(df: pd.DataFrame, remove_zero: bool, fill_level: bool) -> pd.DataFrame:
    df = df.copy()
    if remove_zero and TARGET in df.columns:
        df = df[df[TARGET] != 0]
    if fill_level and "position_level" in df.columns:
        df["position_level"] = df["position_level"].fillna(0)
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros interactivos desde la barra lateral."""
    if df.empty:
        return df
    df = df.copy()

    # Filtro por fecha
    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        min_date = df[DATE_COL].min().date()
        max_date = df[DATE_COL].max().date()
        date_range = st.sidebar.date_input(
            "📅 Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            df = df[
                (df[DATE_COL] >= pd.Timestamp(start))
                & (df[DATE_COL] <= pd.Timestamp(end))
            ]

    # Filtros categóricos
    col_labels = {
        "purpose": "🎯 Propósito",
        "employee_position_norm": "👤 Cargo",
        "city_origin": "🏙️ Ciudad origen",
        "city_destination_main": "📍 Ciudad destino",
    }
    for col, label in col_labels.items():
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            chosen = st.sidebar.multiselect(label, options, default=options)
            if chosen:
                df = df[df[col].isin(chosen)]

    # Filtro contratista
    if "is_contractor" in df.columns:
        contractor_choice = st.sidebar.multiselect(
            "🔖 Es contratista",
            options=[0, 1],
            format_func=lambda x: "Sí" if x else "No",
            default=[0, 1],
        )
        if contractor_choice:
            df = df[df["is_contractor"].isin(contractor_choice)]

    return df


# ─────────────────────────────────────────────────────────────
# Métricas y formateo
# ─────────────────────────────────────────────────────────────
def summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if TARGET not in df.columns or df.empty:
        return {}
    ser = df[TARGET]
    return {
        "rows": float(len(df)),
        "total": float(ser.sum()),
        "mean": float(ser.mean()),
        "median": float(ser.median()),
        "std": float(ser.std()),
        "p25": float(ser.quantile(0.25)),
        "p75": float(ser.quantile(0.75)),
        "max": float(ser.max()),
    }


def format_currency(value: float) -> str:
    try:
        return f"COP {value:,.0f}"
    except Exception:
        return str(value)


# ─────────────────────────────────────────────────────────────
# Gráficas EDA
# ─────────────────────────────────────────────────────────────
def categorical_plot(df: pd.DataFrame, col: str) -> Tuple[go.Figure, go.Figure]:
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    counts = counts.sort_values("count", ascending=False).head(20)

    fig_bar = px.bar(
        counts, x=col, y="count", text="count",
        title=f"Top categorías — {col}",
        color="count", color_continuous_scale="Blues",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(xaxis_title=col, yaxis_title="Registros", coloraxis_showscale=False)

    fig_pie = px.pie(counts, names=col, values="count", title=f"Proporción — {col}", hole=0.35)
    fig_pie.update_traces(textinfo="percent+label")
    return fig_bar, fig_pie


def numeric_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df, x=col, nbins=40, marginal="box", opacity=0.80,
        color_discrete_sequence=[PALETTE[0]],
        title=f"Distribución de {col}",
    )
    fig.update_layout(xaxis_title=col, yaxis_title="Registros")
    return fig


def target_by_category(df: pd.DataFrame, col: str) -> go.Figure:
    """Boxplot del target agrupado por una variable categórica."""
    if col not in df.columns or TARGET not in df.columns:
        return go.Figure()
    top = df[col].value_counts().head(15).index.tolist()
    sub = df[df[col].isin(top)].copy()
    fig = px.box(
        sub, x=col, y=TARGET, color=col,
        title=f"{TARGET} por {col} (top 15 categorías)",
        points=False,
    )
    fig.update_layout(showlegend=False, yaxis_title="Viático total (COP)")
    return fig


def time_series_plot(df: pd.DataFrame) -> go.Figure:
    """Serie de tiempo mensual del viático total."""
    if DATE_COL not in df.columns or TARGET not in df.columns:
        return go.Figure()
    ts = (
        df.groupby("year_month")[TARGET]
        .agg(total="sum", media="mean", registros="count")
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["year_month"], y=ts["total"], mode="lines+markers",
                             name="Total COP", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Bar(x=ts["year_month"], y=ts["registros"], name="Registros",
                         yaxis="y2", opacity=0.35, marker_color=PALETTE[1]))
    fig.update_layout(
        title="Evolución mensual de viáticos",
        xaxis_title="Mes",
        yaxis=dict(title="Total COP", tickformat=",.0f"),
        yaxis2=dict(title="Registros", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def scatter_target(df: pd.DataFrame, x_col: str, color_col: Optional[str] = None) -> go.Figure:
    kwargs = dict(x=x_col, y=TARGET, opacity=0.5, trendline="ols",
                  title=f"{TARGET} vs {x_col}")
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    return px.scatter(df.sample(min(3000, len(df)), random_state=42), **kwargs)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return go.Figure()
    corr = numeric_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        title="Matriz de correlación",
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────
def build_pipeline(feature_cols: List[str]) -> Pipeline:
    numeric_features = [c for c in feature_cols if c in NUM_COLS]
    cat_features = [c for c in feature_cols if c in CAT_COLS]
    flag_features = [c for c in feature_cols if c in FLAG_COLS]

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if cat_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
            cat_features,
        ))
    if flag_features:
        transformers.append(("flag", "passthrough", flag_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])


def split_hgb_categoricals(df: pd.DataFrame, candidates: List[str]) -> Tuple[List[str], List[str]]:
    low_card, high_card = [], []
    for col in candidates:
        if col not in df.columns:
            continue
        (low_card if df[col].nunique(dropna=True) <= HGB_MAX_CARDINALITY else high_card).append(col)
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
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_features,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    cat_start = len(numeric_features) + len(flag_features)
    cat_idx = list(range(cat_start, cat_start + len(low_card_cat))) if low_card_cat else None

    regressor = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1,
        categorical_features=cat_idx, random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> Dict[str, object]:
    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)
    y_pred_orig = np.exp(y_pred_log)
    y_test_orig = np.exp(y_test)

    residuals = y_test.values - y_pred_log

    return {
        "pipeline": pipeline,
        "metrics": {
            "r2_log": r2_score(y_test, y_pred_log),
            "mae_log": mean_absolute_error(y_test, y_pred_log),
            "mae": mean_absolute_error(y_test_orig, y_pred_orig),
            "rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            "mape": mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100,
        },
        "residuals": residuals,
        "y_test_log": y_test.values,
        "y_pred_log": y_pred_log,
    }


def train_models(df: pd.DataFrame) -> Dict[str, object]:
    df = df[df[TARGET] > 0].copy()
    df["log_target"] = np.log(df[TARGET])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train, y_test = train_df["log_target"], test_df["log_target"]

    baseline_cols = [c for c in NUM_COLS + CAT_COLS + FLAG_COLS if c in df.columns]
    hgb_cat_low, hgb_cat_high = split_hgb_categoricals(
        df, [c for c in HGB_CAT_COLS if c in df.columns]
    )
    hgb_cols = [c for c in NUM_COLS + FLAG_COLS + hgb_cat_low + hgb_cat_high if c in df.columns]

    baseline = evaluate_model(build_pipeline(baseline_cols),
                              train_df[baseline_cols], y_train, test_df[baseline_cols], y_test)
    baseline["feature_cols"] = baseline_cols

    hgb = evaluate_model(
        build_hgb_pipeline(
            [c for c in NUM_COLS if c in hgb_cols],
            [c for c in FLAG_COLS if c in hgb_cols],
            hgb_cat_low, hgb_cat_high,
        ),
        train_df[hgb_cols], y_train, test_df[hgb_cols], y_test,
    )
    hgb["feature_cols"] = hgb_cols
    hgb["cat_low"] = hgb_cat_low
    hgb["cat_high"] = hgb_cat_high

    return {"baseline": baseline, "hgb": hgb}


def predict_single(pipeline: Pipeline, feature_cols: List[str], payload: Dict) -> float:
    row = pd.DataFrame([{col: payload.get(col) for col in feature_cols}])
    return float(pipeline.predict(row)[0])


# ─────────────────────────────────────────────────────────────
# Gráficas de diagnóstico de modelos
# ─────────────────────────────────────────────────────────────
def residuals_plot(result: Dict) -> go.Figure:
    residuals = result["residuals"]
    y_pred = result["y_pred_log"]
    fig = px.scatter(
        x=y_pred, y=residuals, opacity=0.4,
        labels={"x": "Predicción (log)", "y": "Residuo (log)"},
        title="Residuos vs Predicción",
        color_discrete_sequence=[PALETTE[2]],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def actual_vs_predicted_plot(result: Dict) -> go.Figure:
    y_test = np.exp(result["y_test_log"])
    y_pred = np.exp(result["y_pred_log"])
    lim = max(y_test.max(), y_pred.max())
    fig = px.scatter(
        x=y_test, y=y_pred, opacity=0.45,
        labels={"x": "Real (COP)", "y": "Predicho (COP)"},
        title="Real vs Predicho",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(color="red", dash="dash"))
    return fig


def feature_importance_plot(result: Dict, model_key: str) -> Optional[go.Figure]:
    pipeline = result["pipeline"]
    feature_cols = result["feature_cols"]
    regressor = pipeline.named_steps["regressor"]

    if model_key == "baseline":
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            coefs = regressor.coef_
            fi = pd.DataFrame({"feature": names, "importance": np.abs(coefs)})
        except Exception:
            return None
    elif model_key == "hgb":
        try:
            fi = pd.DataFrame({
                "feature": feature_cols,
                "importance": regressor.feature_importances_,
            })
        except Exception:
            return None
    else:
        return None

    fi = fi.sort_values("importance", ascending=True).tail(20)
    fig = px.bar(fi, x="importance", y="feature", orientation="h",
                 title="Importancia de características (top 20)",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────────────────────
# Pruebas estadísticas
# ─────────────────────────────────────────────────────────────
def run_kruskal(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[float, float]:
    groups = [df[df[group_col] == cat][value_col].dropna()
              for cat in df[group_col].dropna().unique()]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos 2 grupos.")
    return stats.kruskal(*groups)


def run_mann_whitney(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[float, float]:
    groups = df[group_col].dropna().unique()
    g1, g2 = groups[0], groups[1]
    data1 = df[df[group_col] == g1][value_col].dropna()
    data2 = df[df[group_col] == g2][value_col].dropna()
    return stats.mannwhitneyu(data1, data2, alternative="two-sided")


# ─────────────────────────────────────────────────────────────
# Exportación
# ─────────────────────────────────────────────────────────────
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# App principal
# ─────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Registraduría – Viáticos",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Encabezado ──────────────────────────────────────────
    st.title("📊 Registraduría: Análisis de Viáticos")
    st.caption("Exploración, modelado y predicción de viáticos de comisión.")

    # ── Sidebar: carga de datos ──────────────────────────────
    st.sidebar.header("📁 Fuente de datos")
    source = st.sidebar.radio("Origen", ["Archivo local (ruta)", "Subir archivo"])

    raw_df: Optional[pd.DataFrame] = None
    try:
        if source == "Subir archivo":
            uploaded = st.sidebar.file_uploader("Sube tu Excel o CSV", type=["xlsx", "xls", "csv"])
            if uploaded:
                raw_df = load_data(file_bytes=uploaded.read(), file_name=uploaded.name)
        else:
            data_path_input = st.sidebar.text_input("Ruta de datos", value=str(DEFAULT_DATA_PATH))
            data_path = Path(data_path_input)
            raw_df = load_data(path=data_path)
    except FileNotFoundError:
        st.error("⚠️ Archivo no encontrado. Actualiza la ruta o sube un archivo.")
        st.stop()
    except Exception as exc:
        st.error(f"⚠️ Error al leer el archivo: {exc}")
        st.stop()

    if raw_df is None:
        st.info("👆 Selecciona o sube un archivo de datos para comenzar.")
        st.stop()

    # ── Sidebar: limpieza ───────────────────────────────────
    st.sidebar.header("🧹 Limpieza")
    remove_zero = st.sidebar.checkbox("Eliminar filas con viático = 0", value=True)
    fill_level = st.sidebar.checkbox("Rellenar 'position_level' nulo con 0", value=True)

    df = apply_cleaning(raw_df, remove_zero, fill_level)
    df = add_date_features(df)

    # ── Sidebar: filtros ────────────────────────────────────
    st.sidebar.header("🔍 Filtros")
    df = filter_data(df)

    if df.empty:
        st.warning("⚠️ No hay registros con los filtros actuales. Ajusta los filtros.")
        st.stop()

    # ── Exportar datos filtrados ─────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.download_button(
            label="⬇️ Exportar datos filtrados (.xlsx)",
            data=df_to_excel_bytes(df),
            file_name="viaticos_filtrados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ═══════════════════════════════════════════════════════
    # Pestañas
    # ═══════════════════════════════════════════════════════
    tabs = st.tabs(["📋 Resumen", "📈 EDA", "🔬 Hipótesis", "🤖 Modelado", "📝 Conclusiones"])

    # ── Tab 0: Resumen ───────────────────────────────────────
    with tabs[0]:
        st.subheader("Resumen del conjunto de datos")
        metrics = summary_metrics(df)

        if metrics:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Registros", f"{int(metrics['rows']):,}")
            c2.metric("Total (COP)", format_currency(metrics["total"]))
            c3.metric("Media (COP)", format_currency(metrics["mean"]))
            c4.metric("Mediana (COP)", format_currency(metrics["median"]))
            c5.metric("Desv. estándar", format_currency(metrics["std"]))

            c6, c7, c8 = st.columns(3)
            c6.metric("P25 (COP)", format_currency(metrics["p25"]))
            c7.metric("P75 (COP)", format_currency(metrics["p75"]))
            c8.metric("Máximo (COP)", format_currency(metrics["max"]))

        # Serie de tiempo si hay fecha
        if DATE_COL in df.columns and df[DATE_COL].notna().any():
            st.plotly_chart(time_series_plot(df), use_container_width=True)

        with st.expander("📄 Vista previa de datos (primeras 50 filas)"):
            st.dataframe(df.head(50), use_container_width=True)

        with st.expander("📊 Estadísticas descriptivas completas"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── Tab 1: EDA ───────────────────────────────────────────
    with tabs[1]:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Variables categóricas")
            cat_choices = [c for c in ["purpose", "employee_position_norm", "city_origin", "city_destination_main"]
                           if c in df.columns]
            if cat_choices:
                cat_col = st.selectbox("Variable categórica", cat_choices, key="eda_cat")
            else:
                st.info("No hay columnas categóricas disponibles.")
                cat_col = None

        if cat_col:
            fig_bar, fig_pie = categorical_plot(df, cat_col)
            ec1, ec2 = st.columns(2)
            ec1.plotly_chart(fig_bar, use_container_width=True)
            ec2.plotly_chart(fig_pie, use_container_width=True)

            if TARGET in df.columns:
                st.plotly_chart(target_by_category(df, cat_col), use_container_width=True)

        st.markdown("---")
        st.subheader("Variables numéricas")
        num_choices = [c for c in [TARGET, "commission_days", "distance_km", "destination_count"]
                       if c in df.columns]
        if num_choices:
            nc1, nc2 = st.columns([1, 2])
            with nc1:
                num_col = st.selectbox("Variable numérica", num_choices, key="eda_num")
                color_opt = st.selectbox(
                    "Color por (scatter)", ["Ninguno"] + [c for c in cat_choices or [] if c != num_col],
                    key="scatter_color",
                )
            st.plotly_chart(numeric_histogram(df, num_col), use_container_width=True)
            if num_col != TARGET and TARGET in df.columns:
                color_col = None if color_opt == "Ninguno" else color_opt
                st.plotly_chart(scatter_target(df, num_col, color_col), use_container_width=True)
        else:
            st.info("No hay columnas numéricas disponibles.")

        st.markdown("---")
        st.subheader("Matriz de correlación")
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    # ── Tab 2: Hipótesis ─────────────────────────────────────
    with tabs[2]:
        st.subheader("🔬 Pruebas estadísticas no paramétricas")

        value_choices = [c for c in [TARGET, "commission_days", "distance_km", "destination_count"]
                         if c in df.columns]
        group_choices = [c for c in ["purpose", "employee_position_norm", "position_level", "is_contractor"]
                         if c in df.columns]

        if not (value_choices and group_choices):
            st.info("No hay columnas suficientes para pruebas de hipótesis.")
        else:
            hc1, hc2 = st.columns(2)
            with hc1:
                value_col = st.selectbox("Variable de valor", value_choices, key="h_val")
            with hc2:
                group_col = st.selectbox("Variable de agrupación", group_choices, key="h_grp")

            col_kw, col_mw = st.columns(2)
            with col_kw:
                if st.button("▶ Ejecutar Kruskal-Wallis"):
                    try:
                        h_stat, p_val = run_kruskal(df, group_col, value_col)
                        sig = "✅ Diferencia significativa (p < 0.05)" if p_val < 0.05 else "❌ Sin diferencia significativa"
                        st.metric("H-estadístico", f"{h_stat:.4f}")
                        st.metric("p-valor", f"{p_val:.6f}")
                        st.info(sig)
                    except Exception as e:
                        st.error(f"Error: {e}")

            with col_mw:
                if "is_contractor" in df.columns and df["is_contractor"].nunique() == 2:
                    if st.button("▶ Ejecutar Mann-Whitney"):
                        try:
                            u_stat, p_val = run_mann_whitney(df, "is_contractor", value_col)
                            sig = "✅ Diferencia significativa (p < 0.05)" if p_val < 0.05 else "❌ Sin diferencia significativa"
                            st.metric("U-estadístico", f"{u_stat:.4f}")
                            st.metric("p-valor", f"{p_val:.6f}")
                            st.info(sig)
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.markdown("---")
        st.subheader("Correlación de Pearson y Spearman")
        numeric_candidates = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]

        if len(numeric_candidates) >= 2:
            pc1, pc2 = st.columns(2)
            with pc1:
                x_col = st.selectbox("Variable X", numeric_candidates, key="pearson_x")
            with pc2:
                y_col = st.selectbox("Variable Y", [c for c in numeric_candidates if c != x_col], key="pearson_y")

            if st.button("▶ Ejecutar Pearson / Spearman"):
                try:
                    valid = df[[x_col, y_col]].dropna()
                    pearson_r, pearson_p = stats.pearsonr(valid[x_col], valid[y_col])
                    spearman_r, spearman_p = stats.spearmanr(valid[x_col], valid[y_col])

                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.metric("Pearson r", f"{pearson_r:.4f}")
                        st.metric("p-valor (Pearson)", f"{pearson_p:.6f}")
                    with rc2:
                        st.metric("Spearman ρ", f"{spearman_r:.4f}")
                        st.metric("p-valor (Spearman)", f"{spearman_p:.6f}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No hay suficientes columnas numéricas para pruebas de correlación.")

    # ── Tab 3: Modelado ──────────────────────────────────────
    with tabs[3]:
        st.subheader("🤖 Modelado predictivo")
        st.caption("Entrena Regresión Lineal y HistGradientBoosting sobre log(viático).")

        if st.button("🚀 Entrenar modelos", type="primary"):
            with st.spinner("Entrenando... esto puede tomar unos segundos."):
                try:
                    st.session_state["model_result"] = train_models(df)
                    st.success("¡Modelos entrenados exitosamente!")
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

        model_result = st.session_state.get("model_result")

        if model_result:
            # Tabla de métricas
            st.subheader("Comparativa de métricas")
            rows = []
            for key, label in MODEL_LABELS.items():
                result = model_result.get(key)
                if not result:
                    continue
                m = result["metrics"]
                rows.append({
                    "Modelo": label,
                    "R² (log)": round(m["r2_log"], 4),
                    "MAE (log)": round(m["mae_log"], 4),
                    "MAE (COP)": format_currency(m["mae"]),
                    "RMSE (COP)": format_currency(m["rmse"]),
                    "MAPE (%)": round(m["mape"], 2),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Modelo"), use_container_width=True)

            # Gráficas de diagnóstico
            st.subheader("Diagnóstico visual")
            model_key_diag = st.selectbox(
                "Modelo a diagnosticar",
                list(MODEL_LABELS.keys()),
                format_func=lambda k: MODEL_LABELS[k],
                key="diag_model",
            )
            diag_result = model_result[model_key_diag]
            dc1, dc2 = st.columns(2)
            dc1.plotly_chart(actual_vs_predicted_plot(diag_result), use_container_width=True)
            dc2.plotly_chart(residuals_plot(diag_result), use_container_width=True)

            fi_fig = feature_importance_plot(diag_result, model_key_diag)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)

            # Detalle HGB
            hgb_result = model_result.get("hgb")
            if hgb_result:
                with st.expander("ℹ️ Detalle de variables HGB"):
                    st.write("**Baja cardinalidad (tratadas como categóricas):**", hgb_result.get("cat_low", []))
                    st.write("**Alta cardinalidad (codificadas numéricamente):**", hgb_result.get("cat_high", []))

            # Predicción individual
            st.markdown("---")
            st.subheader("🔮 Predicción individual")
            model_choice = st.selectbox(
                "Modelo para predecir",
                list(MODEL_LABELS.keys()),
                format_func=lambda k: MODEL_LABELS[k],
                key="pred_model",
            )
            chosen = model_result[model_choice]
            feature_cols = chosen["feature_cols"]

            payload = {}
            input_cols = st.columns(min(4, len(feature_cols)))
            for i, col in enumerate(feature_cols):
                with input_cols[i % len(input_cols)]:
                    if col in ["purpose", "employee_position_norm", "city_origin", "city_destination_main"]:
                        options = df[col].dropna().unique().tolist()
                        payload[col] = st.selectbox(col, options, key=f"pred_{col}")
                    elif col in ["month", "day_of_week", "quarter", "year"]:
                        min_v, max_v = int(df[col].min()), int(df[col].max())
                        payload[col] = st.number_input(col, min_value=min_v, max_value=max_v,
                                                       value=min_v, key=f"pred_{col}")
                    elif col == "is_contractor":
                        raw = st.selectbox(col, ["No", "Sí"], key=f"pred_{col}")
                        payload[col] = 1 if raw == "Sí" else 0
                    else:
                        payload[col] = st.number_input(col, value=float(df[col].median()),
                                                       key=f"pred_{col}")

            if st.button("⚡ Predecir", type="primary"):
                try:
                    pipeline = chosen["pipeline"]
                    residuals = chosen["residuals"]
                    log_pred = predict_single(pipeline, feature_cols, payload)
                    sigma_log = float(np.std(residuals))

                    pred = np.exp(log_pred)
                    ci_low = np.exp(log_pred - 1.96 * sigma_log)
                    ci_high = np.exp(log_pred + 1.96 * sigma_log)

                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("Predicción", format_currency(pred))
                    pc2.metric("IC 95% inferior", format_currency(ci_low))
                    pc3.metric("IC 95% superior", format_currency(ci_high))
                except Exception as e:
                    st.error(f"Error al predecir: {e}")
        else:
            st.info("👆 Entrena los modelos para ver métricas y hacer predicciones.")

    # ── Tab 4: Conclusiones ───────────────────────────────────
    with tabs[4]:
        st.subheader("📝 Conclusiones")
        model_result = st.session_state.get("model_result")

        if model_result:
            baseline = model_result.get("baseline")
            hgb = model_result.get("hgb")
            if baseline and hgb:
                b_m = baseline["metrics"]
                h_m = hgb["metrics"]
                best_key = "baseline" if b_m["r2_log"] >= h_m["r2_log"] else "hgb"
                best_label = MODEL_LABELS[best_key]
                best_r2 = model_result[best_key]["metrics"]["r2_log"]

                st.success(f"**Mejor modelo:** {best_label} con R² = {best_r2:.3f}")

                conclusion_items = [
                    f"**R² (log):** Regresión lineal {b_m['r2_log']:.3f} | HGB {h_m['r2_log']:.3f}",
                    f"**MAE (COP):** Regresión lineal {format_currency(b_m['mae'])} | HGB {format_currency(h_m['mae'])}",
                    f"**MAPE (%):** Regresión lineal {b_m['mape']:.2f}% | HGB {h_m['mape']:.2f}%",
                    "Los resultados dependen de los filtros y opciones de limpieza seleccionadas.",
                    f"Se recomienda usar **{best_label}** para análisis y planificación.",
                ]
                for item in conclusion_items:
                    st.markdown(f"- {item}")

                # Comparativa en gráfica de barras
                compare_df = pd.DataFrame({
                    "Modelo": [MODEL_LABELS["baseline"], MODEL_LABELS["hgb"]],
                    "R² (log)": [b_m["r2_log"], h_m["r2_log"]],
                    "MAPE (%)": [b_m["mape"], h_m["mape"]],
                })
                cc1, cc2 = st.columns(2)
                cc1.plotly_chart(px.bar(compare_df, x="Modelo", y="R² (log)",
                                        color="Modelo", title="R² por modelo",
                                        color_discrete_sequence=PALETTE), use_container_width=True)
                cc2.plotly_chart(px.bar(compare_df, x="Modelo", y="MAPE (%)",
                                        color="Modelo", title="MAPE (%) por modelo",
                                        color_discrete_sequence=PALETTE), use_container_width=True)
        else:
            st.info("👆 Entrena los modelos para generar conclusiones automáticas.")

        st.markdown("---")
        notes = st.text_area(
            "✍️ Notas narrativas",
            placeholder="Agrega aquí tus conclusiones, observaciones o hallazgos...",
            height=180,
        )
        if notes:
            st.download_button(
                "⬇️ Descargar notas (.txt)",
                data=notes,
                file_name="conclusiones_viaticos.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()

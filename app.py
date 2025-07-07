import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Rendimiento de Arroz - Babahoyo", layout="wide")
st.title("🌾 Análisis Predictivo de Rendimiento de Arroz - Babahoyo 2024")

@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_arroz_babahoyo_completo.csv", parse_dates=["fecha"])

df = cargar_datos()

tabs = st.tabs([
    "📁 Datos", 
    "📈 Tendencias Climáticas", 
    "📉 Modelo Predictivo", 
    "🔍 Predicción Interactiva",
    "📌 Correlaciones", 
    "📊 Análisis por Variable",
    "📤 Exportar Predicción",
    "📅 Evolución Temporal",
    "🗺️ Mapa Geográfico",
    "🔄 Comparación Predicción vs Real"
])

# ---------------------- 📁 TAB 1 - Datos ----------------------
with tabs[0]:
    st.subheader("📁 Dataset de Entrada - Año 2024")
    st.dataframe(df, use_container_width=True)

# ---------------------- 📈 TAB 2 - Clima ----------------------
with tabs[1]:
    st.subheader("📈 Comportamiento Climático Diario")
    fig, ax = plt.subplots(figsize=(12, 4))
    df.set_index("fecha")["temperatura_promedio_C"].plot(ax=ax, label="Temperatura (°C)", color="orangered")
    df.set_index("fecha")["precipitacion_mm"].plot(ax=ax, secondary_y=True, label="Precipitación (mm)", color="skyblue")
    ax.set_ylabel("Temperatura (°C)")
    ax.right_ax.set_ylabel("Precipitación (mm)")
    ax.set_title("Temperatura y Precipitación diaria - Babahoyo")
    ax.legend(loc="upper left")
    ax.right_ax.legend(loc="upper right")
    st.pyplot(fig)

# ---------------------- 📉 TAB 3 - Modelo ----------------------
with tabs[2]:
    st.subheader("📉 Modelo de Regresión Lineal para Predecir el Rendimiento")
    X = df[["temperatura_promedio_C", "precipitacion_mm", "humedad_relativa_%", "ph_suelo", "materia_organica_%"]]
    y = df["rendimiento_toneladas_ha"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    st.write("### 📌 Coeficientes del Modelo:")
    coef_df = pd.DataFrame({
        "Variable": X.columns,
        "Coeficiente": modelo.coef_
    })
    st.dataframe(coef_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Intercepto", f"{modelo.intercept_:.2f}")
    col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
    col3.metric("R²", f"{r2_score(y_test, y_pred):.3f}")

# ---------------------- 🔍 TAB 4 - Predicción ----------------------
with tabs[3]:
    st.subheader("🔍 Predicción Personalizada del Rendimiento")
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    temp = col1.slider("🌡️ Temperatura Promedio (°C)", 20.0, 35.0, 27.0)
    precip = col2.slider("🌧️ Precipitación (mm)", 0.0, 100.0, 10.0)
    hum = col3.slider("💧 Humedad Relativa (%)", 50.0, 100.0, 80.0)
    ph = col4.slider("🧪 pH del Suelo", 4.5, 7.5, 5.8)
    mo = col5.slider("🌱 Materia Orgánica (%)", 1.0, 4.0, 2.5)

    input_data = pd.DataFrame([[temp, precip, hum, ph, mo]], columns=X.columns)
    prediccion = modelo.predict(input_data)[0]

    st.success(f"✅ El rendimiento estimado del arroz es: **{prediccion:.2f} toneladas/ha**")

# ---------------------- 📌 TAB 5 - Correlaciones ----------------------
with tabs[4]:
    st.subheader("📌 Mapa de Correlación entre Variables")
    corr = df.select_dtypes(include=["float64", "int64"]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ---------------------- 📊 TAB 6 - Variable vs Rendimiento ----------------------
with tabs[5]:
    st.subheader("📊 Análisis Individual de Variables vs Rendimiento")
    variable = st.selectbox("Selecciona una variable:", X.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=variable, y="rendimiento_toneladas_ha", ax=ax, color="green")
    ax.set_title(f"{variable} vs Rendimiento (ton/ha)")
    st.pyplot(fig)

# ---------------------- 📤 TAB 7 - Exportar predicción ----------------------
with tabs[6]:
    st.subheader("📤 Exportar Predicción a CSV")
    exportar_df = input_data.copy()
    exportar_df["rendimiento_estimado"] = prediccion
    st.dataframe(exportar_df)

    csv = exportar_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Descargar como CSV",
        data=csv,
        file_name="prediccion_arroz.csv",
        mime="text/csv"
    )

# ---------------------- 📅 TAB 8 - Evolución semanal/mensual ----------------------
with tabs[7]:
    st.subheader("📅 Evolución del Rendimiento por Semana o Mes")
    df["semana"] = df["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
    df["mes"] = df["fecha"].dt.to_period("M").apply(lambda r: r.start_time)

    opcion_tiempo = st.radio("Agrupar por:", ["Semana", "Mes"], horizontal=True)

    if opcion_tiempo == "Semana":
        agrupado = df.groupby("semana")["rendimiento_toneladas_ha"].mean().reset_index()
        x = "semana"
    else:
        agrupado = df.groupby("mes")["rendimiento_toneladas_ha"].mean().reset_index()
        x = "mes"

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=agrupado, x=x, y="rendimiento_toneladas_ha", marker="o", ax=ax, color="darkblue")
    ax.set_title(f"Evolución del Rendimiento por {opcion_tiempo.lower()}")
    ax.set_ylabel("Toneladas por hectárea")
    ax.set_xlabel("Fecha")
    st.pyplot(fig)

# ---------------------- 🗺️ TAB 9 - Mapa geográfico ----------------------
with tabs[8]:
    st.subheader("🗺️ Visualización Geográfica del Rendimiento")
    if "latitud" in df.columns and "longitud" in df.columns:
        st.map(df.rename(columns={"latitud": "latitude", "longitud": "longitude"}))
        st.caption("Cada punto representa una observación del rendimiento en una ubicación de Babahoyo.")
    else:
        st.warning("El dataset no contiene columnas de latitud y longitud para mostrar el mapa.")

# ---------------------- 🔄 TAB 10 - Comparación predicción vs real ----------------------
with tabs[9]:
    st.subheader("🔄 Comparación entre Predicción y Valores Reales")
    comparacion_df = pd.DataFrame({
        "Real": y_test.values,
        "Predicción": y_pred
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=comparacion_df, x="Real", y="Predicción", color="purple")
    ax.plot([comparacion_df.min().min(), comparacion_df.max().max()],
            [comparacion_df.min().min(), comparacion_df.max().max()],
            color="gray", linestyle="--", label="Ideal")
    ax.set_title("Comparación de Predicciones vs Valores Reales")
    ax.set_xlabel("Rendimiento Real (ton/ha)")
    ax.set_ylabel("Rendimiento Predicho (ton/ha)")
    ax.legend()
    st.pyplot(fig)

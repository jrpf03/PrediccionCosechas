
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Análisis de Rendimiento de Arroz - Babahoyo", layout="wide")

st.title("📊 Análisis Predictivo de Rendimiento de Arroz en Babahoyo - Año 2024")

@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_arroz_babahoyo_2024.csv", parse_dates=["fecha"])

df = cargar_datos()

st.subheader("📁 Datos de Entrada - 2024")
st.dataframe(df)

# Visualización básica
st.subheader("📈 Tendencias Climáticas Diarias")
fig, ax = plt.subplots(figsize=(12, 4))
df.set_index("fecha")["temperatura_promedio_C"].plot(ax=ax, label="Temperatura (°C)")
df.set_index("fecha")["precipitacion_mm"].plot(ax=ax, secondary_y=True, label="Precipitación (mm)", color="skyblue")
ax.set_ylabel("Temperatura (°C)")
ax.right_ax.set_ylabel("Precipitación (mm)")
ax.set_title("Temperatura y Precipitación diaria - 2024")
st.pyplot(fig)

# Modelo de regresión lineal
st.subheader("📉 Modelo de Regresión Lineal para predecir el rendimiento")

X = df[["temperatura_promedio_C", "precipitacion_mm", "humedad_relativa_%", "ph_suelo", "materia_organica_%"]]
y = df["rendimiento_toneladas_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

st.write("### Coeficientes del modelo:")
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
})
st.dataframe(coef_df)

st.write(f"**Intercepto:** {modelo.intercept_:.2f}")
st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.3f}")
st.write(f"**R²:** {r2_score(y_test, y_pred):.3f}")

# Predicción personalizada
st.subheader("🔍 Predicción Personalizada")
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

temp = col1.slider("Temperatura Promedio (°C)", 20.0, 35.0, 27.0)
precip = col2.slider("Precipitación (mm)", 0.0, 100.0, 10.0)
hum = col3.slider("Humedad Relativa (%)", 50.0, 100.0, 80.0)
ph = col4.slider("pH del Suelo", 4.5, 7.5, 5.8)
mo = col5.slider("Materia Orgánica (%)", 1.0, 4.0, 2.5)

input_data = pd.DataFrame([[temp, precip, hum, ph, mo]], columns=X.columns)
prediccion = modelo.predict(input_data)[0]

st.success(f"🌾 El rendimiento estimado del arroz es: {prediccion:.2f} toneladas/ha")

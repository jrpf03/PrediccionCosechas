
# 📊 Proyecto de Análisis Predictivo del Rendimiento de Arroz en Babahoyo

Este proyecto utiliza datos meteorológicos simulados diarios del año 2024 para estimar el rendimiento del cultivo de arroz en Babahoyo, Ecuador, mediante un modelo de regresión lineal con una interfaz gráfica desarrollada en Streamlit.

## 🗂 Contenido del proyecto

- `app.py`: Aplicación principal de Streamlit.
- `dataset_arroz_babahoyo_2024.csv`: Dataset diario para el año 2024 con variables climáticas y de suelo.

## ▶️ Cómo ejecutar


### 1. Instala los requerimientos
```
pip install pandas scikit-learn matplotlib streamlit
```

### 2. Ejecuta la aplicación
```
streamlit run app.py
```

La aplicación se abrirá por defecto en [http://localhost:8501](http://localhost:8501)

## 🔍 ¿Qué hace la app?

- Carga y visualiza datos diarios climáticos y edáficos del año 2024.
- Aplica un modelo de regresión lineal para estimar el rendimiento del arroz (toneladas/hectárea).
- Permite probar escenarios personalizados para ver su efecto en el rendimiento.

## 🧠 Modelo usado

- **Modelo**: Regresión lineal
- **Variables independientes**:
  - Temperatura promedio diaria (°C)
  - Precipitación diaria (mm)
  - Humedad relativa (%)
  - pH del suelo
  - Materia orgánica (%)
- **Variable dependiente**:
  - Rendimiento de arroz (ton/ha)

---

📍 Ubicación: Babahoyo, Ecuador  
📅 Año: 2024  
📈 Datos generados con base en promedios históricos del portal [meteoblue.com](https://www.meteoblue.com)


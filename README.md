# Proyecto de análisis de viáticos - Registraduría

Este repositorio reúne el trabajo realizado para analizar los viáticos y comisiones de viaje de la Registraduría Nacional del Estado Civil. Incluye el notebook principal de estadística, los datos procesados, y una app en Streamlit para explorar la información de forma interactiva.

## Objetivo del proyecto

El propósito fue entender qué factores explican el gasto en viáticos, identificar patrones de comportamiento en las comisiones de viaje y construir modelos predictivos que permitan estimar el valor total del viático en función de variables operativas como la duración del viaje, la distancia, el cargo, la ciudad de origen y el destino.

## Dataset

El notebook trabaja sobre un dataset histórico de comisiones y viáticos de la Registraduría. En su versión analizada se reportan estas características principales:

- 14 variables.
- 15.753 observaciones.
- 1.272 valores faltantes en total, concentrados principalmente en `position_level`.
- 0 filas duplicadas.
- 416 registros con valor cero en `total_travel_allowance`.

### Variables principales

Las columnas más importantes son:

- `total_travel_allowance`: valor total del viático, que funciona como variable objetivo.
- `commission_days`: cantidad de días de la comisión.
- `distance_km`: distancia entre origen y destino.
- `destination_count`: número de ciudades de destino en la ruta.
- `purpose`: motivo del viaje.
- `employee_position_norm`: cargo normalizado del empleado.
- `position_level`: nivel jerárquico del cargo.
- `city_origin` y `city_destination_main`: ciudad de origen y destino principal.
- `is_contractor`: indica si la persona es contratista.
- `full_date`: fecha del viaje, usada para generar variables de tiempo como mes, día de la semana y trimestre.

## Qué se hizo en el notebook

El notebook principal es [Proyecto_Final_Estadistica.ipynb](Estadística%20para%20analitica%20de%20datos/notebooks/Proyecto_Final_Estadistica.ipynb) y resume el flujo completo del análisis:

1. Se presentó la motivación del estudio y el objetivo general.
2. Se documentaron las variables del dataset.
3. Se hizo limpieza y preprocesamiento de datos.
4. Se realizó análisis exploratorio de datos con tablas, gráficos y distribución de variables.
5. Se revisó la calidad del dataset con un reporte automático de perfilado.
6. Se analizaron correlaciones entre variables numéricas y relaciones entre variables categóricas y numéricas.
7. Se aplicaron pruebas de hipótesis y correlación, incluyendo Pearson, Spearman, Kruskal-Wallis y Mann-Whitney.
8. Se entrenaron dos modelos predictivos para estimar el viático:
	- Regresión lineal como línea base.
	- HistGradientBoosting como modelo más flexible.
9. Se evaluaron los modelos con métricas como R², MAE, RMSE y MAPE.
10. Se incluyó una función para predecir un viático nuevo y obtener un intervalo de confianza aproximado.

### Hallazgos destacados

Algunos hallazgos útiles del análisis fueron:

- `commission_days` muestra una relación alta con `total_travel_allowance`.
- `purpose`, `employee_position_norm`, `position_level` e `is_contractor` presentan desbalance en sus categorías.
- `position_level` concentra los valores faltantes del dataset.
- Se detectó que varias variables categóricas están relacionadas entre sí, especialmente cargo y tipo de vinculación.

## App en Streamlit 

La app ubicada en [apps/streamlit_app.py](apps/streamlit_app.py) permite explorar el dataset sin abrir el notebook. Incluye:

- Resumen general del dataset.
- Gráficos de distribuciones categóricas y numéricas.
- Matriz de correlación.
- Pruebas estadísticas no paramétricas.
- Entrenamiento y comparación de modelos.
- Predicción individual de viáticos.
- Un espacio para dejar conclusiones narrativas.

## Como correrla de manera local

- Python 3.10 o superior. Se recomienda Python 3.12.
- Dependencias listadas en [requirements.txt](requirements.txt).

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecución de la app

```bash
streamlit run apps/streamlit_app.py
```

### Ruta de datos

Por defecto la app intenta cargar:

```text
Estadística para analitica de datos/data/processed/registraduria_database_clean.xlsx
```

Si necesitas usar otro archivo, cambia la ruta desde la barra lateral de la app.

## Enlace de la app usando Streamlit deploy

[Análisis de Viaticos Registraduría](https://registradur-a-viaticos-3f9vi8hoyxiv5snt3vnaxq.streamlit.app/)

## Enlace Notebook Colab

[Google Drive - Colab](https://drive.google.com/file/d/1V91nWS5MC-6PZplbrOIyh5C-borQHV1P/view?usp=sharing)

## Estructura del repositorio

- [apps/streamlit_app.py](apps/streamlit_app.py): app interactiva.
- [Estadística para analitica de datos/notebooks/Proyecto_Final_Estadistica.ipynb](Estad%C3%ADstica%20para%20analitica%20de%20datos/notebooks/Proyecto_Final_Estadistica.ipynb): notebook principal del proyecto.
- [Estadística para analitica de datos/data/](Estad%C3%ADstica%20para%20analitica%20de%20datos/data/raw/): datos originales usados para el análisis.



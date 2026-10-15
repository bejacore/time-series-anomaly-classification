# Clasificación Automatizada de Series Temporales Astronómicas (MANTRA / CRTS)

**Autor:** Carlos Bejarano

**Enfoque:** Data Analytics, Data Governance & Time-Series Feature Engineering.

---

## Contexto y Problema

Los estudios astronómicos modernos de dominio temporal (como el *Catalina Real-Time Transient Survey* - CRTS o el Observatorio Vera C. Rubin) generan volúmenes masivos de alertas fotométricas cada noche. Sin embargo, confirmar la naturaleza exacta de un evento mediante espectroscopía requiere tiempo de telescopio especializado, el cual es un recurso extremadamente escaso y costoso.

**Objetivo del Proyecto:**

Desarrollar un pipeline analítico y de clasificación supervisada capaz de identificar y categorizar automáticamente cinco tipos principales de fenómenos astrofísicos (Supernovas [SN], Núcleos Galácticos Activos [AGN], Variables Cataclísmicas [CV], Blazares y Estrellas de Alto Movimiento Propio [HPM]) utilizando únicamente la morfología estadística de sus series de tiempo (curvas de luz). Esto permite priorizar alertas críticas y optimizar la asignación de recursos operativos de observación.

---

## Arquitectura del Pipeline Analítico

### Ingesta, Integración y Gobierno de Datos (*Data Integrity*)

Se procesaron e integraron tres fuentes independientes con esquemas de identificación heterogéneos:

- **`transient_lightcurves.csv` (440,469 registros):** Series longitudinales de brillo (Magnitud) e incertidumbre de medición a lo largo del tiempo (MJD).
- **`transient_labels.csv` (5,539 registros):** Catálogo histórico de etiquetas de referencia (*ground truth*).
- **`transient_info.txt` (5,539 registros):** Metadatos espaciales y coordenadas celestes (RA, Dec).

> **Decisión Analítica:** Ante la ausencia de una llave primaria unificada nativa (`TransientID` vs. prefijos `TranID` vs. columna `CSS images`), se implementó una normalización de cadenas y una auditoría de integridad referencial mediante teoría de conjuntos, garantizando un 100% de trazabilidad sobre 4,869 objetos únicos con fotometría verificada.
> 

### Control de Calidad y Embudo de Datos (*Data Quality Funnel*)

Para evitar que el ruido de etiqueta degradara las fronteras de decisión de los clasificadores, se diseñó un embudo de filtrado en lugar de forzar imputaciones sobre clases ambiguas:

| **Etapa del Embudo Analítico** | **Registros**  | **Criterio de Calidad Aplicado** |
| --- | --- | --- |
| **Catálogo Crudo Inicial** | 5,539 | Total de registros en el catálogo de referencia. |
| **Filtro de Etiquetas Confirmadas** | 4,040 | Descarte de 1,499 objetos (27.1%) con etiquetas ambiguas o contaminadas (`SN?`, `Unknown`, `SN/CV`, `AGN?`), aislando las 5 clases objetivo. |
| **Filtro de Suficiencia Estadística** | 3,123 | Retención exclusiva de series temporales con curva de luz verificada y un umbral mínimo de **`MIN_OBS >= 5`** puntos fotométricos para garantizar validez en momentos estadísticos de orden superior. |

### Hallazgos Clave del Análisis Exploratorio (EDA)

- **Sesgo de selección operativo (Desbalance ~5.5:1):** La clase dominante es `SN` (1,293 objetos), seguida de `CV` (862), `AGN` (425), `HPM` (306) y `Blazar` (237), reflejando la prioridad operativa del survey CRTS por capturar eventos explosivos extragalácticos.
- **Asimetría en la cadencia temporal:** Los fenómenos persistentes (`AGN`, `Blazar`) acumulan decenas de mediciones durante años, mientras que eventos transitorios fugaces (`SN`) registran una mediana de observaciones significativamente menor antes de desvanecerse.
- **Prevención del sesgo de supervivencia:** La inspección visual de curvas de luz se realizó seleccionando objetos ubicados en la mediana de observaciones **(`n_points`)** de cada clase y no en los valores extremos superiores, evitando graficar objetos contaminados por el brillo constante de su galaxia anfitriona.

### Ingeniería de Características (*Time-Series Feature Engineering*)

Para transformar series temporales de longitud variable y muestreo irregular en una matriz tabular modelable, se construyó un motor de extracción que ordena cronológicamente por `MJD`, elimina *timestamps* duplicados y calcula 21 descriptores por objeto:

- **Escala y Robustez:** Media (`mean_mag`), desviación estándar (`std_mag`), desviación absoluta mediana (`mad_mag`), amplitud total (`amplitude`), amplitud interdecil robusta a *outliers* (`amp_10_90`) y percentiles (`pct_5`, `pct_25`, `pct_75`, `pct_95`).
- **Forma de la Distribución:** Asimetría (`skewness`) y curtosis (`kurtosis`).
- **Dinámica Temporal y Coherencia Física:** Pendiente lineal y ajuste (`slope`, `slope_r2`), índice $\eta$ de Von Neumann (suavidad de variación punto a punto), índice $J$ de Stetson (variabilidad real ponderada frente al ruido fotométrico), cruces por la mediana (`median_crossings`) y fracción de tiempo hasta el pico máximo de brillo (`time_to_peak_frac`).
- **Métricas Instrumentales:** Error fotométrico medio (`mean_magerr`), correlación error-magnitud (`err_mag_corr`), duración del monitoreo (`time_span`) y conteo de capturas (`n_points`).

---

## Modelado Supervisado y Resultados

Se aplicó una partición estratificada 80% entrenamiento / 20% prueba. Para prevenir fuga de información, el escalador z-score (`StandardScaler`) se ajustó exclusivamente sobre el conjunto de entrenamiento. El desbalance de clases se gestionó penalizando la función de costo mediante pesos inversamente proporcionales (`class_weight='balanced'`), evitando distorsiones físicas por datos sintéticos.

### Comparativa de Desempeño

| **Modelo Supervisado** | **F1-Score Macro (Train)** | **F1-Score Macro (Test)** | **Accuracy (Test)** | **Rol en la Experimentación** |
| --- | --- | --- | --- | --- |
| **SVM (Kernel RBF)** | 0.9008 | 0.7919 | 81.0% | **Modelo Ganador:** Mejor generalización en fronteras no lineales de alta dimensión. |
| **Random Forest** *(Regularizado)* | 0.9590 | 0.7639 | 82.0% | **Modelo de Ensamble:** Captura interacciones complejas (`max_depth=8`, `min_samples_leaf=3`) y provee importancia de variables. |
| **Logistic Regression** | 0.7004 | 0.7462 | 78.0% | **Baseline Lineal:** Línea base de referencia para evaluar separabilidad lineal. |

### Interpretación Analítica y del Dominio

1. **Mayor Facilidad de Clasificación — Objetos `HPM` (F1-Score Test: 0.90 - 1.00):** Al tratarse de estrellas cercanas que se desplazan rápidamente a través del píxel del detector, generan tendencias lineales monótonas y firmas altamente predecibles capturadas por `slope`, `slope_r2` y un bajo número de `median_crossings`.
2. **Desempeño Sólido — `AGN`, `CV` y `SN` (F1-Score Test: 0.81 - 0.86):** Las Supernovas (`SN`) destacan por su pulso único asimétrico (`time_to_peak_frac`, `skewness`), mientras que las Variables Cataclísmicas (`CV`) son identificadas por sus estallidos recurrentes de gran amplitud (`amplitude`, `kurtosis`).
3. **Principal Reto Operativo — `Blazar` (F1-Score Test: 0.40 - 0.47):** Presenta alta confusión con la clase `AGN`. Físicamente, un Blazar es un subtipo de AGN cuyo chorro relativista apunta hacia el observador; ambos exhiben variabilidad estocástica continua en escalas de años, haciendo que sus distribuciones estadísticas unibanda se solapen.
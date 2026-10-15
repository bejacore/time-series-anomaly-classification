import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

COLORS = {'SN': '#E63946', 'AGN': '#457B9D', 'CV': '#2A9D8F', 'Blazar': '#E9C46A', 'HPM': '#F4A261'}
MAIN5 = ['SN', 'AGN', 'CV', 'Blazar', 'HPM']

def plot_eda_summary(df_pool: pd.DataFrame) -> None:
    """Grafica el desbalance de clases y la mediana de observaciones por categoría."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    counts = df_pool['label'].value_counts()
    axes[0].bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    axes[0].set_ylabel('Número de objetos')
    axes[0].set_title('Distribución de clases')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center')

    pts_by_class = df_pool.groupby('label')['n_points'].median().sort_values()
    axes[1].barh(pts_by_class.index, pts_by_class.values, color=[COLORS[c] for c in pts_by_class.index])
    axes[1].set_xlabel('Mediana de puntos fotométricos por curva')
    axes[1].set_title('Cadencia mediana por clase')
    
    plt.tight_layout()
    plt.show()


def plot_representative_lightcurves(df_pool: pd.DataFrame, df_lc: pd.DataFrame) -> None:
    """Grafica 2 curvas de luz representativas (cercanas a la mediana de puntos) para cada clase."""
    fig, axes = plt.subplots(5, 2, figsize=(13, 18))

    for row, cls in enumerate(MAIN5):
        sub_cls = df_pool[df_pool['label'] == cls].copy()
        med_n = sub_cls['n_points'].median()
        sub_cls['dist_to_median'] = (sub_cls['n_points'] - med_n).abs()
        ids_cls = sub_cls.sort_values('dist_to_median')['TransientID'].values
        chosen = ids_cls[:2] if len(ids_cls) >= 2 else ids_cls
        
        for col, oid in enumerate(chosen):
            ax = axes[row, col]
            g = df_lc[df_lc['TransientID'] == oid].sort_values('MJD')
            ax.errorbar(g['MJD'] - g['MJD'].min(), g['Mag'], yerr=g['Magerr'],
                        fmt='o', ms=3, color=COLORS[cls], alpha=0.7, ecolor='gray', elinewidth=0.5)
            ax.invert_yaxis()
            ax.set_title(f"{cls}  (n={len(g)} pts, mediana={int(med_n)})", fontsize=10)
            if row == 4:
                ax.set_xlabel('Días desde 1ra observación')
            if col == 0:
                ax.set_ylabel('Magnitud')
                
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(fitted_models: dict, X_te_sc, y_test, classes: list) -> None:
    """Grafica las matrices de confusión normalizadas en el conjunto de prueba para todos los modelos."""
    n_models = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, clf) in zip(axes, fitted_models.items()):
        yp_te = clf.predict(X_te_sc)
        cm = confusion_matrix(y_test, yp_te, labels=classes, normalize='true')
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
        ax.set_title(f"{name}\n(Test Normalizada)")
        ax.set_xlabel("Predicción")
        if ax == axes[0]:
            ax.set_ylabel("Clase Real")

        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if cm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(rf_model, feature_cols: list, top_n: int = 12) -> None:
    """Grafica las variables con mayor poder predictivo según el modelo Random Forest."""
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(9, 5))
    plt.barh(np.array(feature_cols)[idx], importances[idx], color='#2A9D8F')
    plt.xlabel('Importancia Relativa (Reducción de Impureza Gini)')
    plt.title(f'Top {top_n} Características Más Determinantes (Random Forest)')
    plt.tight_layout()
    plt.show()
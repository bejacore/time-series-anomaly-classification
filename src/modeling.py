import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

def split_and_scale(df_model: pd.DataFrame, test_size: float = 0.20, random_state: int = 23):
    """Divide en train/test estratificado y aplica StandardScaler evitando data leakage."""
    feature_cols = [c for c in df_model.columns if c not in ('TransientID', 'label')]
    X = df_model[feature_cols].values
    y = df_model['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    return X_tr_sc, X_te_sc, y_train, y_test, feature_cols, scaler


def train_and_evaluate_models(classifiers: dict, X_tr_sc, y_train, X_te_sc, y_test, classes: list):
    """Entrena un diccionario de clasificadores, imprime reportes y retorna una tabla comparativa."""
    fitted = {}
    summary = []

    for name, clf in classifiers.items():
        clf.fit(X_tr_sc, y_train)
        fitted[name] = clf
        yp_tr, yp_te = clf.predict(X_tr_sc), clf.predict(X_te_sc)
        
        f1_tr = f1_score(y_train, yp_tr, average='macro')
        f1_te = f1_score(y_test, yp_te, average='macro')
        summary.append({'Modelo': name, 'F1_Macro_Train': round(f1_tr, 4), 'F1_Macro_Test': round(f1_te, 4)})

        print(f"\n{'='*60}\nCLASIFICADOR: {name}\n{'='*60}")
        print(f"F1 macro — Train: {f1_tr:.4f} | Test: {f1_te:.4f}")
        print("\n-- TEST REPORT --")
        print(classification_report(y_test, yp_te, labels=classes, zero_division=0))

    return fitted, pd.DataFrame(summary)
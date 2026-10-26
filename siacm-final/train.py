"""Treinamento reproduzível do SIACM.

O desenvolvimento (80%) é usado na validação cruzada e nos exemplos/estatísticas;
o teste (20%) fica isolado e é usado somente para as métricas finais.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
d = load_breast_cancer()
X = pd.DataFrame(d.data, columns=[str(f) for f in d.feature_names])
y = pd.Series(d.target)
Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
f1_maligno = make_scorer(f1_score, pos_label=0)
make_pipeline = lambda model: Pipeline([("scaler", StandardScaler()), ("model", model)])
grids = {
    "Regressão Logística": (LogisticRegression(random_state=42, max_iter=1000),
                            {"model__C": [.01, .1, 1, 10]}),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=42),
                          {"model__max_depth": [3, 5, 7, None],
                           "model__min_samples_split": [2, 5, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"model__n_estimators": [50, 100, 200],
                       "model__max_depth": [5, 10, None]}),
}
models, metrics = {}, {}
for name, (model, params) in grids.items():
    search = GridSearchCV(make_pipeline(model), params,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring=f1_maligno, n_jobs=-1, refit=True)
    search.fit(Xdev, ydev)
    best = search.best_estimator_
    models[name] = best
    pred = best.predict(Xte)
    p_maligno = best.predict_proba(Xte)[:, 0]
    metrics[name] = dict(
        acuracia=accuracy_score(yte, pred),
        precisao_maligno=precision_score(yte, pred, pos_label=0),
        recall_maligno=recall_score(yte, pred, pos_label=0),
        f1_maligno=f1_score(yte, pred, pos_label=0),
        auc=roc_auc_score(yte == 0, p_maligno),
        f1_cv_maligno=search.best_score_,
        params={key: str(value) for key, value in search.best_params_.items()},
    )

p_ensemble = np.mean([model.predict_proba(Xte)[:, 0] for model in models.values()], axis=0)
ensemble_pred = np.where(p_ensemble >= .5, 0, 1)
metrics["Conjunto (média dos 3)"] = dict(
    acuracia=accuracy_score(yte, ensemble_pred),
    precisao_maligno=precision_score(yte, ensemble_pred, pos_label=0),
    recall_maligno=recall_score(yte, ensemble_pred, pos_label=0),
    f1_maligno=f1_score(yte, ensemble_pred, pos_label=0),
    auc=roc_auc_score(yte == 0, p_ensemble),
    f1_cv_maligno=None,
    params={},
)

# Componentes auxiliares nunca usam o conjunto de teste.
p_dev = models["Regressão Logística"].predict_proba(Xdev)[:, 0]
indices = {"Maligno típico": int(np.argmax(p_dev)),
           "Benigno típico": int(np.argmin(p_dev)),
           "Caso limítrofe": int(np.argmin(abs(p_dev - .5)))}
examples = {
    name: dict(zip(d.feature_names, map(float, Xdev.iloc[index]))) |
    {"_real": "maligno" if ydev.iloc[index] == 0 else "benigno"}
    for name, index in indices.items()
}
stats = {feature: dict(mean=float(Xdev[feature].mean()), min=float(Xdev[feature].min()),
                       max=float(Xdev[feature].max())) for feature in d.feature_names}

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(dict(models=models, features=list(d.feature_names)), MODEL_DIR / "bundle.joblib")
(MODEL_DIR / "meta.json").write_text(
    json.dumps(dict(metrics=metrics, examples=examples, stats=stats),
               ensure_ascii=False, indent=1),
    encoding="utf-8",
)
for name, values in metrics.items():
    print(name, {key: round(value, 4) for key, value in values.items()
                 if isinstance(value, float)})

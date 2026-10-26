"""Modelo principal, comparação do ensemble e motor de decisão por regras."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
B = joblib.load(MODEL_DIR / "bundle.joblib")
META = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
FEATURES, MODELS = B["features"], B["models"]
PRIMARY_NAME = "Regressão Logística"
PRIMARY = MODELS[PRIMARY_NAME]
MEANS = PRIMARY.named_steps["scaler"].mean_

RULES = dict(baixo=0.30, alto=0.70, inconclusivo=(0.40, 0.60), max_desacordo=0.25)


def risk_level(p):
    return "Baixo" if p < RULES["baixo"] else "Alto" if p >= RULES["alto"] else "Intermediário"


def _ensemble(df):
    return np.mean([m.predict_proba(df)[:, 0] for m in MODELS.values()], axis=0)


def explain(x, top=6):
    """Contribuições por oclusão do modelo principal, na escala log-odds de malignidade.
    Na escala de probabilidade a explicação satura em casos de ~0% ou ~100%; em log-odds
    a contribuição de cada atributo é aditiva e continua informativa."""
    base = np.tile(x, (len(FEATURES) + 1, 1))
    for i in range(len(FEATURES)):
        base[i + 1, i] = MEANS[i]
    logit = -PRIMARY.decision_function(pd.DataFrame(base, columns=FEATURES))  # classe 1 = benigno
    delta = logit[0] - logit[1:]
    total = abs(delta).sum() or 1.0
    order = np.argsort(-abs(delta))[:top]
    return [
        dict(feature=FEATURES[i], contribuicao=float(delta[i]), peso=float(abs(delta[i]) / total),
             direcao="aumenta a estimativa" if delta[i] > 0 else "reduz a estimativa")
        for i in order
    ]


def analyze(values):
    x = np.array([float(values[f]) for f in FEATURES])
    df = pd.DataFrame([x], columns=FEATURES)
    per = {name: float(model.predict_proba(df)[0, 0]) for name, model in MODELS.items()}
    p = per[PRIMARY_NAME]
    ensemble_p = float(_ensemble(df)[0])
    spread = max(per.values()) - min(per.values())
    lo, hi = RULES["inconclusivo"]
    reasons = []
    if lo <= p <= hi:
        reasons.append("estimativa próxima de 50%")
    if spread > RULES["max_desacordo"]:
        reasons.append("modelos em desacordo")
    if (min(per.values()) < .5) != (max(per.values()) < .5):
        reasons.append("modelos divergem na classe")
    if any(not META["stats"][f]["min"] * .5 <= x[i] <= META["stats"][f]["max"] * 1.5
           for i, f in enumerate(FEATURES)):
        reasons.append("valores fora da faixa do conjunto de desenvolvimento")
    conf = "Baixa" if reasons else ("Alta" if abs(p - .5) > .35 else "Média")
    if reasons:
        cls, rec = "Resultado inconclusivo", "Recomenda-se avaliação complementar."
    elif p >= .5:
        cls, rec = "Provável tumor maligno", "Encaminhar para avaliação clínica prioritária."
    else:
        cls, rec = "Provável tumor benigno", "Manter acompanhamento conforme avaliação médica."
    return dict(classificacao=cls, prob_maligno=p, nivel_risco=risk_level(p), confianca=conf,
                motivos=reasons, recomendacao=rec, por_modelo=per, ensemble_prob_maligno=ensemble_p,
                modelo_principal=PRIMARY_NAME, explicacao=explain(x),
                aviso="Ferramenta acadêmica de apoio à decisão. Não substitui avaliação médica.")

import os, math, sqlite3, datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import engine, genai
ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "static")); app.config["MAX_CONTENT_LENGTH"] = 32_000
app.config["DB"] = os.environ.get("SIACM_DB", "history.db")
def db():
    c = sqlite3.connect(app.config["DB"])
    c.execute("create table if not exists h(id integer primary key, ts text, label text, p real, risk text, conf text, cls text)")
    return c

@app.get("/")
def index(): return send_from_directory(app.static_folder, "index.html")
@app.get("/api/schema")
def schema(): return jsonify(features=engine.FEATURES, stats=engine.META["stats"], examples=engine.META["examples"], rules=engine.RULES)
@app.get("/api/metrics")
def metrics(): return jsonify(engine.META["metrics"])

def validar(d):
    if not isinstance(d, dict): return None, "Corpo da requisição inválido."
    out = {}
    for f in engine.FEATURES:
        try: v = float(d[f])
        except (KeyError, TypeError, ValueError): return None, f"Campo ausente ou não numérico: {f}"
        if not math.isfinite(v) or v < 0: return None, f"Valor inválido (deve ser finito e ≥ 0): {f}"
        out[f] = v
    return out, None

@app.post("/api/predict")
def predict():
    raw = request.get_json(silent=True); vals, err = validar(raw)
    if err: return jsonify(erro=err), 400
    label = "".join(ch for ch in str(raw.get("_label") or "Exame") if ch.isprintable())[:40]   # só demonstração: sem dados reais
    r = engine.analyze(vals); r["explicacao_ia"] = genai.explicar(r)
    with db() as c: c.execute("insert into h(ts,label,p,risk,conf,cls) values(?,?,?,?,?,?)",
        (datetime.datetime.now().strftime("%d/%m %H:%M"), label, r["prob_maligno"], r["nivel_risco"], r["confianca"], r["classificacao"]))
    return jsonify(r)
@app.get("/api/history")
def history():
    with db() as c: rows = c.execute("select ts,label,p,risk,conf,cls from h order by id desc limit 15").fetchall()
    return jsonify([dict(ts=a, label=b, p=p, risk=r, conf=cf, cls=cl) for a, b, p, r, cf, cl in rows])
if __name__ == "__main__": app.run(debug=os.environ.get("SIACM_DEBUG") == "1")   # debug desligado por padrão

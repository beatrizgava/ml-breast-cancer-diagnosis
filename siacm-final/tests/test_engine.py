import engine
E = engine.META["examples"]
def vals(k): return {f: v for f, v in E[k].items() if f != "_real"}
def test_maligno(): r = engine.analyze(vals("Maligno típico")); assert r["prob_maligno"] > .9 and r["nivel_risco"] == "Alto"
def test_benigno(): r = engine.analyze(vals("Benigno típico")); assert r["prob_maligno"] < .1 and r["nivel_risco"] == "Baixo"
def test_limitrofe(): assert engine.analyze(vals("Caso limítrofe"))["confianca"] in ("Baixa", "Média")
def test_niveis(): assert [engine.risk_level(x) for x in (.1, .5, .9)] == ["Baixo", "Intermediário", "Alto"]
def test_inconclusivo():
    v = {f: engine.META["stats"][f]["mean"] for f in engine.FEATURES}; v["mean radius"] *= 20
    assert engine.analyze(v)["classificacao"] == "Resultado inconclusivo"
def test_explicacao_do_conjunto():
    r = engine.analyze(vals("Maligno típico")); e = r["explicacao"]
    assert len(e) == 6 and e[0]["contribuicao"] > 0 and abs(sum(x["peso"] for x in e)) <= 1.0001
def test_explicacao_nao_satura():
    for k in ("Maligno típico", "Benigno típico"):
        e = engine.analyze(vals(k))["explicacao"]; assert abs(e[0]["contribuicao"]) > 0.5 and e[0]["peso"] < 0.9
def test_pipeline_sem_vazamento():
    from sklearn.pipeline import Pipeline
    assert all(isinstance(m, Pipeline) for m in engine.MODELS.values())

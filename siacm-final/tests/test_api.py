import pytest, engine, app as appmod
@pytest.fixture
def c(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False); appmod.app.config["DB"] = str(tmp_path / "t.db"); return appmod.app.test_client()
def ex(k="Maligno típico"): return {f: v for f, v in engine.META["examples"][k].items() if f != "_real"}
def test_rotas_basicas(c):
    assert c.get("/").status_code == 200 and len(c.get("/api/schema").get_json()["features"]) == 30
    assert "Conjunto (média dos 3)" in c.get("/api/metrics").get_json()
def test_predict_ok_e_historico(c):
    d = ex() | {"_label": "Exame 001"}; r = c.post("/api/predict", json=d); j = r.get_json()
    assert r.status_code == 200 and j["nivel_risco"] == "Alto" and j["explicacao_ia"]["texto"]
    h = c.get("/api/history").get_json(); assert h[0]["label"] == "Exame 001" and h[0]["cls"] == j["classificacao"]
@pytest.mark.parametrize("mut", [lambda d: d.pop("mean area"), lambda d: d.update({"mean area": "abc"}),
    lambda d: d.update({"mean area": -5}), lambda d: d.update({"mean area": "nan"}), lambda d: d.update({"mean area": "inf"})])
def test_entradas_invalidas(c, mut):
    d = ex(); mut(d); assert c.post("/api/predict", json=d).status_code == 400
def test_corpo_invalido(c):
    assert c.post("/api/predict", data="x", content_type="text/plain").status_code == 400
    assert c.post("/api/predict", json=[1, 2]).status_code == 400
def test_label_truncado(c):
    c.post("/api/predict", json=ex() | {"_label": "<b>" + "a" * 100}); assert len(c.get("/api/history").get_json()[0]["label"]) == 40

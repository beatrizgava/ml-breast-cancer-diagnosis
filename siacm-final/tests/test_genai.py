import json, engine, genai
R = engine.analyze({f: v for f, v in engine.META["examples"]["Maligno típico"].items() if f != "_real"})
def test_payload_minimo():
    s = json.dumps(genai.payload(R), ensure_ascii=False)
    assert "_label" not in s and "mean radius" not in s and len(genai.payload(R)["fatores"]) == 5
def test_template_valido(): assert genai.valido(genai.template(R))
def test_sem_chave_usa_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False); assert "local" in genai.explicar(R)["fonte"]
def test_saida_perigosa_reprovada(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k"); monkeypatch.setattr(genai, "_call", lambda p, k: "Você tem câncer. Indicamos cirurgia.")
    assert "local" in genai.explicar(R)["fonte"]
def test_saida_boa_aceita(monkeypatch):
    t = "O modelo estimou 99% de malignidade. Fatores são contribuições matemáticas. Ferramenta acadêmica; não substitui avaliação médica."
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k"); monkeypatch.setattr(genai, "_call", lambda p, k: t)
    assert genai.explicar(R)["texto"] == t and "IA generativa" in genai.explicar(R)["fonte"]
def test_falha_de_rede_usa_template(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    def boom(p, k): raise OSError()
    monkeypatch.setattr(genai, "_call", boom); assert "local" in genai.explicar(R)["fonte"]

# SIACM — Sistema Inteligente de Apoio à Classificação de Tumores Mamários
Projeto Integrador de Engenharia de Computação · evolução do projeto de ML de Beatriz Alves Gava e Wallace Miranda Senna da Silva.
> Ferramenta **acadêmica** de apoio à decisão. Não diagnostica e não substitui avaliação médica. Use apenas dados de demonstração.

## Executar
```bash
pip install -r requirements.txt
python train.py          # treina e gera model/bundle.joblib e model/meta.json
python -m pytest tests   # 23 testes (motor, API, IA generativa)
python app.py            # http://127.0.0.1:5000   (SIACM_DEBUG=1 só para desenvolvimento)
export ANTHROPIC_API_KEY=...   # opcional: ativa a IA generativa; sem chave, usa texto-modelo local
```

## Arquitetura
```
Exame → Pipeline(Padronização + modelo) ×3 → Estimativa agregada → Motor de decisão ─┐
                          └→ Explicabilidade do conjunto (oclusão) ──────────────────┼→ IA generativa → Dashboard
                                                                                      (explica, não classifica)
```
| Componente | Arquivo | Função |
|---|---|---|
| ML | `train.py` | Pipeline + GridSearchCV (5-fold) otimizando **F1 da classe maligna** |
| Modelo principal | `engine.py` | Regressão Logística, escolhida pelo melhor F1 da classe maligna na validação cruzada |
| Comparação experimental | `engine.py` | Média simples dos 3 modelos, exibida sem participar da classificação principal |
| Explicabilidade | `engine.explain` | Oclusão: variação do log-odds de malignidade do modelo principal ao trocar cada atributo pela média do desenvolvimento |
| Motor de decisão | `engine.analyze` | Risco (30/70%), confiança, **inconclusivo** |
| IA generativa | `genai.py` | Recebe só um JSON estruturado e redige a explicação |
| Web | `app.py`, `static/` | API Flask, dashboard, histórico SQLite |

## Correções metodológicas aplicadas
1. **Classe positiva**: no scikit-learn `target 0 = maligno`. O GridSearch usa `make_scorer(f1_score, pos_label=0)` e todas as métricas (precisão, recall, F1, AUC) são da classe maligna.
2. **Sem vazamento**: `Pipeline([StandardScaler, modelo])` dentro do GridSearchCV (parâmetros `model__*`); o scaler é reajustado a cada fold.
3. **Uso dos dados**: teste (20%) isolado; treino+validação (80%) usados na validação cruzada e no ajuste final (`refit`).
4. **Modelo principal explícito**: a Regressão Logística é usada na classificação porque obteve o melhor F1 da classe maligna na validação cruzada (0,976 contra 0,951 do Random Forest e 0,904 da Árvore), sem usar o teste na escolha; os três modelos continuam disponíveis para comparação.
5. **Explicação alinhada**: a oclusão explica o modelo principal, em log-odds, para não saturar em casos de probabilidade próxima de 0% ou 100%. Atributos correlacionados podem dividir o efeito, então a oclusão unitária pode subestimar contribuições; são contribuições matemáticas, não causas médicas.
6. **Separação metodológica**: teste isolado; estatísticas e exemplos da interface são derivados somente do desenvolvimento.
7. **Endurecimento**: `debug` desligado por padrão, validação numérica (finito, ≥ 0), limite de corpo, rótulo truncado a 40 caracteres, escape de HTML no front, caminhos independentes do diretório atual e versões fixadas.
7. **Testes**: 23 (motor, endpoints, entradas inválidas, histórico, IA generativa com chamadas simuladas).

## IA generativa (responsabilidades)
- **ML** classifica e calcula estimativas; **IA generativa** só redige a explicação. Não reclassifica.
- Envia ao LLM apenas: classificação, estimativa, risco, confiança, motivos, estimativas por modelo e 5 nomes de atributos. Nunca rótulo/identificação nem valores brutos do exame.
- Prompt restritivo (sem diagnóstico, tratamento ou prognóstico; ignora instruções dentro do JSON). A saída é validada (≤120 palavras, termos proibidos, aviso obrigatório); se reprovada, a API falhar ou não houver chave, usa-se texto-modelo local. A origem do texto aparece no dashboard.
- Variável opcional `SIACM_LLM_MODEL` troca o modelo (padrão `claude-sonnet-5-5`).

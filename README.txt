# Diagnóstico de Câncer de Mama com Machine Learning

**Dupla:** Beatriz Alves Gava e Wallace Miranda Senna da Silva

## Visão Geral
Este projeto aplica três algoritmos de classificação para prever se um tumor de mama é benigno ou maligno, utilizando o dataset Wisconsin Breast Cancer (scikit-learn).

## Requisitos atendidos
- [x] Definição de problema real (classificação médica)
- [x] Dados públicos (UCI)
- [x] Pré-processamento: padronização, divisão treino/validação/teste (60/20/20)
- [x] Dois+ algoritmos (Regressão Logística, Árvore de Decisão, Random Forest)
- [x] Grid Search com validação cruzada para ajuste de hiperparâmetros
- [x] Métricas: Acurácia, Precisão, Recall, F1-Score, AUC
- [x] Comparação e discussão dos resultados

## Como reproduzir
1. Clone o repositório  
2. Instale as dependências: `pip install -r requirements.txt`  
3. Execute o notebook `trabalho_ml.ipynb` (célula por célula)  
   ou rode o script `python trabalho_ml.py`

## Resultados (conjunto de teste)
| Modelo              | Acurácia | Precisão | Recall | F1-Score | AUC   |
|---------------------|----------|----------|--------|----------|-------|
| Regressão Logística | 0.9737   | 0.9726   | 0.9861 | 0.9793   | 0.9976|
| Árvore de Decisão   | 0.9474   | 0.9444   | 0.9722 | 0.9581   | 0.9789|
| Random Forest       | 0.9649   | 0.9722   | 0.9722 | 0.9722   | 0.9958|

## Gráficos e matrizes de confusão
Disponíveis no repositório.

## Link do repositório
https://github.com/beatrizgava/ml-breast-cancer-diagnosis
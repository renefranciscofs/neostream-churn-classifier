---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - churn
  - logistic-regression
  - tabular
  - pt-br
  - scikit-learn
  - pipeline
metrics:
  - accuracy: 0.5896
  - f1: 0.35
  - recall: 0.64
---

# Neostream Churn Classifier (v2)

Pipeline de regressão logística para previsão de churn de assinantes de um serviço de streaming fictício (NeoStream) — projeto acadêmico da disciplina de Machine Learning (Instituto Infnet, DR3).

## Sobre o modelo

Pipeline único do Scikit-learn que recebe dados crus e retorna a probabilidade de churn: `ColumnTransformer` (StandardScaler nas features numéricas + OneHotEncoder com `handle_unknown="ignore"` nas categóricas) + `LogisticRegression` com `class_weight="balanced"`.

**Por que v2?** A versão original (v1) otimizava acurácia — com ~17,5% de churn, o modelo aprendeu a prever "não-churn" quase sempre (**recall de 1%** na classe minoritária, inútil para decisão de retenção). O v2 muda o objetivo de otimização para F1 e equilibra o custo das classes:

| Versão | Otimização | Acurácia (teste) | Precisão (churn) | **Recall (churn)** | F1 (churn) |
|---|---|---|---|---|---|
| v1 | acurácia | 0,825 | 0,56 | **0,01** | 0,02 |
| **v2** | **F1 + class_weight balanced** | 0,590 | 0,24 | **0,64** | **0,35** |

*Split estratificado 80/20 (random_state=42), 9.600 treino / 2.400 teste, GridSearchCV 5-fold para C ∈ {0.01, 0.1, 1, 10, 100}.*

## Como usar

```python
import joblib
import pandas as pd

modelo = joblib.load("pipeline_final_churn_v2.joblib")

# dados crus exatamente como na base de treino
novo_assinante = pd.DataFrame([{
    "tempo_assinatura_meses": 8,
    "uso_medio_semanal_horas": 3.5,
    "numero_reclamacoes": 4,
    "plano": "Premium",
    "metodo_pagamento": "Pix",
    "dispositivo_principal": "Celular",
}])

prob_churn = modelo.predict_proba(novo_assinante)[:, 1]
classe = modelo.predict(novo_assinante)
print(f"Probabilidade de churn: {prob_churn[0]:.2%} | Classe: {classe[0]}")
```

## Dados de treinamento

- **Fonte:** base sintética `assinantes_neostream.csv` (exercício acadêmico; sem dados pessoais reais)
- **Tamanho:** 12.000 assinantes | **Churn:** ~17,5%
- **Features:** `tempo_assinatura_meses`, `uso_medio_semanal_horas`, `numero_reclamacoes` (numéricas); `plano`, `metodo_pagamento`, `dispositivo_principal` (categóricas)

## Decisões de design

1. **Pipeline único:** pré-processamento serializado junto com o modelo — produção chama `predict` sobre dados crus, eliminando risco de data leakage e divergência entre ambientes.
2. **`handle_unknown="ignore"`:** categorias inéditas em produção não quebram a inferência.
3. **Estratificação no split:** garante a mesma proporção de churn em treino e teste.
4. **`class_weight="balanced"` + `scoring="f1"`:** o desbalanceamento é tratado no custo do modelo, não na base.

## Limitações

- Dados sintéticos — métricas não são transferíveis para bases reais.
- Precisão da classe churn (0,24) ainda é baixa: para uso real, calibrar o threshold de decisão via `precision_recall_curve` e priorizar recall ou precisão conforme o custo de retenção.
- Métricas de um único split com seed fixo; intervalos de confiança exigiriam validação cruzada aninhada ou múltiplas seeds.

## Autor

Rene Francisco Ferreira Santos — estudante de Engenharia de Software (Infnet) e Ciência de Dados (UVV).
Repositório com o código completo: [github.com](https://github.com/SEU_USERNAME/neostream-churn-classifier)

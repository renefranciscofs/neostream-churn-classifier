# Pipeline de Churn — NeoStream (DR3)

Pipeline de classificação de churn para assinantes de um serviço de streaming fictício (NeoStream) — projeto da disciplina de Machine Learning (Instituto Infnet). O pipeline recebe dados crus e entrega a probabilidade de churn com um único artefato serializado, pronto para produção.

**Modelo publicado no Hugging Face:** [huggingface.co/SEU_USERNAME/neostream-churn-classifier](https://huggingface.co/SEU_USERNAME/neostream-churn-classifier)

## Resultados

| Versão | Otimização | Acurácia (teste) | Precisão (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|---|
| v1 | acurácia | 0,825 | 0,56 | 0,01 | 0,02 |
| v2 | F1 + `class_weight="balanced"` | 0,590 | 0,24 | **0,64** | **0,35** |

Split estratificado 80/20 (seed 42) · GridSearchCV 5-fold · C ∈ {0.01, 0.1, 1, 10, 100}

O v1 mostra a armadilha clássica do desbalanceamento: otimizar acurácia com 17,5% de churn produz um modelo que prevê "não-churn" quase sempre (recall de 1% — inútil para retenção). O v2 muda o objetivo de otimização para F1 e equilibra o custo das classes: recall 64%.

## Estrutura

```
├── assinantes_neostream.csv        # base sintética (12.000 assinantes)
├── pipeline_final_churn.py         # v1 — otimiza acurácia
├── pipeline_final_churn_v2.py      # v2 — class_weight balanced + scoring f1
├── preprocessamento_heterogeneo.py # análise/experimentos de preprocessamento
├── pipeline_classificacao.py       # primeiros experimentos de classificação
├── renefrancisco_ferreirasantos_DR3_TP1.ipynb  # notebook do TP
├── model_card.md                   # card do Hugging Face
└── hf_push.py                      # script de publicação no Hub
```

## Como rodar

```bash
cd DR3_TP1
python3 -m venv ../.venv           # se ainda não existir
../.venv/bin/pip install scikit-learn joblib pandas
../.venv/bin/python pipeline_final_churn_v2.py   # treina e serializa
../.venv/bin/python -c "
import joblib, pandas as pd
m = joblib.load('pipeline_final_churn_v2.joblib')
print(m.predict_proba(pd.DataFrame([{
    'tempo_assinatura_meses': 8, 'uso_medio_semanal_horas': 3.5,
    'numero_reclamacoes': 4, 'plano': 'Premium',
    'metodo_pagamento': 'Pix', 'dispositivo_principal': 'Celular'}]))[:, 1])
"
```

## Decisões de design

1. **Pipeline único (sklearn):** pré-processamento serializado junto com o modelo — produção chama `predict` sobre dados crus, eliminando risco de data leakage e divergência entre ambientes.
2. **`handle_unknown="ignore"`:** categorias inéditas em produção não quebram a inferência.
3. **Estratificação no split:** mesma proporção de churn em treino e teste.
4. **Desbalanceamento tratado no custo, não na base:** `class_weight="balanced"` + `scoring="f1"`.

## Limitações

- Dados sintéticos — métricas não são transferíveis para bases reais.
- Precisão da classe churn (0,24) ainda é baixa: em produção, calibrar o threshold via `precision_recall_curve` conforme o custo de retenção.
- Métricas de um único split; para intervalos de confiança seriam necessárias múltiplas seeds.

## Stack

Python · scikit-learn · pandas · numpy · joblib

"""Pré-processamento heterogêneo da base de assinantes da NeoStream.

Aplica StandardScaler às features numéricas e OneHotEncoder às categóricas
em um único objeto ColumnTransformer, reutilizável no treino e em produção.

Notas de design:
1. remainder="drop" descarta cliente_id (identificador sem sinal preditivo)
   e ltv_alto (rótulo do TP2, não é feature legítima para prever churn).
2. handle_unknown="ignore" faz o encoder ignorar categorias inéditas que
   surjam em produção, em vez de falhar.
3. sparse_threshold=0 força saída densa, simplificando inspeção e consumo.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNAS_NUMERICAS = [
    "tempo_assinatura_meses",
    "uso_medio_semanal_horas",
    "numero_reclamacoes",
]
COLUNAS_CATEGORICAS = ["plano", "metodo_pagamento", "dispositivo_principal"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), COLUNAS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
    ],
    remainder="drop",
    sparse_threshold=0,
    verbose_feature_names_out=False,
)

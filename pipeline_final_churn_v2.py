"""Pipeline v2 de classificação de churn da NeoStream.

Evolução do pipeline_final_churn.py (v1). O v1 otimizava a acurácia — com
~17,5% de churn, o modelo aprendeu a prever "não-churn" quase sempre
(recall de 1% na classe minoritária, inútil para a decisão de retenção).

MUDANÇAS DO V2
1. class_weight="balanced": o erro nas duas classes passa a ter o mesmo
   custo, compensando o desbalanceamento sem alterar a base de dados.
2. scoring="f1" no GridSearchCV: o hiperparâmetro C passa a ser escolhido
   pela métrica que importa para o negócio (equilíbrio precisão × recall
   na classe churn), não pela acurácia global.

Todo o restante — ColumnTransformer heterogêneo, estratificação,
handle_unknown="ignore", artefato único serializado — é idêntico ao v1.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAMINHO_DADOS = "assinantes_neostream.csv"
CAMINHO_MODELO = "pipeline_final_churn_v2.joblib"
COLUNAS_NUMERICAS = [
    "tempo_assinatura_meses",
    "uso_medio_semanal_horas",
    "numero_reclamacoes",
]
COLUNAS_CATEGORICAS = ["plano", "metodo_pagamento", "dispositivo_principal"]
RANDOM_STATE = 42


def carregar_dados(caminho):
    return pd.read_csv(caminho)


def construir_preprocessador():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUNAS_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def construir_pipeline(preprocessador):
    return Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            (
                "classificador",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def buscar_melhor_c(pipeline, X_train, y_train):
    grade = {"classificador__C": [0.01, 0.1, 1, 10, 100]}
    busca = GridSearchCV(
        pipeline, param_grid=grade, cv=5, scoring="f1", n_jobs=-1
    )
    busca.fit(X_train, y_train)
    return busca


def main():
    df = carregar_dados(CAMINHO_DADOS)
    X = df.drop(columns="churn")
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")
    print(f"Churn no treino: {y_train.mean():.2%} | Churn no teste: {y_test.mean():.2%}")

    pipeline = construir_pipeline(construir_preprocessador())

    busca = buscar_melhor_c(pipeline, X_train, y_train)
    print(f"Melhor C: {busca.best_params_['classificador__C']}")
    print(f"F1 médio na validação cruzada: {busca.best_score_:.4f}")

    modelo_final = busca.best_estimator_
    y_pred = modelo_final.predict(X_test)
    print(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(modelo_final, CAMINHO_MODELO)
    print(f"Modelo serializado em {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()

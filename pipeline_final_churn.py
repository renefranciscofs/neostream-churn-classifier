"""Pipeline final de classificação de churn da NeoStream.

Script único e autocontido: carrega a base bruta assinantes_neostream.csv,
divide os dados de forma estratificada, otimiza o hiperparâmetro C do
LogisticRegression por GridSearchCV e entrega um Pipeline treinado e
serializado, pronto para receber novos dados em produção.

DECISÕES DE DESIGN
1. Divisão estratificada (stratify=y): o churn representa cerca de 18% da
   base. Um sorteio puramente aleatório poderia deixar treino e teste com
   proporções diferentes da classe alvo, enviesando o treinamento ou a
   avaliação. A estratificação garante a mesma proporção de churn nos dois
   conjuntos.
2. ColumnTransformer: a base mistura features numéricas e categóricas.
   Um único objeto aplica StandardScaler às numéricas e OneHotEncoder às
   categóricas, sem nenhum passo manual fora do Pipeline. O argumento
   handle_unknown="ignore" torna o encoder tolerante a categorias inéditas
   em produção, e remainder="drop" descarta cliente_id (sem sinal preditivo)
   e ltv_alto (rótulo do TP2, não é feature de churn).
3. Pipeline em vez de passos manuais: fit e predict passam a ser chamadas
   únicas sobre dados crus. Isso elimina a possibilidade de esquecer de
   transformar um conjunto, de ajustar um transformador no conjunto errado
   ou de divergir os parâmetros de pré-processamento entre treino e
   produção.
4. GridSearchCV com cv=5: a busca do melhor C é sistemática e barata
   (5 valores x 5 folds), compatível com o orçamento computacional limitado.
   Com refit=True (padrão), o GridSearchCV reajusta o melhor modelo em todo
   o X_train, então busca.best_estimator_ já é o modelo final a ser
   serializado.
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
CAMINHO_MODELO = "pipeline_final_churn.joblib"
COLUNAS_NUMERICAS = [
    "tempo_assinatura_meses",
    "uso_medio_semanal_horas",
    "numero_reclamacoes",
]
COLUNAS_CATEGORICAS = ["plano", "metodo_pagamento", "dispositivo_principal"]
RANDOM_STATE = 42


def carregar_dados(caminho):
    """Carrega a base bruta em um DataFrame."""
    return pd.read_csv(caminho)


def construir_preprocessador():
    """Monta o pré-processamento heterogêneo em um único objeto."""
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
    """Encapsula pré-processamento e classificador em um Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            (
                "classificador",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def buscar_melhor_c(pipeline, X_train, y_train):
    """Otimiza o C por validação cruzada de 5 folds em uma grade fixa."""
    grade = {"classificador__C": [0.01, 0.1, 1, 10, 100]}
    busca = GridSearchCV(
        pipeline, param_grid=grade, cv=5, scoring="accuracy", n_jobs=-1
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
    print(f"Acurácia média na validação cruzada: {busca.best_score_:.4f}")

    # refit=True (padrão do GridSearchCV): o melhor estimador já foi
    # reajustado com todo o X_train, então é o modelo final
    modelo_final = busca.best_estimator_
    y_pred = modelo_final.predict(X_test)
    print(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(modelo_final, CAMINHO_MODELO)
    print(f"Modelo serializado em {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()

# PARÁGRAFO FINAL: POR QUE ESTA ORGANIZAÇÃO É MAIS SEGURA PARA PRODUÇÃO
# Transformar os dados manualmente antes de treinar exige repetir os mesmos
# passos, na mesma ordem e com os mesmos parâmetros, toda vez que um lote
# novo chega para previsão. Qualquer esquecimento ou divergência (um scaler
# reajustado em dados novos, um encoder treinado em outro conjunto de
# categorias, uma coluna fora de ordem) muda silenciosamente o significado
# das features e corrompe as previsões. O Pipeline torna o pré-processamento
# parte do próprio modelo: tudo o que foi aprendido no treino fica
# serializado junto com o classificador em um único artefato, e a produção
# só precisa chamar predict sobre dados crus. Assim, o caminho percorrido
# pelos dados em produção é exatamente o mesmo do treino, o que elimina por
# construção o risco de data leakage e de inconsistência entre ambientes.

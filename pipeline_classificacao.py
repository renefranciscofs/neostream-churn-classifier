"""Pipeline de classificação de churn da NeoStream.

Encapsula o pré-processamento heterogêneo (ColumnTransformer do Exercício 4)
e o classificador LogisticRegression em um único objeto treinável, aplicável
a dados crus tanto no treino quanto em produção.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessamento_heterogeneo import preprocessor

pipeline = Pipeline(
    steps=[
        ("preprocessamento", preprocessor),
        ("classificador", LogisticRegression(max_iter=2000, random_state=42)),
    ]
)

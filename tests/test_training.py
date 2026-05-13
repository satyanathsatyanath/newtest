from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors


def test_model_training(spark):

    data = [
        (0.0, Vectors.dense([0.0, 1.0])),
        (1.0, Vectors.dense([1.0, 0.0]))
    ]

    df = spark.createDataFrame(
        data,
        ["label", "features"]
    )

    lr = LogisticRegression()

    model = lr.fit(df)

    assert model is not None
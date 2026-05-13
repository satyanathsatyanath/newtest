from pyspark.ml.feature import VectorAssembler


def test_feature_vector(spark):

    data = [
        (1, 50.0, 100.0)
    ]

    columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df = spark.createDataFrame(data, columns)

    assembler = VectorAssembler(
        inputCols=columns,
        outputCol="features"
    )

    transformed_df = assembler.transform(df)

    assert "features" in transformed_df.columns
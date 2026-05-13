def test_data_loading(spark):

    df = spark.read.csv(
        "/opt/spark-data/customer_churn.csv",
        header=True,
        inferSchema=True
    )

    assert df.count() > 0
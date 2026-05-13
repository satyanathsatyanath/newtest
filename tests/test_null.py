from pyspark.sql.functions import col


def test_no_null_churn(spark):

    df = spark.read.csv(
        "app/customer_churn.csv",
        header=True,
        inferSchema=True
    )

    null_count = df.filter(
        col("Churn").isNull()
    ).count()

    assert null_count == 0
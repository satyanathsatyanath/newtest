def test_required_columns(spark):

    df = spark.read.csv(
        "app/customer_churn.csv",
        header=True,
        inferSchema=True
    )

    required_columns = [
        "gender",
        "tenure",
        "MonthlyCharges",
        "Churn"
    ]

    for col_name in required_columns:
        assert col_name in df.columns
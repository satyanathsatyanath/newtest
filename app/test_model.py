from pyspark.sql import SparkSession

from pyspark.ml import PipelineModel


# =========================================
# CREATE SPARK SESSION
# =========================================

spark = SparkSession.builder \
    .appName("ChurnModelTesting") \
    .master("spark://spark-master:7077") \
    .getOrCreate()


# =========================================
# LOAD SAVED MODEL
# =========================================

model = PipelineModel.load(
    "/opt/spark-data/churn_model"
)

print("\nMODEL LOADED SUCCESSFULLY")


# =========================================
# CREATE TEST DATA
# =========================================

sample_data = [
    (
        "Female",
        0,
        12,
        90.5,
        1200.7,
        "Month-to-month",
        "Electronic check"
    ),

    (
        "Male",
        0,
        72,
        45.2,
        5000.3,
        "Two year",
        "Credit card (automatic)"
    )
]


columns = [
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod"
]


test_df = spark.createDataFrame(
    sample_data,
    columns
)

print("\nTEST DATA")
test_df.show()


# =========================================
# RUN PREDICTIONS
# =========================================

predictions = model.transform(test_df)


# =========================================
# SHOW RESULTS
# =========================================

print("\nPREDICTIONS")

predictions.select(
    "gender",
    "tenure",
    "MonthlyCharges",
    "prediction",
    "probability"
).show(truncate=False)


# =========================================
# INTERPRET RESULTS
# =========================================

print("\nINTERPRETED RESULTS")

results = predictions.collect()

for row in results:

    churn_prediction = (
        "CHURN"
        if row["prediction"] == 1.0
        else "NO CHURN"
    )

    probability = max(row["probability"])

    print(
        f"""
Customer:
  Gender: {row['gender']}
  Tenure: {row['tenure']}
  MonthlyCharges: {row['MonthlyCharges']}

Prediction:
  {churn_prediction}

Confidence:
  {round(probability * 100, 2)}%
"""
    )


# =========================================
# STOP SPARK
# =========================================

spark.stop()
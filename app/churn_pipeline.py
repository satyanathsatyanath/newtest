from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler
)

from pyspark.ml.classification import LogisticRegression

from pyspark.ml import Pipeline

from pyspark.ml.evaluation import BinaryClassificationEvaluator


# =========================================
# CREATE SPARK SESSION
# =========================================

spark = SparkSession.builder \
    .appName("CustomerChurnPrediction") \
    .master("spark://spark-master:7077") \
    .getOrCreate()


# =========================================
# LOAD DATA
# =========================================

df = spark.read.csv(
    "/opt/spark-data/customer_churn.csv",
    header=True,
    inferSchema=True
)

print("\nDATA SAMPLE")
df.show(5)

print("\nSCHEMA")
df.printSchema()


# =========================================
# SELECT IMPORTANT COLUMNS
# =========================================

df = df.select(
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "Churn"
)


# =========================================
# CLEAN DATA
# =========================================

df = df.na.drop()

df = df.withColumn(
    "TotalCharges",
    col("TotalCharges").cast("double")
)

df = df.na.drop()


# =========================================
# CONVERT TARGET COLUMN
# =========================================

label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label"
)


# =========================================
# ENCODE CATEGORICAL FEATURES
# =========================================

gender_indexer = StringIndexer(
    inputCol="gender",
    outputCol="gender_index"
)

contract_indexer = StringIndexer(
    inputCol="Contract",
    outputCol="contract_index"
)

payment_indexer = StringIndexer(
    inputCol="PaymentMethod",
    outputCol="payment_index"
)


# =========================================
# FEATURE ASSEMBLER
# =========================================

assembler = VectorAssembler(
    inputCols=[
        "gender_index",
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "contract_index",
        "payment_index"
    ],
    outputCol="features"
)


# =========================================
# MODEL
# =========================================

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)


# =========================================
# PIPELINE
# =========================================

pipeline = Pipeline(stages=[
    label_indexer,
    gender_indexer,
    contract_indexer,
    payment_indexer,
    assembler,
    lr
])


# =========================================
# TRAIN TEST SPLIT
# =========================================

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)


# =========================================
# TRAIN MODEL
# =========================================

model = pipeline.fit(train_df)


# =========================================
# PREDICTIONS
# =========================================

predictions = model.transform(test_df)

print("\nPREDICTIONS")
predictions.select(
    "features",
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)


# =========================================
# EVALUATION
# =========================================

evaluator = BinaryClassificationEvaluator(
    labelCol="label"
)

auc = evaluator.evaluate(predictions)

print(f"\nAUC SCORE: {auc}")


# =========================================
# SAVE MODEL
# =========================================

model.write().overwrite().save(
    "/opt/spark-data/churn_model"
)

print("\nMODEL SAVED SUCCESSFULLY")


# =========================================
# STOP SPARK
# =========================================

spark.stop()
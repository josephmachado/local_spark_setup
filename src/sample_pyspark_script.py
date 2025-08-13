#!/usr/bin/env python3
"""
Simple PySpark Sample Script
Demonstrates basic Spark operations with sample data
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def create_spark_session():
    """Create and configure Spark session with Iceberg support"""
    spark = (
        SparkSession.builder.appName("PySpark Sample Application")
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config("spark.sql.catalog.local.warehouse", "/home/iceberg/warehouse")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")  # Reduce log verbosity
    return spark


def create_sample_data(spark):
    """Create sample employee data"""

    # Define schema
    schema = StructType(
        [
            StructField("employee_id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("department", StringType(), True),
            StructField("salary", DoubleType(), True),
            StructField("experience_years", IntegerType(), True),
            StructField("hire_date", StringType(), True),
        ]
    )

    # Sample data
    data = [
        (1, "Alice Johnson", "Engineering", 85000.0, 3, "2021-03-15"),
        (2, "Bob Smith", "Marketing", 65000.0, 2, "2022-01-10"),
        (3, "Carol Williams", "Engineering", 95000.0, 5, "2019-08-20"),
        (4, "David Brown", "Sales", 55000.0, 1, "2023-02-28"),
        (5, "Eva Martinez", "Engineering", 78000.0, 2, "2022-06-01"),
        (6, "Frank Wilson", "Marketing", 72000.0, 4, "2020-11-12"),
        (7, "Grace Lee", "Sales", 62000.0, 3, "2021-09-05"),
        (8, "Henry Davis", "Engineering", 88000.0, 4, "2020-04-18"),
        (9, "Ivy Chen", "Marketing", 68000.0, 2, "2022-03-22"),
        (10, "Jack Thompson", "Sales", 58000.0, 1, "2023-01-15"),
    ]

    # Create DataFrame
    df = spark.createDataFrame(data, schema)

    # Convert hire_date string to date
    df = df.withColumn("hire_date", col("hire_date").cast(DateType()))

    return df


def analyze_data(df):
    """Perform various data analysis operations"""

    print("=" * 50)
    print("PYSPARK SAMPLE DATA ANALYSIS")
    print("=" * 50)

    # 1. Experience-based categorization
    print("\n1. Employees by Experience Level:")
    experience_df = df.withColumn(
        "experience_level",
        when(col("experience_years") <= 1, "Junior")
        .when(col("experience_years") <= 3, "Mid-level")
        .otherwise("Senior"),
    )
    print(experience_df.show(10, False))

    return experience_df


def save_to_iceberg(spark, df):
    print("\n2. Saving to Iceberg Table:")

    try:
        # Create database if not exists
        spark.sql("CREATE DATABASE IF NOT EXISTS sample_db")

        # Write to Iceberg table
        df.write.format("iceberg").mode("overwrite").saveAsTable(
            "local.sample_db.employees"
        )

        print("✅ Data successfully saved to Iceberg table: sample_db.employees")

        # Verify the save
        print("\nReading back from Iceberg table:")
        iceberg_df = spark.table("local.sample_db.employees")
        print(f"Records in Iceberg table: {iceberg_df.count()}")

    except Exception as e:
        print(f"❌ Error saving to Iceberg: {str(e)}")
        print("Note: This is normal if Iceberg is not fully configured")


def main():
    """Main function"""
    # Create Spark session
    spark = create_spark_session()

    try:
        print("🚀 Starting PySpark Sample Application...")

        # Create sample data
        df = create_sample_data(spark)

        # Analyze the data
        processed_df = analyze_data(df)

        # Try to save to Iceberg (optional)
        save_to_iceberg(spark, processed_df)

        print("\n✅ Sample script completed successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    finally:
        # Stop Spark session
        spark.stop()
        print("🛑 Spark session stopped")


if __name__ == "__main__":
    main()

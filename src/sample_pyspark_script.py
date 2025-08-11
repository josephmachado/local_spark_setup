#!/usr/bin/env python3
"""
Simple PySpark Sample Script
Demonstrates basic Spark operations with sample data
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, max, min, when
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
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.iceberg.spark.SparkSessionCatalog",
        )
        .config("spark.sql.catalog.spark_catalog.type", "hive")
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

    # 1. Show the data
    print("\n1. Sample Employee Data:")
    df.show()

    # 2. Basic statistics
    print("\n2. Dataset Info:")
    print(f"Total records: {df.count()}")
    print(f"Total columns: {len(df.columns)}")
    df.printSchema()

    # 3. Department analysis
    print("\n3. Employees by Department:")
    dept_analysis = (
        df.groupBy("department")
        .agg(
            count("*").alias("employee_count"),
            avg("salary").alias("avg_salary"),
            max("salary").alias("max_salary"),
            min("salary").alias("min_salary"),
        )
        .orderBy(desc("employee_count"))
    )

    dept_analysis.show()

    # 4. Salary analysis
    print("\n4. Salary Analysis:")
    salary_stats = df.select(
        avg("salary").alias("average_salary"),
        max("salary").alias("highest_salary"),
        min("salary").alias("lowest_salary"),
    )
    salary_stats.show()

    # 5. Experience-based categorization
    print("\n5. Employees by Experience Level:")
    experience_df = df.withColumn(
        "experience_level",
        when(col("experience_years") <= 1, "Junior")
        .when(col("experience_years") <= 3, "Mid-level")
        .otherwise("Senior"),
    )

    experience_analysis = (
        experience_df.groupBy("experience_level")
        .agg(count("*").alias("count"), avg("salary").alias("avg_salary"))
        .orderBy("avg_salary")
    )

    experience_analysis.show()

    # 6. High earners
    print("\n6. High Earners (Salary > $70,000):")
    high_earners = (
        df.filter(col("salary") > 70000)
        .select("name", "department", "salary", "experience_years")
        .orderBy(desc("salary"))
    )

    high_earners.show()

    return experience_df


def save_to_iceberg(spark, df):
    print("\n7. Saving to Iceberg Table:")

    try:
        # Create database if not exists
        spark.sql("CREATE DATABASE IF NOT EXISTS sample_db")

        # Write to Iceberg table
        df.write.format("iceberg").mode("overwrite").saveAsTable(
            "local.sample_db.employees"
        )

        print("✅ Data successfully saved to Iceberg table: local.sample_db.employees")

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

    """
    pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version (3, 10) than that in driver 3.9, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.
"""

import pyspark
from config.db_configs import db_config
from pyspark.sql import SparkSession

# Build JDBC URL from db_config
db_url = f"jdbc:postgresql://{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("PostgresConnectionTest") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.2.27") \
    .getOrCreate()

try:
    # Test connection by reading a simple query
    df = spark.read.format("jdbc") \
        .option("url", db_url) \
        .option("dbtable", "(SELECT 1 AS test_col) AS test_table") \
        .option("user", db_config['user']) \
        .option("password", db_config['password']) \
        .option("driver", "org.postgresql.Driver") \
        .load()
    df.show()
    print("Connection to PostgreSQL via PySpark is successful!")
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    spark.stop()


########jessica_koktail_local_spark_building.py########
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()
########jessica_koktail_local_spark_building.py########

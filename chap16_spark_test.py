import findspark
import random

findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import json


conf = SparkConf()
conf.setAppName('dispel4py')
conf.set("spark.storage.memoryFraction", "0.5")
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
rows = sqlContext.read.load("/home/dev/hadoop/data_frame/TEST2", "parquet")
tbl = rows.registerTempTable("TEST2")
result = sqlContext.sql("select * from TEST2").toJSON(False).map(lambda x : x).collect()

print(result)

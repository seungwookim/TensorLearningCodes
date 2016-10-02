import findspark
import random

findspark.init()

import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import json
from pyspark.sql import SQLContext, DataFrameWriter, DataFrame, HiveContext
from boto.s3.connection import S3Connection
from pyspark.sql import *

conf = SparkConf()
conf.setMaster('spark://fb493782e0e9:7077')
conf.setAppName('dispel4py')
# conf.set("spark.storage.memoryFraction", "0.5")

conf.set("aws-java-sdk-1.7.4.jar", "/home/dev/spark/jars/aws-java-sdk-1.7.4.jar")
conf.set("hadoop-aws-2.7.1.jar", "/home/dev/spark/jars/hadoop-aws-2.7.1.jar")
conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("spark.hadoop.fs.s3a.access.key", "AKIAJRIB7FPFX62MOLEA")
conf.set("spark.hadoop.fs.s3a.secret.key", "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")
conf.set("aws_access_key_id", "AKIAJRIB7FPFX62MOLEA")
conf.set("aws_secret_access_key", "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")
conf.set("fs.s3a.awsAccessKeyId", "AKIAJRIB7FPFX62MOLEA")
conf.set("fs.s3a.awsSecretAccessKey", "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")
conf.set("aws.accessKeyId", "AKIAJRIB7FPFX62MOLEA")
conf.set("aws.secretKey", "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")
conf.set("AWS_ACCESS_KEY_ID", "AKIAJRIB7FPFX62MOLEA")
conf.set("AWS_SECRET_ACCESS_KEY", "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")

sc = SparkContext(conf=conf)
sc._jsc.hadoopConfiguration().set("fs.s3a.fast.upload", "true")
sc._jsc.hadoopConfiguration().set("fs.s3a.buffer.dir", "/root/spark/work,/tmp")
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "/root/spark/work,/tmp")

# sc._jsc.hadoopConfiguration().set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
# sc._jsc.hadoopConfiguration().set("fs.s3a.awsAccessKeyId", "AKIAJRIB7FPFX62MOLEA")
# sc._jsc.hadoopConfiguration().set("fs.s3a.awsSecretAccessKey", "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")
# sc._jsc.hadoopConfiguration().set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
# sc._jsc.hadoopConfiguration().set("spark.hadoop.fs.s3n.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
# sc._jsc.hadoopConfiguration().set("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3.S3FileSystem")
# sc._jsc.hadoopConfiguration().set("fs.s3a.awsAccessKeyId", "AKIAJRIB7FPFX62MOLEA")
# sc._jsc.hadoopConfiguration().set("fs.s3a.awsSecretAccessKey", "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")
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", "AKIAJRIB7FPFX62MOLEA")
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", "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")
# sc._jsc.hadoopConfiguration().set("fs.s3.awsAccessKeyId", "AKIAJRIB7FPFX62MOLEA")
# sc._jsc.hadoopConfiguration().set("fs.s3.awsSecretAccessKey", "spark.hadoop.fs.s3a.secret.key")
os.environ['S3_USE_SIGV4'] = 'True'
AWS_KEY = 'AKIAJRIB7FPFX62MOLEA'
AWS_SECRET = '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'
conn = S3Connection(AWS_KEY, AWS_SECRET, host ='s3.amazonaws.com')
file_name = 's3a://s3testworks/world_bank.json'
rdd = sc.textFile(file_name)
print rdd.count()


buckets = conn.get_all_buckets()
print(buckets)
print(conn.get_bucket('s3testworks'))

sqlContext = HiveContext(sc)
# bucket = "s3testworks"
# prefix = "world_bank.json"
# filename = "s3a://{}/{}".format(bucket, prefix)
#
# df_load = sc.textFile(filename)
#
# print(df_load)
# print(df_load.collect())

## path to S3 bucket containing my files
path = "s3a://s3testworks/*"

## get those fields we need to create the schema. file is tab delimited
lines = sc.textFile(path)
parts = lines.map(lambda l: l.split("\t"))
weblogs_hit = parts.map(lambda p: Row(url=p[12], city=p[49], country = p[50], state = p[52]))

## create DataFrame
schema_weblogs_hit = sqlContext.createDataFrame(weblogs_hit)

## register DataFrame as a temporary table
schema_weblogs_hit.registerTempTable("weblogs_hit")

## RANK pageview count by geographic location - which areas generate the most traffic in terms of page views
rows = sqlContext.sql("SELECT m.location, m.page_view_count, RANK() OVER (ORDER BY m.page_view_count DESC) AS ranking FROM (SELECT CONCAT(UPPER(city),',',UPPER(country),',',UPPER(state)) AS location, count(1) AS page_view_count FROM weblogs_hit GROUP BY city, country, state ORDER BY page_view_count) m LIMIT 10")

## run SQL command and display output
output = rows.collect()
for row in output:
  row = str(row)
  print "%s" % (row)



#
# buckets = conn.get_all_buckets()
#
# for b in buckets:
#     print(b.name)
#     boc = conn.get_bucket(b.name)
#     print(boc)
#
#
# print(conn.lookup('tensormsa'))
# bucket = conn.get_bucket(buckets[0])
#
# print len(list(bucket.list()))
# for key in bucket.list():
#     file_name = 's3n://tensormsa/'+key.name
#     print file_name
#     rdd = sc.textFile(file_name
#         ,'org.apache.hadoop.mapred.TextInputFormat',
#         'org.apache.hadoop.io.Text',
#         'org.apache.hadoop.io.LongWritable'
#         )
#     print rdd.count()
#
#
# my_rdd = sc.parallelize(xrange(100))
# print my_rdd.collect()
#
#
# bucket = "s3testworks"
# prefix = "world_bank.json"
# filename = "s3n://{}/{}".format(bucket, prefix)
#
# df_load = sc.textFile(filename)
#
# print(df_load)
# #print(df_load.collect())
#
# sqlContext = SQLContext(sc)
# df = sqlContext.read.json("/home/dev/hadoop/world_bank.json")
# df.write.save("s3n://s3testworks/test.csv", format="csv")





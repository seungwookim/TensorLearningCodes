import json, requests, textwrap



# server connection info
# Docker Container Id, Public DNS aloso works
host = "http://481bf68ee6d9:8998"
headers = {'Content-Type': 'application/json'}

def test_case1():
    """
    create session, get session id form return, run long code with that session
    :return:
    """
    data = {'kind': 'pyspark',
            "name": "Hive Test",
            "executorCores": 1,
            "executorMemory": "512m",
            "driverCores": 1,
            "driverMemory": "512m"}
    r = requests.post(host + "/sessions", data=json.dumps(data), headers=headers)
    print(r.json())
    print(r.json()['id'])

    return r.json()['id']



def test_case2_1(session_id):
    """
    read data form 'users.parquet' , select name, fav color and save it on  namesAndFavColors.parquet
    :param session_id:
    :return:
    """


    data = {
        'code': textwrap.dedent("""
               from pyspark.sql import SQLContext
               sqlContext = SQLContext(sc)
               df = sqlContext.read.load("/home/dev/spark/examples/src/main/resources/users.parquet")
               df.select("name", "favorite_color").write.save("namesAndFavColors.parquet")
           """)
    }
    print(host + "/sessions/" + str(session_id) + "/statements")
    r = requests.post(host + "/sessions/" + str(session_id) + "/statements", data=json.dumps(data), headers=headers)
    print(r.json())

def test_case2_2(session_id):
    """
    read json data and create parquet file
    :param session_id:
    :return:
    """

    data = {
            'code': textwrap.dedent("""
                   from pyspark.sql import SQLContext
                   sqlContext = SQLContext(sc)
                   df = sqlContext.read.load("/home/dev/spark/examples/src/main/resources/people.json", format="json")
                   result = df.select("name", "age").write.save("namesAndAges.parquet", format="parquet")
               """)
    }
    print(host + "/sessions/" + str(session_id) + "/statements")
    r = requests.post(host + "/sessions/" + str(session_id) + "/statements", data=json.dumps(data), headers=headers)
    print(r.json())

def test_case2_3(session_id):
    """
    read data form parquet and return
    :param session_id:
    :return:
    """

    data = {
            'code': textwrap.dedent("""
                   from pyspark.sql import SQLContext
                   sqlContext = SQLContext(sc)
                   df = sqlContext.read.load("/home/dev/spark/examples/src/main/resources/users.parquet")
                   df.registerAsTable("users")
                   result = sqlContext.sql("SELECT * FROM users").collect()
                   result
               """)
    }
    print(host + "/sessions/" + str(session_id) + "/statements")
    r = requests.post(host + "/sessions/" + str(session_id) + "/statements", data=json.dumps(data), headers=headers)
    print(r.json())

def test_case3(session_id):
    """
    delete session
    :param session_id:
    :return:
    """
    r = requests.delete(host + "/sessions/"+ str(session_id), headers=headers)
    print(r.json())


def test_case4(sessio_id, time):
    """
    get result of request
    :param sessio_id:
    :param time:
    :return:
    """
    r = requests.get(host + "/sessions/" + str(sessio_id) + "/statements/" + str(time),  headers=headers)
    print(r.json())


#test_case1()
#test_case2_1(6)
#test_case2_2(5)
#test_case2_3(1)
test_case4(1,7)
#test_case3(0)
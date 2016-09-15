import json, requests, textwrap


# server connection info
# Docker Container Id, Public DNS aloso works
host = "http://481bf68ee6d9:8998"
headers = {'Content-Type': 'application/json'}


def test_case1():
    """
    search alive sessions
    :return:
    """
    r = requests.post(host + '/sessions')
    print(r.json())

def test_case2():
    """
    create session with pyspark type
    :return:
    """
    data = {'kind': 'pyspark'}
    r = requests.post(host + '/sessions', data=json.dumps(data), headers=headers)
    print(r.json())

def test_case3():
    """
    if case2 works nomally we can see session 0 is created
    :return:
    """
    r = requests.get(host + '/sessions/0', headers=headers)
    print(r.json())

def test_case4():
    """
    you can put real python code on "code" parm 1+4 will return 5
    :return:
    """
    data = {"code":"1 + 4"}
    r = requests.post(host + "/sessions/0/statements", data=json.dumps(data), headers=headers)
    print(r.json())

def test_case5():
    """
    after than you can search the result of session 0's first statesments result
    :return:
    """
    r = requests.get(host + "/sessions/0/statements/0", headers=headers)
    print(r.json())

def test_case6():
    """
    on the same session you can execute multiple lines seperatly
    :return:
    """
    data = {"code":"a=10"}
    r = requests.post(host + "/sessions/0/statements", data=json.dumps(data), headers=headers)
    print(r.json())
    data = {"code": "a+1"}
    r = requests.post(host + "/sessions/0/statements", data=json.dumps(data), headers=headers)
    print(r.json())

def test_case7():
    """
    you can kill the session like bellow
    :return:
    """
    r = requests.get(host + "/sessions/0", headers=headers)
    print(r.json())
    r = requests.delete(host + "/sessions/0", headers=headers)
    print(r.json())
    r = requests.get(host + "/sessions/0", headers=headers)
    print(r.json())

def test_case8():
    """
    create session with proxyUser , this works on HUE
    :return:
    """
    data = {"kind": "pyspark", "proxyUser": "bob"}
    r = requests.post(host + "/sessions", data=json.dumps(data),  headers=headers)
    print(r.json())
    r = requests.get(host + "/sessions/3", headers=headers)
    print(r.json())

def test_case9():
    """
    create session, get session id form return, run long code with that session
    :return:
    """
    data = {'kind': 'pyspark',
            "name": "Livy Pi Example",
            "executorCores": 1,
            "executorMemory": "512m",
            "driverCores": 1,
            "driverMemory": "512m"}
    r = requests.post(host + "/sessions", data=json.dumps(data), headers=headers)
    print(r.json())
    print(r.json()['id'])

    data = {
        'code': textwrap.dedent("""
        import random
        NUM_SAMPLES = 10
        def sample(p):
          x, y = random.random(), random.random()
          return 1 if x*x + y*y < 1 else 0
        count = sc.parallelize(xrange(0, NUM_SAMPLES)).map(sample).reduce(lambda a, b: a + b)
        print "Pi is roughly %f" % (4.0 * count / NUM_SAMPLES)
        """)
    }
    print(host + "/sessions/"+str(r.json()['id'])+"/statements")
    r = requests.post(host + "/sessions/"+str(r.json()['id'])+"/statements", data=json.dumps(data), headers=headers)
    print(r.json())


test_case1()
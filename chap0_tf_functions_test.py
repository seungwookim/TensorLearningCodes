# -*- coding: utf-8 -*-
import tensorflow as tf
import json

# truncated_normal
# tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 원하는 차원 형태에 맞춰서 지정한 평균, 분산등을 만족하는 랜덤 데이터를 채워서 리턴
def test_truncated_normal():
    shape = [5, 5, 1, 32]
    initial = tf.truncated_normal(shape, stddev=0.1)
    out = tf.Variable(initial)
    model = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(model)
        print(session.run(out))

# Json - python 변환 테스트
def json_test():
    list = {"aaa": [[1, 2], [3, 4]] , "bbb" : [1]} # Note that the 3rd element is a tuple (3, 4)
    js = json.dumps(list)  # '[1, 2, [3, 4]]'
    print("jsoon-format:{0}".format(js))

    py_list = json.loads(js)
    print("python-format:{0}".format(py_list))

test_truncated_normal()
json_test()
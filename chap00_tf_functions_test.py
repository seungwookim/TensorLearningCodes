# -*- coding: utf-8 -*-
import tensorflow as tf
import json
import traceback
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


# scope method
def my_op_with_vars_scope_a(a, b, scope=None):
  with tf.variable_op_scope([a, b], scope, "MyOp") as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")

    print("scope a : {0}".format(a.name))
    print("scope b : {0}".format(b.name))
    return tf.mul(a, b)

# scope method
def my_op_with_vars_scope_b(a, b, scope=None):
  with tf.variable_op_scope([a, b], scope, "MyXX") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")

      print("scope a : {0}".format(a.name))
      print("scope b : {0}".format(b.name))
      return tf.mul(a, b)

# none scope method
def my_op_with_vars_none_scope1(a, b):
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    print("none scope a : {0}".format(a.name))
    print("none scope b : {0}".format(b.name))
    return tf.mul(a, b)

# none scope method
def my_op_with_vars_none_scope2(a, b):
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    print("none scope a : {0}".format(a.name))
    print("none scope b : {0}".format(b.name))
    return tf.mul(a, b)

def python_to_json():
    return_data = {"status": "ok", "result": "on dev"}
    print(json.dumps(return_data))

def main(unused_argv):

  # Test CASE1
  test_truncated_normal()

  # Test CASE2
  json_test()

  # TEst CASE3
  with tf.Session() as session:
      for i in range(1, 3):
        print(session.run(my_op_with_vars_scope_a(my_op_with_vars_scope_a(i, 2), 3)))
        print(session.run(my_op_with_vars_scope_b(my_op_with_vars_scope_b(i, 2), 3)))
        print(session.run(my_op_with_vars_none_scope1(my_op_with_vars_none_scope1(i, 2), 3)))
        print(session.run(my_op_with_vars_none_scope2(my_op_with_vars_none_scope2(i, 2), 3)))

  # Test CASE4
  python_to_json()

if __name__ == '__main__':
  tf.app.run()


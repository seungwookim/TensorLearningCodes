# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np

# 테스트 데이터 생성
data = np.random.randint(1000, size=10000)

# CASE 1 : with session syntax
# 아래와 같이 사용시 자동으로 세션이 종료됨, 단 Sessin.run 을 일일히 실행 필요
try:

    x = tf.constant(data, name='x')
    y = tf.Variable(x**2 + 5*x + 5, name='y')

    model = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(model)
        print(session.run(y))
except:
    print("ERROR-1")

# CASE 2 : InteractiveSession
# 매번 session.run 을 할 필요가 없음, 단 마지막에 session.close() 필수
try:
    session = tf.InteractiveSession()
    x = tf.constant(data, name='x')
    y = x ** 2 + 5 * x + 5
    print(y.eval())

    session.close()

except:
    print("ERROR-2")

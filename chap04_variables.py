# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# CASE1
# 아래처럼은 동작하지 않음, 왜 냐면 Tensor 연산은 Session 위에서 이루어져야 하기 때문
try :
    # 일반적인 변수처럼 처리하려고 하면 발생한느 현상
    x = tf.constant(35, name='x')
    y = tf.Variable(x + 5, name='y')

    #<tensorflow.python.ops.variables.Variable object at 0x7fabe98cced0>
    print(y)
except:
    print("ERROR-1")

# CASE2
# 아래처럼 실행하면 40이 리턴됨
try :
    x = tf.constant([35, 40, 45], name='x')
    y = tf.Variable(x + 5, name='y')

    # 지금까지 정의한 변수를 Session 에서 사용할 수 있도록 초기화 합니다
    model = tf.initialize_all_variables()

    # with Session as 구분 사용시 Session 종료는 자동
    with tf.Session() as session:
        session.run(model)
        print(session.run(y))
except:
    print("ERROR-2")

# CASE3
# numpy 를 사용해서 사이즈가 큰 데이터를 생성하고 식을 조금 복잡하게 수행
try:
    data = np.random.randint(1000, size=10000)
    x = tf.constant(data, name='x')
    y = tf.Variable(x**2 + 5*x + 5, name='y')

    model = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(model)
        print(session.run(y))
except:
    print("ERROR-3")

# CASE4
# for i in range(5): 을 사용한 루프 연산
try:
    x = tf.Variable(0, name='x')

    model = tf.initialize_all_variables()

    with tf.Session() as session:
        for i in range(5):
            session.run(model)
            x = x + 1
            print(session.run(x))

except:
    print("ERROR-4")


# CASE5
# TensorBoard 를 활용한 그래프 출력
try:
    data = np.random.randint(1000, size=10000)
    x = tf.constant(data, name='x')
    y = tf.Variable(x**2 + 5*x + 5, name='y')

    with tf.Session() as session:

        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/basic", session.graph)
        model = tf.initialize_all_variables()
        session.run(model)
        print(session.run(y))

        # 실행 방법 :  tensorboard --logdir=/tmp/basic
        # http://localhost:6006
except:
    print("ERROR-5")
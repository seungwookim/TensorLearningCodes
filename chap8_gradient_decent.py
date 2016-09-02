# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# threadhold (5) 보다 값이 커지기 전까지 X 를 하나씩 증가시키면서 출력한다
def simple_loop_exp():

    x = tf.Variable(0., name='x')
    threshold = tf.constant(5.)

    model = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(model)
        while session.run(tf.less(x, threshold)):
            x = x + 1
            x_value = session.run(x)
            print(x_value)


# 테스트를 위한 랜덤 값을 생성한다.
def make_data_set(weight, bias, train_num, max_x):

    data = np.random.randint(max_x, size=train_num)
    x = tf.constant(data, name='x')
    result = []

    for i in range(0, train_num):
        result.append([data[i], data[i] * weight + bias])

    return result



# 최적의 회귀식을 찾아내는 기능이다. 최적의 회귀식이란 모든 점을 가장 잘 설명할 수 있는
# 각 점으로부터의 직선거리가 최소화되는 하나의 선을 구하는 것이다라고 보면 된다
def graident_descent(sets):

    # 테스트 데이터를 위한  placeholder 를 생성한다.
    # x, y 는 각 2차원 그래프의 x, y 좌표 값이라고 보면 된다.
    # 아래에서 for i in range(1000): 100번을 반복하면서 랜덤 값을 연산하기 위한
    # 변수로 사용할 것이다.
    x = tf.placeholder("float")
    y = tf.placeholder("float")

    # 최초의 추축 값 , 초기 값이라고 보면 된다.
    w = tf.Variable([1.0, 2.0], name="w")
    # 우리 모델은 아주 간단한 1차 방적식이다  y = a*x + b
    y_model = tf.mul(x, w[0]) + w[1]

    # error 는 식으로 구한 y 값과 원래 가지고 있는 Y 값의 차이이다
    error = tf.square(y - y_model)

    # 애러 값을 최소화 할 수 있는 방향으로 0.01 의 학습도로 공식을 수정해 간다
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

    # 위에 정의한 변수들을 모두 등록한다.
    model = tf.initialize_all_variables()

    errors = []
    # 1000번을 반복하여 학습도를 조금식 올려 보자
    with tf.Session() as session:
        session.run(model)
        for set in sets:
            #print("{0} , {1}".format(set[0], set[1]))
            #session.run(train_op, feed_dict={x: set[0], y: set[1]})
            _, error_value = session.run([train_op, error], feed_dict={x: set[0], y: set[1]})
            errors.append(error_value)

        # for i in range(1000):
        #     x_value = np.random.rand()
        #     y_value = x_value * 2 + 6
        #     print("{0},{1}".format(x_value,y_value))
        #     session.run(train_op, feed_dict={x: x_value, y: y_value})

        w_value = session.run(w)

        print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
        return errors, w_value

# 애러율이 감소하는 것을 시각화
def draw_errorr_graph(errors):
    plt.plot([np.mean(errors[i - 50:i]) for i in range(len(errors))])
    plt.show()
    plt.savefig("errors.png")

# 최종적으로 추측한 그래프의 시각화
def draw_opt_line(all_samples, fomula, x_axis_max_num):

    start = [0  , fomula[1]]
    end = [x_axis_max_num, x_axis_max_num * fomula[0] + fomula[1]]

    for set in all_samples:
        plt.scatter(set[0], set[1], color='b')

    plt.plot(start , end, color='r')
    plt.show()


#생성하려고 하는 초기 데이터 변수
weight = 2
bias = 6
iter = 1000
x_axis_max_num = 5

# 초기 데이터를 생성합니다
data = make_data_set(weight , bias , iter, x_axis_max_num)

# 학습을 시켜 최적의 공식을 찾는다
errors, w_value = graident_descent(data)

# 애러율이 감소하는 그래프
draw_errorr_graph(errors)

# 최종 결과 , 데이터와, 유추한 선
draw_opt_line(data, w_value, x_axis_max_num)
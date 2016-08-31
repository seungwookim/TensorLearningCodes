# -*- coding: utf-8 -*-

import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf
#
# from tensorflow.contrib.session_bundle import exporter
# from tensorflow_serving.example import mnist_input_data


def main(_):
    # 탠소플로우 변수 생성 , 파이선 리턴 값은 tf.Variable 이 된다.
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                          name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")

    # 값을 찍어 보면 탠소 플로우 베리어블 객체인 것을 확인 할 수 있다
    print("weight : " + str(weights))
    print("biases : " + str(biases))

    # 위에 정의한 변수를 변형하여 다른 이름으로 재 정의 할 수 있다
    w2 = tf.Variable(weights.initialized_value(), name="w2")
    w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

    # 위에 정의한 변수들을 실제로 탠소플로우 모델 메모리에 등록한다
    init_op = tf.initialize_all_variables()

    # 값을 직어 보면 아래와 같다
    # name: "init"
    # op: "NoOp"
    # input: "^weights/Assign"
    # input: "^biases/Assign"
    # input: "^w2/Assign"
    # input: "^w_twice/Assign"
    print(init_op)

    #저장에는 tf.train.Saver 를 사용한다
    saver = tf.train.Saver()

    # 실제 메모리와 코어를 사용할 세션을 생성
    with tf.Session() as sess:
        # 메모리에 등록하는 액션을 실제로 실행
        result = sess.run(init_op)

        #지정한 경로에 해당 세션의 모든 정보를 저장한다
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
  tf.app.run()
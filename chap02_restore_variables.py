# -*- coding: utf-8 -*-

import tensorflow as tf


def main(_):
    # 저장할때와 마찬가지로 변수명 자체는 생성 필요 . 이니셜 값 자체는 동일해야 함
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")
    w2 = tf.Variable(weights.initialized_value(), name="w2")
    w_twice = tf.Variable(weights.initialized_value() * 5.0, name="w_twice")
    temp1 = tf.constant(0.0)
    temp2 = tf.constant(0.0)
    temp3 = tf.add(temp1, temp2)

    # 값을 찍어 보면 탠소 플로우 베리어블 객체의 내용을 확인 가능
    print("weight : " + str(weights.initialized_value()))
    print("biases : " + str(biases.initialized_value()))
    print("w2 : " + str(w2.initialized_value()))
    print("w_twice : " + str(w_twice.initialized_value()))
    print("temp1 : " + str(temp1))
    print("temp2 : " + str(temp2))
    print("temp3 : " + str(temp3))

    #저장에는 tf.train.Saver 를 사용한다
    saver = tf.train.Saver()

    # 실제 메모리와 코어를 사용할 세션을 생성
    with tf.Session() as sess:
        #
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")

        # 값을 찍어 보면 탠소 플로우 베리어블 객체의 내용을 확인 가능
        # 더하기 연산 수행과 그 결과
        result2 = sess.run(temp3)
        print("result2 : " + str(result2))

if __name__ == '__main__':
  tf.app.run()
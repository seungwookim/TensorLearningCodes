# -*- coding: utf-8 -*-

import tensorflow as tf

def main(_):
    # 탠소플로우 변수 생성 , 파이선 리턴 값은 tf.Variable 이 된다.
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")

    # 위에 정의한 변수를 변형하여 다른 이름으로 재 정의 할 수 있다
    w2 = tf.Variable(weights.initialized_value(), name="w2")
    w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
    temp1 = tf.constant(5.0)
    temp2 = tf.constant(10.0)
    temp3 = tf.add(temp1, temp2)

    # 값을 찍어 보면 탠소 플로우 베리어블 객체의 내용을 확인 가능
    print("weight : " + str(weights.initialized_value()))
    print("biases : " + str(biases.initialized_value()))
    print("w2 : " + str(w2.initialized_value()))
    print("w_twice : " + str(w_twice.initialized_value()))
    print("temp1 : " + str(temp1))
    print("temp2 : " + str(temp2))
    print("temp3 : " + str(temp3))

    # 위에 정의한 변수들을 실제로 탠소플로우 모델 메모리에 등록한다
    init_op = tf.initialize_all_variables()

    print(init_op)
    # 값을 직어 보면 아래와 같다
    # name: "init"
    # op: "NoOp"
    # input: "^weights/Assign"
    # input: "^biases/Assign"
    # input: "^w2/Assign"
    # input: "^w_twice/Assign"


    #저장에는 tf.train.Saver 를 사용한다
    saver = tf.train.Saver()

    # 실제 메모리와 코어를 사용할 세션을 생성
    with tf.Session() as sess:
        # 메모리에 등록하는 액션을 실제로 실행
        result = sess.run(init_op)

        # 더하기 연산 수행과 그 결과
        result2 = sess.run(temp3)
        print("result2 : " + str(result2))

        #지정한 경로에 해당 세션의 모든 정보를 저장한다
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
  tf.app.run()
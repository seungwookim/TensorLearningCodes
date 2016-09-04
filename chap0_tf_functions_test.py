import tensorflow as tf

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


test_truncated_normal()
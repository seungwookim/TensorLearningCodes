# -*- coding: utf-8 -*-
# Reference : niektemme/tensorflow-mnist-predict


import tensorflow as tf
import json
from tensorflow.examples.tutorials.mnist import input_data

# JSON Object 변수에 직접 접근할 수 있도록 합니다
class JsonObject:
  def __init__(self, d):
    self.__dict__ = d


# JSON to Test Data 변환 처리
# 현재 버전으로는 정해진 포맷을 처리하도록 개발중
# 향후 가변적인 데이터 처리 가능하도록 수정예정
class CustomDataCNNTest:

    def convert_json_to_matrix(self, data):
        train_data = []
        train_tag = []
        result_data = []
        #result_data.append(train_data, train_tag)
        
        json_data = json.loads(data, object_hook=JsonObject)

        for row in json_data:

            temp2 = []
            temp1 = row.tag
            train_tag.append(temp1)

            for line in row.data:
                temp2.append(line)


            train_data.append(temp2)

        result_data.append(train_data)
        result_data.append(train_tag)

        return result_data
        

# 테스트 데이터 셋
json_data = """
             [
                 {
                     "name": "nn0001",
                     "data" : [ 0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ],
                     "tag": [ 0, 1]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ,
                                0 , 1 , 0, 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 ],
                     "tag": [ 0, 1]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ],
                     "tag": [ 1, 0]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ],
                     "tag": [ 1, 0]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ],
                     "tag": [ 1, 0]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ],
                     "tag": [ 1, 0]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ],
                     "tag": [ 1, 0]
                 },
                 {
                     "name": "nn0001",
                     "data" : [ 1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ,
                                1 , 0 , 1, 0 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0  ],
                     "tag": [ 1, 0]
                 }
             ]
             """

# 테스트 데이터를 사용해서 CNN 학습을 시작한다.
test = CustomDataCNNTest()
data = test.convert_json_to_matrix(json_data)

print("Customized Input Data : {0}".format(data))


sess = tf.InteractiveSession()

"""
 JSON 변환 테스트 셋의 데이터의 형태는 아래와 같다
 96 개의 데이터로 이루어 졌으며, 12 X 8 형태의 메트릭스로
 변환하여 작업 예정. 아웃풋은 1, 0 둘중에 하나를 성택하도록 한다
"""
x = tf.placeholder(tf.float32, [None, 96])
y_= tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.zeros([96, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 원하는 행렬 사이즈로 초기 값을 만들어서 리턴하는 메서드
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# 0.1 로 초기값 지정하여 원하는 사이즈로 리턴
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# CNN 동작에 대한 정의
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max Pool 동작에 대한 정의
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


"""
 [레이어 1]
  2X2 사이즈의 필터로 가로세로 한칸씩 이동하면서 16가지 종류의 필터를 적용하고
  2X2 사이즈의 Max Pool을 적용하여 메트릭스 차원을 절반으로 축소
  입력 : 12X8 사이즈 메트릭스 하나
  출력 : 6X4 사이즈 메트릭스 16개
"""
W_conv1 = weight_variable([2, 2, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1,12,8,1])       # 96개 데이터 12 - 8 로 변환
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
 [레이어 2]
  2X2 사이즈의 필터로 가로세로 한칸씩 이동하면서 16가지 종류의 필터를 적용하고
  2X2 사이즈의 Max Pool을 적용하여 메트릭스 차원을 절반으로 축소
  입력 : 6X4 사이즈 메트릭스 16개
  출력 : 3X2 사이즈 메트릭스 32개
"""
W_conv2 = weight_variable([2, 2, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
 [레이어 3]
  Drop Out 데이터를 버림으로써 과적합을 예방한다.
  입력 : 3X2 사이즈 메트릭스 32개
  출력 : 100 사이즈 리스트
"""
W_fc1 = weight_variable([3 * 2 * 32, 100])
b_fc1 = bias_variable([100])
h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 2 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# drop out 연산의 결과를 담을 변수
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


"""
Out put Layer
  최종 판단은 SoftMax 를 사용하여 확률을 구해준다
  입력 : 100 사이즈 리스트
  출력 : 각 0과 1일 확률
"""
W_fc2 = weight_variable([100, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 평가를 위한 계산식
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"""
Train & Save Model
"""
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

#50개씩, 20000번 반복학습
for i in range(3):
  batch = data
  #batch = data.train.next_batch(5)
  # 10회 단위로 한번씩 모델 정합성 테스트
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  # batch[0] 28X28 이미지,  batch[1] 숫자태그, keep_prob : Dropout 비율
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 모델에 사용된 모든 변수 값을 저장한다
save_path = saver.save(sess, "model3.ckpt")
print ("Model saved in file: ", save_path)

#최종적으로 모델의 정합성을 체크한다
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))



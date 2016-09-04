# -*- coding: utf-8 -*-

"""
niektemme/tensorflow-mnist-predict 를 참조하였음

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#레파지토리에서 테스트 프로그램에 필요한 데이터 다운로드
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
"""
모델 생성에 필요한 데이터 정의
x : 인풋레이어에 사용할 변수 정의
y : 아웃풋레이어에 사용할 변수 정의
w : 784 X 10 개의 초기값 0을 갖는 메트릭스 생성
b : 10개짜리 배열 생성
y = x * w + b
x (784) * w(784*10) = x*w(10)
x*w(10) + b(10) = y(10)
위에처럼 메트릭스 연산이 수행되기 때문에 위와 같이 데이터 사이즈를 잡은 것이다.
"""
x = tf.placeholder(tf.float32, [None, 784])
y_= tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 원하는 행렬 사이즈로 초기 값을 만들어서 리턴하는 메서드
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# 0.1 로 초기값 지정하여 원하는 사이즈로 리턴
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""
필터에 대해서 설명하고자 하면 CNN 의 동작 원리를 설명해야만한다.
[5, 5, 1, 32] 는 5X5 사이즈의 단위 필터를 사용해서 인풋데이터
(여기서는 28 X 28 사이즈 메트릭스)를 CNN연산을 하겠다는 것이다.
Stride 가 [1,1] 이라고 하면 28X28크기 행렬을 5X5 사이즈의
메트릭스로가로세로 한칸씩 이동하면서 필터에 연산하겠다는 의미가 된다.
결과적으로 아웃풋은 24X24 사이즈가 된다. 왜냐하면 5X5 사이즈의
메트릭스로 이동할 수 있는 한계가 있기 때문이다.
(메트릭스 끝부분 까지 이동할 수 없음)
이러한 경우 패딩 옵션을 사용하여 0으로 태두리를 채워넣어 메특릭스
사이즈를 동일하게 유지할 수도 있다
참조:http://deeplearning4j.org/convolutionalnets.html

"""
def conv2d(x, W):
# tf.nn.conv2d(input, filter, strides, padding, use_cudnn
# _on_gpu=None, data_format=None, name=None)
# strides= [1 , stride, stride, 1] 차원축소 작업시 마스크 메트릭스를 이동하는 보복
# padding='SAME' 다음 레벨에서도 메특릭스가 줄어들지 않도록 패딩을 추가한다
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


"""
보통은 이렇게 생성한 메트릭스를 max pooling 을 사용하여 다시 한번 간소화한다.
위에서 필터링이 마스크에 대한 & 연산이었다면, max Pooling은 메트릭스에서 가장
큰 값 하나만 뽑아서 사용하는 방법이다. 아래와 같은 max pooling 정의
(mask [2,2] , stride[2,2] )를 4X4 메트릭스에 적용하면 2X2 메트릭스가 될 것이다
"""
# x :  [batch, height, width, channels]
# 2x2 행열에 가장 큰 값을 찾아서 추출, 가로세로 2칸씩이동
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# [filter_height, filter_width, in_channels, out_channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

"""
Layer 1
아래의 3줄로써 인풋 레이어에 대한 정의는 완료된다. 28X28 행렬 하나를 넣고
28X28행렬(원래는 24X24가 되지만 Padding 적용) 32개를 만들고 다시 max pool
(2,2)를 사용하여 14X14 메트릭스 32개를 리턴하는 레이어를 정의하였다
메트릭스 단위로 정리하면 인풋 1개, 아웃풋 32개 이다 트
"""
#인풋 데이터터 메트릭스를 변형한다. 784 개의 행렬을 갖는 복수의 데이터를
#[-1, 28, 28,1] 로 의 형태로 변형한다. 테스트 데이터 수 만큼 (-1) ,
#[28x28] 행렬로 만드는데 각 픽셀데이터는 rgb 데이터가 아니고 하나의 값만 갖도 변환
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
Layer 2
1번 레이어에서 아웃풋을 32개를 전달하였음으로 2번 레이어의 인풋은
14X14 메트릭스 32개 그리고 아웃풋은 동일한 max pool 을 적용하여 8x8 메트릭스
64개를 출력한다. 정리하면 인풋 32개(14X14) 아웃풋 64개(7X7) 이 된다
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
Layer 3
현재 최종 데이터의 수는 7 X 7 X 64 = 3136 개 이지만 1024 개 를 사용한다
1024는 임의의 선택 값이다
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


"""
Drop Out
Dropout 은 데이터 간의 연과 관계가 큰 데이터들을 제거함으로써 과적합 문제를
해결하는 기법의 하나이다.
"""
# drop out 연산의 결과를 담을 변수
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


"""
Out put Layer
마지막으로 1024개의 노드에서 10개의 (0~9까지 숫자)에 대한 확률을 Soft Max 를
이용하여 도출할 수 있도록 한다
"""
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
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
for i in range(20000):
  batch = mnist.train.next_batch(50)
  # 10회 단위로 한번씩 모델 정합성 테스트
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  # batch[0] 28X28 이미지,  batch[1] 숫자태그, keep_prob : Dropout 비율
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 모델에 사용된 모든 변수 값을 저장한다
save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)

#최종적으로 모델의 정합성을 체크한다
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close()
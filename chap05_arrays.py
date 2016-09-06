# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import matplotlib.image as mpimg

# 이미지를 로딩한다
filename = os.path.dirname(__file__) + "/MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# 이미지의 메트릭스 형태 출력
print("1. Initial : height :{0} , width :{1} , depth :{2}".format(height, width, depth))


# 축을 변경한다
try:
    x = tf.Variable(image, name='x')

    model = tf.initialize_all_variables()

    with tf.Session() as session:
        # Array 위치가 축 (x,y,z), 거기에 입력하는 숫자가 바꾸고 싶은 차원
        x = tf.transpose(x, perm=[0, 2, 1])
        #x = tf.transpose(x, perm=[0 ,1, 2])
        session.run(model)
        result = session.run(x)

    height, width, depth = result.shape
    print("2. Transpose : height :{0} , width :{1} , depth :{2}".format(height, width, depth))

except Exception as err:
    print(err)


# 배열의 순서를 변경한다
try:
    #보기 좋게 데이터를 축소해보자
    tem_len = 2
    temp_img = image[0:tem_len][0:tem_len]

    x = tf.Variable(temp_img, name='x')
    model = tf.initialize_all_variables()

    #변경전 데이터 출력
    for i in range(tem_len):
        for j in range(tem_len):
            print("Before :[{0},{1}] : {2}".format(i, j, temp_img[i][j]))


    #다른 메서드들 : https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html
    #x는 인풋 데이터
    #데이터 사이즈 [width] * height = [5, 5, 5, 5, 5] 생성
    # seq_dim 1 : 데이터의 순서가 역순으로 정렬 됨
    # seq_dim 2 : 데이터 안의 내용이 역순 정렬
    #seq_dim 3 : seq_dim must be < input.dims() (3 vs 3)
    #batch_dim = 0 위에서 아래로 연산 수행
    with tf.Session() as session:
        x = tf.reverse_sequence(x, [tem_len] * tem_len, 2, batch_dim=0)
        session.run(model)
        result = session.run(x)

    #변경후 데이터 출력
    for i in range(tem_len):
        for j in range(tem_len):
            print("After :[{0},{1}] : {2}".format(i, j, result[i][j]))

except Exception as err:
    print(err)
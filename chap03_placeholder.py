# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


# 데이터 타입, 사이즈 정의 ,None은 무한대
x1 = tf.placeholder("float", 3)
x2 = tf.placeholder("float", None)
x3 = tf.placeholder("float", 3)
y = x1 * 2 + x2 + x3

# 데이터의 형태는 아래와 같이 이중 배열로로 정의 가능
x4 = tf.placeholder("float", [None, 3])
y2 = x4 + 1

# 기본적으로 연산하려면 세션에서 해야됨
with tf.Session() as session:

    # 데이터 직접 삽입
    result = session.run(y, feed_dict={x1: [1, 2, 3], x2: [1, 2, 3], x3: [1, 2, 3] })
    print(result)

    # 데이터 직접 삽입
    x_data = [[1, 2, 3], [4, 5, 6], ]
    result2 = session.run(y2, feed_dict={x4: x_data})
    print(result2)


# 이미지와 동일하게 가로, 세로, RGB 처럼 표현하는 테스트 데이터를 만들어 보자
data1 = np.random.randint(10)    # 이미지로 치면 빨간색
data2 = np.random.randint(10)    # 초록색
data3 = np.random.randint(10)    # 파란색

# 이미지로 치면 3색 표현 , 가로 5, 세로 5 인 이미지가 되겠다
raw_image_data = [[[data1, data2, data3]] * 5 ] * 5

# 데이터! 출력
print("1. Original : {0}".format(raw_image_data))

# 홀더를 만든다 이홀더는 가로 세로 사이즈 제한은 없고 색은 3가지 RGB 로 표현
# 하는 모든 데이터를 커버 할 수 있다
image = tf.placeholder("uint8", [None, None, 3])

# 메트릭스를 자른다, 첫번째는 소스 메트릭스, 두번재는 시작 주소, 세번째는 끝 메트릭스
# 시작과 끝 사이의 데이터만 리턴한다, -1은 데이터 최대 길이라는 이야기다
slice = tf.slice(image, [0, 0, 0], [-1, 1, -1])

# 슬라이스를 연산한다
with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})

# 슬라이스 연산후 데이터
print("2. Transformed : {0}".format(result))


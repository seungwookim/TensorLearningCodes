# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


# 랜덤하게 클러스터 데이터를 구성합니다.
def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):

        # 평균이 0이고 표준편차가 5이고 2개의 데이터로 이루어진 500개의 랜덤 데이터 Pool 을 생성합니다
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                               mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))

        # 랜덤으로 중앙값을 만듭니다. 형태는 [X , Y] 가 되겠습니다.
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)

        # 각 그룹들의 중앙값을 무엇으로 만들었는지 그 이력을 저장합니다.
        centroids.append(current_centroid)

        # 이렇게 생성된 중앙 값을 500개의 랜덤 데이터에 더함으로써 해당 분포를 전체적으로 옮기도록 합니다.
        samples += current_centroid

        #최종적으로 완성된 각 클러스터별 데이터 셋을 저장합니다
        slices.append(samples)

    # 모든 클러스터 그룹의 그룹 데이터와 중앙값을 탠소플로우 변수로 정의하여 리턴합니다.
    samples = tf.concat(0, slices, name='samples')
    centroids = tf.concat(0, centroids, name='centroids')
    return centroids, samples

# all_samples 전체 샘플 배열( x, y 좌표로 구성) , 각 그룹의 중앙값 배열 , 클러스터의 수
def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    import matplotlib.pyplot as plt

    # 0 ~ 1 구간을 len(centroids) 센터의 수 만큼으로 균등하게 나눈 배열을 생성함
    # linespace :[ 0.   0.5  1. ]
    print("linespace :{0}".format(np.linspace(0,1,len(centroids))))

    # 0 ~ 1 구간에 대응하는 색상 스팩트럼에서 대응하는 색상을 리턴
    # colur:[[1.00000000e+00   3.03152674e-01   1.53391655e-01   1.00000000e+00]
    #        [1.00000000e+00   1.22464680e-16   6.12323400e-17   1.00000000e+00]
    print("colur :{0}".format(plt.cm.rainbow([0.9, 1, 2, 3])))

    # 각 그룹의 색상을 다르게 해주기 위하여  RGB 색상 추출
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))


    # 중앙 값 배열을 기준으로 중앙점과 각 그룹에 해당하는 점을 표시
    for i, centroid in enumerate(centroids):
        # 각 배열에 해당하는 0 ~ 500 , 500 ~ 1000 , 1000 ~ 1500 구간을 나누어 색상을 지정하고 출력한다
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0], samples[:,1], c=colour[i])

        # 중앙점을 표시한다
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()

# https://www.tensorflow.org/versions/r0.10/api_docs
# 클러스터 중심좌표 3개를 랜덤으로 섞어서 다시 리턴
def choose_random_centroids(samples, n_clusters):

    # tf.shape(input, name=None)
    # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    # shape(t) == > [3]
    n_samples = tf.shape(samples)[0]

    # shape Test
    print("tf.shape Test! ")
    t = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]

    with tf.Session() as session:
        tt = tf.shape(t)
        rt = session.run(tt[0])
        print("tf.shape Test Result: {0} , {1}".format(tt, rt))

    # tf.random_shuffle
    # [[1, 2],       [[5, 6],
    # [3, 4],  ==>   [1, 2],
    # [5, 6]]        [3, 4]]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))

    begin = [0, ]
    size = [n_clusters, ]
    size[0] = n_clusters
    # tf.slice ( input, form array , to array )
    centroid_indices = tf.slice(random_indices, begin, size)

    # gather mixed cluster again
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids

#
def assign_to_nearest(samples, centroids):

    # expand_dims test
    # 't2' is a tensor of shape [2, 3, 5]
    # expand_dims ( data , location to add dim)
    with tf.Session() as session:
        t2 = [[2,3,5,6,7,8,9],[9,8,7,6,5,4,3,]]
        print("expand_before shape : {0}".format(session.run(tf.shape(t2))))
        print("expand_dims test : {0}, {1}".format(session.run(tf.shape(tf.expand_dims(t2, 0))),
                                                   session.run(tf.expand_dims(t2, 0))))
        print("expand_dims test : {0}, {1}".format(session.run(tf.shape(tf.expand_dims(t2, 1))),
                                                   session.run(tf.expand_dims(t2, 1))))
        print("expand_dims test : {0}, {1}".format(session.run(tf.shape(tf.expand_dims(t2, -1))),
                                                   session.run(tf.expand_dims(t2, -1))))


    # 차원을 추가한다
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    # (1) tf.reduce_sum
    ## 'x' is [[1, 1, 1]
    #         [1, 1, 1]]
    #tf.reduce_sum(x) ==> 6
    #tf.reduce_sum(x, 0) ==> [2, 2, 2]
    #tf.reduce_sum(x, 1) ==> [3, 3]
    #tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
    #tf.reduce_sum(x, [0, 1]) ==> 6

    # (2) tf.square(X) =  x^2

    # 모든 점들과 위에서 구한 랜덤 중앙점간의 거리의 합을 구한다
    distances = tf.reduce_sum( tf.square(
               tf.sub(expanded_vectors, expanded_centroids)), 2)

    # 가장 가까운 거리를 구한다
    mins = tf.argmin(distances, 0)
    #
    nearest_indices = mins
    return nearest_indices

# 새롭게 구한 중앙점 데이터를 업데이트
def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat(0, [tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions])
    return new_centroids

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

# 복수 그룹의 군집 데이터를 만든다 의
data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
# 랜덤으로 중앙점을 생성한다
initial_centroids = choose_random_centroids(samples, n_clusters)
# 랜덤한 중앙점이 가장 가까운 그룹을 찾는다
nearest_indices = assign_to_nearest(samples, initial_centroids)
#
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

# 데이터 확인
with tf.Session() as session:
    print("[step 1] cent : {0} , sample : {1}".format(session.run(data_centroids), session.run(data_centroids)))
    print("[step 2] cent : {0} ".format(session.run(initial_centroids)))
    print("[step 3] cent : {0} ".format(session.run(nearest_indices)))
    print("[step 4] cent : {0} ".format(session.run(updated_centroids)))

# 세션에서 연산을 수행한다
model = tf.initialize_all_variables()
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)

# 그래프를 출력한다
# 맷랩은 슈퍼유저로 실행하면 애러난다
plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)

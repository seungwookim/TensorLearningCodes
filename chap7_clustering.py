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

def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    import matplotlib.pyplot as plt
    # Plot out the different clusters
    # Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0], samples[:,1], c=colour[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()

#
def choose_random_centroids(samples, n_clusters):
    # Step 0: Initialisation: Select `n_clusters` number of random points
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0, ]
    size = [n_clusters, ]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids


def assign_to_nearest(samples, centroids):
    # Finds the nearest centroid for each sample

    # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(
               tf.sub(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    nearest_indices = mins
    return nearest_indices

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


data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

model = tf.initialize_all_variables()
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)

print(sample_values)
print(updated_centroid_value)
print(n_samples_per_cluster)
#plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)


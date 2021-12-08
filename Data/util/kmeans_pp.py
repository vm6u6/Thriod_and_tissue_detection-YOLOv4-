import numpy as np
from kmeans import iou

def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)
    
    centroids.append(boxes[centroid_index])
    centroid_num = len(centroids)

    for centroid_index in range(0,n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid in range(centroid_num):
                distance = (1 - iou(box, centroids[centroid]))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                
                centroids.append(boxes[i]) 
                break
    return centroids
  
def kmeans_pp(boxes, k, dist=np.median):
    #len(boxes)
    rows = boxes.shape[0] 
    # 計算每个ground truth和k個Anchor的距離
    distances = np.empty((rows, k))
    # 上一次每個ground truth"距離"最近的Anchor索引
    last_clusters = np.zeros((rows,))

    # kmean++ 後的centeriod
    cluster = init_centroids( boxes, k )
    clusters =[]
    clusters.append( [cluster[0][0][0], cluster[0][0][1]] )
    for i in range(len(cluster)-1):
        a = cluster[i+1][0]
        b = cluster[i+1][1]
        clusters.append( [a, b] )
    clusters = np.array(clusters)
    
    while True:
        
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters
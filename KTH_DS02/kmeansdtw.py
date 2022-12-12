import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from numpy import inf
from math import isinf,sqrt
from sklearn.metrics import classification_report, adjusted_rand_score
from scipy.io import arff
from scipy.special import comb
import random
import time

def DTWdist(x, y, w):
    # x (m*K) array; y (n*K) array
    # int w: window size limiting the maximal distance
    DTW ={}
    m = x.shape[0]
    n = y.shape[0]

    w = max(w, abs(m-n))
    for i in range(-1,m):
        for j in range(-1,n):
            DTW[(i, j)] = inf
    DTW[(-1, -1)] = 0
    for i in range(m):
        for j in range(max(0, i-w), min(n, i+w)):
            dist = (x[i]-y[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
    return sqrt(DTW[m-1, n-1])


def LB_Keogh(x, y, b):
    # x (m*K) array; y (n*K) array
    # b is the window size
    LB_sum = 0
    for ind, i in enumerate(x):
        lower_bound = min(y[(ind-b if ind-b>=0 else 0):(ind+b)])
        upper_bound = max(y[(ind-b if ind-b>=0 else 0):(ind+b)])
        if i > upper_bound:
            LB_sum = LB_sum + (i-upper_bound)**2
        elif i < lower_bound:
            LB_sum = LB_sum + (i-lower_bound)**2

    return sqrt(LB_sum)


def EUCdist(x, y):
    dist = np.linalg.norm(x - y)
    return dist


def knn(train, test, w, b):
    preds = []
    for ind,i in enumerate(test):
        min_dist = inf
        closest_seq = []
        for j in train:
            if LB_Keogh(i[:-1], j[:-1], b) < min_dist:
                dist = DTWdist(i[:-1], j[:-1], w)
                if dist < min_dist:
                    min_dist = dist
                    closest_seq = j
        preds.append(closest_seq[-1])
    return classification_report(test[:, -1], preds)


def dataloader(filename):
    data = arff.loadarff(filename)[0]
    data1 = np.array(list(data[0]))
    for i in data[1:]:
        data1 = np.vstack((data1,np.array(list(i))))

    return data1.astype('float')


def kmeans(data, k, max_iter, w, b, dm=1):
    cluster = np.random.choice(np.arange(data.shape[0]), size = k, replace=False)
    centroid = data[cluster,:]
    start_time = time.time()
    count = 0
    for n in range(max_iter):
        print(count)
        count = count + 1
        assignment = {}
        if count == max_iter:
            inf_start = time.time()
        for ind,i in enumerate(data):
            min_dist = inf
            closest_cen = None
            if dm is 1:
                for ind2,j in enumerate(centroid):
                    if LB_Keogh(i, j, b) < min_dist:
                        dist = DTWdist(i, j, w)
                        if dist < min_dist:
                            min_dist = dist
                            closest_cen = ind2
            elif dm is 2:
                for ind2,j in enumerate(centroid):
                    dist = EUCdist(i, j)
                    if dist < min_dist:
                        min_dist = dist
                        closest_cen = ind2
            if closest_cen in assignment:
                assignment[closest_cen].append(ind)
            else:
                assignment[closest_cen] = []
                assignment[closest_cen].append(ind)

        if count == max_iter:
            inf_end = time.time()
            print('Infering done in {:.6f} seconds.'.format(inf_end - inf_start))

        for key in assignment:
            sum = 0
            for p in assignment[key]:
                sum = sum + data[p]
            centroid[key] = [m/len(assignment[key]) for m in sum]

    end_time = time.time()
    print('Training done in {:.6f} seconds.'.format(end_time - start_time))
    locations = np.zeros((data.shape[0]), dtype=np.int32)
    for key in assignment:
        for p in assignment[key]:
            locations[p] = key

    return centroid, locations


if __name__ == '__main__':
    w = 100
    b = 5
    name = 'CricketZ'
    # https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    train = dataloader('./Univariate_arff/' + name + '/' + name + '_TRAIN.arff')
    test = dataloader('./Univariate_arff/' + name + '/' + name + '_TEST.arff')
    data = np.vstack((train[:, :-1], test[:, :-1]))
    #knnreport = knn(train, test, w, b)
    #print(knnreport)
    print(train[:, -1].shape, test[:, -1].shape)
    label = np.hstack((train[:, -1], test[:, -1]))
    centroid, locations = kmeans(data=data, k = 12, max_iter = 10, w = w, b = b, dm = 1)

    samples = data.shape[0]
    count = 0
    for i in range(samples):
        for j in range(i, samples):
            if i==j:
                count = count - 1
            if (label[i] == label[j]) and (locations[i] == locations[j]):
                count = count + 1
            elif (label[i] != label[j]) and (locations[i] != locations[j]):
                count = count + 1

    cn = comb(samples, 2)
    ri = count / cn

    print(cn, count, ri)

    print(adjusted_rand_score(label, locations))

    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['figure.dpi'] = 500
    #for i in centroid:
    #   plt.plot(i)

    fig = plt.figure(tight_layout=True)
    # grid row : x, coloum : y
    size1 = 3
    size2 = 4
    gs = GridSpec(size1, size2)

    for x in range(size1):
        for y in range(size2):
            ax = fig.add_subplot(gs[x, y])
            ki = y * size1 + x
            if ki < len(centroid):
                ax.plot(centroid[ki])
                ax.set_title('Cluster %d' % (ki+1))
    plt.show()

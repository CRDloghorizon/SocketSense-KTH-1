import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import adjusted_rand_score
from numpy import inf
import math
import time
from scipy.io import arff
from scipy.special import comb
from collections import defaultdict

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
    return math.sqrt(DTW[m-1, n-1])


def LB_Keogh(x,y,b):
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

    return math.sqrt(LB_sum)


def dataloader(filename):
    data = arff.loadarff(filename)[0]
    data1 = np.array(list(data[0]))
    for i in data[1:]:
        data1 = np.vstack((data1,np.array(list(i))))

    return data1.astype('float')


def lr_decay(lr, t, n_iteration):
    return lr / (1 + t/(n_iteration/2))


class SOMdtw(object):
    def __init__(self, width, height, channel, sigma = 1.0, learning_rate=0.5, w=100, b=5):
        # Set up the SOM map (w,h,ch) and lr
        self.x = width
        self.y = height
        self.ch = channel
        self.lr = learning_rate
        self.sigma = sigma
        self.w = w
        self.b = b
        self.trained = False

    def initialize_map(self):
        # Initialize map weight vectors Kohonen random initialization
        ds_mul = np.mean(self.data) / 0.5
        self.node_vectors = np.random.rand(self.y, self.x, self.ch) * ds_mul

    def save(self, filename):
        # Save SOM to .npy file.
        if self.trained:
            np.savez(filename, x = self.node_vectors, w = np.array(self.w), b = np.array(self.b))
            return True
        else:
            return False

    def load(self, filename):
        # Load SOM from .npy file.
        npzfile = np.load(filename)
        self.node_vectors = npzfile['x']
        self.w = npzfile['w'][0]
        self.b = npzfile['b'][0]
        self.x = self.node_vectors.shape[1]
        self.y = self.node_vectors.shape[0]
        self.ch = self.node_vectors.shape[2]
        self.trained = True
        return True

    def get_map_vectors(self):
        # Returns the map vectors.
        if self.trained:
            return self.node_vectors
        else:
            return False

    def euc_distance(self, x, y):
        dist = np.linalg.norm(x - y)
        return dist

    def find_maching_nodes(self, input_sample):
        # This is to be called only when the map is trained.
        if self.trained == False:
            return False
        inf_start = time.time()
        n_data = input_sample.shape[0]
        locations = np.zeros((n_data, 3), dtype=np.int32)
        distances = np.zeros((n_data), dtype=np.float32)

        print_step = int(n_data / 20)
        print_count = 0
        for idx in range(n_data):

            if idx % print_step == 0:
                print_count += 1
                print('Finding mathing nodes:' + str(print_count))

            data_vect = input_sample[idx]
            min_dist = inf
            x = None
            y = None
            if self.dm == 1:
                for y_idx in range(self.y):
                    for x_idx in range(self.x):
                        node_vect = self.node_vectors[y_idx, x_idx]
                        if LB_Keogh(data_vect, node_vect, self.b) < min_dist:
                            dist = DTWdist(data_vect, node_vect, self.w)
                            if dist < min_dist:
                                min_dist = dist
                                x = x_idx
                                y = y_idx
            elif self.dm == 2:
                for y_idx in range(self.y):
                    for x_idx in range(self.x):
                        node_vect = self.node_vectors[y_idx, x_idx]
                        dist = self.euc_distance(data_vect, node_vect)
                        if dist < min_dist:
                            min_dist = dist
                            x = x_idx
                            y = y_idx

            locations[idx, 0] = y
            locations[idx, 1] = x
            locations[idx, 2] = x + self.x * y
            distances[idx] = min_dist

        print('Done')
        inf_end = time.time()
        print('Infering done in {:.6f} seconds.'.format(inf_end - inf_start))

        return locations, distances

    def train(self, data, n_iteration=10, batch_size=10, dm = 1):
    # Train the som, neighborhood function = gaussian, distance method = dtw
        self.data = data
        self.n_iter = n_iteration
        self.batch_size =batch_size
        self.dm = dm

        start_time = time.time()
        self.initialize_map()

        self.nb_dist = math.floor(min(self.x, self.y) / 1.5)
        n_samples = self.data.shape[0]
        data_index = np.arange(n_samples)

        # Pad the vector map to allow easy array processing.
        tmp_node_vects = np.zeros((self.y + 2 * self.nb_dist, self.x + 2 * self.nb_dist, self.ch))
        tmp_node_vects[self.nb_dist: self.nb_dist + self.y,
        self.nb_dist: self.nb_dist + self.x] = self.node_vectors.copy()
        self.node_vectors = tmp_node_vects

        batch_count = math.ceil(n_samples / self.batch_size)

        #self.neighbor_function()
        for iteration in range(self.n_iter):
            self.neighbor_function()
            np.random.shuffle(data_index)
            total_dist = 0

            # batch processing
            for batch in range(batch_count):
                steps_left = n_samples - batch * self.batch_size
                if steps_left < self.batch_size:
                    steps_in_batch = steps_left
                else:
                    steps_in_batch = self.batch_size
                # each sample is assigned with the best match node
                best_node = np.zeros((steps_in_batch, 3), dtype=np.int32)

                for step in range(steps_in_batch):
                    input_idx = data_index[batch * self.batch_size + step]
                    input_vect = self.data[input_idx]
                    y, x, dist = self.find_best_matching_node(input_vect)
                    best_node[step, 0] = y
                    best_node[step, 1] = x
                    best_node[step, 2] = input_idx
                    total_dist += dist
                # update the weights for each batch
                self.update_node_vectors(best_node)

            self.sigma = lr_decay(self.sigma, iteration+1, self.n_iter)
            self.lr = lr_decay(self.lr, iteration+1, self.n_iter)
            print('Iteration = {:d}, Average distance = {:.2f}.'.format(iteration+1,total_dist/n_samples))

        self.node_vectors = self.node_vectors[self.nb_dist : self.nb_dist + self.y, self.nb_dist : self.nb_dist + self.x]

        end_time = time.time()
        self.trained = True
        print('Training done in {:.6f} seconds.'.format(end_time - start_time))


    def find_best_matching_node(self, data_vect):
        # This method is used to find best matching node for data vector.
        # The node coordinates and distance are returned.
        min_dist = inf
        x = 0
        y = 0
        for y_idx in range(self.y):
            for x_idx in range(self.x):
                node_vect = self.node_vectors[y_idx + self.nb_dist, x_idx + self.nb_dist]
                if self.dm == 1:
                    if LB_Keogh(data_vect, node_vect, self.b) < min_dist:
                        dist = DTWdist(data_vect, node_vect, self.w)
                        if dist < min_dist:
                            min_dist = dist
                            x = x_idx
                            y = y_idx
                elif self.dm == 2:
                    dist = self.euc_distance(data_vect, node_vect)
                    if dist < min_dist:
                        min_dist = dist
                        x = x_idx
                        y = y_idx
        return y, x, min_dist

    def update_node_vectors(self, best_node):
        # This method updates the map node weights.
        for ind in range(best_node.shape[0]):
            node_y = best_node[ind, 0]
            node_x = best_node[ind, 1]
            input_ind = best_node[ind, 2]
            input_vec = self.data[input_ind]
            old_vec = self.node_vectors[node_y + self.y_delta + self.nb_dist, node_x + self.x_delta + self.nb_dist]

            update_vec = self.neighbor_weights * self.lr * (np.expand_dims(input_vec, axis=0) - old_vec)

            self.node_vectors[node_y + self.y_delta + self.nb_dist,
                              node_x + self.x_delta + self.nb_dist, :] += update_vec

    def neighbor_function(self):
        # Create a Guassian distribution matrix with (x,y,channel) and sigma is decayed according to iteration number
        # modify self.node_weights
        size = self.nb_dist * 2
        if size == 0:
            size = size + 2
        self.neighbor_weights = np.full((size * size, self.ch), 0.0)
        cp = size / 2.0
        p1 = 1.0 / (2 * math.pi * self.sigma ** 2)
        pdiv = 2.0 * self.sigma ** 2

        y_delta = []
        x_delta = []
        for y in range(size):
            for x in range(size):
                ep = -1.0 * ((x - cp) ** 2.0 + (y - cp) ** 2.0) / pdiv
                value = p1 * math.e ** ep
                self.neighbor_weights[y * size + x] = value
                y_delta.append(y - int(cp))
                x_delta.append(x - int(cp))
        self.x_delta = np.array(x_delta, dtype=np.int32)
        self.y_delta = np.array(y_delta, dtype=np.int32)
        #print(self.neighbor_weights, self.nb_dist)
        self.neighbor_weights -= self.neighbor_weights[size // 2]
        self.neighbor_weights[self.neighbor_weights < 0] = 0
        self.neighbor_weights /= np.max(self.neighbor_weights)


if __name__ == '__main__':
    w = 15
    b = 5
    name = 'BME'
    train = dataloader('./Univariate_arff/' + name + '/' + name + '_TRAIN.arff')
    test = dataloader('./Univariate_arff/' + name + '/' + name + '_TEST.arff')
    data = np.vstack((train[:,:-1], test[:,:-1]))
    label = np.hstack((train[:, -1], test[:, -1]))
    ch = data.shape[1]

    som1 = SOMdtw(width=3, height=1, channel=ch, sigma = 1.0, learning_rate=0.5, w=w, b=b)
    # dm = 1 : dtw; dm = 2 : Euc
    som1.train(data=data, n_iteration=10, batch_size=10, dm=2)
    samples = data.shape[0]

    # inference
    locations, distances = som1.find_maching_nodes(data)

    count = 0
    for i in range(samples):
        for j in range(i, samples):
            if i==j:
                count = count - 1
            if (label[i] == label[j]) and (locations[i,2] == locations[j,2]):
                count = count + 1
            elif (label[i] != label[j]) and (locations[i,2] != locations[j,2]):
                count = count + 1

    cn = comb(samples, 2)
    ri = count / cn

    print(cn, count, ri)
    print(adjusted_rand_score(label, locations[:, 2]))

    winmap = defaultdict(list)
    for i in range(samples):
        winmap[locations[i, 2]].append(data[i, :])

    # plt.rcParams['savefig.dpi'] = 500
    # plt.rcParams['figure.dpi'] = 500
    # fig = plt.figure(tight_layout=True)
    # # grid row : x, coloum : y
    # size1 = 3
    # size2 = 4
    # gs = GridSpec(size1, size2)
    # for x in range(size1):
        # for y in range(size2):
            # ax = fig.add_subplot(gs[x, y])
            # ki = y * size1 + x
            # if ki < len(winmap) and winmap[ki]:
                # ax.plot(np.mean(winmap[ki], axis=0))

            # ax.set_title('Cluster %d' % (ki + 1))

    # #plt.savefig('./scri6.png')
    # plt.show()
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import kmeansdtw
import somdtw
from scipy.special import comb
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
import scipy.stats as sc
from scipy.stats.kde import gaussian_kde



def task1(filename):
    w = 15
    b = 5

    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'

    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']

    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)

    for num in range(6):
        # s1 = 746 + 1000 * (num // 2) + (num % 2) *250
        # s2 = 996 + 1000 * (num // 2) + (num % 2) *250
        s1 = 746 + 1000 * num
        s2 = 1247 + 1000 * num
        # 1252
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

        centroid, locations = knndtw.kmeans(data=data, k=2, max_iter=10, w=w, b=b, dm=1)
        samples = data.shape[0]
        count = 0

        winloc = defaultdict(list)
        for i in range(samples):
            winloc[locations[i]].append(i + 1)
        print(winloc)

    # plt.rcParams['savefig.dpi'] = 500
    # plt.rcParams['figure.dpi'] = 500
    # # for i in centroid:
    # #   plt.plot(i)
    #
    # fig = plt.figure(tight_layout=True)
    # # grid row : x, coloum : y
    # size1 = 2
    # size2 = 2
    # gs = GridSpec(size1, size2)
    #
    # for x in range(size1):
    #     for y in range(size2):
    #         ax = fig.add_subplot(gs[x, y])
    #         ki = y * size1 + x
    #         if ki < len(centroid):
    #             ax.plot(centroid[ki])
    #             ax.set_title('Cluster %d' % (ki + 1))
    # #plt.savefig(filename)
    # plt.show()


def task2(filename):
    dataset1 = np.load('ss53A.npz')
    time = dataset1['t']
    force = dataset1['f']
    voltage = dataset1['v']
    datalen = int(max(force[:, 1]))
    label1 = force[:, 0]

    data = force[:, 2:(datalen + 2)]
    fig = plt.figure(figsize=(10, 12), dpi=400)
    gs = GridSpec(4, 1)
    for j in range(0, 4):
        ax = fig.add_subplot(gs[j % 4, j // 4])
        for i in range(3):
            ax.plot(data[j*3+i, :], linestyle="-", linewidth=0.5, label="test" + str(i + 1))
        ax.set_title('Sensor SS53 - %d' % label1[j*3+i])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, labelspacing=0.)
    fig.text(0.5, 0.04, 'Time (0.02s)', va='center', ha='center', fontsize=matplotlib.rcParams['axes.labelsize'])
    fig.text(0.04, 0.5, 'Force (N)', va='center', ha='center', rotation='vertical', fontsize=matplotlib.rcParams['axes.labelsize'])
    plt.savefig(filename)
    plt.show()


def task3(filename):
    w = 15
    b = 5
    '''
    data = np.load('datamat.npz')
    sensor1 = data['s1']
    sensor2 = data['s2']
    sensor3 = data['s3']
    sensor4 = data['s4']
    sensor5 = data['s5']
    ts = data['t']
    data_all = np.concatenate((sensor1, sensor2, sensor3, sensor4, sensor5), axis=0)
    data = data_all[:, 5500:5700]

    data[data == 0] += 0.01
    data = 33 / data - 10
    data = 189.67 * (data ** 1.46)
    data = data / 1000
    
    filepath = r'E:/SocketSense/noveldata/'

    name = ['2_don.asc', '3_standing.asc', '4_6MWT.asc', '5_STS.asc', '6_gaitmon1.asc', '7_gaitmon2.asc',
            '8_gaitmon3.asc',
            '9_ramp.asc', '10_ramprep.asc', '11_stairs.asc', '12_stairsrep.asc', '13_doff.asc', '14_dondoff.asc']

    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)

    d1 = data_all[9]
    x = np.transpose(d1[:, 0])
    data = np.transpose(d1[:, 11:21])
    '''
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'

    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']

    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)

    for num in range(6):
        # s1 = 746 + 1000 * (num // 2) + (num % 2) *250
        # s2 = 996 + 1000 * (num // 2) + (num % 2) *250
        s1 = 746 + 1000 * num
        s2 = 1247 + 1000 * num
        # 1252
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, 1:21])
        data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        # data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])
        # data = zscore(data, axis=None)

        ch = data.shape[1]
        som1 = somdtw.SOMdtw(width=3, height=3, channel=ch, sigma=1.0, learning_rate=0.2, w=w, b=b)
        # dm = 1 : dtw; dm = 2 : Euc
        som1.train(data=data, n_iteration=8, batch_size=10, dm=2)
        samples = data.shape[0]

        # inference
        locations, distances = som1.find_maching_nodes(data)


        winmap = defaultdict(list)
        winloc = defaultdict(list)
        for i in range(samples):
            winmap[locations[i, 2]].append(data[i, :])
            winloc[locations[i, 2]].append(i+1)

        print(num)
        # print(locations)
        print(winloc)

    # plt.rcParams['savefig.dpi'] = 500
    # plt.rcParams['figure.dpi'] = 500
    # fig = plt.figure(tight_layout=True)
    # # grid row : x, coloum : y
    # size1 = 3
    # size2 = 3
    # gs = GridSpec(size1, size2)
    # for x in range(size1):
    #     for y in range(size2):
    #         ax = fig.add_subplot(gs[x, y])
    #         ki = x * size2 + y
    #         if ki < len(winmap) and winmap[ki]:
    #             ax.plot(np.mean(winmap[ki], axis=0))
    #
    #         ax.set_title('Cluster %d' % (ki + 1))
    #
    # #plt.savefig(filename, dpi=400)
    # plt.show()


def task4():
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    for num in range(3):
        s1 = 746 + 1000 * (num)
        s2 = 1747 + 1000 * (num)
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, :])
        # data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

        if num == 4:
            m1 = data[[0, 1, 5], :]
            m2 = data[[2, 3, 4,  6, 7, 8, 9], :]
        else:
            m1 = data[[0, 1, 3, 4], :]
            m2 = data[[2, 5, 6, 7, 8, 9], :]

        filepath = 'E:/SocketSense/PilotStudy/1121/data1000r' + str(num+1) + '.npz'
        np.savez(filepath, a=m1, b=m2)

    # distance = np.zeros(shape=(10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         if i != j:
    #             distance[i, j] = knndtw.EUCdist(data[i, :], data[j, :])
    # plt.imshow(distance, cmap='hot')
    # plt.colorbar()
    # plt.show()


def task5():
    filepath = r'E:/SocketSense/PilotStudy/mar10/'  # filepath to the folder
    name = 'QTSS_10032021.txt'  # text filename
    filename = filepath + name
    f = open(filename, 'r')
    linek = f.readlines()
    i = 0
    length = len(linek)
    sensor = np.zeros(shape=(5, 8, length))
    gait = np.zeros(shape=(1, length))
    tag = np.zeros(shape=(1, length))
    time = np.zeros(shape=(1, length))
    for line in linek:
        for j in range(8):
            sensor[0, j, i] = line[12 + 10 * j:16 + 10 * j]
            sensor[1, j, i] = line[92 + 10 * j:96 + 10 * j]
            sensor[2, j, i] = line[172 + 10 * j:176 + 10 * j]
            sensor[3, j, i] = line[252 + 10 * j:256 + 10 * j]
            sensor[4, j, i] = line[332 + 10 * j:336 + 10 * j]

        gait[0, i] = line[409:409 + 15]  # identify the gait (may changed according to real data)
        time[0, i] = line[433:]  # identify the time (may changed according to real data)
        i = i + 1

    f.close()

    # clarify data (eliminate 0 and 3.3)
    sensor[sensor == 0] += 0.005
    sensor[sensor == 3.3] -= 0.005
    sensor = 33 / sensor - 10
    sensor = 100 / sensor
    # sensor data in resistance (kOhm) -> sensor data in pressure (kPa)

    # ss56 -> s1 (E1-E8)
    sensor[0, 0] = 189.67 * (sensor[0, 0] ** -1.46)
    sensor[0, 1] = 187.31 * (sensor[0, 1] ** -1.295)
    sensor[0, 2] = 137.51 * (sensor[0, 2] ** -1.661)
    sensor[0, 3] = 124.65 * (sensor[0, 3] ** -1.418)
    sensor[0, 4] = 157.09 * (sensor[0, 4] ** -1.203)
    sensor[0, 5] = 152.85 * (sensor[0, 5] ** -1.293)
    sensor[0, 6] = 121.64 * (sensor[0, 6] ** -1.395)
    sensor[0, 7] = 109.46 * (sensor[0, 7] ** -1.626)

    # ss59 -> s2 (E1-E8)
    sensor[1, 0] = 149.96 * (sensor[1, 0] ** -1.207)
    sensor[1, 1] = 127.74 * (sensor[1, 1] ** -1.841)
    sensor[1, 2] = 199.87 * (sensor[1, 2] ** -1.238)
    sensor[1, 3] = 152.78 * (sensor[1, 3] ** -1.438)
    sensor[1, 4] = 132.26 * (sensor[1, 4] ** -1.337)
    sensor[1, 5] = 120.37 * (sensor[1, 5] ** -1.465)
    sensor[1, 6] = 94.392 * (sensor[1, 6] ** -1.750)
    sensor[1, 7] = 89.005 * (sensor[1, 7] ** -1.684)

    # ss57 -> s3 (E1-E8)
    sensor[2, 0] = 106.63 * (sensor[2, 0] ** -1.945)
    sensor[2, 1] = 105.86 * (sensor[2, 1] ** -1.804)
    sensor[2, 2] = 141.64 * (sensor[2, 2] ** -1.593)
    sensor[2, 3] = 120.81 * (sensor[2, 3] ** -1.875)
    sensor[2, 4] = 161.32 * (sensor[2, 4] ** -0.691)
    sensor[2, 5] = 146.64 * (sensor[2, 5] ** -1.438)
    sensor[2, 6] = 87.490 * (sensor[2, 6] ** -1.914)
    sensor[2, 7] = 71.587 * (sensor[2, 7] ** -1.808)

    # ss58 -> s4 (E1-E8)
    sensor[3, 0] = 164.66 * (sensor[3, 0] ** -1.104)
    sensor[3, 1] = 118.64 * (sensor[3, 1] ** -1.858)
    sensor[3, 2] = 113.7 * (sensor[3, 2] ** -1.8)
    sensor[3, 3] = 107.69 * (sensor[3, 3] ** -1.659)
    sensor[3, 4] = 148.07 * (sensor[3, 4] ** -1.348)
    sensor[3, 5] = 118.55 * (sensor[3, 5] ** -1.838)
    sensor[3, 6] = 106.16 * (sensor[3, 6] ** -1.929)
    sensor[3, 7] = 117.17 * (sensor[3, 7] ** -1.473)

    # ss60 -> s5 (E1-E8)
    sensor[4, 0] = 182.69 * (sensor[4, 0] ** -1.084)
    sensor[4, 1] = 264.14 * (sensor[4, 1] ** -1.168)
    sensor[4, 2] = 251.56 * (sensor[4, 2] ** -1.111)
    sensor[4, 3] = 209.71 * (sensor[4, 3] ** -1.553)
    sensor[4, 4] = 175.13 * (sensor[4, 4] ** -1.628)
    sensor[4, 5] = 207.28 * (sensor[4, 5] ** -0.81)
    sensor[4, 6] = 230.42 * (sensor[4, 6] ** -1.426)
    sensor[4, 7] = 204.35 * (sensor[4, 7] ** -0.843)


    # output S1E1 -> S5E8, gait, tag, time (40 columns, each row -> each data frame)
    # you can change the order accordingly
    sensor = sensor.reshape(40, length)
    st1 = 11458 + 200
    st2 = 11560 + 200
    data = sensor[:, st1:st2]
    distance = np.zeros(shape=(40, 40))
    for i in range(40):
        for j in range(40):
            if i != j:
                distance[i, j] = knndtw.EUCdist(data[i, :], data[j, :])

    e6 = data[29, :]
    e7 = data[30, :]
    np.savez('E:/SocketSense/PilotStudy/dataloose.npz', a=e6, b=e7)

    plt.imshow(distance, cmap='viridis')
    plt.colorbar()
    plt.show()

# euclidean distance
def task6():
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    for num in range(6):
        s1 = 746 + 1000 * (num+1)
        s2 = 1252 + 1000 * (num+1)
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, :])
        data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        # data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

        m1 = data[[0, 1, 5], :]
        m2 = data[[2, 3, 4, 6, 7, 8, 9], :]

        am1 = np.mean(m1, axis=0)
        am2 = np.mean(m2, axis=0)

        mindist = 200
        d1 = []
        for i in range(3):
            for j in range(i+1, 3):
                t1 = np.mean(m1[[i, j], :], axis=0)
                dist = knndtw.EUCdist(am1, t1)
                if (dist <= mindist):
                    mindist = dist
                    si = i
                    sj = j
                d1.append(dist)
        print("s1 = %d, i = %d, j = %d, dist = %.3f" % (num+1, si, sj, mindist))

        d2 = []
        mindist = 200
        for i in range(7):
            for j in range(i+1, 7):
                t2 = np.mean(m2[[i, j], :], axis=0)
                dist = knndtw.EUCdist(am2, t2)
                if (dist <= mindist):
                    mindist = dist
                    si = i
                    sj = j
                d2.append(dist)
        print("s2 = %d, i = %d, j = %d, dist = %.3f" % (num+1, si, sj, mindist))

        plt.plot(am2, label="Smean")
        plt.plot(np.mean(m2[[2, 5], :], axis=0), label="S" + str(1))
        plt.plot(np.mean(m2[[1, 4], :], axis=0), label="S" + str(2))
        plt.legend(title='', loc='upper right')
        plt.show()


        # np.savetxt('E:/SocketSense/PilotStudy/result3/s' + str(num+1) + 'd1.txt', d1, fmt='%.3f')
        # np.savetxt('E:/SocketSense/PilotStudy/result3/s' + str(num+1) + 'd2.txt', d2, fmt='%.3f')

# euclidean distance
def task7():
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    avg1 = np.zeros(shape=(12, 10))
    avg2 = np.zeros(shape=(12, 10))
    pre1 = np.zeros(shape=(12, 1))
    pre2 = np.zeros(shape=(12, 1))
    for num in range(12):
        s1 = 746 + 1000 * (num // 2) + (num % 2) * 250
        s2 = 996 + 1000 * (num // 2) + (num % 2) * 250
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, :])
        # data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

        m1 = data[[0, 1, 3, 4, 5], :]
        m2 = data[[2, 6, 7, 8, 9], :]

        am1 = np.mean(m1, axis=0)
        am2 = np.mean(m2, axis=0)

        # np.savez('E:/SocketSense/PilotStudy/1121/dataright' + str(num+1) + '.npz', a=m1, b=m2)

        mindist = 200
        d1 = []
        count = 0
        for i in range(5):
            for j in range(i+1, 5):

                t1 = np.mean(m1[[i, j], :], axis=0)
                dist = knndtw.DTWdist(am1, t1, 30)
                if (dist <= mindist):
                    mindist = dist
                    si = i
                    sj = j
                d1.append(dist)
                avg1[num, count] = dist
                count = count + 1
        print("s1 = %d, i = %d, j = %d, dist = %.3f" % (num+1, si, sj, mindist))
        pre1[num, 0] = np.mean(m1[[0, 4], :].flatten())

        d2 = []
        mindist = 200
        count = 0
        for i in range(5):
            for j in range(i+1, 5):
                t2 = np.mean(m2[[i, j], :], axis=0)
                dist = knndtw.DTWdist(am2, t2, 30)
                if (dist <= mindist):
                    mindist = dist
                    si = i
                    sj = j
                d2.append(dist)
                avg2[num, count] = dist
                count = count + 1
        print("s2 = %d, i = %d, j = %d, dist = %.3f" % (num+1, si, sj, mindist))
        pre2[num, 0] = np.mean(m2[[1, 4], :].flatten())

        # np.savetxt('E:/SocketSense/PilotStudy/1121/s' + str(num+1) + 'd1.txt', d1, fmt='%.3f')
        # np.savetxt('E:/SocketSense/PilotStudy/1121/s' + str(num+1) + 'd2.txt', d2, fmt='%.3f')
    # avgt1 = np.mean(avg1, axis=0)
    # avgt2 = np.mean(avg2, axis=0)
    # print(avgt1, avgt2)
    # print(np.argmin(avgt1), np.argmin(avgt2))
    print(np.mean(pre1), np.mean(am1), np.mean(pre2), np.mean(am2))


def task8():
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    for num in range(6):
        s1 = 746 + 1000 * (num+1)
        s2 = 1252 + 1000 * (num+1)
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, :])
        # data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

        m1 = data[[0, 1, 3, 4, 5], :]
        m2 = data[[2, 6, 7, 8, 9], :]

        am1 = np.mean(m1, axis=0)
        am2 = np.mean(m2, axis=0)

        mindist = 1000
        d1 = []
        for i in range(5):
            t1 = m1[[i], :].flatten()
            dist = knndtw.EUCdist(am1, t1)
            if (dist <= mindist):
                mindist = dist
                si = i
            d1.append(dist)
        print("s1 = %d, i = %d, dist = %.3f" % (num+1, si, mindist))

        d2 = []
        mindist = 1000
        for i in range(5):
            t2 = m2[[i], :].flatten()
            dist = knndtw.EUCdist(am2, t2)
            if (dist <= mindist):
                mindist = dist
                si = i

            d2.append(dist)
        print("s2 = %d, i = %d, dist = %.3f" % (num+1, si, mindist))

        # plt.plot(am2, label="Smean")
        # plt.plot(m2[3, :], label="S" + str(1))
        # plt.plot(m2[4, :], label="S" + str(2))
        # plt.legend(title='', loc='upper right')
        # plt.show()

        np.savetxt('E:/SocketSense/PilotStudy/result2/single' + str(num + 1) + 'd1.txt', d1, fmt='%.3f')
        np.savetxt('E:/SocketSense/PilotStudy/result2/single' + str(num + 1) + 'd2.txt', d2, fmt='%.3f')


# plot
def task9():
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    num = 0
    s1 = 746 + 1000 * (num+1)
    s2 = 1452 + 1000 * (num+1)
    d1 = data_all[1]
    x = np.transpose(d1[s1:s2, 0])
    data = np.transpose(d1[s1:s2, [1, 2]])
    # data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
    # data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

    # m1 = data[[0, 1, 5], :]
    # m2 = data[[2, 3, 4, 6, 7, 8, 9], :]
    #
    # am1 = np.mean(m1, axis=0)
    # am2 = np.mean(m2, axis=0)
    #
    # plt.plot(am2, label="Smean")

    plt.plot(data[0, :], label="L" + str(1))
    plt.plot(data[1, :], label="R" + str(1))
    plt.xlabel("Points")
    plt.ylabel("Pressure (kPa)")
    # plt.plot(np.mean(m2[[0, 4], :], axis=0), label="S" + str(2))
    plt.legend(title='', loc='upper right')
    plt.show()


def task10():
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    slp = 6
    avg1 = np.zeros(shape=(slp, 10))
    avg2 = np.zeros(shape=(slp, 10))
    pre1 = np.zeros(shape=(slp, 10))
    pre2 = np.zeros(shape=(slp, 10))
    pre3 = np.zeros(shape=(slp, 1))
    pre4 = np.zeros(shape=(slp, 1))
    mse1 = np.zeros(shape=(10, 2))

    for num in range(slp):
        s1 = 746 + 1000 * num
        s2 = 1252 + 1000 * num
        #1252
        # s1 = 746 + 1000 * (num // 2) + (num % 2) *250
        # s2 = 996 + 1000 * (num // 2) + (num % 2) *250
        d1 = data_all[1]
        x = np.transpose(d1[s1:s2, 0])
        # data = np.transpose(d1[s1:s2, :])
        # data = np.transpose(d1[s1:s2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]])
        data = np.transpose(d1[s1:s2, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])

        m1 = data[[0, 1, 3, 4, 5], :]
        m2 = data[[2, 6, 7, 8, 9], :]
        # m1 = data[[0, 1, 3, 4], :]
        # m2 = data[[2, 5, 6, 7, 8, 9], :]

        am1 = np.mean(m1, axis=0)
        am2 = np.mean(m2, axis=0)

        # np.savez('E:/SocketSense/PilotStudy/1121/dataright' + str(num+1) + '.npz', a=m1, b=m2)

        mindist = 200
        d1 = []
        count = 0
        pre3[num, 0] = np.mean(m1.flatten())
        for i in range(5):
            for j in range(i+1, 5):

                t1 = np.mean(m1[[i, j], :], axis=0)
                dist = knndtw.EUCdist(am1, t1)
                if (dist <= mindist):
                    mindist = dist
                    si = i
                    sj = j
                d1.append(dist)
                avg1[num, count] = dist
                pre1[num, count] = np.mean(m1[[i, j], :].flatten())
                count = count + 1
        print("s1 = %d, i = %d, j = %d, dist = %.3f" % (num+1, si, sj, mindist))

        d2 = []
        mindist = 200
        count = 0
        pre4[num, 0] = np.mean(m2.flatten())
        for i in range(5):
            for j in range(i+1, 5):
                t2 = np.mean(m2[[i, j], :], axis=0)
                dist = knndtw.EUCdist(am2, t2)
                if (dist <= mindist):
                    mindist = dist
                    si = i
                    sj = j
                d2.append(dist)
                avg2[num, count] = dist
                pre2[num, count] = np.mean(m2[[i, j], :].flatten())
                count = count + 1
        print("s2 = %d, i = %d, j = %d, dist = %.3f" % (num+1, si, sj, mindist))
        # pre2[num, 0] = np.mean(m2[[1, 3], :].flatten())

        # np.savetxt('E:/SocketSense/PilotStudy/1121/s' + str(num+1) + 'd1.txt', d1, fmt='%.3f')
        # np.savetxt('E:/SocketSense/PilotStudy/1121/s' + str(num+1) + 'd2.txt', d2, fmt='%.3f')
    avgt1 = np.mean(avg1, axis=0)
    avgt2 = np.mean(avg2, axis=0)
    print(avgt1, avgt2)

    for k in range(10):
        # print(pre1, pre3)
        # print(pre2, pre4)
        mse1[k, 0] = np.square(np.subtract(pre1[:, k].flatten(), pre3)).mean()
        mse1[k, 1] = np.square(np.subtract(pre2[:, k].flatten(), pre4)).mean()
        print(k, mse1[k, 0], mse1[k, 1])
    print(np.argmin(avgt1), np.argmin(avgt2))
    print(np.argmin(mse1[:, 0].flatten()), np.argmin(mse1[:, 1].flatten()))


def kl_divergence(p, q):
    return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))

def JS_divergence(p,q):
    M = (p+q)/2
    return 0.5*sc.entropy(p, M, base=2)+0.5*sc.entropy(q, M, base=2)


def task11(ki=0, kj=1):
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    d1 = data_all[1]
    d1 = d1[:, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
    slp = 5
    s1 = 746
    s2 = 1252
    p1 = [0, 1, 3, 4, 5]
    p2 = [2, 6, 7, 8, 9]
    m1 = d1[s1:s2, p1].flatten()
    m2 = d1[s1:s2, p2].flatten()
    am1 = d1[s1:s2, [p1[ki], p1[kj]]].flatten()
    am2 = d1[s1:s2, [p2[ki], p2[kj]]].flatten()
    # am1 = d1[s1:s2, p1[ki]].flatten()
    # am2 = d1[s1:s2, p2[ki]].flatten()
    for num in range(slp):
        s1 = s1 + 1000 * num
        s2 = s2 + 1000 * num

        m1 = np.append(m1, d1[s1:s2, p1])
        m2 = np.append(m2, d1[s1:s2, p2])
        am1 = np.append(am1, d1[s1:s2, [p1[ki], p1[kj]]])
        am2 = np.append(am2, d1[s1:s2, [p2[ki], p2[kj]]])
        # am2 = np.append(am2, d1[s1:s2, p2[ki]])

    # m1 = data[[0, 1, 5], :]
    # m2 = data[[2, 3, 4, 6, 7, 8, 9], :]

    # kde1 = gaussian_kde(m2.flatten())
    # kde2 = gaussian_kde(am2.flatten())
    kde1, bins1 = np.histogram(m2.flatten(), bins=64, density=True)
    kde2, bins2 = np.histogram(am2.flatten(), bins=bins1, density=True)
    #
    # width = 0.7 * (bins1[1] - bins1[0])
    # center = (bins1[:-1] + bins1[1:]) / 2
    # plt.bar(center, kde1, align='center', width=width)
    # plt.xlabel("Pressure (KPa)")
    # plt.ylabel("Frequency")
    # plt.yticks(np.arange(0, 0.5, 0.05))
    # plt.show()
    #
    # width = 0.7 * (bins2[1] - bins2[0])
    # center = (bins2[:-1] + bins2[1:]) / 2
    # plt.bar(center, kde2, align='center', width=width)
    # plt.xlabel("Pressure (KPa)")
    # plt.ylabel("Frequency")
    # plt.yticks(np.arange(0, 0.5, 0.05))
    # plt.show()
    print(np.mean(m1.flatten()), np.mean(am1.flatten()))
    kl_pq = JS_divergence(kde1, kde2)
    # print('JS: %.4f' % kl_pq)
    edg = knndtw.EUCdist(kde1, kde2)
    return kl_pq, edg


def task12(ki=0, kj=1):
    filepath = r'E:/SocketSense/PilotStudy/mar10/novel/'
    name = ['7_donning.asc', '8_walking.asc', '9_sts.asc', '10_ramp.asc', '11_doffing.asc']
    data_all = []
    for x in name:
        tmppath = filepath + x
        tmpdata = np.loadtxt(tmppath, skiprows=7)
        data_all.append(tmpdata)
    d1 = data_all[1]
    d1 = d1[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
    slp = 5
    s1 = 746
    s2 = 1252
    p1 = [0, 1, 5]
    p2 = [2, 3, 4, 6, 7, 8, 9]
    m1 = d1[s1:s2, p1].flatten()
    m2 = d1[s1:s2, p2].flatten()
    # am1 = d1[s1:s2, [p1[ki], p1[kj]]].flatten()
    am2 = d1[s1:s2, [p2[ki], p2[kj]]].flatten()
    # am1 = d1[s1:s2, p1[ki]].flatten()
    # am2 = d1[s1:s2, p2[ki]].flatten()
    for num in range(slp):
        s1 = s1 + 1000 * num
        s2 = s2 + 1000 * num

        m1 = np.append(m1, d1[s1:s2, p1])
        m2 = np.append(m2, d1[s1:s2, p2])
        # am1 = np.append(am1, d1[s1:s2, [p1[ki], p1[kj]]])
        am2 = np.append(am2, d1[s1:s2, [p2[ki], p2[kj]]])
        # am1 = np.append(am1, d1[s1:s2, p1[ki]])
        # am2 = np.append(am2, d1[s1:s2, p2[ki]])

    # m1 = data[[0, 1, 5], :]
    # m2 = data[[2, 3, 4, 6, 7, 8, 9], :]
    kde1, bins1 = np.histogram(m2.flatten(), bins=64, density=True)
    kde2, bins2 = np.histogram(am2.flatten(), bins=bins1, density=True)

    # width = 0.7 * (bins1[1] - bins1[0])
    # center = (bins1[:-1] + bins1[1:]) / 2
    # plt.bar(center, kde1, align='center', width=width)
    # plt.xlabel("Pressure (KPa)")
    # plt.ylabel("Frequency")
    # plt.yticks(np.arange(0, 0.09, 0.01))
    # plt.show()
    #
    # width = 0.7 * (bins2[1] - bins2[0])
    # center = (bins2[:-1] + bins2[1:]) / 2
    # plt.bar(center, kde2, align='center', width=width)
    # plt.xlabel("Pressure (KPa)")
    # plt.ylabel("Frequency")
    # plt.yticks(np.arange(0, 0.09, 0.01))
    # plt.show()
    print(np.mean(m2.flatten()), np.mean(am2.flatten()))
    kl_pq = JS_divergence(kde1, kde2)
    # print('JS: %.4f' % kl_pq)
    edg = knndtw.EUCdist(kde1, kde2)
    return kl_pq, edg



if __name__ == '__main__':
    print('start')
    # task1('./fig/ss60_k1.png')
    # task2('./fig/ss53_of1.png')
    # task3('./fig/ss60_som12.png')
    # task4()
    # task5()
    # task9()
    # js_pq = []
    # edg = []
    # count = 1
    # for i in range(5):
    #     for j in range(i + 1, 5):
    #         # print(count)
    #         tmp1, tmp2 = task11(i, j)
    #         js_pq.append(tmp1)
    #         edg.append(tmp2)
    #         count = count + 1
    # # for i in range(5):
    # #     # print(count)
    # #     tmp1, tmp2 = task11(i, 0)
    # #     js_pq.append(tmp1)
    # #     edg.append(tmp2)
    # #     count = count + 1
    # print(np.argmin(js_pq) + 1, np.min(js_pq))
    # print(np.argmin(edg) + 1, np.min(edg))
    tmp1, tmp2 = task12(3, 3)
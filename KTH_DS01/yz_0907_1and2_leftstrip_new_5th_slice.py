### readme:
## 1.Before running code
# Change the data location of line 13 to your own address
## 2. Run the code
import  numpy as np
import matplotlib.pylab as plt
k1_end=3
k2_end=7#n2 class contains k2_end sensors/curves
###load tight data
slice_number=3
if (slice_number==1):
    tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal1.npz')
    print("\rleft strip data first\n")
elif  (slice_number==2):
    tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal2.npz')
    print("\rleft strip data second\n")
elif   (slice_number==3):
    tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal3.npz')
    print("\rleft strip data third\n")

elif    slice_number==4:
    tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal4.npz')
    print("\rleft strip data fourth\n")
elif slice_number == 5:
    tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal5.npz')
    print("\rleft strip data fifth\n")
elif   slice_number==6:
    tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal6.npz')
    print("\rleft strip data sixth\n")

#tight_data_1=np.load('/home/yz/PycharmProjects/0807_cluster/clustering/data/data_0914_slice5-recluster/datal5.npz')
#print("\rleft strip data fifth\n")
###############
#############
tight_data_1_a=tight_data_1["a"]
tight_data_1_a_n2temp=tight_data_1_a[0].copy()
for i_sensor in range(1,k1_end):
    tight_data_1_a_n2temp +=tight_data_1_a[i_sensor]
tight_data_1_a_n2=tight_data_1_a_n2temp/k1_end
###########
pcc_1sensor=[]
for i_sensor in range(0,k1_end):
    x=tight_data_1_a_n2
    y=tight_data_1_a[i_sensor]
    pccs = np.corrcoef(x, y)
    pcc_1sensor.append(pccs[0, 1])
    print(pccs[0,1],' i',i_sensor,'pccs of one sensor in "a" and curve a after clustering')
print(np.max(pcc_1sensor),'best pcc for one sensor in first cluster')
m_position = np.argmax(pcc_1sensor)
print("position",m_position," (start from 0)")
####
print("\r\n")
############
pcc_2sensors=[]
for i_sensor in range(0,k1_end):
    for j_sensor in range(0,k1_end):
       x = tight_data_1_a_n2
       y=(tight_data_1_a[i_sensor]+tight_data_1_a[j_sensor])/2
       if (i_sensor <j_sensor  ):
            pccs = np.corrcoef(x, y)
            pcc_2sensors.append(pccs[0, 1])
            print(pccs[0, 1], ' i', i_sensor,'j',j_sensor, 'pccs of two sensors in "a" and curve a after clustering')
print(np.max(pcc_2sensors),'best pcc for two sensors in first cluster')
m_position = np.argmax(pcc_2sensors)
print("position",m_position," (start from 0)")
####
print("\r\n  below second cluster")
##########
tight_data_1_b=tight_data_1["b"]
tight_data_1_b_n2temp=tight_data_1_b[0].copy()

for i_sensor in range(1,k2_end):
    tight_data_1_b_n2temp +=tight_data_1_b[i_sensor]
tight_data_1_b_n2=tight_data_1_b_n2temp/k2_end
##########
####
pcc_1sensor=[]
for i_sensor in range(0,k2_end):
    x=tight_data_1_b_n2
    y=tight_data_1_b[i_sensor]
    pccs = np.corrcoef(x, y)
    pcc_1sensor.append(pccs[0, 1])
    print(pccs[0,1],' i',i_sensor,'pccs of one sensor in "b" and curve b after clustering')
print(np.max(pcc_1sensor),'best pcc for one sensor in first cluster')
m_position = np.argmax(pcc_1sensor)
print("position",m_position," (start from 0)")
#######################
print("  \n 2 sensors  \r")
pcc_2sensors=[]
for i_sensor in range(0,k2_end):
    for j_sensor in range(0,k2_end):
        if (i_sensor<j_sensor):
          x = tight_data_1_b_n2
          y=(tight_data_1_b[i_sensor]+tight_data_1_b[j_sensor])/2
          pccs = np.corrcoef(x, y)
          pcc_2sensors.append(pccs[0, 1])
          print(pccs[0, 1], ' i', i_sensor,'j',j_sensor, 'two sensors pccs of part of sensors in "b" and curve b after clustering')
print(np.max(pcc_2sensors),'best pcc for two sensors in second cluster ')
m_position = np.argmax(pcc_2sensors)
print("position",m_position," (start from 0)")
#################


#####draw figures
plot_switch=1
if plot_switch==1:

    plt.figure(1)
    plt.plot(tight_data_1_b_n2,label='cluster curver b')
    plt.plot((tight_data_1_b[1]+tight_data_1_b[4])/2,label='first one of the 7 senosors in cluster b')
    plt.plot((tight_data_1_b[2] + tight_data_1_b[5]) / 2, label='second of the 7 senosors in cluster b')
    print(np.corrcoef(tight_data_1_b_n2, (tight_data_1_b[1]+tight_data_1_b[4])/2),"1  4 pcc")
    print(np.corrcoef(tight_data_1_b_n2, (tight_data_1_b[2] + tight_data_1_b[5]) / 2), "2  5 pcc")
    plt.legend()
    #plt.title('tight_data_1_b_n2')
    plt.title('curve of 506 data points')
    plt.xlabel('data points')
    plt.ylabel('pressure/kPa')
    plt.show()

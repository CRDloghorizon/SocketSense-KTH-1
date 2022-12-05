import matplotlib.pyplot as plt
import numpy as np
num_list = [0.9991,
0.9997,
0.9991,

]
num_list_b=[
0.9937,
0.9956,
0.9983,
0.9994,
0.9992,
0.9969,
0.9958,
0.9985,
0.9992,
0.9990,
0.9966,
0.9992,
0.9990,
0.9990,
0.9973,
0.9975,
0.9974,
0.9923,
0.9936,
0.9911,
0.9876,]
num_list_all=[0.9991,
0.9997,
0.9991,
0.9937,
0.9956,
0.9983,
0.9994,
0.9992,
0.9969,
0.9958,
0.9985,
0.9992,
0.9990,
0.9966,
0.9992,
0.9990,
0.9990,
0.9973,
0.9975,
0.9974,
0.9923,
0.9936,
0.9911,
0.9876,]
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,]
x_name=['a0+a1',
        'a0+a2',
        'a1+a2',
        'b0+b1',
        'b0+b2',
        'b0+b3',
        'b0+b4',
        'b0+b5',
        'b0+b6',
        'b1+b2',
        'b1+b3',
        'b1+b4',
        'b1+b5',
        'b1+b6',
        'b2+b3',
        'b2+b4',
        'b2+b5',
        'b2+b6',
        'b3+b4',
        'b3+b5',
        'b3+b6',
        'b4+b5',
        'b4+b6',
        'b5+b6',]
print(len(x_name))
plt.xticks(fontsize=4)
plt.bar(x_name, num_list_all, fc='b')
plt.ylim((0.98, 1))
plt.xlabel('Different sensor selection ')
plt.ylabel('PCC')
plt.title('Average PCC for selecting two sensors in dataset L')
plt.show()



y_euc=[
54.3120,
29.6658,
51.9038,
97.3262,
97.3245,
40.8767,
57.9795,
32.1562,
38.3392,
80.9462,
23.4950,
40.2305,
19.7285,
55.5222,
23.6560,
41.3943,
19.1040,
54.2905,
28.6093,
43.9187,
111.2762,
30.9385,
94.2193,
119.0815,]
x=np.arange(len(x_name))

fig, ax1 = plt.subplots()
plt.xticks(fontsize=4)
width=0.4

ax1.set_ylim(0.98, 1)
ax1.bar(x-width/2,num_list_all, width,fc='skyblue',label='PCC')
plt.legend()
plt.xticks(x, labels=x_name) 


ax2 = ax1.twinx()
ax2.set_ylabel('Euc')

ax2.set_ylim(0, 120)
ax2.bar(x+width/2,y_euc, width,fc='darkorange',label='Euc')
plt.tight_layout()
plt.savefig("similarity.png")
plt.legend()
ax1.set_ylabel('PCC')
ax1.legend(loc=2,bbox_to_anchor=(1.1,1.0),borderaxespad=0)
ax2.legend(loc=2,bbox_to_anchor=(1.1,0.9),borderaxespad=0)
for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
plt.tight_layout()
plt.savefig('results_0915.png',dpi=400)
plt.show()

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def round_of_rating(number):
    """Round a number to the closest half integer.
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0"""

    return round(number * 2) / 2




# For the 8D results

G = [4.732481344595529, 0.0008015050731737095, 0.9375537411490803, 1.3983400569047024e-06]
E = [8.603341686611616, 0.0014450276555567749, 0.7936222151596996, 4.6213548851085744e-06]
M3 = [5.187187651822027, 0.0008712950129322388, 0.9249773289648779, 1.679959824892701e-06]
M5 = [4.652556947883, 0.0007949369054138386, 0.9396451744528516, 1.3515072278638205e-06]
EM = [4.627975668015783, 0.0007841498989530843, 0.9402812460401024, 1.3372638705213834e-06]
CKL = [4.321021618927791, 0.000766755319057223, 0.9479403214701506, 1.1657565269308036e-06]
MCKL = [4.378409909289078, 0.000743695751844258, 0.9465483101571535, 1.1969274123360506e-06]

G_tt = [6.064405679702759]
E_tt = [6.502922058105469]
M3_tt = [8.629276275634766]
M5_tt = [5.5541300773620605]
EM_tt = [26.93177318572998]
CKL_tt = [13.691071510314941]
MCKL_tt = [96.53869986534119]

ensemble_weight = [4.83018607e-01,2.50045450e-14, 5.62373283e-03 ,5.11357660e-01]
ckl_weight = [0.88840913, 0.00108002, 1.00000001e-05, 0.11050085]

'''
#select metric
m = 0

max_xnumber = max(G[m],E[m],M3[m],M5[m],EM[m],CKL[m],MCKL[m]) 
# max_xnumber = round_of_rating(max_xnumber) + 0.25
max_xnumber = roundup(max_xnumber) #round up to the nearest 10 
xnumbers = np.linspace(0,max_xnumber,6)
fig, ax = plt.subplots(figsize=(20,18))

methods = ('G','E','$M_{3/2}$','$M_{5/2}$','EM','CKL','MCKL')
y_pos = np.arange(len(methods))
performance = [G[m],E[m],M3[m],M5[m],EM[m],CKL[m],MCKL[m]]
ax.barh(y_pos, performance, align='center',color=['blue','orange','green','red','gold','brown','pink','purple','magenta','cyan'])
ax.set_yticks(y_pos)
ax.set_yticklabels(methods,fontsize=50)
plt.xticks(xnumbers)
ax.set_xticklabels(xnumbers,fontsize=50)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('NRMSE (\%)',fontsize=60)
fig.savefig("single_8D_NRMSE.png",dpi=600)
fig.savefig("single_8D_NRMSE.pdf",dpi=600)

'''

'''
#select metric
m = 2

max_xnumber = max(G[m],E[m],M3[m],M5[m],EM[m],CKL[m],MCKL[m]) 
# max_xnumber = round_of_rating(max_xnumber) + 0.25
# max_xnumber = roundup(max_xnumber) #round up to the nearest 10 
xnumbers = np.linspace(0,1,6)
fig, ax = plt.subplots(figsize=(20,18))

methods = ('G','E','$M_{3/2}$','$M_{5/2}$','EM','CKL','MCKL')
y_pos = np.arange(len(methods))
performance = [G[m],E[m],M3[m],M5[m],EM[m],CKL[m],MCKL[m]]
ax.barh(y_pos, performance, align='center',color=['blue','orange','green','red','gold','brown','pink','purple','magenta','cyan'])
ax.set_yticks(y_pos)
ax.set_yticklabels(methods,fontsize=50)
plt.xticks(xnumbers)
ax.set_xticklabels(xnumbers,fontsize=50)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('$R^2$ score',fontsize=60)
fig.savefig("single_8D_R2.png",dpi=600)
fig.savefig("single_8D_R2.pdf",dpi=600)
'''

# '''
#select metric
m = 0

max_xnumber = max(G_tt[m],E_tt[m],M3_tt[m],M5_tt[m],EM_tt[m],CKL_tt[m],MCKL_tt[m]) 
# max_xnumber = round_of_rating(max_xnumber) + 0.25
max_xnumber = roundup(max_xnumber) #round up to the nearest 10 
xnumbers = np.linspace(0,max_xnumber,6)
fig, ax = plt.subplots(figsize=(20,18))

methods = ('G','E','$M_{3/2}$','$M_{5/2}$','EM','CKL','MCKL')
y_pos = np.arange(len(methods))
performance = [G_tt[m],E_tt[m],M3_tt[m],M5_tt[m],EM_tt[m],CKL_tt[m],MCKL_tt[m]]
ax.barh(y_pos, performance, align='center',color=['blue','orange','green','red','gold','brown','pink','purple','magenta','cyan'])
ax.set_yticks(y_pos)
ax.set_yticklabels(methods,fontsize=50)
plt.xticks(xnumbers)
ax.set_xticklabels(xnumbers,fontsize=50)
plt.xscale('log')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Training time (s)',fontsize=60)
fig.savefig("single_8D_training_time_n.png",dpi=600)
fig.savefig("single_8D_training_time_n.pdf",dpi=600)
# '''



'''
# ensemble weight 

max_xnumber = max(ensemble_weight) 
# max_xnumber = round_of_rating(max_xnumber) + 0.25
max_xnumber = roundup(max_xnumber) #round up to the nearest 10 
xnumbers = np.linspace(0,1,6)
fig, ax = plt.subplots(figsize=(20,18))

methods = ('G','E','$M_{3/2}$','$M_{5/2}$')
y_pos = np.arange(len(methods))
ax.barh(y_pos, ensemble_weight, align='center',color=['blue','orange','green','red','gold','brown','pink','purple','magenta','cyan'])
ax.set_yticks(y_pos)
ax.set_yticklabels(methods,fontsize=50)
plt.xticks(xnumbers)
ax.set_xticklabels(xnumbers,fontsize=50)
plt.xscale('log')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Ensemble weight',fontsize=60)
fig.savefig("single_8D_ensemble_weight.png",dpi=600)
fig.savefig("single_8D_ensemble_weight.pdf",dpi=600)

'''

'''
# CKL weight 

max_xnumber = max(ckl_weight) 
# max_xnumber = round_of_rating(max_xnumber) + 0.25
max_xnumber = roundup(max_xnumber) #round up to the nearest 10 
xnumbers = np.linspace(0,1,6)
fig, ax = plt.subplots(figsize=(20,18))

methods = ('G','E','$M_{3/2}$','$M_{5/2}$')
y_pos = np.arange(len(methods))
ax.barh(y_pos, ckl_weight, align='center',color=['blue','orange','green','red','gold','brown','pink','purple','magenta','cyan'])
ax.set_yticks(y_pos)
ax.set_yticklabels(methods,fontsize=50)
plt.xticks(xnumbers)
ax.set_xticklabels(xnumbers,fontsize=50)
plt.xscale('log')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('CKL weight',fontsize=60)
fig.savefig("single_8D_ckl_weight.png",dpi=600)
fig.savefig("single_8D_ckl_weight.pdf",dpi=600)

'''
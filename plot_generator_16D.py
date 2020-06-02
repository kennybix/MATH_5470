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




# For the 16D results

G = [17.570091527939475, 0.0041356864204981384, 0.06652132649428488, 2.586311741797258e-05]
E = [12.34228322916483, 0.002780191529908852, 0.5393751441376433, 1.2762149871152775e-05]
M3 = [12.974874384443897, 0.0030093308272185677, 0.4909473805727631, 1.4103897648490341e-05]
M5 = [12.987325703194195, 0.002995543385162461, 0.4899698886072944, 1.4130980205591618e-05]
EM = [12.884405427038073, 0.0029867358792916767, 0.49802149961078834, 1.3907900914443859e-05]
CKL = [12.117659036763072, 0.0028486485800194913, 0.5559889185417115, 1.2301845838116405e-05]
MCKL = [11.3923778992064, 0.0026546614792607327, 0.6075492911235915, 1.08733054675137e-05]

G_tt = [10.172569274902344]
E_tt = [47.66709923744202]
M3_tt = [43.337292432785034]
M5_tt = [15.372673273086548]
EM_tt = [117.1505172252655]
CKL_tt = [98.15424585342407]
MCKL_tt = [2074.779541015625]

ensemble_weight = [1.68857979e-08,1.69104298e-02, 5.08915832e-01 ,4.74173722e-01]
ckl_weight = [0.74210491, 1.00000003e-05, 1.0000001e-05, 0.25787509]

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
plt.autoscale(enable=True, axis='x')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('NRMSE (\%)',fontsize=60)
fig.savefig("single_16D_NRMSE_n.png",dpi=600)
fig.savefig("single_16D_NRMSE_n.pdf",dpi=600)

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
fig.savefig("single_16D_R2.png",dpi=600)
fig.savefig("single_16D_R2.pdf",dpi=600)
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
fig.savefig("single_16D_training_time_n.png",dpi=600)
fig.savefig("single_16D_training_time_n.pdf",dpi=600)
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
fig.savefig("single_16D_ensemble_weight.png",dpi=600)
fig.savefig("single_16D_ensemble_weight.pdf",dpi=600)

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
fig.savefig("single_16D_ckl_weight.png",dpi=600)
fig.savefig("single_16D_ckl_weight.pdf",dpi=600)
'''
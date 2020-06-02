#from smt.surrogate_models import KRG
import preprocessing as pre
import halton_sequence as hs
import numpy as np
import random
import kriging_NR
import Benchmark_Problems as BP

import matplotlib.pyplot as plt


# def Obj_function(x):
#     #x = preprocessing.normalize(x)
#     x = x
#     return (x*np.sin(x))
# kernels = ['gaussian','exponential','matern3_2','matern5_2']
# x = np.linspace(0,2*np.pi,10)
# y = Obj_function(x)
# xtt = np.linspace(-0.5,(2*np.pi)+0.5,100)
# x = np.array(x)[:,np.newaxis]
# xtt = np.array(xtt)[:,np.newaxis]
# y = np.array(y)[:,np.newaxis]
# model = kriging_NR.Kriging(x,y,kernels,theta0=[0.5],weights0=[0.25,0.25,0.25,0.25])
# #model.get_theta([0.5])
# model.use(xtt)
# y_pred = model.predict()
# y_exact = Obj_function(xtt)
# RMSE = model.computeNRMSE(y_exact)
# print("RMSE: {0}".format(RMSE))
# plt.figure(1,figsize=(12,9))
# plt.plot(x, y, 'o')
# plt.plot(xtt, y_pred)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])
# plt.show()

#=========================================Successful test=============================
# n = 20
# kernels = ['gaussian','exponential','matern3_2','matern5_2']
# limits = [[-5,10],[0,15]]
# xt_n = hs.halton_sequence(n,2,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# #xt = np.zeros((n,2))
# # xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# # xt[:,1] = pre.denormalize(xt_n[:,1],0,15)
# xt = xt_n
# yt = BP.branin(xt[:,0],xt[:,1])
# model = kriging_NR.Kriging(xt,yt,kernels,theta0=[0.5,0.5],weights0=[0.25,0.25,0.25,0.25])
# num = 100
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-5,10,num)
# xtest[:,1] = np.linspace(0,15,num)

# l = len(xtest[:,0])
# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((l,l))
# for i in range(l):
#     x = np.vstack((X[i,:],Y[i,:]))
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(l,)
#     grid_Z[i,:] = y
# y_exact = BP.branin(xtest[:,0],xtest[:,1])


# g = np.array(grid_Z)
# d = np.diag(g)
# yhat = np.array(d)
# model.y_output = yhat
# RMSE = model.computeNRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))

# plt.figure(1,figsize=(9,6))
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Branin Function (My prediction)',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# #plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()





















# def Obj_function(x):
#     #x = preprocessing.normalize(x)
#     x = x
#     return (np.sin(x))

# n = 10
# limit = [0,(2*np.pi)]
# #x = hs.halton_sequence(n,1)
# # x = hs.halton_sequence(n,1,limit)
# # x = np.array(x)
# # #x = np.sort(x)
# # x = x.T
# #x = pre.denormalize(x,0,(2*np.pi))
# # x = [7.5,3.75,11.25,1.875,9.375,5.625,13.125,0.9375,8.4375,4.6875]
# x = [3.142,1.5710,4.7130,0.7855,3.9275,2.3565,5.4985,0.3928,3.5348,1.9638]
# x = np.array(x)[:,np.newaxis]
# #x = x.T
# y = Obj_function(x)
# #xtt = np.linspace(-0.5,(2*np.pi)+0.5,1000)
# xtt = np.linspace(-0.5,(2*np.pi)+0.5,100)
# #x = np.array(x)[:,np.newaxis]
# xtt = np.array(xtt)[:,np.newaxis]
# #y = np.array(y)[:,np.newaxis]
# #print(y)
# model = kriging_NR.Kriging(x,y,kernel='exponential',theta0=[0.5])
# model.get_theta([0.5])
# print("theta: {0}".format(model.theta))
# si = model.sigma2
# print('Sigma2: {0}'.format(si))
# print("Likelihood: {0}".format(model.likelihood))
# print("R determinant: {0}".format(model.detR))
# model.use(xtt)
# y_pred = model.predict()
# y_exact = Obj_function(xtt)
# RMSE = model.computeRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))
# b = model.Beta
# print("Beta: {0}".format(b))

# fileN = "1DOutputM.csv"
# csv = open(fileN,"w")
# Header = "Y_hat\n"
# csv.write(Header)
# for i in range(len(y_pred)):
#     a = y_pred[i]
#     b = str(a)
#     b +="\n"
#     csv.write(b)
# csv.close()


# plt.plot(x, y, 'o')
# #plt.plot(x, y)
# plt.plot(xtt, y_pred)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])
# plt.title("My Prediction $y = x sin(x)$")
# plt.show()
# #print(model.R)
# #print(model.K)

#============================================End Here================================

#N = [5,8,10,20,25,30,35,43,45,50,55,65]
#N = [2,5,7,8,10,12,15,17,19,20,22,23,25,30]
# N = [7,8,10,12,15,17,19,20,22,23,25,30]
# Kernels = ['gaussian','exponential','matern3_2','matern5_2']
# RMSE_data = np.zeros(len(N))
# for i in range(len(Kernels)):
#     for j in range(len(N)):
#         n = N[j]
#         xt_n = hs.halton_sequence(n,2)
#         xt_n = np.array(xt_n)
#         xt_n = xt_n.T
#         xt = np.zeros((n,2))
#         xt[:,0] = pre.denormalize(xt_n[:,0],-6,6)
#         xt[:,1] = pre.denormalize(xt_n[:,1],-6,6)

#         yt = BP.himmelblau(xt[:,0],xt[:,1])
#         model = kriging_NR.Kriging(xt,yt,Kernels[i],theta0=[0.5,0.5])
#         num = 100
#         xtest = np.zeros((num,2))
#         xtest[:,0] = np.linspace(-6,6,num)
#         xtest[:,1] = np.linspace(-6,6,num)
#         x = np.vstack((xtest[:,0],xtest[:,1]))
#         model.use(x.T)
#         y = model.predict()
#         y_exact = BP.himmelblau(xtest[:,0],xtest[:,1])
#         RMSE = model.computeRMSE(y_exact)
#         RMSE_range = RMSE/(max(y_exact)-min(y_exact))
#         RMSE_data[j] = RMSE_range
#     fig,ax = plt.subplots()
#     plt.autoscale(enable=True, axis='both', tight=None)
#     plt.scatter(N,RMSE_data)
#     plt.xlabel('Sample size')
#     plt.ylabel('RMSE/range(f)')
#     ax.legend(Kernels[i],loc='best',fancybox=True)
#     plt.title('Himmelblau function  (Eval point: 100)')
#     plt.show()

#=====================================================================
# n = 25
# limits = [[-5,10],[0,15]]
# xt_n = hs.halton_sequence(n,2,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# #xt = np.zeros((n,2))
# # xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# # xt[:,1] = pre.denormalize(xt_n[:,1],0,15)
# xt = xt_n
# yt = BP.branin(xt[:,0],xt[:,1])
# model = kriging_NR.Kriging(xt,yt,'exponential',theta0=[0.5,0.5])
# model.get_theta([0.5,0.5])
# print("theta: {0}".format(model.theta))
# si = model.sigma2
# print('Sigma2: {0}'.format(si))
# print("Likelihood: {0}".format(model.likelihood))
# print("R determinant: {0}".format(model.detR))

# num = 1000
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-5,10,num)
# xtest[:,1] = np.linspace(0,15,num)

# l = len(xtest[:,0])
# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((l,l))
# for i in range(l):
#     x = np.vstack((X[i,:],Y[i,:]))
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(l,)
#     grid_Z[i,:] = y
# y_exact = BP.branin(xtest[:,0],xtest[:,1])


# g = np.array(grid_Z)
# d = np.diag(g)
# yhat = np.array(d)

# fileN = "2DOutputM.csv"
# csv = open(fileN,"w")
# Header = "Y_exact\n"
# csv.write(Header)
# for i in range(len(y_exact)):
#     a = y_exact[i]
#     b = str(a)
#     b +="\n"
#     csv.write(b)

# Header1 = "Y_hat\n"
# csv.write(Header1)
# for i in range(len(yhat)):
#     a = yhat[i]
#     b = str(a)
#     b +="\n"
#     csv.write(b)
# csv.close()

# model.y_output = yhat
# RMSE = model.computeRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))

# plt.figure(1,figsize=(9,6))
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Branin Function (My prediction)',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# #plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()

#===============================================================================



Nn = [20,25,30,35]
kernels = ['gaussian','exponential','matern3_2','matern5_2']
k = 2
Obj_func = BP.himmelblau
theta0 = [0.1,0.1]
#write code to automatically get the limits
'''
Obj_func = BP.problem("Branin")
limits = Obj_func.limits
'''
# limits = [[-5,10],[0,15]]
limits = [[-6,6],[-6,6]]
for o in range(len(Nn)):
    xt_n = hs.halton_sequence(Nn[o],k,limits)
    xt_n = np.array(xt_n)
    xt_n = xt_n.T
    xt = xt_n
    yt = Obj_func(xt[:,0],xt[:,1])
    yt_min = np.min(yt)
    yt_max = np.max(yt)
    xt[:,0] = pre.normalize_values(xt[:,0],limits[0][0],limits[0][1])
    xt[:,1] = pre.normalize_values(xt[:,1],limits[1][0],limits[1][1])
    yt = pre.normalize_values(yt,yt_min,yt_max)


    num = 100
    xtest = np.zeros((num,2))
    xtestn = np.copy(xtest)

    xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
    xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)

    xtestn[:,0] = pre.normalize_values(xtest[:,0],limits[0][0],limits[0][1])
    xtestn[:,1] = pre.normalize_values(xtest[:,1],limits[1][0],limits[1][1])

    x = np.vstack((xtestn[:,0],xtestn[:,1]))
    y_exact = Obj_func(xtest[:,0],xtest[:,1])

    # model = kriging_NR.Kriging(xt,yt,kernels,theta0=theta0,weights0=[0.25]*4)
    model = kriging_NR.Kriging(xt,yt,kernels,theta0=theta0)
    model.use(x.T)
    y = model.predict()
    y = y.reshape(num,)
    y = pre.denormalize(y,yt_min,yt_max)
    model.y_output = y
    RMSE = 100 * model.computeNRMSE(y_exact)
    

    print(Nn[o])
    print(RMSE)
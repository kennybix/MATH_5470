#from smt.surrogate_models import KRG
import preprocessing as pre
import halton_sequence as hs
import numpy as np
import random
import kriging_MK
import Benchmark_Problems as BP
import re
import string
import permutate as pm

import matplotlib.pyplot as plt



# N = {8,15,20,25,30,35,40,45,50,55,60}
# Kernels = {'gaussian','exponential','matern3_2','matern5_2'}


# n = 20
# xt_n = hs.halton_sequence(n,2)
# xt_n = np.array(xt_n)
# print(xt_n.shape)
# xt_n[0] = pre.denormalize(xt_n[0],-5,10)
# xt_n[1] = pre.denormalize(xt_n[1],-5,10)
# yt = BP.rosenbrock(xt_n[0],xt_n[1])
# #print(yt)
# xt = xt_n.T
# model = kriging_N.Kriging(xt,yt,'gaussian')


# num = 1000
# random.seed(1)
# a = np.random.uniform(-5,10,num)
# b = np.random.uniform(-5,10,num)
# x = np.vstack((a,b))
# x = x.T
# model.train(x)
# y= model.predict()
# print(y)
# y_exact = BP.branin(a,b)
# error = model.computeRMSE(y_exact)
# print('Error: {0}'.format(error))
# theta = model.get_theta()
# print(theta)
# model.plot_nll_theta()






# N = {8,15,20,25,30,35,40,45,50,55,60}
# Kernels = {'gaussian','exponential','matern3_2','matern5_2'}


# n = 20
# xt_n = hs.halton_sequence(n,2)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = np.zeros((n,2))
# xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# xt[:,1] = pre.denormalize(xt_n[:,1],0,15)
# xtt = np.vstack(xt[:,0],xt[:,1])

# yt = BP.rosenbrock(xtt[0],xtt[1])
# print(yt)
# model = kriging_NN.Kriging(xt,yt,'gaussian')

# num = 1000
# random.seed(1)
# x = np.zeros((num,2))
# for i in range(num):
#     x[i,0] = random.uniform(-5,10)
#     x[i,1] = random.uniform(0,15)

# model.train(x)

# y= model.predict()
# #print(y.shape)
# y_exact = BP.rosenbrock(x[:,0],x[:,1])
# error = model.computeRMSE(y_exact)
# print('Error: {0}'.format(error))

# theta = model.get_theta()
# print(theta)



# plt.plot(xt, yt, 'o')
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])
# plt.show()
#print(yt)


# #model.plotTrend(y_exact)#
# print(model.get_theta())
# plt.plot(x,y,'o')
# plt.plot(xtt,y_y,'-r')


#plt.plot(xtt,y_exact,'-r')
#plt.plot(y_exact,y_y)
#model.plot_nll_theta()

# #plt.plot(x, y, 'o') use this much later to check the efficiency of the model
# plt.plot(x,y, 'o')
# plt.plot(xtt, y_y)
# #plt.plot(xtt,y_exact.reshape(1000,1),'-b')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction', 'exact'])
# plt.show()

# n = 100
# xt_n = hs.halton_sequence(n,2)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = np.zeros((n,2))
# xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# xt[:,1] = pre.denormalize(xt_n[:,1],0,15)
# xtt = np.vstack((xt[:,0],xt[:,1]))

# yt = BP.branin(xt[:,0],xt[:,1])
# #print(yt)
# model = kriging_NR.Kriging(xt,yt,'gaussian')
# #model.plot_nll_theta()
# num = 500
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-5,10,num)
# xtest[:,1] = np.linspace(0,15,num)
# x = np.vstack((xtest[:,0],xtest[:,1]))
# print(xt_n.shape)
# model.train(x.T)
# print(model.get_theta())
# #print(model.sigma2)
# model.plot_nll_theta()
# y = model.predict()
# y_exact = BP.branin(xtest[:,0],xtest[:,1])
# plt.plot(y_exact,y)
# plt.show()
# error = model.computeRMSE(y_exact)
# print(error)




# yt = BP.branin(xt_n[:,0],xt_n[:,1])
# #print(yt)
# model = kriging_NR.Kriging(xt_n,yt,'matern3_2')
# #model.plot_nll_theta()
# num = 1000
# xtest = np.zeros((num,2))
# xtest[:,0] = pre.normalize(np.linspace(-5,10,num))
# xtest[:,1] = pre.normalize(np.linspace(0,15,num))
# x = np.vstack((xtest[:,0],xtest[:,1]))
# print(xt_n.shape)
# model.train(xtest)
# print(model.get_theta())
# print(model.sigma2)
# model.plot_nll_theta()



# N = [20,25,30,35,40,45,50,55,60]
# Kernels = ['gaussian','exponential','matern3_2','matern5_2']
# RMSE_data = np.zeros(len(N))
# for i in range(len(Kernels)):
#     for j in range(len(N)):
#         n = N[j]
#         xt_n = hs.halton_sequence(n,2)
#         xt_n = np.array(xt_n)
#         xt_n = xt_n.T
#         xt = np.zeros((n,2))
#         xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
#         xt[:,1] = pre.denormalize(xt_n[:,1],0,15)

#         yt = BP.branin(xt[:,0],xt[:,1])
#         model = kriging_NR.Kriging(xt,yt,Kernels[i])
#         num = 500
#         xtest = np.zeros((num,2))
#         xtest[:,0] = np.linspace(-5,10,num)
#         xtest[:,1] = np.linspace(0,15,num)
#         x = np.vstack((xtest[:,0],xtest[:,1]))
#         model.train(x.T)
#         y = model.predict()
#         y_exact = BP.branin(xtest[:,0],xtest[:,1])
#         RMSE = model.computeRMSE(y_exact)
#         RMSE_range = RMSE/(max(y_exact)-min(y_exact))
#         RMSE_data[j] = RMSE_range
#     fig,ax = plt.subplots()
#     plt.scatter(N,RMSE_data)
#     plt.xlabel('Sample size')
#     plt.ylabel('RMSE/range(f)')
#     ax.legend(Kernels[i],loc='best',fancybox=True)
#     plt.title('Branin function  (Eval point: 500)')
#     plt.show()



# N = [20,25,30,35,40,45,50,55,60]
# Kernels = ['gaussian','exponential','matern3_2','matern5_2']
# RMSE_data = np.zeros(len(N))
# for i in range(len(Kernels)):
#     for j in range(len(N)):
#         n = N[j]
#         xt_n = hs.halton_sequence(n,2)
#         xt_n = np.array(xt_n)
#         xt_n = xt_n.T
#         xt = np.zeros((n,2))
#         xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
#         xt[:,1] = pre.denormalize(xt_n[:,1],-5,10)

#         yt = BP.rosenbrock(xt[:,0],xt[:,1])
#         model = kriging_NR.Kriging(xt,yt,Kernels[i])
#         num = 500
#         xtest = np.zeros((num,2))
#         xtest[:,0] = np.linspace(-5,10,num)
#         xtest[:,1] = np.linspace(-5,10,num)
#         x = np.vstack((xtest[:,0],xtest[:,1]))
#         model.train(x.T)
#         y = model.predict()
#         y_exact = BP.rosenbrock(xtest[:,0],xtest[:,1])
#         RMSE = model.computeRMSE(y_exact)
#         RMSE_range = RMSE/(max(y_exact)-min(y_exact))
#         RMSE_data[j] = RMSE_range
#     fig,ax = plt.subplots()
#     plt.scatter(N,RMSE_data)
#     plt.xlabel('Sample size')
#     plt.ylabel('RMSE/range(f)')
#     ax.legend(Kernels[i],loc='best',fancybox=True)
#     plt.title('Rosenbrock function  (Eval point: 500)')
#     plt.show()

func = BP.himmelblau
xLimit0 = [-6,-6]
xLimit1 = [-6,-6]

#N = [5,8,10,20,25,30,35,43,45,50,55,65]
# N = [2,5,7,8,10,12,15,17,19,20,22,23,25,30]
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
#         model = kriging_NR.Kriging(xt,yt,Kernels[i])
#         num = 500
#         xtest = np.zeros((num,2))
#         xtest[:,0] = np.linspace(-6,6,num)
#         xtest[:,1] = np.linspace(-6,6,num)
#         x = np.vstack((xtest[:,0],xtest[:,1]))
#         model.train(x.T)
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
#     plt.title('Himmelblau function  (Eval point: 500)')
#     plt.show()


# ========SMT===================
# n = 25
# xt_n = hs.halton_sequence(n,2)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = np.zeros((n,2))
# xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# xt[:,1] = pre.denormalize(xt_n[:,1],-5,10)
# yt = BP.rosenbrock(xt[:,0],xt[:,1])
# sm = KRG(theta0=[0.01,0.01])
# sm.set_training_values(xt, yt)
# sm.train()
# num = 500
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-5,10,num)
# xtest[:,1] = np.linspace(-5,10,num)
# #sm.get_theta()
# l = len(xtest[:,0])
# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((l,l))
# for i in range(l):
#     x = np.vstack((X[i,:],Y[i,:]))
#     y = sm.predict_values(x.T)
#     y = y.reshape(l,)
#     grid_Z[i,:] = y
# y_exact = BP.rosenbrock(xtest[:,0],xtest[:,1])
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Rosenbrock Function',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()
#=======================SMT Ends==========================



# n = 40
# xt_n = hs.halton_sequence(n,2)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = np.zeros((n,2))
# xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# xt[:,1] = pre.denormalize(xt_n[:,1],0,15)

# yt = BP.branin(xt[:,0],xt[:,1])
# model = kriging_NR.Kriging(xt,yt,'gaussian',theta0=[0.01,0.01])
# th = model.get_theta([0.01,0.01])
# print(th)

# num = 500
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-5,10,num)
# xtest[:,1] = np.linspace(0,15,num)



# l = len(xtest[:,0])
# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((l,l))
# for i in range(l):
#     x = np.vstack((X[i,:],Y[i,:]))
#     model.train(x.T)
#     y = model.predict()
#     y = y.reshape(l,)
#     grid_Z[i,:] = y
# y_exact = BP.branin(xtest[:,0],xtest[:,1])
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Himmelblau Function',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()





# def Obj_function(x):
#     #x = preprocessing.normalize(x)
#     x = x
#     return (np.sin(x))

# x = np.linspace(0,2*np.pi,10)
# y = Obj_function(x)
# xtt = np.linspace(-0.5,(2*np.pi)+0.5,1000)
# x = np.array(x)[:,np.newaxis]
# xtt = np.array(xtt)[:,np.newaxis]
# y = np.array(y)[:,np.newaxis]
# model = kriging_NR.Kriging(x,y,kernel='matern5_2',theta0=[0.001])
# th = model.get_theta([0.001])
# print(th)
# model.train(xtt)
# y_pred = model.predict()
# plt.plot(x, y, 'o')
# plt.plot(xtt, y_pred)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])
# plt.show()

#=========================================Successful test=============================

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
# model = kriging_MK.Kriging(x,y,kernel=['gaussian'],theta0=[0.5])
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
# # #print(model.R)
# # #print(model.K)

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
# n = 20
# kernel = ['matern5_2','gaussian']
# limits = [[-5,10],[0,15]]
# xt_n = hs.halton_sequence(n,2,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# #xt = np.zeros((n,2))
# # xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
# # xt[:,1] = pre.denormalize(xt_n[:,1],0,15)
# xt = xt_n
# yt = BP.branin(xt[:,0],xt[:,1])
# model = kriging_MK.Kriging(xt,yt,kernel,theta0=[0.5,0.5])
# model.get_theta([0.5,0.5])
# print("theta: {0}".format(model.theta))
# si = model.sigma2
# print('Sigma2: {0}'.format(si))
# print("Likelihood: {0}".format(model.likelihood))
# print("R determinant: {0}".format(model.detR))

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

# fileN = "2DOutputBranin_M_5_2_G_25_100.csv"
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
# plt.title('Branin Function',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# #plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()
#==============================================================================
# n = 35
# kernel = ['gaussian','gaussian']
# limits = [[-5,10],[0,15]]
# xt_n = hs.halton_sequence(n,2,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = xt_n
# yt = BP.branin(xt[:,0],xt[:,1])
# model = kriging_MK.Kriging(xt,yt,kernel,theta0=[0.5,0.5])
# model.get_theta([0.5,0.5])
# # print("theta: {0}".format(model.theta))
# si = model.sigma2
# # print('Sigma2: {0}'.format(si))
# # print("Likelihood: {0}".format(model.likelihood))
# # print("R determinant: {0}".format(model.detR))

# num = 100
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-5,10,num)
# xtest[:,1] = np.linspace(0,15,num)

# l = len(xtest[:,0])
# x = np.vstack((xtest[:,0],xtest[:,1]))
# model.use(x.T)
# y = model.predict()
# y = y.reshape(l,)

# y_exact = BP.branin(xtest[:,0],xtest[:,1])


# model.y_output = y
# RMSE = model.computeRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))


#===============================================================================

#Multikernel brute force test
n = 35
k = 2
Obj_func = BP.branin

limits = [[-5,10],[0,15]]
xt_n = hs.halton_sequence(n,k,limits)
xt_n = np.array(xt_n)
xt_n = xt_n.T
xt = xt_n
yt = Obj_func(xt[:,0],xt[:,1])

num = 100
xtest = np.zeros((num,2))
xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)
x = np.vstack((xtest[:,0],xtest[:,1]))
y_exact = Obj_func(xtest[:,0],xtest[:,1])

kernels = [r"'gaussian'",r"'matern3_2'",r"'matern5_2'"]
kernel = pm.perm(k,kernels)
results = {}
for i in range(len(kernel)):
    kernel_string = kernel[i]
    kernel_mix = re.sub("[^\w]", " ",  kernel_string).split()
    model = kriging_MK.Kriging(xt,yt,kernel_mix,theta0=[0.5,0.5])
    model.get_theta([0.5,0.5])
    # print("theta: {0}".format(model.theta))
    #si = model.sigma2
    # print('Sigma2: {0}'.format(si))
    # print("Likelihood: {0}".format(model.likelihood))
    # print("R determinant: {0}".format(model.detR))
    model.use(x.T)
    y = model.predict()
    y = y.reshape(num,)
    model.y_output = y
    RMSE = model.computeRMSD(y_exact)
    #RMSE_range = RMSE/(max(y_exact)-min(y_exact))
    #results[kernel_string] = str(RMSE) + "%"
    results[kernel_string] = "{0:.2f}%".format(RMSE)
    #print("RMSE: {0}".format(RMSE))
print(results)

#===================================================================================

# #Multikernel brute force test
# n = 40
# k = 3
# Obj_func = BP.Rosenbrock_nD
# limits = [[-2,2],[-2,2],[-2,2]]
# xt_n = hs.halton_sequence(n,k,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = xt_n
# yt = Obj_func(np.vstack((xt[:,0],xt[:,1],xt[:,2])))

# num = 100
# xtest = np.zeros((num,k))
# xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
# xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)
# xtest[:,2] = np.linspace(limits[2][0],limits[2][1],num)
# x = np.vstack((xtest[:,0],xtest[:,1],xtest[:,2]))
# y_exact = Obj_func(x)

# kernels = [r"'gaussian'",r"'matern3_2'",r"'matern5_2'"]
# kernel = pm.perm(k,kernels)
# results = {}
# for i in range(len(kernel)):
#     kernel_string = kernel[i]
#     kernel_mix = re.sub("[^\w]", " ",  kernel_string).split()
#     model = kriging_MK.Kriging(xt,yt,kernel_mix,theta0=[0.5,0.5,0.5])
#     model.get_theta([0.5,0.5,0.5])
#     # print("theta: {0}".format(model.theta))
#     #si = model.sigma2
#     # print('Sigma2: {0}'.format(si))
#     # print("Likelihood: {0}".format(model.likelihood))
#     # print("R determinant: {0}".format(model.detR))
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(num,)
#     model.y_output = y
#     RMSE = model.computeRMSE(y_exact)
#     #RMSE_range = RMSE/(max(y_exact)-min(y_exact))
#     results[kernel_string] = RMSE
#     #print("RMSE: {0}".format(RMSE))
# print(results)

#==============================================================================
# n = 30
# kernel = ['gaussian','gaussian']
# k = 2
# Obj_func = BP.Rosenbrock_nD
# limits = [[-2,2],[-2,2]]
# xt_n = hs.halton_sequence(n,k,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = xt_n
# yt = Obj_func(np.vstack((xt[:,0],xt[:,1])))

# num = 100
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
# xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)
# x = np.vstack((xtest[:,0],xtest[:,1]))
# y_exact = Obj_func(x)

# model = kriging_MK.Kriging(xt,yt,kernel,theta0=[0.5,0.5])
# model.get_theta([0.5,0.5])
# # print("theta: {0}".format(model.theta))
# si = model.sigma2
# # print('Sigma2: {0}'.format(si))
# # print("Likelihood: {0}".format(model.likelihood))
# # print("R determinant: {0}".format(model.detR))
# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((num,num))
# for i in range(num):
#     x = np.vstack((X[i,:],Y[i,:]))
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(num,)
#     grid_Z[i,:] = y
# g = np.array(grid_Z)
# d = np.diag(g)
# yhat = np.array(d)

# model.y_output = yhat
# RMSE = model.computeRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))

# plt.figure(1,figsize=(9,6))
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Rosenbrock Function',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# #plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()
#==========================================================================================
# n = 35
# k=2
# Obj_func = BP.himmelblau
# kernel = ['gaussian','gaussian']
# limits = [[-6,6],[-6,6]]
# xt_n = hs.halton_sequence(n,k,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = xt_n
# yt = Obj_func(xt[:,0],xt[:,1])
# num = 100
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
# xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)
# y_exact = Obj_func(xtest[:,0],xtest[:,1])
# model = kriging_MK.Kriging(xt,yt,kernel,theta0=[0.5,0.5])
# model.get_theta([0.5,0.5])

# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((num,num))
# for i in range(num):
#     x = np.vstack((X[i,:],Y[i,:]))
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(num,)
#     grid_Z[i,:] = y
# g = np.array(grid_Z)
# d = np.diag(g)
# yhat = np.array(d)
# model.y_output = yhat
# RMSE = model.computeRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))

# plt.figure(1,figsize=(9,6))
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Himmelblau Function',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# #plt.scatter(xt[:,0],xt[:,1],20,c=yt,cmap="RdGy")
# plt.show()
#=============================================================================

# #Multikernel brute force test
# n = 40
# k = 2
# Obj_func = BP.himmelblau
# limits = [[-6,6],[-6,6]]
# xt_n = hs.halton_sequence(n,k,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = xt_n
# yt = Obj_func(xt[:,0],xt[:,1])

# num = 100
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
# xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)
# x = np.vstack((xtest[:,0],xtest[:,1]))
# y_exact = Obj_func(xtest[:,0],xtest[:,1])

# kernels = [r"'gaussian'",r"'matern3_2'",r"'matern5_2'"]
# kernel = pm.perm(k,kernels)
# results = {}
# for i in range(len(kernel)):
#     kernel_string = kernel[i]
#     kernel_mix = re.sub("[^\w]", " ",  kernel_string).split()
#     model = kriging_MK.Kriging(xt,yt,kernel_mix,theta0=[0.5,0.5])
#     model.get_theta([0.5,0.5])
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(num,)
#     model.y_output = y
#     RMSE = model.computeRMSE(y_exact)
#     results[kernel_string] = RMSE
# print(results)

#=====================================================================================================
# #Multikernel brute force test
# n = 20
# k = 3
# Obj_func = BP.RobotArm_nD
# limits = [[0,2*np.pi],[0,2*np.pi],[0,2*np.pi]]
# xt_n = hs.halton_sequence(n,k,limits)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = xt_n
# yt = Obj_func(np.vstack((xt[:,0],xt[:,1],xt[:,2])))

# num = 100
# xtest = np.zeros((num,k))
# xtest[:,0] = np.linspace(limits[0][0],limits[0][1],num)
# xtest[:,1] = np.linspace(limits[1][0],limits[1][1],num)
# xtest[:,2] = np.linspace(limits[2][0],limits[2][1],num)
# x = np.vstack((xtest[:,0],xtest[:,1],xtest[:,2]))
# y_exact = Obj_func(x)

# kernels = [r"'gaussian'",r"'matern3_2'",r"'matern5_2'"]
# kernel = pm.perm(k,kernels)
# results = {}
# for i in range(len(kernel)):
#     kernel_string = kernel[i]
#     kernel_mix = re.sub("[^\w]", " ",  kernel_string).split()
#     model = kriging_MK.Kriging(xt,yt,kernel_mix,theta0=[0.5,0.5,0.5])
#     model.get_theta([0.5,0.5,0.5])
#     # print("theta: {0}".format(model.theta))
#     #si = model.sigma2
#     # print('Sigma2: {0}'.format(si))
#     # print("Likelihood: {0}".format(model.likelihood))
#     # print("R determinant: {0}".format(model.detR))
#     model.use(x.T)
#     y = model.predict()
#     y = y.reshape(num,)
#     model.y_output = y
#     RMSE = model.computeRMSE(y_exact)
#     #RMSE_range = RMSE/(max(y_exact)-min(y_exact))
#     results[kernel_string] = RMSE
#     #print("RMSE: {0}".format(RMSE))
# print(results)

#============================================================================================================
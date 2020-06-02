from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import preprocessing as pre
import halton_sequence as hs
import numpy as np
import Benchmark_Problems as BP


class Kriging: 
    """ To build and use Kriging """

    def __init__(self, x, y, GE, globalType):
        self.x = x    # training points to construct the model 
        self.GE = GE  # GE = false --> Ordinary Kriging
                      # GE = true --> Gradient-enhanced Kriging 
        self.y = y    # function evaluations at training points
                      # size: [Ns x 1] for Kriging
                      #       [Ns*(Ndv+1) x 1] for Gradient-enhanced Kriging
        self.globalType = globalType 
                      # type of global model:
                      # 'constant'
                      # 'linear'
                      # 'quadratic'

        self.Ns = self.x.shape[0]
        self.Ndv = self.x.shape[1]
        self.Nr = len(self.y) # for Kriging: Nr = Ns 
                         # for GEK: Nr = Ns(Ndv+1)

        globalOptions = {'constant': self._computefconstant, 
                         'linear': self._computeflinear, 
                         'quadratic': self._computefquadratic}

        # compute f
        globalOptions['constant']()

    def _computefconstant(self):

        f1 = ones(self.Ns)

        if not self.GE:
            self.f = f1
        elif self.GE:
            f0 = zeros(self.Ns*self.Ndv)
            self.f = append(f1,f0)

    def _computeflinear(self):

        f1 = vstack([ones(self.Ns), self.x[:,0], self.x[:,1]])
        f1 = f1.T
        if not self.GE:
            self.f = f1
        elif self.GE:
            f00 = zeros(self.Ns*self.Ndv)
            f0 = vstack([f00, f00, f00])
            f0 = f0.T
            self.f = vstack([f1, f0])

    def _computefquadratic(self):

        f1 = vstack([ones(self.Ns), self.x[:,0], (self.x[:,0])**2, self.x[:,1], (self.x[:,1])**2, self.x[:,0]*self.x[:,1]])
        f1 = f1.T
        if not self.GE:
            self.f = f1
        elif self.GE:
            f00 = zeros(self.Ns*self.Ndv)
            f0 = vstack([f00, f00, f00, f00, f00, f00])
            f0 = f0.T
            self.f = vstack([f1, f0])

    def _computeBeta(self, R):

        if R.shape[0] == self.Nr: 
            yy = self.y 
            ff = self.f
        elif R.shape[0] != self.Nr: 
            yy = self.y[0:self.Ns] 
            ff = self.f[0:self.Ns]


        Ry = linalg.solve(R, yy)
        Rf = linalg.solve(R, ff)

        if self.globalType == 'constant':
            self.beta = dot(ff.T, Ry)/dot(ff.T, Rf)
        else:
            temp = dot(ff.T, Rf)
            invtemp = array(matrix(temp).I)
            self.beta = dot(invtemp, dot(ff.T, Ry))

    def _computeB(self):

        if self.globalType == 'constant':
            self.B = self.beta
        elif self.globalType == 'linear':
            self.B = self.beta[0] + self.beta[1]*self.x[0] + self.beta[2]*self.x[1]
        elif self.globalType == 'quadratic':
            self.B = self.beta[0] + self.beta[1]*self.x[0] + self.beta[2]*(self.x[0]**2) + self.beta[3]*self.x[1] + self.beta[4]*(self.x[1]**2) + self.beta[5]*(self.x[0]*self.x[1])
        return self.B
    def _computePhi(self, xi, xj, theta, flag):

        phi = 0
        dphidx = zeros(self.Ndv)
        dphi2dxdy = zeros((self.Ndv, self.Ndv))

        dx = xi - xj
        adx = abs(dx)
        self.d = adx
        phi = exp(-1.0 * sum(theta * (adx**2)))

        if not flag:
            # when derivative information is not required
            return phi
        elif flag:
            dphidx = 2.0 * phi * (theta * dx)

            for k in range(self.Ndv):
                for l in range(self.Ndv):
                    if k == l:
                        dphi2dxdy[k,l] = -2.0*theta[k]*(-1.0 + 2.0*theta[k]*(dx[k]**2))*phi
                    else:
                        dphi2dxdy[k,l] = -4.0*theta[k]*theta[l]*dx[k]*dx[l]*phi

            return phi, dphidx, dphi2dxdy

    def _correlation(self, theta, flag):
        # correlation function

        if flag == self.GE:
            R = zeros((self.Nr, self.Nr))
        elif flag != self.GE: 
            R = zeros((self.Ns, self.Ns))

        # go through each sample pair
        for i in range(self.Ns):
            xi = self.x[i,:]
            for j in range(self.Ns):
                xj = self.x[j,:]

                if not flag:
                    R[i,j] = self._computePhi(xi, xj, theta, flag)
                elif flag:
                    [phi, dphidx, dphi2dxdy] = self._computePhi(xi, xj, theta, flag)
                    R[i,j] = phi
                    R[i, self.Ns+j*self.Ndv:self.Ns+(j+1)*self.Ndv] = dphidx
                    R[self.Ns+j*self.Ndv:self.Ns+(j+1)*self.Ndv, i] = dphidx
                    R[self.Ns+i*self.Ndv:self.Ns+(i+1)*self.Ndv, self.Ns+j*self.Ndv:self.Ns+(j+1)*self.Ndv] = dphi2dxdy
        return R

    def _HJfun_obj(self, theta):
        # compute likelihood function

        # compute R
        R = self._correlation(theta, self.GE)

        try:
            # compute beta 
            self._computeBeta(R)
            f = np.copy(self.f)
            f = f[:,np.newaxis]
            f = f.reshape(self.Ns,1)
            self.beta = (self.beta).reshape(1,)
            self.y = (self.y).reshape(self.Ns,1)
            # compute sigma2
            temp = self.y - np.dot(f, self.beta[:,np.newaxis]) 
            sigma2 = 1.0/self.Nr * dot(temp.T, linalg.solve(R, temp))
            self.sigma2 = sigma2
            self.detR = linalg.det(R)
            #sigma2 = 1.0/self.Nr * dot(temp.T, linalg.solve(R, temp)) #debug line
            # objective function -- likelihood
            f_obj = 1.0/2.0 * ((self.Nr * log(sigma2)) + log(linalg.det(R)))

            if f_obj == -inf or math.isnan(f_obj):
                f_obj = inf
        except LinAlgError:
            print('Oops!')
            f_obj = inf

        return f_obj

    def _checkPositive(self, x):
        # check that all design variables are > 0

        Ndv = len(x)
        count = 0 
        for i in range(Ndv):
            if x[i] > 0:
                # positive entry, increment count
                count = count + 1

        if count == Ndv:
            self.constraint = True
        else:
            self.constraint = False 

    def _explore(self, xk, sk, delta, minf, rhok, fxk): 
        n = len(xk)
        for i in range(n):
            e = zeros(n)
            e[i] = 1
            ski = sk + delta*e
            xki = xk + ski 
            fxki = self._HJfun_obj(xki) 

            if fxki < minf:
                rhok = fxk - fxki 
                minf = fxk
                i
                sk = ski
            else: 
                ski = sk - delta*e
                xki = xk + ski
                self._checkPositive(xki) 
                if self.constraint:
                    fxki = self._HJfun_obj(xki)
                    if fxki < minf:
                        rhok = fxk - fxki 
                        minf = fxki
                        sk = ski 

        return rhok, sk, minf

    def _hj(self, xk, delta, rhok, sk, limDelta, shrink):
        xk = np.array(xk)
        #xk = float(xk[0])
        k = 0
        while delta > limDelta: 
            fxk = self._HJfun_obj(xk)
            minf = fxk
            f0 = minf
            test = xk + sk
            self._checkPositive(test)
            if rhok > 0 and self.constraint:
                trialF = self._HJfun_obj(xk + sk)
                if trialF != inf:
                    rhok = fxk - trialF
                    minf = trialF
                    rhok, sk, minf = self._explore(xk, sk, delta, minf, rhok, fxk)
                else:
                    rhok = 0
                    sk = 0
            else: 
                sk = 0
                rhok = 0
                minf = fxk 

                rhok, sk, minf = self._explore(xk, sk, delta, minf, rhok, fxk)

                if minf >= f0 and minf != inf:
                    delta = shrink*delta

            # Pattern move
            xk = xk + sk
            #print 'xk: ', xk, 'sk: ', sk, 'fxk: ', self._HJfun_obj(xk)
            k = k+1
            if minf == inf: 
                xk = xk + delta

        self.theta = xk
        self.likelihood = self._HJfun_obj(xk)

    def getTheta(self, theta0):
        # theta0: initial values for MLE 
        # theta: hyperparameter (correlation parameter) 

        xk  = theta0
        delta = 1
        rhok = 0 
        sk = zeros(self.Ndv)
        #limDelta = 1e-2
        #shrink = 0.2
        limDelta = 1e-4
        shrink = 0.5


        # obtain theta 
        self._hj(xk, delta, rhok, sk, limDelta, shrink)  

    def build(self): 
        # compute R
        R = self._correlation(self.theta, self.GE) 
        self.R = R

        # compute beta
        self._computeBeta(R)

        # V_gek 
        f = np.copy(self.f)
        f = f[:,np.newaxis]
        n = f.shape[0]
        f = f.reshape(n,1)
        temp = self.y - dot(f, self.beta[:,np.newaxis]) 
        self.V_gek = linalg.solve(R, temp)

        # FOR CHECKING WEIGHTS 

        fmat = matrix(self.f)
        Rmat = matrix(self.R)

        M = vstack((hstack((Rmat, fmat.T)), hstack((fmat, matrix(0)))))
        self.M = array(M)

    def use(self, xtest):

        Ntest = xtest.shape[0]

        self.yhat = zeros(Ntest)
        self.yweight = zeros(Ntest)
        
        for i in range(Ntest):
            xi = xtest[i,:]
            rVec = zeros(self.Nr)
            for j in range(self.Ns):
                xj = self.x[j,:]
                if not self.GE:
                    rVec[j] = self._computePhi(xi, xj, self.theta, self.GE) 
                elif self.GE:
                    [phi, dphidx, dphi2dxdy] = self._computePhi(xi, xj, self.theta, self.GE)
                    rVec[j] = phi
                    rVec[self.Ns+j*self.Ndv:self.Ns+(j+1)*self.Ndv] = dphidx

            # global model
            B = self._computeB()
            self.yhat[i] = self.B + dot(rVec, self.V_gek)
            self.y = (self.y).reshape(self.Ns,)
            weight = linalg.solve(self.M, hstack((rVec,1)))
            self.yweight[i] = dot(weight, hstack((self.y,0)))

    def computeRMS(self, yexact):
        
        # compute RMS values
        n = len(self.yhat)

        if len(self.yhat) != len(yexact): 
            print('length error')

        self.RMSerror = sqrt(1.0/n * sum((self.yhat - yexact)**2))
        self.RMSerrorNorm = sqrt(1.0/n * sum(((self.yhat - yexact)/yexact)**2))
        return self.RMSerror

    def computeRMSE(self,y_exact):
        m = len(self.yhat) 
        sum = 0.0
        for i in range(m):
            sum += np.power((y_exact[i] - self.yhat[i]),2)
        self.RMSE = np.sqrt(sum / m)
        return self.RMSE

    def computeRAME(self, yexact):

        n = len(self.yhat)

        if len(self.yhat) != len(yexact): 
            print('length error')

        absDiff = absolute(yexact - self.yhat)
        num = absDiff.max()

        ybar = sum(yexact)/n
        denom = sqrt(1.0/n * sum((yexact - ybar)**2))
        self.RAMEerror = num/denom

    def krigingGradient(self, xtest):

        Ntest = xtest.shape[0]
        dr = zeros([Ntest, self.Ndv]) 

        if not self.GE: 
            for i in range(Ntest):
                xi = xtest[i,:]
                rVec = zeros((self.Ns, self.Ndv))
                for j in range(self.Ns):
                    xj = self.x[j,:]
                    [phi, dphidx, dphi2dxdy] = self._computePhi(xi, xj, self.theta, True)
                    rVec[j,:] = -dphidx
                temp = dot(rVec.T, self.V_gek)
                dr[i,:] = temp.T
        elif self.GE: 
            for i in range(Ntest):
                xi = xtest[i,:]
                rVec = zeros((self.Ns*(self.Ndv+1), self.Ndv))
                for j in range(self.Ns):
                    xj = self.x[j,:]
                    [phi, dphidx, dphi2dxdy] = self._computePhi(xi, xj, self.theta, True)
                    rVec[j,:] = -dphidx
                    rVec[self.Ns+j*self.Ndv:self.Ns+(j+1)*self.Ndv,:] = dphi2dxdy

                temp = dot(rVec.T, self.V_gek)
                dr[i,:] = temp.T
                
        self.gradient = dr
        return dr


# def Obj_function(x):
#     x = x
#     return (np.sin(x))


# x = np.linspace(0,2*np.pi,10)
# y = Obj_function(x)
# xtt = np.linspace(-0.5,(2*np.pi)+0.5,1000)
# x = np.array(x)[:,np.newaxis]
# xtt = np.array(xtt)[:,np.newaxis]
# y = np.array(y)[:,np.newaxis]

# model = Kriging(x, y, GE=False, globalType='constant')
# model.getTheta([[0.5]])
# print("Theta: {0}".format(model.theta))
# print("Sigma: {0}".format(model.sigma2))
# print("Beta: {0}".format(model.beta))
# model.build()
# model.use(xtt)
# plt.plot(x, y, 'o')
# plt.plot(xtt,  model.yhat)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])

# #plt.scatter(yt, model.yhat)
# plt.show()



#=================================Passed Test============================================

# def Obj_function(x):
#     #x = preprocessing.normalize(x)
#     x = x
#     return (x*np.sin(x))

# n = 10
# x = hs.halton_sequence(n,1)
# x = np.array(x)
# #x = np.sort(x)
# x = x.T
# x = pre.denormalize(x,0,(2*np.pi))
# print(x)
# #print out the value of x
# #x = [7.5,3.75,11.25,1.875,9.375,5.625,13.125,0.9375,8.4375,4.6875]
# #x = np.sort(x)
# y = Obj_function(x)
# #xtt = np.linspace(-0.5,(2*np.pi)+0.5,1000)
# xtt = np.linspace(-0.5,(2*np.pi)+0.5,100)
# x = np.array(x)[:,np.newaxis]
# xtt = np.array(xtt)[:,np.newaxis]
# y = np.array(y)[:,np.newaxis]

# model = Kriging(x, y, GE=False, globalType='constant')
# model.getTheta([[0.5]])
# print("Theta: {0}".format(model.theta))
# print("Sigma: {0}".format(model.sigma2))
# print("Beta: {0}".format(model.beta))
# print("Likelihood: {0}".format(model.likelihood))
# print("R determinant: {0}".format(model.detR))
# model.build()
# model.use(xtt)
# y_exact = Obj_function(xtt)
# RMSE = model.computeRMSE(y_exact)
# RMSE_range = RMSE/(max(y_exact)-min(y_exact))
# print("RMSE: {0}".format(RMSE))

# fileN = "1DOutput.csv"
# csv = open(fileN,"w")
# Header = "Y_hat\n"
# csv.write(Header)
# for i in range(len(model.yhat)):
#     a = model.yhat[i]
#     b = str(a)
#     b +="\n"
#     csv.write(b)
# csv.close()

# plt.plot(x, y, 'o')
# plt.plot(xtt,  model.yhat)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Training data', 'Prediction'])
# plt.title("Prof's Prediction $y = x sin(x)$")
# #plt.scatter(yt, model.yhat)
# plt.show()


#===================================End Here=================================================

# N = [2,5,7,8,10,12,15,17,19,20,22,23,25,30]
# Kernels = ['gaussian','exponential','matern3_2','matern5_2']
# RMSE_data = np.zeros(len(N))
# for i in range(len(Kernels)):
#     for j in range(len(N)):
#         n = N[j]
#         xt_n = hs.halton_sequence(n,2)
#         xt_n = np.array(xt_n)
#         #xt_n = xt_n.T
#         xt = np.zeros((n,2))
#         xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
#         xt[:,1] = pre.denormalize(xt_n[:,1],0,15)

#         yt = BP.branin(xt[:,0],xt[:,1])
#         model = Kriging(xt,yt,GE=False, globalType='constant')
#         model.getTheta([[0.5,0.5]])
#         model.build()
#         num = 100
#         xtest = np.zeros((num,2))
#         xtest[:,0] = np.linspace(-5,10,num)
#         xtest[:,1] = np.linspace(0,15,num)
#         x = np.vstack((xtest[:,0],xtest[:,1]))
#         model.use(x.T)
#         y = model.yhat
#         y_exact = BP.branin(xtest[:,0],xtest[:,1])
#         RMSE = model.computeRMS(y_exact)
#         RMSE_range = RMSE/(max(y_exact)-min(y_exact))
#         RMSE_data[j] = RMSE_range
#     fig,ax = plt.subplots()
#     plt.autoscale(enable=True, axis='both', tight=None)
#     plt.scatter(N,RMSE_data)
#     plt.xlabel('Sample size')
#     plt.ylabel('RMSE/range(f)')
#     ax.legend(Kernels[i],loc='best',fancybox=True)
#     plt.title('Branin function  (Eval point: 100)')
#     plt.show()

#======================================Next=============================================

n = 25
xt_n = hs.halton_sequence(n,2)
xt_n = np.array(xt_n)
xt_n = xt_n.T
xt = np.zeros((n,2))
xt[:,0] = pre.denormalize(xt_n[:,0],-5,10)
xt[:,1] = pre.denormalize(xt_n[:,1],0,15)

yt = BP.branin(xt[:,0],xt[:,1])
model = Kriging(xt,yt,GE=False, globalType='constant')
model.getTheta([0.01,0.01])

print("Theta: {0}".format(model.theta))
print("Sigma: {0}".format(model.sigma2))
print("Beta: {0}".format(model.beta))
print("Likelihood: {0}".format(model.likelihood))
print("R determinant: {0}".format(model.detR))

model.build()
num = 100
xtest = np.zeros((num,2))
xtest[:,0] = np.linspace(-5,10,num)
xtest[:,1] = np.linspace(0,15,num)


l = len(xtest[:,0])
X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
grid_Z = np.zeros((l,l))
for i in range(l):
    x = np.vstack((X[i,:],Y[i,:]))
    model.use(x.T)
    y = model.yhat
    y = y.reshape(l,)
    grid_Z[i,:] = y

y_exact = BP.branin(xtest[:,0],xtest[:,1])

g = np.array(grid_Z)
d = np.diag(g)
yhat = np.array(d)

fileN = "2DOutput.csv"
csv = open(fileN,"w")
Header = "Y_exact\n"
csv.write(Header)
for i in range(len(y_exact)):
    a = y_exact[i]
    b = str(a)
    b +="\n"
    csv.write(b)

Header1 = "Y_hat\n"
csv.write(Header1)
for i in range(len(yhat)):
    a = yhat[i]
    b = str(a)
    b +="\n"
    csv.write(b)
csv.close()

model.yhat = yhat
RMSE = model.computeRMSE(y_exact)
RMSE_range = RMSE/(max(y_exact)-min(y_exact))
print("RMSE: {0}".format(RMSE))

plt.figure(1,figsize=(9,6))
plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
plt.title("Branin Function (Prof's prediction)",fontsize=20)
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()
#plt.scatter(xt[:,0],xt[:,1],c=yt,cmap="RdGy")
plt.show()

#========================================End Batch=============================


# n = 35
# xt_n = hs.halton_sequence(n,2)
# xt_n = np.array(xt_n)
# xt_n = xt_n.T
# xt = np.zeros((n,2))
# xt[:,0] = pre.denormalize(xt_n[:,0],-6,6)
# xt[:,1] = pre.denormalize(xt_n[:,1],-6,6)

# yt = BP.himmelblau(xt[:,0],xt[:,1])
# model = Kriging(xt,yt,GE=False, globalType='constant')
# model.getTheta([0.01,0.01])
# model.build()
# num = 100
# xtest = np.zeros((num,2))
# xtest[:,0] = np.linspace(-6,6,num)
# xtest[:,1] = np.linspace(-6,6,num)


# l = len(xtest[:,0])
# X,Y = np.meshgrid(xtest[:,0],xtest[:,1])
# grid_Z = np.zeros((l,l))
# for i in range(l):
#     x = np.vstack((X[i,:],Y[i,:]))
#     model.use(x.T)
#     y = model.yhat
#     y = y.reshape(l,)
#     grid_Z[i,:] = y
# y_exact = BP.branin(xtest[:,0],xtest[:,1])
# plt.figure(1,figsize=(9,6))
# plt.contourf(X,Y,grid_Z,20,cmap='RdGy')
# plt.title('Himmelblau Function',fontsize=20)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.colorbar()
# plt.scatter(xt[:,0],xt[:,1],c=yt,cmap="RdGy")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import halton_sequence as hs
import lhs
import pattern_search
import preprocessing
import nearestPD as NPD


class Kriging:

    def __init__(self,x,y,kernel,theta=0.1,sampleGenerator='Hal',eps=1.49e-08):
        self.x = x
        self.y = y
        self.kernel = kernel
        self.theta = theta
        self.eps = eps
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]

    def compute_distances(self,X1,X2): #squared euclidean distance
        n = X2.shape[0] #if not for this bug, I wont need to do the unnecessary transpose for K_s
        k = X1.shape[0]  #fix when all works well
        X1 = X1.reshape(k,1) #must reflect fix
        X2 = X2.reshape(n,1) #must reflect fix
        dists = np.zeros((n,k))
        for i in range(n):
            dists[i,:] = np.sqrt(np.sum((X1 - (X2[i,:]))**2, axis=1)) #effort to compute D instead
        return dists

    def compute_K(self,X,Y,theta=1,sigma2=1):
        d = self.compute_distances(X,Y)
        if self.kernel == "exponential":
            K = (sigma2 * np.exp(-d/theta))
        elif self.kernel == "matern3_2":
            K = (1 + (np.sqrt(3)*d)/theta)*np.exp(((-np.sqrt(3))*d)/theta) 
        elif self.kernel == "gaussian":
            K = (sigma2 * np.exp(-0.5 * ((d/theta) ** 2))) #on the suspicion that d is already squared
        elif self.kernel == "matern5_2":
            K = (1 + (np.sqrt(5)*d)/theta + (5/3*(d/theta)**2))*np.exp(((-np.sqrt(5))*d)/theta) 
        return K



    def nll(self,length_scale,DistMat,Y):
        '''
        Function to obtain the negative log likelihood for the length scale and noise 
        sig_n: noise standard deviation
        DistMat: Pairwise Euclidean Distance Matrix
        Y: List of targets / Observations
        '''
        n = len(Y)
        if (self.kernel=='gaussian'):
            K = (np.exp(-0.5 * ((DistMat/length_scale) ** 2))) #investigate, D is already a square, I just need to take the square of theta
        elif (self.kernel=='exponential'):
            K = (np.exp(-DistMat/length_scale))
        elif (self.kernel=='matern3_2'):
            K = ((1 + (np.sqrt(3)*DistMat)/length_scale)*np.exp(((-np.sqrt(3))*DistMat)/length_scale))
        elif (self.kernel=='matern5_2'):
            K = ((1 + (np.sqrt(5)*DistMat)/length_scale + (5/3*(DistMat/length_scale)**2))*np.exp(((-np.sqrt(5))*DistMat)/length_scale))
        if NPD.isPD(K)==False:
            K = NPD.nearestPD(K)
        else:
            K = K
        c = np.linalg.inv(np.linalg.cholesky(K))           
        Ki = np.dot(c.T,c)
        (sign, logdetK) = np.linalg.slogdet(K)  # returns the sign and natural logarithm of the determinant of an array
        self.sigma2 = np.dot(Y.T,(np.dot(Ki,Y)))
        sigma2 = self.sigma2
        #ll = -(n/2) * np.log(2*np.pi) - (n/2)*np.log(sigma2) - 1/2 * logdetK - (1/(2*sigma2))*np.dot(Y.T,(np.dot(Ki,Y)))
        ll = -n/2 * np.log(np.dot(Y.T,(np.dot(Ki,Y)))) - 1/2 * logdetK #k is the number of observations
        return -ll



    def get_theta(self,xk=[0.8],delta=0.5):
        '''
        xk: starting point of the pattern search
        delta: the increment
        X: the multidimensional input. shape could be (20,3) for 3 Dimensional problems
        '''
        #k = self.Nk #dimension of the sample 
        theta = np.zeros(self.Nk)
        iteration = np.zeros(self.Nk)
        for i in range(self.Nk):
            if self.Nk ==1:
                Xtrain = self.x_normal #debug line
                #Xtrain = preprocessing.normalize(self.x)
                DistMat = self.compute_distances(Xtrain,Xtrain)
            else:
                Xtrain = np.zeros((self.Ns,self.Nk))
                #Xtrain[i] =preprocessing.normalize(self.x[i]) #debug line
                Xtrain[:,i] =self.x_normal[:,i]
            #Xtrain = (np.array(Xtrain))[:,np.newaxis]
                Xtrain[:,i] = (np.array(Xtrain[:,i]))
                DistMat = self.compute_distances(Xtrain[:,i],Xtrain[:,i])
        # kernels: exponential gaussian matern3_ matern5_2
            NT = lambda theta: self.nll(theta,DistMat=DistMat,Y=self.y_normal)
            [x,k]= pattern_search.hooke_jeeves(xk,delta,NT)
            theta[i] = x
            iteration[i] = k
        self.theta = theta
        return theta

    def K_components(self,Xt):
        K = 1 #initialization
        K_s = 1 #initialization
        K_ss = 1 #initialization
        theta = self.get_theta()
        Xt = np.array(Xt)
        k = self.Nk #dimension of the sample 
        for i in range(k): #gets the number of dimensions
            if k ==1:
                #Xtrain = preprocessing.normalize(self.x) #debug line
                #Xtest = preprocessing.normalize(Xt) #debug line
                Xtrain = self.x_normal
                Xtest = Xt
            else:
                Xtrain = np.copy(self.x_normal)
                Xtest = np.copy(Xt)
                # Xtrain[i] = preprocessing.normalize(Xtrain[i]) #normalizing the inputs
                # Xtest[i] = preprocessing.normalize(Xtest[i])   #normalizing the inputs
                Xtrain[i] = Xtrain[i] #normalizing the inputs
                Xtest[i] = Xtest[i]  #normalizing the inputs
            #K *= (self.compute_K(Xtrain[:,i],Xtrain[:,i],theta[i]) * np.exp(np.diag(self.eps*np.eye(self.Ns))))
            K *= (self.compute_K(Xtrain[:,i],Xtrain[:,i],theta[i]))
            K_s *= self.compute_K(Xtrain[:,i],Xtest[:,i],theta[i])
            K_ss *= self.compute_K(Xtest[:,i],Xtest[:,i],theta[i])
        return K,K_s.T,K_ss  #had to use transpose for this to work, take note


    def train(self,testdata):
        #estimate hyperparameter
        #train the model
        #Xtest is the test data 
        
        self.testdata = testdata
        if(self.Nk==1):
            self.testdata = preprocessing.normalize(testdata)
        else: 
            for i in range(self.Nk):
                self.testdata[:,i] = preprocessing.normalize(self.testdata[:,i])

        theta = self.get_theta()
        #self.Xtest = np.array(testdata)[:,np.newaxis]
        self.Xtest = np.array(self.testdata)
        self.Nsx = self.Xtest.shape[0] #number of test data
        self.Nkx = self.Nk
        #dimension of testdata must be equal to dimension of sample data
        
        # for i in range(self.Nkx):
        #     #normalize the test data
        #     if(self.Nkx==1):
        #         self.Xtest = preprocessing.normalize(self.Xtest)
        #     else:
        #         self.Xtest[i] = preprocessing.normalize(self.Xtest[i])


        Kc = self.K_components(self.Xtest)

        K = Kc[0] # K(X,X)
        self.K_s = Kc[1] # K(X,X*)
        K_ss = Kc[2] # K(X*,X*)

        if(NPD.isPD(K)): #sane check
            self.K = K
        else:
            self.K = NPD.nearestPD(K)
        if(NPD.isPD(K_ss)): #sane check
            self.K_ss = K_ss
        else:
            self.K_ss = NPD.nearestPD(K_ss)

        self.L = np.linalg.cholesky(self.K) # lower cholesky factor
        self.Lk = np.linalg.solve(self.L, self.K_s)

    
    def predict(self):
        mu = np.dot(self.Lk.T, np.linalg.solve(self.L,self.y_normal)).reshape((self.Nsx,))
        self.y_predict = mu.reshape(-1,1)
        s2 = np.diag(self.K_ss) - np.sum(self.Lk**2,axis=0) # variance
        #stdv = np.sqrt(s2) #standard deviation
        L = np.linalg.cholesky(self.K_ss+self.eps*np.eye(self.Nsx) - np.dot(self.Lk.T,self.Lk)) # add small jitter to keep the kernel psd
        self.f_post = mu.reshape(-1,1) + np.dot(L,np.random.normal(size=(self.Nsx,1)))
        #[minY,maxY] = self.getYScale() #debug line
        #self.y_output = (self.y_mean + (self.y_std*self.y_predict)) #denormalized here
        self.y_output = preprocessing.denormalize(self.y_predict,self.y_min,self.y_max)
        return self.y_output

    def computeRMSE(self,y_exact):
        m = len(self.y_output) 
        sum = 0.0
        for i in range(m):
            sum += np.power((y_exact[i] - self.y_output[i]),2)
        self.RMSE = sum / m
        return self.RMSE
    
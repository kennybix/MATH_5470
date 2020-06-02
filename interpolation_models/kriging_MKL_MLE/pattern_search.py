import numpy as np

def checkPositive(x):
    k = len(x)
    count = 0
    for i in range(k):
        if x[i] > 0:
            # positive entry, increment count
            count+=1
        if count == k:
            constraint = True
        else:
            constraint = False
    return constraint


def explore(xk,delta,obj_func,alpha=2):
    '''
    This method aims at finding the best nearby points
    xk: starting point
    delta: usually the increment vector
    obj_func: objective function 
    '''
    n = len(xk) # get the length of the vector 
    xc = np.copy(xk) # not correct but this should be the base point
    for i in range(n):
        xk = np.copy(xc)
        farray = np.zeros(3)
        xkp = np.copy(xk)
        xkm = np.copy(xk)
        xkp[i] = xkp[i]+delta
        #print(xkp[i])
        #print(xkp)
        xkm[i] = xkm[i]-delta
        fplus = obj_func(xkp)
        f = obj_func(xk)
        fmin = obj_func(xkm)
        farray[0]= fmin
        farray[1] = f
        farray[2] = fplus
        indexmin = np.argmin(farray)
        if indexmin == 0:
            xc = xkm
        elif indexmin ==1:
            xc = xk
        else:
            xc = xkp
        if (checkPositive(xc)): #added to force the point to be positive
            fminimum = farray[indexmin] #indexed to tally with previous line
        else:
            [fminimum,xc] = explore(xk+0.5,delta,obj_func)
    return fminimum,xc  

def hooke_jeeves(xk,delta,obj_func,eps=0.000001,itermax=500,sensitivity=0.1):
    steplength = delta
    k = 0
    xk = np.array(xk)
    xbefore = xk
    fbefore = obj_func(xk)
    newx = np.copy(xk)
    n = len(xbefore) 
    while (steplength > eps):
        k +=1
        [newf,newx] = explore(newx,delta,obj_func)  #initial exploration
        while(newf<fbefore and k < itermax): #for every successful exploratory, perform a pattern move
            for i in range(len(newx)):
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                newx[i] = newx[i]*2 - tmp
            fbefore = newf #hold before exploration
            [newf,newx] = explore(newx,delta,obj_func) # obtain the new points by exploratory move
            for i in range(len(newx)): #debug lines
                xbefore[i] = newx[i]    #debug lines
            k +=1
            for i in range(len(newx)):
                if (abs(newx[i] - xbefore[i]))<sensitivity*abs(delta): #no much improvement, terminate
                    break
            if (abs(newf- fbefore))<sensitivity*abs(delta): #no much improvement, terminate
                steplength = eps #hack - test passed
                break
        if (steplength > eps and newf >= fbefore):
            steplength /= 2
            delta /=2
    x = xbefore
    return x,k
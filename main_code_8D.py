import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sampling_plan_for_real_data as sp


#import the different models 
from interpolation_models.core import kriging_NR as KCORE
from interpolation_models.ensemble import kriging_ensemble as KRE
from interpolation_models.ensemble import ensemble as KEM
from interpolation_models.kriging_MKL_MLE  import kriging_NR as KCKL
from interpolation_models.kriging_Multilayer_CKL import kriging_NR as KMCKL

import time 

data = pd.read_excel(r'../data/single_8D.xlsx')

x = data.drop('cd',axis=1)
y = data['cd']

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, random_state=30,train_size=0.8) #split the dataset into training data and truthset for validation

y_test = np.array(ytest)
#set parameters
file1 = open("single_8D.txt","a") 
n = [20,40,60]
kernels = ['gaussian','exponential','matern3_2','matern5_2']

k = x.shape[1] #dimension of data
theta0 = [0.5] * k

G = []
E = []
M3 = []
M5 = []
EM = []
CKL = []
MCKL = []
MiKL = []

G_tt = []
E_tt = []
M3_tt = []
M5_tt = []
EM_tt = []
CKL_tt = []
MCKL_tt = []
MiKL_tt = []

G_pt = []
E_pt = []
M3_pt = []
M5_pt = []
EM_pt = []
CKL_pt = []
MCKL_pt = []
MiKL_pt = []


for i in range(len(n)):

    # reset 
    G = []
    E = []
    M3 = []
    M5 = []
    EM = []
    CKL = []
    MCKL = []
    MiKL = []

    G_tt = []
    E_tt = []
    M3_tt = []
    M5_tt = []
    EM_tt = []
    CKL_tt = []
    MCKL_tt = []
    MiKL_tt = []

    G_pt = []
    E_pt = []
    M3_pt = []
    M5_pt = []
    EM_pt = []
    CKL_pt = []
    MCKL_pt = []
    MiKL_pt = []


    file1.write("Sample size: {0}".format(n[i]))

    plan = sp.Sampling_plan(Xtrain,ytrain,Ns=n[i])
    x_train,y_train = plan.create_samples()

    x_scaler = MinMaxScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)

    y_train = np.array(y_train)
    y_train = y_train.reshape(n[i],1)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)

    
    #transform test input for prediction
    x_test = x_scaler.transform(Xtest)

    #build model and predict with it 
    g_model = KCORE.Kriging(x_train,y_train,kernels[0],optimizer="nelder-mead-c",theta0=theta0)
    start_time = time.time()
    g_model.train()
    elapsed_time = time.time() - start_time
    G_tt.append(elapsed_time)

    start_time = time.time()
    g_y = g_model.predict(x_test)
    elapsed_time = time.time() - start_time
    G_pt.append(elapsed_time)

    g_y = y_scaler.inverse_transform(g_y)
    g_model.y_output = g_y

    RMSE = 100 * g_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,g_y)
    R2 = r2_score(y_test,g_y)
    MSE = mean_squared_error(y_test,g_y)

    errors = [RMSE[0],MAE,R2,MSE]
    G.append(errors)

    e_model = KCORE.Kriging(x_train,y_train,kernels[1],optimizer="nelder-mead-c",theta0=theta0)
    start_time = time.time()
    e_model.train()
    elapsed_time = time.time() - start_time
    E_tt.append(elapsed_time)

    start_time = time.time()       
    e_y = e_model.predict(x_test)
    elapsed_time = time.time() - start_time
    E_pt.append(elapsed_time)

    e_y = y_scaler.inverse_transform(e_y)
    e_model.y_output = e_y

    RMSE = 100 * e_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,e_y)
    R2 = r2_score(y_test,e_y)
    MSE = mean_squared_error(y_test,e_y)

    errors = [RMSE[0],MAE,R2,MSE]
    E.append(errors)


    # matern 3/2
    m3_model = KCORE.Kriging(x_train,y_train,kernels[2],optimizer="nelder-mead-c",theta0=theta0)
    start_time = time.time()
    m3_model.train()
    elapsed_time = time.time() - start_time
    M3_tt.append(elapsed_time)

    start_time = time.time()
    m3_y = m3_model.predict(x_test)
    elapsed_time = time.time() - start_time
    M3_pt.append(elapsed_time)

    m3_y = y_scaler.inverse_transform(m3_y)
    m3_model.y_output = m3_y
    RMSE = 100 * m3_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,m3_y)
    R2 = r2_score(y_test,m3_y)
    MSE = mean_squared_error(y_test,m3_y)
    errors = [RMSE[0],MAE,R2,MSE]
    M3.append(errors)


    m5_model = KCORE.Kriging(x_train,y_train,kernels[3],optimizer="nelder-mead-c",theta0=theta0)
    start_time = time.time()
    m5_model.train()
    elapsed_time = time.time() - start_time
    M5_tt.append(elapsed_time)

    start_time = time.time()
    m5_y = m5_model.predict(x_test)
    elapsed_time = time.time() - start_time
    M5_pt.append(elapsed_time)

    m5_y = y_scaler.inverse_transform(m5_y)
    m5_model.y_output = m5_y
    RMSE = 100 * m5_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,m5_y)
    R2 = r2_score(y_test,m5_y)
    MSE = mean_squared_error(y_test,m5_y)
    errors = [RMSE[0],MAE,R2,MSE]
    M5.append(errors)

    # ensemble model using AICc global weighting
    em_model = KEM.ensemble(x_train,y_train,kernels,theta0,method='AICc',optimizer="nelder-mead-c")
    
    start_time = time.time()
    em_model.train()
    elapsed_time = time.time() - start_time
    EM_tt.append(elapsed_time)

    start_time = time.time()
    em_y = em_model.predict(x_test)
    elapsed_time = time.time() - start_time
    EM_pt.append(elapsed_time)

    # hack 
    em_y = np.array(em_y)
    em_y = em_y.reshape(len(em_y),1)
    # hack ends 

    em_y = y_scaler.inverse_transform(em_y)
    em_model.y_output = em_y
    RMSE = 100 * em_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,em_y)
    R2 = r2_score(y_test,em_y)
    MSE = mean_squared_error(y_test,em_y)
    errors = [RMSE[0],MAE,R2,MSE]
    EM.append(errors)


    ckl_model = KCKL.Kriging(x_train,y_train,kernels,theta0=theta0)
    start_time = time.time()
    ckl_model.train()
    elapsed_time = time.time() - start_time
    CKL_tt.append(elapsed_time)

    start_time = time.time()
    ckl_y = ckl_model.predict(x_test)
    elapsed_time = time.time() - start_time
    CKL_pt.append(elapsed_time)

    ckl_y = y_scaler.inverse_transform(ckl_y)
    ckl_model.y_output = ckl_y
    RMSE = 100 * ckl_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,ckl_y)
    R2 = r2_score(y_test,ckl_y)
    MSE = mean_squared_error(y_test,ckl_y)
    errors = [RMSE[0],MAE,R2,MSE]
    CKL.append(errors)

    # mikl_model = KMKL.Kriging(x_train,y_train,kernels,theta0=theta0,optimizer="nelder-mead-c")
    # start_time = time.time()
    # mikl_model.train()
    # elapsed_time = time.time() - start_time
    # MiKL_tt.append(elapsed_time)

    # start_time = time.time()
    # mikl_y = mikl_model.predict(x_test)
    # elapsed_time = time.time() - start_time
    # MiKL_pt.append(elapsed_time)

    # mikl_y = pre.denormalize(mikl_y,y_min,y_max)
    # mikl_model.y_output = mikl_y
    # RMSE = 100 * mikl_model.computeNRMSE(y_test)
    # MiKL.append(RMSE[0])



    mckl_model = KMCKL.Kriging(x_train,y_train,kernels,theta0=theta0)

    start_time = time.time()
    mckl_model.train()
    elapsed_time = time.time() - start_time
    MCKL_tt.append(elapsed_time)

    start_time = time.time()
    mckl_y = mckl_model.predict(x_test)
    elapsed_time = time.time() - start_time
    MCKL_pt.append(elapsed_time)

    mckl_y = y_scaler.inverse_transform(mckl_y)
    mckl_model.y_output = mckl_y
    RMSE = 100 * mckl_model.computeNRMSE(y_test)
    MAE = mean_absolute_error(y_test,mckl_y)
    R2 = r2_score(y_test,mckl_y)
    MSE = mean_squared_error(y_test,mckl_y)
    errors = [RMSE[0],MAE,R2,MSE]
    MCKL.append(errors)

    file1.write("\n")
    file1.write("\n")
    file1.writelines("Sample size: {0}".format(n[i]))
    file1.write("\n")
    file1.write("\n")

    file1.writelines("G = {0}".format(G))
    file1.write("\n")
    file1.writelines("E = {0}".format(E))
    file1.write("\n")
    file1.writelines("M3 = {0}".format(M3))
    file1.write("\n")
    file1.writelines("M5 = {0}".format(M5))
    file1.write("\n")
    file1.writelines("EM = {0}".format(EM))
    file1.write("\n")
    file1.writelines("CKL = {0}".format(CKL))
    file1.write("\n")
    file1.writelines("MCKL = {0}".format(MCKL))
    file1.write("\n")
    file1.writelines("MiKL = {0}".format(MiKL))
    file1.write("\n")


    file1.writelines("G_tt = {0}".format(G_tt))
    file1.write("\n")
    file1.writelines("E_tt = {0}".format(E_tt))
    file1.write("\n")
    file1.writelines("M3_tt = {0}".format(M3_tt))
    file1.write("\n")
    file1.writelines("M5_tt = {0}".format(M5_tt))
    file1.write("\n")
    file1.writelines("EM_tt = {0}".format(EM_tt))
    file1.write("\n")
    file1.writelines("CKL_tt = {0}".format(CKL_tt))
    file1.write("\n")
    file1.writelines("MCKL_tt = {0}".format(MCKL_tt))
    file1.write("\n")
    file1.writelines("MiKL_tt = {0}".format(MiKL_tt))

    file1.write("\n")



    file1.writelines("G_pt = {0}".format(G_pt))
    file1.write("\n")
    file1.writelines("E_pt = {0}".format(E_pt))
    file1.write("\n")
    file1.writelines("M3_pt = {0}".format(M3_pt))
    file1.write("\n")
    file1.writelines("M5_pt = {0}".format(M5_pt))
    file1.write("\n")
    file1.writelines("EM_pt = {0}".format(EM_pt))
    file1.write("\n")
    file1.writelines("CKL_pt = {0}".format(CKL_pt))
    file1.write("\n")
    file1.writelines("MCKL_pt = {0}".format(MCKL_pt))
    file1.write("\n")
    # file1.writelines("MiKL_pt = {0}".format(MiKL_pt))
    file1.write("\n")
    # file1.writelines("BMKL = {0}".format(BMKL))
    # file1.write("\n")
    file1.write("\n")
file1.close()


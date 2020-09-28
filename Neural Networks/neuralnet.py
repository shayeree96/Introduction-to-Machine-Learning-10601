#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:33:25 2020

@author: shayereesarkar
"""
import random
import numpy as np
import scipy as sp
import sys

train_table=np.loadtxt(sys.argv[1],dtype='int',delimiter=',')    
test_table=np.loadtxt(sys.argv[2],dtype='int',delimiter=',')
X=np.loadtxt(sys.argv[1],dtype='int',delimiter=',')  
X_test=np.loadtxt(sys.argv[2],dtype='int',delimiter=',')

gamma=float(sys.argv[9])
epochs=int(sys.argv[6])
D=int(sys.argv[7])
init_flag=int(sys.argv[8])
K=10#No of outoput classes
M=train_table.shape[1]
if init_flag==2:
    alpha=np.zeros((D,M))
    beta=np.zeros((K,D+1))
else:    
    alpha=np.random.uniform(-0.1,0.1,(D,M))#For random nos
    beta=np.random.uniform(-0.1,0.1,(K,D+1))
X[:,0] = 1#Initializing all bias terms

X_test[:,0] = 1

#def stochastic_gradient(train_table,X,alpha,beta):
    #During Training
def sigmoid(theta_transpose_x):#Sigmoid working fine
    
    sig=np.exp(theta_transpose_x)/(1+np.exp(theta_transpose_x))
    
    return sig

def CROSSENTROPYBACKWARD(y_actual,y_pred,J,gJ):
    gy=np.zeros((len(y_pred),1))
    
    for i in range(0,len(y_pred)):
        gy[i,0]=-y_actual[i]/y_pred[i]
        
    return gy    
            
def SOFTMAXBACKWARD(b,y_pred,gyˆ):
    gb=np.zeros((len(y_pred),len(y_pred)))
#    gb_vec=np.matmul(y_pred,(1-y_pred))
    for i in range(0,len(y_pred)):
        for j in range(0,len(y_pred)):
            if(i==j):
                 gb[i,j]=(y_pred[j]*(1-y_pred[j]))
            else:
                gb[i,j]=-y_pred[j]*y_pred[i]
        
    return gb

def LINEARBACKWARD_z(z, beta, gb):
    g_beta=z
    g_z=beta[:,1:]
    #print(g_z)
    
    return g_beta,g_z

def SIGMOIDBACKWARD(a, z, gz):
    ga=np.zeros((len(z),1))
    #print(ga.shape)
    for i in range(0,len(z)):
        ga[i,0]=z[i]*(1-z[i])
    
    ga=np.delete(ga,0,0)   
    return ga

def LINEARBACKWARD_a(x, a, ga):
    g_alpha=x
    gx=a
    return g_alpha,gx
def neural_net_predict(data,alpha,beta,X_test,J_entropy):
    
    row_len=data.shape[0]
    for i in range(0,1):#1.For epoch
       J=[]
       y_label=[]
       sum=0
       for j in range(0,row_len):#For each training example pair
            #Compute a_j, z_j,b_k ,y_pred and J_objective
            #Forward Propagation
            X_new=np.zeros((len(X_test[j,:]),1))
            X_new[:,0]=np.transpose(X_test[j,:])
            #alpha[:,0]=1
            a=np.matmul(alpha,X_new)#1.Linear layer
            z=sigmoid(a)#2.Sigmoid layer #include bias term
            z=np.insert(z,0,[1],axis=0)#New z with the bias term
            
            b=np.matmul(beta,z)
            y_pred=np.zeros((K,1))
            for i in range(0,K):
                
                y_pred[i]=np.exp(b[i])/np.sum(np.exp(b))
            
            y_label.append(np.argmax(y_pred))
            
            y_actual=np.zeros((K,1))
            y_actual[data[j,0]]=1#turning y_actual into a hot encoding vector
            sum+=-np.sum(np.multiply(y_actual,np.log(y_pred)))
            
            
    J_entropy.append(np.sum(sum)/row_len)

    return J_entropy,y_pred,y_label


def neural_net(data,alpha,beta,X,gamma,X_test):
    J_entropy=[]
    J_entropy_test=[]
    row_len=data.shape[0]
    for i in range(0,epochs):#1.For epoch
       J=[]
       y_label=[]
       
       for j in range(0,row_len):#For each training example pair
            #Compute a_j, z_j,b_k ,y_pred and J_objective
            #Forward Propagation
            X_new=np.zeros((len(X[j,:]),1))
            X_new[:,0]=np.transpose(X[j,:])
            #alpha[:,0]=1
            a=np.matmul(alpha,X_new)#1.Linear layer
            z=sigmoid(a)#2.Sigmoid layer #include bias term
            z=np.insert(z,0,[1],axis=0)#New z with the bias term
            
            b=np.matmul(beta,z)
            #b[0,0]=1
            y_pred=np.zeros((K,1))
            for i in range(0,K):
                
                y_pred[i]=np.exp(b[i])/np.sum(np.exp(b))#3.Softmax layer
                #print(y_pred)
            #y_label.append(np.argmax(y_pred))
            y_actual=np.zeros((K,1))
            y_actual[data[j,0]]=1#turning y_actual into a hot encoding vector
            #1: procedureNNBACKWARD(Trainingexample(x,y),Parametersα,β,Intermediateso)
            #2: Place intermediate quantities x, a, z, b, yˆ, J in o in scope
            gJ=1 #3: gJ = dJ =1 dJ
            gyˆ=CROSSENTROPYBACKWARD(y_actual,y_pred,J,gJ)# 4: gyˆ = CROSSENTROPYBACKWARD(y, yˆ, J, gJ )
            gb = SOFTMAXBACKWARD(b,y_pred,gyˆ)#5: gb = SOFTMAXBACKWARD(b,yˆ,gyˆ)
            g_beta, gz = LINEARBACKWARD_z(z, beta, gb)#6: gβ, gz = LINEARBACKWARD(z, b, gb)
            ga = SIGMOIDBACKWARD(a, z, gz)#7: ga = SIGMOIDBACKWARD(a, z, gz)
            g_alpha, gx = LINEARBACKWARD_a(X_new, alpha, ga)#8: gα, gx = LINEARBACKWARD(x, a, ga)
            mul1=np.matmul(np.transpose(gyˆ),gb)
            mul2=np.matmul(mul1,gz)
            mul3=np.multiply(mul2,np.transpose(ga))
            mul4=np.transpose(np.matmul(g_alpha,mul3))
            grad_alpha=mul4
            grad_beta=np.transpose(np.matmul(g_beta,mul1))
            
            alpha=alpha-gamma*grad_alpha
            beta=beta-gamma*grad_beta
       sum=0     
       for j in range(0,row_len):
                X_new=np.zeros((len(X[j,:]),1))
                X_new[:,0]=np.transpose(X[j,:])
            #alpha[:,0]=1
                a=np.matmul(alpha,X_new)#1.Linear layer
                z=sigmoid(a)#2.Sigmoid layer #include bias term
                z=np.insert(z,0,[1],axis=0)#New z with the bias term
                b=np.matmul(beta,z)
                #b[0,0]=1
                y_pred=np.zeros((K,1))
                for i in range(0,K):
                    
                    y_pred[i]=np.exp(b[i])/np.sum(np.exp(b))#3.Softmax layer
                #print(y_pred)
                y_actual=np.zeros((K,1))
                y_actual[data[j,0]]=1
                y_label.append(np.argmax(y_pred))
                sum+=-np.sum(np.multiply(y_actual,np.log(y_pred)))
                

       J_entropy.append((sum)/row_len)              
       J_entropy_test,y_pred_test,y_label_test=neural_net_predict(test_table,alpha,beta,X_test,J_entropy_test)     
       #print(np.sum(sum))     
            #grad_alpha=np.transpose(np.matmul(g_alpha,(np.multiply(np.matmul(np.transpose(gyˆ),gb),gz),np.transpose(ga))))

    return alpha,beta,J_entropy,y_label,y_pred,J_entropy_test,y_pred_test,y_label_test

alpha_train,beta_train,J_entropy_train,y_label_train,y_pred_train,J_entropy_test,y_pred_test,y_label_test=neural_net(train_table,alpha,beta,X,gamma,X_test) 

error_train=0.000000000000000

for i in range(0,len(train_table)):
    
    if int(train_table[i][0])!=y_label_train[i]:
        error_train+=1
error_train=error_train/len(train_table) 
       
error_test=0.0000000000000000
for i in range(0,len(test_table)):
    if int(test_table[i][0])!=y_label_test[i]:
        error_test+=1
error_output=[]      
error_test=error_test/len(test_table) 
k=''
for i in range(0,epochs):
    k+='epoch='+str(i+1)+' crossentropy(train): '+str(J_entropy_train[i])+'\n'+'epoch='+str(i+1)+' crossentropy(test): '+str(J_entropy_test[i])+'\n' 
k+='error(train): '+str(error_train)+'\n'+'error(test): '+str(error_test)
l=[]      
#erroroutput=[[epoch=1 crossentropy(train): 2.18506276114['error(train):',error_train],['error(test):',error_test]] 
l.append(k)

with open(sys.argv[3], 'w') as train_out:
    for i in y_label_train:
        train_out.write('%s\n' % i)
        
with open(sys.argv[4], 'w') as test_out:
    for i in y_label_test:
        test_out.write('%s\n' % i)        

with open(sys.argv[5], 'w') as metrics_out:
    metrics_out.write('%s' % k) 



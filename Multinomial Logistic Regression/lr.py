#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:36:20 2020

@author: shayereesarkar
"""

"""
Created on Thu Feb 27 03:51:57 2020

@author: shayereesarkar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:21:12 2020

@author: shayereesarkar
"""
import sys
import timeit
import numpy as np
import csv
import time
#train_table=np.loadtxt('/Users/shayereesarkar/Desktop/10601/hw4/handout/smalloutput/model1_formatted_train.tsv',dtype='str',delimiter='\t',comments=None)
#valid_table=np.loadtxt('/Users/shayereesarkar/Desktop/10601/hw4/handout/smalloutput/model1_formatted_valid.tsv',dtype='str',delimiter='\t',comments=None)
#test_table=np.loadtxt('/Users/shayereesarkar/Desktop/10601/hw4/handout/smalloutput/model1_formatted_test.tsv',dtype='str',delimiter='\t',comments=None)

start=time.time()
with open(sys.argv[1]) as d :
    train_table=list(csv.reader(d,delimiter='\t'))
with open(sys.argv[2]) as d :
    valid_table=list(csv.reader(d,delimiter='\t')) 
with open(sys.argv[3]) as d :
    test_table=list(csv.reader(d,delimiter='\t'))     

with open(sys.argv[4]) as d :
    dict_data=list(csv.reader(d,delimiter='\t'))
epochs=int(sys.argv[8])
theta=np.zeros((len(dict_data)+1,1))

def dictionary(data):
    X={}
    
    feature=[]
    for i in range(0,len(data)):
        X={'1':int(data[i][0])}
        
        for c in range(1,len(data[i])):
           split_out=[] 
           
           split_out=data[i][c].split(':')
           
           X.update({int(split_out[0]):int(split_out[-1])})
           
        X.update({-1:1})
             
        feature.append(X)       
         
    return feature
       
train_dict=dictionary(train_table)
test_dict=dictionary(test_table)#Dictionary for test  

def sigmoid(theta_transpose_x):#Sigmoid working fine
    
    sig=np.exp(theta_transpose_x)/(1+np.exp(theta_transpose_x))
    
    return sig

def stochastic_gradient(data,theta):
    
    alpha=0.1
    for i in range(0,epochs):
        for j in data:
            diff,t=(gradient(j,theta))#Calculating the gradient for that example and getting back a vector of g
            
                
            for k, v in j.items():
                if k!='1':
                    theta[k]=theta[k]+(alpha*diff)
                            
    return theta,t       

def gradient(example,theta):#gradient for each example
    #print(example)
    theta_transpose_x=0
    
    for k, v in example.items():   
        if k == '1':
            
            y=v
        else:
           theta_transpose_x+=np.dot(v,theta[k])       
    sig=sigmoid(theta_transpose_x)
    
    diff=y-sig
    
    return diff,theta_transpose_x      

theta_train,t=stochastic_gradient(train_dict,theta)
#NOW FOR PREDICTION WE TAKE THIS THETA AND TRAVERSE THROUGH EVERY EXAMPLE
def predict(data):
    theta_transpose_x=0
    l=[]
    for example in data:
        g,theta_transpose_x=gradient(example,theta_train)
        if(sigmoid(theta_transpose_x)>0.5):
            l.append(1)
        else:
            l.append(0)
            
    return l   

     
predict_train=predict(train_dict) 
predict_test=predict(test_dict)
    
#erroroutput=[['error(train):',error_train],['error(test):',error_test]]         
error_train=0.000000000000000

for i in range(0,len(train_table)):
    
    if int(train_table[i][0])!=predict_train[i]:
        error_train+=1
error_train=error_train/len(train_table) 
       
error_test=0.0000000000000000
for i in range(0,len(test_table)):
    if int(test_table[i][0])!=predict_test[i]:
        error_test+=1
      
error_test=error_test/len(test_table) 
       
erroroutput=[['error(train):',error_train],['error(test):',error_test]] 

with open(sys.argv[5], 'w') as train_out:
    for listitem in predict_train:
        train_out.write('%s\n' % listitem)
with open(sys.argv[6], 'w') as test_out:
    for listitem in predict_test:
        test_out.write('%s\n' % listitem)
with open(sys.argv[7],"w") as metrics_out:
    csv.writer(metrics_out,delimiter=' ').writerows(erroroutput) 
    
stop=time.time()  
print(stop-start)     
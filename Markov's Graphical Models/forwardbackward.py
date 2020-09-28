#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:31:22 2020

@author: shayereesarkar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:19:41 2020

@author: shayereesarkar
"""

import numpy as np
import scipy as sp
import csv
import sys

error=0
J=0
total_words=0

b=np.loadtxt(sys.argv[5],dtype='float',delimiter=' ')    
a=np.loadtxt(sys.argv[6],dtype='float',delimiter=' ')
prior=np.loadtxt(sys.argv[4],dtype='float',delimiter=' ')
test_words=np.loadtxt(sys.argv[1],dtype='str',delimiter='\t')
ind_tag=np.loadtxt(sys.argv[3],dtype='str',delimiter='\t')    
ind_word=np.loadtxt(sys.argv[2],dtype='str',delimiter='\t')

if test_words.shape==():
    
    test_words=test_words.reshape(1,)
    
k=''
#Initial Value of the 1st state
alpha_initial=0
y_pred=[]

for i in test_words:#len(train_words)):
    
    index_word=np.zeros((1,1))
    index_tag=np.zeros((1,1))
    temp=i.split(" ")
    for i in temp:
        word,tag=i.split('_')#Splits wrt word and tag
        if np.any(ind_word==word):
                w=np.argwhere(ind_word==word)
                
                index_word=np.append(index_word,w[0][0])
          
        if np.any(ind_tag==tag):
                t=np.argwhere(ind_tag==tag)
                index_tag=np.append(index_tag,t[0][0])
                
    index_word=np.delete(index_word,0) 
    index_tag=np.delete(index_tag,0)
    
    alpha=np.zeros((b.shape[0],index_word.shape[0]))
    alpha_initial=np.multiply(prior,b[:,int(index_word[0])])#Initial Values for every example
    alpha[:,0]=alpha_initial
    
    #Forward propagartion without using log
    for i in range(1,index_word.shape[0]):
        if i==index_word.shape[0]:
            break
        
        else:
            alpha[:,i]=np.multiply(b[:,int(index_word[i])],np.matmul(np.transpose(a),alpha[:,i-1]))
            
    alpha_new=np.zeros((b.shape[0],index_word.shape[0]))
    alpha_initial=np.multiply(prior,b[:,int(index_word[0])])#Initial Values for every example
    alpha_new[:,0]=np.log(alpha_initial)        
    #Forward propagartion using log      
    alpha_new=np.log(alpha)
    
    beta=np.ones((b.shape[0],index_word.shape[0]))  
    i=index_word.shape[0]-1 
    
    beta_new=np.log(np.ones((b.shape[0],index_word.shape[0])))  
    i=index_word.shape[0]-1 
    
    #Backward propagation without log       
    while(i-1>=0):
        
        beta[:,i-1]=np.matmul(a,np.multiply(b[:,int(index_word[i])],beta[:,i]))  
        i=i-1
        
    #Backward propagation with log 
    i=index_word.shape[0]-1   
    beta_new=np.log(beta)
    
    y_pred=np.argmax((alpha_new+beta_new),axis=0)
    
    for i in range(0,len(y_pred)):
        
        k+=str(ind_word[int(index_word[i])])+'_'+str(ind_tag[y_pred[i]])+' '
        
        if np.equal(y_pred[i],index_tag[i])==False:
            error+=1
    k=k[:-1]       
    k+='\n'
    
    #Accuracy
        
    J+=np.log((np.sum(alpha[:,-1])))
    total_words+=len(index_word)
 
J=J/test_words.shape # Average log likelihood  
Accuracy=1-(error/total_words)

text_file = open(sys.argv[7], "w")
n = text_file.write(k)#Output for predicttest
text_file.close() 
 
metrics_out='Average Log-Likelihood: '+str(J[0])+'\n'+'Accuracy: '+str(Accuracy)+'\n'
     
text_file = open(sys.argv[8], "w")
n = text_file.write(metrics_out)#Output for predicttest
text_file.close() 

        
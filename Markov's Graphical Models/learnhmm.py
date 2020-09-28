#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:30:44 2020

@author: shayereesarkar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:12:11 2020

@author: shayereesarkar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:27:18 2020

@author: shayereesarkar
"""

import numpy as np
import scipy as sp
import csv
import sys

ind_tag=np.loadtxt(sys.argv[3],dtype='str',delimiter='\t')    
ind_word=np.loadtxt(sys.argv[2],dtype='str',delimiter='\t')
train_words=np.loadtxt(sys.argv[1],dtype='str',delimiter='\t')

l=np.zeros((len(train_words),1))
C=np.ones(len(ind_tag),)
a=np.ones((len(ind_tag),len(ind_tag)))
b=np.ones((len(ind_tag),len(ind_word)))

store=''

for i in train_words:#len(train_words)):
    
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
    C[int(index_tag[0])]+=1
    
    #Now we calculate the emission matrix
    for i in range(0,len(index_word)):
        
        b[int(index_tag[i]),int(index_word[i])]+=1
         #Now we calculate the transiition matrix
    for i in range(0,len(index_tag)):
        if (i+1)!=len(index_tag):
        
            a[int(index_tag[i]),int(index_tag[i+1])]+=1
    
    temp=[]#Clears the output of list at the end of every iteration

C=np.true_divide(C, sum(C)) #This is the prior matrix
b_sum=np.sum(b,axis=1)
for i in range(0,len(b_sum)):
    b[i,]=np.true_divide(b[i,],b_sum[i])
a_sum=np.sum(a,axis=1)
for i in range(0,len(a_sum)):
    a[i,]=np.true_divide(a[i,],a_sum[i])
    
np.savetxt(sys.argv[6],a)
np.savetxt(sys.argv[5],b)
np.savetxt(sys.argv[4],C)
   
        
                
    
        
    
    


        
    
    
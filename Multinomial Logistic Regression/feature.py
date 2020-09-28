#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:22:53 2020

@author: shayereesarkar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:00:09 2020

@author: shayereesarkar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:32:30 2020

@author: shayereesarkar
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
'''
with open('/Users/shayereesarkar/Desktop/10601/hw4/handout/largedata/train_data.tsv') as d :
    #train_data=list(csv.reader(d,delimiter='\t'))
with open('/Users/shayereesarkar/Desktop/10601/hw4/handout/smalldata/valid_data.tsv') as d :
    valid_data=list(csv.reader(d,delimiter='\t')) 
with open('/Users/shayereesarkar/Desktop/10601/hw4/handout/smalldata/test_data.tsv') as d :
    test_data=list(csv.reader(d,delimiter='\t'))     

with open('/Users/shayereesarkar/Desktop/10601/hw4/handout/smalldata/dict.txt') as d :
    c=list(csv.reader(d,delimiter='\t'))
'''


l=[]
with open(sys.argv[1], 'r') as reader:
    # Note: readlines doesn't trim the line endings
    train_data = reader.readlines()
with open(sys.argv[3], 'r') as reader:
    # Note: readlines doesn't trim the line endings
    test_data = reader.readlines()
with open(sys.argv[2], 'r') as reader:
    # Note: readlines doesn't trim the line endings
    valid_data = reader.readlines()    
with open(sys.argv[4], 'r') as reader:
    # Note: readlines doesn't trim the line endings
    dict_data = reader.readlines()    
    
model=int(sys.argv[8])
for i in dict_data:
    
    split_out_dict=i.split(' ')
    l.append(split_out_dict[:-1][0])
    

feature=[] 

def dictionary_model1(data):#Model 1
    feature=[] 
    for i in data:
        k=''
        split_first=[]
        split_first=i.split('\t')
        k=k+split_first[:-1][0]+'\t'
    #k={'1':train_data[i][0]}
        split_out=[] 
        split_out=split_first[1].split(' ')
        
        repeat=[]
        for c in split_out:
            if c in l:
                repeat.append(c)
                if repeat.count(c)<=1:
                    k=k+str(l.index(c))+':1'+'\t'
                
    #print(k)'
        #feature.append(data[i][0])
        k=k[:-1]
        k+='\n'
        feature.append(k)
    return feature   

       
        
def dictionary_model2(data):#Model 2
    
    feature=[]
    for i in data:
        k=''
        #print(i)
        split_first=[]
        split_first=i.split('\t')
        k=k+split_first[:-1][0]+'\t'
        
    #k={'1':train_data[i][0]}
        split_out=[] 
        split_out=split_first[1].split(' ')
        
        repeat=[]
        for c in split_out:
            if c in l:
                if(split_out.count(c)<4):
                    repeat.append(c)
                    if repeat.count(c)<=1:
                        k=k+str(l.index(c))+':1'+'\t'
                        
        k=k[:-1]
        k+='\n'            
        feature.append(k)
        
        
        
    return feature  
        
if model==1:
        
    train_out=dictionary_model1(train_data)#Here the first key has the label
    test_out=dictionary_model1(test_data)#Here the first key has the label
    valid_out=dictionary_model1(valid_data)#Here the first key has the label 
    
else:
    train_out=dictionary_model2(train_data)#Here the first key has the label
    test_out=dictionary_model2(test_data)#Here the first key has the label
    valid_out=dictionary_model2(valid_data)#Here the first key has the label 

with open(sys.argv[5], 'w') as formatted_train:
    for listitem in train_out:
        formatted_train.write('%s' % listitem)
with open(sys.argv[7], 'w') as formatted_test:
    for listitem in test_out:
        formatted_test.write('%s' % listitem)
with open(sys.argv[6], 'w') as formatted_valid:
    for listitem in valid_out:
        formatted_valid.write('%s' % listitem)        
                 
          
stop=time.time()  
print(stop-start)
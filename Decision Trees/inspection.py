import numpy as np
import scipy as sp
import csv
from scipy import stats
from scipy.stats import itemfreq 
import sys

train_table=np.loadtxt(sys.argv[1],dtype='str',skiprows=1,delimiter='\t')    

row_len=train_table.shape[0] 
column_len=train_table.shape[1]  

count=0
v1=[]
v2=[]

train_error=itemfreq(train_table[:,column_len-1])

def Gini_Gain():
    error_rate=0
    max_1=np.amax(train_error[:,-1].astype('int'),axis=0)
    #print(max_1)
    error_rate=(1-max_1/row_len)#For a particular attribute
           
    GI=1-(max_1/row_len)**2-(1-max_1/row_len)**2
    #print('gini_impurity: {}'.format(GI))

    #print('Gini Gain is:{}'.format(GG))
    #print('error:',error_rate)
    erroroutput=[['gini_impurity:',GI],['error:',error_rate]]
    return erroroutput

w=Gini_Gain()


with open(sys.argv[2], "w") as outputtrain:
  csv.writer(outputtrain,delimiter=' ').writerows(w)

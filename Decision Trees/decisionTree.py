# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:56:53 2020

@author: sonia
"""
import numpy as np
import scipy as sp
import csv
from scipy import stats
from scipy.stats import itemfreq
import sys


train_table=np.loadtxt(sys.argv[1],dtype='str',skiprows=0,delimiter='\t') 
train_new_table= np.loadtxt(sys.argv[1],dtype='str',skiprows=0,delimiter='\t')       
#train_attribute=(train_new_table[0,:])
stack_l=np.array(train_new_table[0,:])
stack_r=np.reshape(stack_l,(1,-1))
stack_l=np.reshape(stack_l,(1,-1))
#train_attribute=np.reshape(train_attribute,(1,-1))
row_len=train_table.shape[0]
column_len=train_table.shape[1]  
count_g=0

def Gini_Gain(split_index,p):
    GG=0
    GI=0
    GI_v1=0
    GI_v2=0
    v1=[]
    v2=[]
    count=1
    #print(p)
    #print(f'split_index:{split_index}')
    column_len=p.shape[1]
    row_len=p.shape[0]-1
    #print('split_index :',split_index)
    #print(n)
        #print('Total length :',row_len)
    train_error=itemfreq(p[1:,column_len-1])
   
    max_1=np.amax(train_error[:,1].astype('int'),axis=0)
    GI=1-(max_1/row_len)**2-(1-max_1/row_len)**2
    #print('Gini Impurity:',GI)
    train_label=itemfreq(p[1:,split_index])
    
    if train_label.shape[0]==1:GG=0
        
    elif train_error.shape[0]==1:GI=0
    
    elif(train_label.shape[0]>1):
        #print(train_label[0,0])
        for i in p[1:,split_index]:
            
            if i==train_label[0,0]:
                v1.append((str)(p[count,column_len-1]))
           
            else:
                v2.append((str)(p[count,column_len-1]))
           
            count+=1
            
        v1array=np.array(v1)
        v2array=np.array(v2)
    
        v1_freq=stats.itemfreq(v1array)
        v2_freq=stats.itemfreq(v2array)
        #print('v1_len:',len(v1))
        #print(v1)
        #print('v2_len:',len(v2))
        max_v1=np.amax(v1_freq[:,1].astype('int'),axis=0)#For a particular attribute
        #print('max_v1:',max_v1)
        #print('v1array:',v1array)
        #print('v2array',v2array)
        max_v2=np.amax(v2_freq[:,1].astype('int'),axis=0)#For a particular attribute
        #print('max_v2:',max_v2)
        GI_v1=(max_v1/len(v1))*(1-max_v1/len(v1))+(max_v1/len(v1))*(1-max_v1/len(v1))
        #print('GI_v1 ',GI_v1)
        GI_v2=(max_v2/len(v2))*(1-max_v2/len(v2))+(max_v2/len(v2))*(1-max_v2/len(v2))
        #print('GI_v2 ',GI_v2)
        
        Wt_1=GI_v1*(len(v1)/(row_len))
        Wt_2=GI_v2*(len(v2)/(row_len))
        
        #print('Gini Impurity',GI) 
        GG=GI-(Wt_1+Wt_2)
        #print('Gini Gain is:{}'.format(GG))
        
    return GG 
   
def data_set(count_g,f):
    
    #print('max_column',count_g)
    #print('Split Index :{}'.format(split_index))
    count_g=count_g
    train_label=itemfreq(f[1:,count_g])
    count=0
    posn1=[]
    posn2=[]
    #print('Inside data_set:',f)
    
    stack_new=np.array(f[0,:])
    #print('First row for attributes:',stack_new)
    stack_r=np.reshape(stack_new,(1,-1))
    stack_l=np.reshape(stack_new,(1,-1))
    
    for i in f[1:,count_g]:
        if i==train_label[0,0]:
            posn1.append(count+1)
        else:
            posn2.append(count+1)
        count+=1
    
    data_set_left=(f[posn1,:])
    
    data_set_right=(f[posn2,:])    
        
    attribute_set_l=np.vstack((stack_l, data_set_left))
    #print('Stack left :',attribute_set_l)
    attribute_set_r=np.vstack((stack_r, data_set_right))
    #if(depth<=2):
     #   print('Stack right :',attribute_set_r)
    #if(depth<=2):
     #   print('Stack left :',attribute_set_l)
    data_set_left=np.delete(attribute_set_l,count_g,1)
    data_set_right=np.delete(attribute_set_r,count_g,1)
    #if(depth<=2):
       # print('Stack after deletion right :',data_set_right)
    
    #print('Stack after deletion left:',data_set_left)
    
    return data_set_left,data_set_right,train_label,f[0,count_g]


def max_gini_gain(f) :
    column_len=f.shape[1]
    #print('Column in max:',column_len)
    max_column_g=Gini_Gain(0,f)
    max_column=0
    #print(f)
    i=1
    #print('in max im right',f)
    #print('max_column_g first',max_column_g)
    #print('Im in max')
    for i in range(0,column_len-1):
        #print('count_g :{}'.format(count_g))
      #  print('Inside here')
      
        if (max_column_g<Gini_Gain(i,f)):
            max_column=i
            max_column_g=Gini_Gain(i,f)
           # print('max_column in f,:',max_column)
        
         
    #print('max column:',max_column)
    #print('max_column gain',max_column_g)
    (left_data,right_data,train_label,parent_attribute)=data_set(max_column,f) 
    #print('Attribute in max:',parent_attribute)
    
    return max_column,left_data,right_data,train_label,parent_attribute

#(max_new_column,left_new_data,right_new_data)=max_gini_gain(max_column_g,count_g,max_column)
 
def decision_Stump(q):
    train_label=itemfreq(q[1:,-1])
    max_v1=np.argmax(train_label[:,1].astype('int'),axis=0)#For a particular attribute
    return train_label[max_v1,0]  
       
   
    #print(f'train_label:{train_label}')
    #max_v1=np.argmax(train_label[:,1].astype('int'),axis=0)#For a particular attribute
    #print(f'max_v1:{max_v1}')    
     
    return label 
        
   
    #print(f'train_label:{train_label}')
    #max_v1=np.argmax(train_label[:,1].astype('int'),axis=0)#For a particular attribute
    #print(f'max_v1:{max_v1}')    
     
    return label 

def tree_traversal_preorder(z):
    if z!=None:
            #print(z)
            print(" ",z.attribute)
            print('|  |')
            tree_traversal_preorder(z.left)
            tree_traversal_preorder(z.right)

   
class Table:
    def __init__(self,data,attribute):
        self.data =data
        self.left= None
        self.right = None
        self.attribute=attribute
        self.label=None
        self.right_val=None
        self.left_val=None
    #def assign_data_to_table(self,data):
        #self.data=data

    def linkright(self,newtable):
        self.right = newtable  

    def linkleft(self,newtable):
        self.left = newtable
       
    def __str__(self):#This is called whenever a print function to print object attribute invokes a str type
        return f'{self.data}'  
#copy.deepcopy  
global n   
n=Table(train_table,0)

depth=0
count_g=0
max_column=0
max_column_g=0    
max_depth=3
column_len_new=column_len
#print(n.data)

def create_Tree(n,max_column,column_len_new,depth):
    global max_column_g
    global max_depth
    global r
    #print('Dataset in create Tree',n)
    r=n.data.shape[0]
    if max_depth>=column_len-1:
        max_depth=column_len
    #print('In create Tree') 
    if column_len_new==1 or r==1 or depth>=max_depth or Gini_Gain(max_column,n.data)==0 :
        #print("Im here now bitches")
        n.label=decision_Stump(n.data)
        return None        
    else:
        depth+=1
        
        (max_new_column,left_table,right_table,train_label,parent_attribute)=max_gini_gain(n.data)
        #print(f'depth:{depth}')
        #Get max column to split on
        #print('max column',max_new_column)
        n.left_val=train_label[0,0]
        n.right_val=train_label[1,0]
        n.attribute=parent_attribute
        #print("Attribute in create_Tree:",n.attribute)
        #print('left-->',left_table)
        left_tree=Table(left_table,n.attribute)
        column_len_left_table=left_table.shape[1]
        #print('Left table column length:',column_len_left_table)
        n.linkleft(left_tree)
        create_Tree(left_tree,0,column_len_left_table,depth)
        
        
        #print("Attribute ta dakh at left:",n.attribute)
        #print('right-->',right_table)
        right_tree=Table(right_table,n.attribute)
        n.linkright(right_tree)
        column_len_right_table=right_table.shape[1]
        #print('Right table column length:',column_len_right_table)
        create_Tree(right_tree,0,column_len_right_table,depth)
        return n
        
w=create_Tree(n,max_column,column_len_new,depth)
#z=tree_traversal_preorder(create_Tree(n,max_column,column_len_new,depth))
d=n

#gini_imp_tot_tbl=np.insert(train_table,finaldatalen,0,axis=1)
#z=tree_traversal_preorder(create_Tree(n,max_column,column_len_new,depth))
d=n

#gini_imp_tot_tbl=np.insert(train_table,finaldatalen,0,axis=1)

train_table_comp=np.insert(train_table,column_len,0,axis=1)
column_len=train_table_comp.shape[1]
row_len=train_table_comp.shape[0]
r=1

def move_train(n,train_table_comp,r):
       
        if n==None:
            return
        elif ((np.where(train_table_comp[0,:]==n.attribute))!=None and n.left_val!=None and n.right_val!=None):
            posn2=np.where(train_table_comp[0,:]==n.attribute)
            #print('posn',posn2)
            #print('left_value:',n.left_val)
            #print('row:',r)
            if train_table_comp[r,posn2]==n.left_val:
                #print('Hey im in left')
                #print('attribute:',n.attribute) 
                #print(train_table_comp[r,posn2])
                move_train(n.left,train_table_comp,r)
               
            else:
                
                #print('Hey im in right')
                #print('attribute:',n.attribute)
                #print(train_table_comp[r,posn2])
                move_train(n.right,train_table_comp,r)
                
        else:
           #print('Im here at the leaf')
           train_table_comp[r,column_len-1]=n.label
           #print(n.label)
           
r=1         
while(r<row_len-1):
    move_train(n,train_table_comp,r)
    r+=1
 
new_table=move_train(n,train_table_comp,r)
c=0
train_table_comp[0,column_len-1]=train_table_comp[0,column_len-2]

#Train Error Calculation
error=0
for i in train_table_comp:
    if(train_table_comp[c,column_len-2]!=train_table_comp[c,column_len-1]):
        error+=1
    c+=1
error_train=error/(row_len-1) 

test_table=np.loadtxt(sys.argv[2],dtype='str',skiprows=0,delimiter='\t') 
column_len_new=test_table.shape[1]
test_table_comp=np.insert(test_table,column_len_new,0,axis=1)


column_len_test=test_table_comp.shape[1]
row_len_test=test_table_comp.shape[0]
test_table_comp[0,column_len_test-1]=test_table_comp[0,column_len_test-2]
r=1
c=0
def move_test(n,test_table_comp,r):
       
        if n==None:
            return
        elif ((np.where(test_table_comp[0,:]==n.attribute))!=None and n.left_val!=None and n.right_val!=None):
            posn2=np.where(test_table_comp[0,:]==n.attribute)
            
            if test_table_comp[r,posn2]==n.left_val:
                #print('Hey im in left')
                #print('attribute:',n.attribute) 
               # print(train_table_comp[r,posn2])
                move_test(n.left,test_table_comp,r)
               
            else:
                
              #  print('Hey im in right')
             #   print('attribute:',n.attribute)
            #    print(train_table_comp[r,posn2])
                move_test(n.right,test_table_comp,r)
                
        else:
           #print('Im here at the leaf')
           test_table_comp[r,column_len_test-1]=n.label
           #print(test_table_comp[r,column_len_test-1])
           #print(n.label)
           
r=1         
while(r<row_len_test-1):
    move_test(n,test_table_comp,r)
    r+=1
    
new_test_table=move_test(n,test_table_comp,r)    
#Train Error Calculation
error_test=0
c=0
for i in test_table_comp:
    if(test_table_comp[c,column_len_test-2]!=test_table_comp[c,column_len_test-1]):
        error_test+=1
    c+=1
error_test=error_test/(row_len_test-1) 
#print(error_test)

erroroutput=[['error(train):',error_train],['error(test):',error_test]]

def traverse(rootnode):
  thislevel = [rootnode]
  a ='                                    '
  while thislevel:
    nextlevel = list()
    c=(int)(len(a)/2)
    a = a[:c]
    for n in thislevel:
      print(a+str(n.attribute))
      if n.left: nextlevel.append(n.left)
      if n.right: nextlevel.append(n.right)
      
      thislevel = nextlevel
j=traverse(n)

with open(sys.argv[4], "w") as outputtrain:
   csv.writer(outputtrain,delimiter='\n').writerows([train_table_comp[1:,column_len-1]])

with open(sys.argv[5], "w") as outputtest:
   csv.writer(outputtest,delimiter="\n").writerows([test_table_comp[1:,column_len_test-1]])

with open(sys.argv[6], "w") as outputerror:
   csv.writer(outputerror, delimiter=' ').writerows(erroroutput)
   
   
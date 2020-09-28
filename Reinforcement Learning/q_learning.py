from environment import MountainCar
import sys
import numpy as np

m=MountainCar(mode=sys.argv[1])#State representation of the object

episodes=int(sys.argv[4])
max_iterations=int(sys.argv[5])
epsilon=float(sys.argv[6])
gamma=float(sys.argv[7])
learning_rate=float(sys.argv[8])
    
action_space=m.action_space        
action_taken=[]
weight_track=[]
reward_sum=[]
#intialize weight matrix with bias folded in

q=np.zeros((action_space,))# This is the initial q
w=np.zeros((action_space,m.state_space))
#print('q output shape :',q.shape)
# For bias term
mode=sys.argv[1] 
b=0
for i in range(0,episodes):
    gradient=np.zeros((m.action_space,m.state_space))#
    s=m.reset()
    s_new=np.zeros((m.state_space,1))
    
    
    if mode=='raw':
        s_dict=np.fromiter(s.values(), dtype=float)
        print(s_dict)
        print(np.fromiter(s.keys(), dtype=float))
        for k in range(0,m.state_space):
            s_new[k,]=s_dict[k]
            
            
    else:
        s_dict=np.fromiter(s.keys(), dtype=int)
        for k in s_dict:
            s_new[k,]=1
    
    sum=0
    
    for j in range(0,max_iterations):
        
        q=np.matmul(w,s_new)+b
        
        gradient=s_new
        
        if np.random.uniform(0,1)<epsilon and epsilon!=0:#Choice of no of outcomes
            action=np.random.choice([0,1,2])
            
            #print('Choose action if epsilon:',action)
        else:
            action=np.argmax(q)
            #print('Choose action if not epsilon:',action)
            
        state_dict,r,situation=m.step(action)#We choose the best action
        
        s_old=np.zeros(m.state_space,)
        
        if mode=='raw':
            s_dict=np.fromiter(state_dict.values(), dtype=float)
            #print()
            for k in range(0,m.state_space):
                s_old[k,]=s_dict[k]
            #print('s_old in if:',s_old)
        else:
            s_dict_new=np.fromiter(state_dict.keys(),dtype=int)
        
            for l in s_dict_new:
                s_old[l,]=1
        
        q_target=np.matmul(w,s_old)+b
        
        mul1=learning_rate*(q[action]-(r+gamma*(max(q_target))))
        
        w[action,:]=w[action,:]-mul1*np.transpose(gradient)#We update the weights
        b=b-mul1
        
        weight_track.append(w)
        action_taken.append(action)
        
        sum+=r
        s_new=s_old
        
        if situation==True:
            break
        #m.render()   
    m.reset()  

    reward_sum.append(sum)
        
final_weight=np.transpose(w) 
final_weight=np.insert(final_weight,0,b) 

np.savetxt(sys.argv[2],final_weight)
np.savetxt(sys.argv[3],reward_sum)
m.close()
 
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:54:26 2019

@author: anita
"""


import numpy as np
import pandas as pd

import random

from sklearn import preprocessing

from sklearn.preprocessing import normalize

from math import e

import math

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import poisson

from scipy.stats import multivariate_normal
import scipy.stats as stats

import scipy

plt.rcParams["figure.figsize"] = (10,8)


#####################################################################
## Problem 1 
#####################################################################
#print (" ************ Problem 1 *************** ")

score_data = np.genfromtxt('hw4_data/CFB2018_scores.csv', delimiter=',')

team_names = np.loadtxt('hw4_data/TeamNames.txt', dtype='str', delimiter= '\n')


#print "Shape of score_data:", score_data.shape
#print "Shape of team names:", team_names.shape


# number of teams = 767
# Initialize all M_hat to 767x767 all zeros 
n = 767 
M_hat = np.zeros((n,n))
#print "Shape of M_hat is :", M_hat.shape

for i in range(0,score_data.shape[0]):
    Team_A_index = int(score_data[i][0]) - 1
    Team_A_score = score_data[i][1]
    Team_B_index = int(score_data[i][2]) - 1
    Team_B_score = score_data[i][3]
    
    #print Team_A_index, Team_A_score, Team_B_index, Team_B_score
    
    if (Team_A_score > Team_B_score):
        M_hat[Team_A_index][Team_A_index] = M_hat[Team_A_index][Team_A_index] + 1 + ( Team_A_score / (Team_A_score + Team_B_score)) 
        M_hat[Team_A_index][Team_B_index] = M_hat[Team_A_index][Team_B_index] + 0 + ( Team_B_score / (Team_A_score + Team_B_score)) 
        M_hat[Team_B_index][Team_A_index] = M_hat[Team_B_index][Team_A_index] + 1 + (Team_A_score / (Team_A_score + Team_B_score))
        M_hat[Team_B_index][Team_B_index] = M_hat[Team_B_index][Team_B_index] + 0 + (Team_B_score / (Team_A_score + Team_B_score)) 
        
    if (Team_B_score > Team_A_score):
        M_hat[Team_A_index][Team_A_index] = M_hat[Team_A_index][Team_A_index] + 0 + (Team_A_score / (Team_A_score + Team_B_score))
        M_hat[Team_A_index][Team_B_index] = M_hat[Team_A_index][Team_B_index] + 1 + (Team_B_score / (Team_A_score + Team_B_score))
        M_hat[Team_B_index][Team_A_index] = M_hat[Team_B_index][Team_A_index] + 0 + (Team_A_score / (Team_A_score + Team_B_score))
        M_hat[Team_B_index][Team_B_index] = M_hat[Team_B_index][Team_B_index] + 1 + (Team_B_score / (Team_A_score + Team_B_score))
        
    # Corner case, not required for this homework problem     
    if (Team_A_score == Team_B_score): # no team wins 
        M_hat[Team_A_index][Team_A_index] = M_hat[Team_A_index][Team_A_index] + 0 + Team_A_score / (Team_A_score + Team_B_score)
        M_hat[Team_B_index][Team_B_index] = M_hat[Team_B_index][Team_B_index] + 0 + Team_B_score / (Team_A_score + Team_B_score)

    #print M_hat[Team_A_index][Team_A_index]
    #print M_hat[Team_A_index][Team_B_index]
    #print M_hat[Team_B_index][Team_B_index]
    
    #print M_hat[Team_B_index][Team_A_index]
    
# Normalize M_hat row wise and assign to M     
M = preprocessing.normalize(M_hat, norm='l1')   
## Alternate way to 
#M = M_hat/np.sum(M_hat, axis=1)




#================================================================
# Problem 1 a
#================================================================

uniform = float(1)/n
#print uniform
#w0 = np.random.uniform(size=n)
#w0 = np.full((767,), uniform)
w0 = np.full((1,767), uniform)


#--------------------------------------------------------------
# To calculate Wt when t = 10
#--------------------------------------------------------------
top_25_list = []
M_10 = np.linalg.matrix_power(M, 10)
wt_10 = np.dot(w0,M_10)

#print(wt_10.shape)

print("wt_10", wt_10)

top_teams = np.argsort(np.multiply(wt_10,-1))


#print " top 25 team index ", top_teams[0:25] 
print( " ----------------------------------------------")
print( " The top 25 Team Names for wt when t=10 are :")
print( " ----------------------------------------------")

for i in range(0,25):
    #print i+1 , ":" , team_names[top_teams[i]]
    top_25_list.append([ i+1, team_names[top_teams[0,i]], wt_10[0,top_teams[0,i]] ])

top_25_df = pd.DataFrame(top_25_list, columns=["Rank","Team Name", "wt t=10 value"])    
#print( top_25_df)
print(top_25_df.to_string(index=False))


#--------------------------------------------------------------
# To calculate Wt when t = 100
#--------------------------------------------------------------

top_25_list = []
M_100 = np.linalg.matrix_power(M, 100)
wt_100 = np.dot(w0,M_100)

top_teams = np.argsort(np.multiply(wt_100,-1))

#print " top 25 team index ", top_teams[0:25] 
print(" ----------------------------------------------")
print(" The top 25 Team Names for wt when t=100 are :")
print(" ----------------------------------------------")

for i in range(0,25):
    #print i+1 , ":" , team_names[top_teams[i]]
    top_25_list.append([ i+1, team_names[top_teams[0,i]], wt_100[0,top_teams[0,i]] ])

top_25_df = pd.DataFrame(top_25_list, columns=["Rank","Team Name", "wt t=100 value"])     
#print(top_25_df)   
print(top_25_df.to_string(index=False))



    
top_25_list = []
M_1000 = np.linalg.matrix_power(M, 1000)
wt_1000 = np.dot(w0,M_1000)

top_teams = np.argsort(np.multiply(wt_1000, -1))

#print " top 25 team index ", top_teams[0:25] 
print(" ----------------------------------------------")
print(" The top 25 Team Names for wt when t=1000 are :")
print(" ----------------------------------------------")

for i in range(0,25):
    #print i+1 , ":" , team_names[top_teams[i]]
    top_25_list.append([ i+1, team_names[top_teams[0,i]], wt_1000[0,top_teams[0,i]] ])

top_25_df = pd.DataFrame(top_25_list, columns=["Rank", "Team Name", "wt t=10 value"])  
#print(top_25_df)
print(top_25_df.to_string(index=False))

  

top_25_list = []
M_10000 = np.linalg.matrix_power(M, 10000)
wt_10000 = np.dot(w0,M_10000)

top_teams = np.argsort(np.multiply(wt_10000, -1))

#print " top 25 team index ", top_teams[0:25] 
print(" ----------------------------------------------")
print(" The top 25 Team Names for wt when t=10000 are :")
print(" ----------------------------------------------")

for i in range(0,25):
    #print i+1 , ":" , team_names[top_teams[i]]
    top_25_list.append([ i+1,team_names[top_teams[0,i]], wt_10000[0,top_teams[0,i]] ])

top_25_df = pd.DataFrame(top_25_list, columns=["Rank","Team Name", "wt t=10 value"])
#print(top_25_df)  
print(top_25_df.to_string(index=False))


#------------------------------------
# Problem 1 b 
#------------------------------------
M_t = np.transpose(M)

eigenValues, eigenVectors = np.linalg.eig(M_t)

#idx = eigenValues.argsort()[::-1]   
#eigenValues = eigenValues[idx]
#eigenVectors = eigenVectors[:,idx]

#print ("max argmax", eigenValues.argmax())

first_eigenVector = eigenVectors[:, eigenValues.argmax()]

#print ("first eigenVector :" , first_eigenVector)
#print (" Shape first eigenVector", first_eigenVector.shape)
real_ev = []
for i in range(0,first_eigenVector.shape[0]):
    real_ev.append(first_eigenVector[i].real)
        

real_ev = np.array(real_ev).reshape(1,-1)
#print ("Real EV  :" , real_ev)

sum_row_ev = np.sum(real_ev, axis =1 ) 
#print ("sum_row_ev", sum_row_ev)

wt_inf = real_ev/np.sum(real_ev, axis=1)
#print ("wt_inf", wt_inf)

#print("sum wt_inf", np.sum(wt_inf,axis=1))
#wt_inf = np.divide(real_ev, sum_row_ev)

#print( "shape of wt_inf:, ", wt_inf.shape)
#first_eigenVector = list(first_eigenVector)

idx_first = np.argsort(np.multiply(wt_inf,-1))
#idx_first = wt_inf.argsort()[::-1]

#print (" idx_first", idx_first)
print( " ----------------------------------------------")
print( " The top 25 Team Names for wt when t=inf are :")
print (" ----------------------------------------------")


top_25_list = []


#print("shape of idx_first", idx_first.shape)

for i in range(0,25):
    #print(idx_first[i])
    #print ( " team names: ", team_names[idx_first[0,i]])
    #print ("Weight Infinity", wt_inf[0,idx_first[0,i]])
    top_25_list.append([i+1, team_names[idx_first[0,i]],wt_inf[0,idx_first[0,i]]])
    #top_25_list.append([team_names[idx_first[0,i],wt_inf[0,idx_first[0,i]]] ])
    #top_25_list.append([ team_names[top_teams[i]], wt_inf[top_teams[i]] ])
 
#print( top_25_list)
top_25_df = pd.DataFrame(top_25_list, columns=["Rank","Team Name", "wt t=inf value"])
#top_25_df.reset_index()
#top_25_df.rename(index={range(1,26)}, inplace=True)
print(top_25_df.to_string(index=False))
print("  ")
#print (top_25_df )   

#==========================================================
#Plotting 
#==========================================================


l1_dist = []

wt_prev = w0
#wt_t = w0

#print( "shape of wt_t", wt_t.shape)
#print( "shape of wt_inf", wt_inf.shape)
#l1_dist.append(np.linalg.norm((wt_t[0:,] - wt_inf[0:,]), ord =1))



for i in range(1,10001):
    #print "Iteration:" ,i
    wt_t = np.dot(wt_prev,M)
    dist = np.linalg.norm((wt_t[0:,] - wt_inf[0:,]), axis=1, ord =1)
   
    l1_dist.append(dist)
    wt_prev = wt_t
    
  

#l1_dist = []
#
#for i in range(1,501):
#    print "Iteration ", i
#    M_i = np.linalg.matrix_power(M, i)
#    w_i = np.dot(w0,M_i)
#    dist = np.linalg.norm((w_i[0:,]- my_wt_inf[0:,]), ord =1)
#    l1_dist.append(dist)
#
#print l1_dist
#
#
# 
def plot_l1_dist(l1_dist):    
    #import matplotlib.pyplot as plt
    #idx  = np.arange(1, 10000, 999)
    plt.plot(l1_dist)
    #plt.xticks(idx)
    
    #plt.legend(loc='best')
    plt.ylabel('L1 dist')
    plt.xlabel('Iterations')
    #plt.xticks(range(len(l1_dist)))
    plt.title('L1 Dist || wt_i - wt_inf ||  for 10000 iterations')
    
    plt.show()
    
plot_l1_dist(l1_dist)


#####################################################################
## Problem 2
#####################################################################
print ( " ************ Problem 2 *************** ")

nyt_data = np.genfromtxt('hw4_data/nyt_data.txt', dtype = 'str', delimiter='\n')
print ("Shape of nyt_data : " , nyt_data.shape)

nyt_voc = np.loadtxt('hw4_data/nyt_vocab.dat', dtype='str')
print ("Shape of nyt_voc : ", nyt_voc.shape)


# No of Vocalbury N
N= 3012

# No of Documents N
M = 8447

# Rank K 
K = 25

X = np.zeros((N,M))
print ("Shape of X:", X.shape)

# Populating X matrix based on nyt_data
print (" Populating X Matrix based on NYT Data ")

for j in range(0, nyt_data.shape[0]):
    #print nyt_data[i]
    for w in nyt_data[j].split(','):
    #word_data = nyt_data[i].split(',')
        #print w
        data = w.split(':')
        #print data[0], data[1]
        word_index = int(data[0]) - 1 
        word_cnt = int(data[1])
        X[word_index,j] = word_cnt
        
W = np.random.uniform(low=1, high=2, size=N*K).reshape(N,K)
H = np.random.uniform(low=1, high=2, size=K*M).reshape(K,M)


def calculate_H(W,H):
    # row of transposed W should sum to 1
    W_t_norm = normalize(np.transpose(W), axis = 1 , norm = 'l1')
    WH = np.matmul(W,H)
    WH = np.add(WH, 10 ** (-16))
    #print "Shape of WH", WH.shape
    #WH_inverse = np.linalg.pinv(WH)
    #print "Shape of WH_inverse", WH_inverse.shape
    X_by_WH = np.divide(X, WH)
    #print "Shape of X by WH", X_by_WH.shape
    W_term = np.matmul(W_t_norm, X_by_WH)
    new_H = np.multiply(H, W_term)
    
    return new_H

#new_H = calculate_H(W,H)

def calculate_W(W,H):
    H_t_norm = normalize(np.transpose(H), axis = 0 , norm = 'l1')
    WH = np.matmul(W,H)
    WH = np.add(WH, 10 ** (-16))
    #print "Shape of WH", WH.shape
    #WH_inverse = np.linalg.pinv(WH)
    #print "Shape of WH_inverse", WH_inverse.shape
    X_by_WH = np.divide(X, WH)
    
    H_term = np.matmul(X_by_WH, H_t_norm)
    
    new_W = np.multiply(W,H_term)
    
    return new_W


def run_topic_modeling(new_W):
    print ("Topic Modeling with W matrix")
    W_col_norm = normalize(new_W, axis = 0 , norm = 'l1')
    word_group_list = []
    for k in range(0, W_col_norm.shape[1]) :
        W_vec = W_col_norm[:,k]
        #print "Shape of W_vec" , W_vec.shape
        top_words = np.argsort(np.multiply(W_vec, -1))
        #print top_words
        top_word_list = []
       
        for i in range(0,10):
            #print  "Top word index", top_words[i]
            #print nyt_voc[top_words[i]]
            #print W_vec[top_words[i]]
            
            top_word_list.append([nyt_voc[top_words[i]],W_vec[top_words[i]]])
        
        top_10_df = pd.DataFrame(top_word_list, columns=["Word", "Weightage"])
        print (top_10_df)
        #print(top_10_df.to_string(index=False))
        word_group_list.append(top_10_df)
    
    return word_group_list
        
        
        
    
    
    
#calculate_W(W,new_H)
obj_fn_list = []  


def run_nmf(W,H):
    
    this_W = W
    this_H = H
    
    for i in range(0,100):
        print ("NMF Iteration Number :", i)
        new_H = calculate_H(this_W,this_H)
        new_W = calculate_W(this_W,new_H)
        
        
        this_H = new_H
        this_W = new_W
        
        WH = np.matmul(this_W, this_H)
        ln_WH = np.log(WH)
        X_ln_WH = np.multiply(X,ln_WH)
        #X_ln_WH = np.matmul(X,ln_WH)
        obj_fn = np.nansum(np.subtract(WH,X_ln_WH))
        obj_fn_list.append(obj_fn)
        

    word_group_list = run_topic_modeling(new_W)
    return word_group_list    


word_group_list = run_nmf(W,H)

print (obj_fn_list)
        

def plot_obj_fn(obj_fn_list):    
    #import matplotlib.pyplot as plt
    plt.plot(obj_fn_list)
    
    plt.legend(loc='best')
    plt.ylabel('Divergence Penality')
    plt.xlabel('Iterations')
    #plt.xticks(range(len(l1_dist)))
    plt.title('Divergence Penality for 100 iterations')
    
    plt.show()
    
plot_obj_fn(obj_fn_list)

new_group_list =[]
for i in range(0, 5) : #len(word_group_list)):
    sub_list = []
    print (i)
    for j in range(0,5):
        print ( j)
        print ( (i*5)+ j)
        sub_list.append([word_group_list[(i*5)+j]])
    new_group_list.append(sub_list)

print (new_group_list)

fig = plt.figure()
#fig.subplots_adjust(left=0.2,top=0.8, wspace=1)

#Table - Main table
ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)
ax.table(cellText=new_group_list) 
          #rowLabels=rows,
          #colLabels=columns, loc="upper center")

ax.axis("off")

#Gold Scatter - Small scatter to the right
#plt.subplot2grid((4,3), (0,2))
#plt.scatter(scatter_x, scatter_y)
#plt.ylabel('Gold Last')

fig.set_size_inches(w=25, h=90)
plt.show()
    
    



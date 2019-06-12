# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:55:44 2019

@author: Anita
"""

import numpy as np
import pandas as pd

import random

from sklearn import preprocessing
from math import e

import math


import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import poisson

from scipy.stats import multivariate_normal
import scipy.stats as stats

plt.rcParams["figure.figsize"] = (10,8)

#####################################################################
## Problem 1 
#####################################################################

####################################################################
#  ************ Generate Data ***************
# 3 Gaussians 
# Clusters C1, C2, C3 
# Mixing weights pi = [0.2,0.5,0.3]
####################################################################

mean_c1 = [0,0]
cov_c1 = [[1,0],[0,1]]


mean_c2 = [3,0]
cov_c2 = [[1,0],[0,1]]


mean_c3 = [0,3]
cov_c3 = [[1,0],[0,1]]

x_data = []

#============================================================================
# This step is generating Cluster based on Cluster Prior probability 
# based on which clutser, we need to generate x, y as corresponding gaussain
#============================================================================

for i in range(1,501) :
    cluster = np.random.choice(np.arange(1,4), p=[0.2, 0.5, 0.3])
    #print cluster
    
    if (cluster == 1) :
        mean = mean_c1
        cov = cov_c1 
    if (cluster == 2) :
        mean = mean_c2
        cov = cov_c2
    if (cluster == 3) :
        mean = mean_c3
        cov = cov_c3
    
    x1, x2 = np.random.multivariate_normal(mean, cov).T
     
    #print x1, x2
    x_data.append([x1,x2, cluster])

x_data = np.array(x_data)
#print "shape of x data " , x_data.shape  

##########################################################################
# K means algorithm 
##########################################################################

#=========================================================================
# Calculate cluster for each data point based on eucledian distance from mu
#=========================================================================

def cal_cluster(x_data, mu,k): 
    clust_x = []
    objective_fn = 0 
    for i in range(0,500) :
        dist_arr = []
        
        for j in range(0, k):
            dist = ( (x_data[i][0] - mu[j][0]) **2 ) +  ( (x_data[i][1] - mu[j][1]) **2 )
            #print "distance" , dist
            dist_arr.append(dist)
            
        #print "dist_arr", dist_arr   
        min_dist = dist_arr.index(min(dist_arr))
        objective_fn = objective_fn + min(dist_arr)
        #print "min_dist" , min_dist
        cluster = min_dist + 1
        clust_x.append(cluster)
    #print clust_x
    return clust_x, objective_fn

#=========================================================================
# Calculate mu for each cluster using new cluster assignment  
#=========================================================================
def cal_mu_k(x_data, clus_x, k) :
    
    new_mu_val = np.zeros((k,2))
    sum_mu = np.zeros((k,2))
    cnt_mu = np.zeros(k)
    #print "shape new_mu_val", new_mu_val.shape
    
        
    for i in range(0,500) : 
        
        for j in range(0,k):
            
            if (clus_x[i] == j+1):
                
                sum_mu[j][0] = sum_mu[j][0] + x_data[i][0]
                sum_mu[j][1] = sum_mu[j][1] + x_data[i][1]
                cnt_mu[j] = cnt_mu[j] + 1
                
           
    #print cnt_mu_3

    for i in range(0,k) :
        new_mu_val[i][0] = sum_mu[i][0] / cnt_mu[i]
        new_mu_val[i][1] = sum_mu[i][1] / cnt_mu[i]
  

    #mu_val[2][0] = sum_mu_3[0]/cnt_mu_3
    #mu_val[2][1] = sum_mu_3[1]/cnt_mu_3
    
    #print "new_mu_val" , new_mu_val
    return new_mu_val

#=========================================================================
# Plot Objective fn for each of the 20 iterations  
#=========================================================================
 
def plot_k_mean(objective_fn, k):    
    #import matplotlib.pyplot as plt
    plt.plot(objective_fn)
    
    plt.legend(loc='best')
    plt.ylabel('Objective  Function')
    plt.xlabel('Iterations')
    plt.xticks(range(len(objective_fn)))
    plt.title('K Means Objective Function k=%i for 20 iterations' %k)
    
    plt.show()    
#k = 3

#=========================================================================
# Plot Data points color coded by cluster number 
#=========================================================================
def plot_data(x_data, clust_x,k):
   
    cdict = {1: 'red', 2: 'blue', 3: 'green', 4:'purple', 5:'orange'}

    fig, ax = plt.subplots()
    for g in np.unique(clust_x):
        ix = np.where(clust_x == g)
        ax.scatter(x_data[:,0][ix], x_data[:,1][ix], c = cdict[g], label =  g)
    ax.legend()
    plt.title("Data points color coded by each cluster k=%i" %k)
    plt.show()
        
    #colors = ['red','orange','blue','purple','green']
    
    #plt.scatter(x_data[:,0], x_data[:,1], c=clust_x, cmap=mpl.colors.ListedColormap(colors), label=colors)
    #plt.legend(loc='best')
    #plt.show()

#=========================================================================
# Main Function for calling k 2 to 5 , and run for 20 iterations
#========================================================================= 
print " --------------------------------------------"
print " ######### Problem 1 #######################"
print " --------------------------------------------"
def run_problem_1(x_data) :
    
    for k in range(2,6): 
        # Centoid for each cluster , first step
        mu = []
        # random mu
        for i in range(0,k):
            index = random.randint(0,500)
            #print index
            #print x_data[index][0], x_data[index][1]
            mu.append([x_data[index][0], x_data[index][1]])
        
        #print mu
        obj_fn_arr = []    
        
        clust_x, obj_fn = cal_cluster(x_data, mu,k)
        obj_fn_arr.append(obj_fn)
        
        for i in range (1,20) : 
            new_mu = cal_mu_k(x_data,clust_x, k)
            clust_x, obj_fn = cal_cluster(x_data, new_mu,k)
            obj_fn_arr.append(obj_fn)
        
        
        #print obj_fn_arr
       
        plot_k_mean(obj_fn_arr,k)
        
        if ( k==3 or k==5):
            plot_data(x_data,np.array(clust_x),k)
            


run_problem_1(x_data) 


       
        
#####################################################################
## Problem 2
# Train - 4140 , Test 460 
# X has 10 parameters
#####################################################################

# Data reading from csv files          
x_test = np.genfromtxt('hw3-data/Prob2_Xtest.csv', delimiter=',')
x_test_shape = x_test.shape
#print " Shape of X Test " , x_test_shape

## y = 1 -> Spam , else 0 
y_test = np.genfromtxt('hw3-data/Prob2_ytest.csv')
y_test_shape = y_test.shape
#print " Shape of Y Test " , y_test_shape

x_train = np.genfromtxt('hw3-data/Prob2_Xtrain.csv', delimiter=',')
x_train_shape = x_train.shape
#print " Shape of X Train " , x_train_shape

## y = 1 -> Spam , else 0 
y_train = np.genfromtxt('hw3-data/Prob2_ytrain.csv')
y_train_shape = y_train.shape
#print " Shape of Y Train " , y_train_shape

train_data = np.hstack(( x_train, y_train[:,None],))
test_data = np.hstack(( x_test, y_test[:,None],))
#print "Shape of train data", train_data.shape
#print "Shape of test data", test_data.shape

#==============================================================================
# Do some pandas for easy data transformation code
#==============================================================================
df = pd.DataFrame(train_data)
#print(df.head())

class_0 = df[df[10] == 0]
class_1 = df[df[10] == 1]

class_0 = np.array(class_0)
class_1 = np.array(class_1)

class_0_shape = class_0.shape
#print "Class 0 Shape", class_0_shape 

class_1_shape = class_1.shape
#print "Class 1 Shape", class_1_shape

# Refer -> https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
# column wise , axis =0 
mean = np.mean(train_data[:,0:10],axis=0)
#print "mean of training data:" , mean 

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
sigma =  np.cov(train_data[:,0:10], rowvar = False)
#print "covariance of training data:", sigma



#==================================================================================
# Get Initial Class Parameters based on K , mean and sigma are global and shared
# returns array of K list of sigma and mean parameters, also intial pi values 
# pi values is equal weight i.e 1/k
#==================================================================================
def get_initial_class_parameters(k, mean, sigma):
    
    mu_k = []
    sigma_k = []
    for i in range(0,k):
        x_random_guass = np.random.multivariate_normal(mean, sigma, 1000) 
        mu = mean = np.mean(x_random_guass,axis=0)
        mu_k.append(mu)
        sigma_k.append(sigma)
    
    pi_k = np.empty(k)
    pi_k.fill(float(1)/k)
    
    return mu_k, sigma_k, pi_k



#print "mu_k :",  mu_k
#print "sigma_k : ", sigma_k
#print "pi_k :", pi_k
    
#===================================================================================
# E Step of EM algorithm 
# Refernce : https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
# 
#===================================================================================
 
 
def e_step(x_data, mu_k, sig_k, pi_k,k): 
    phi_k = []
    y_list = []
 
    for i in range(0,k):
        y = multivariate_normal.pdf(x_data[:,0:10], mu_k[i], sig_k[i], allow_singular = True)
        y_list.append(y)
        
    y_list = np.array(y_list)
    #print "Shape of Y:" , y_list.shape
    
    for i in range(0,x_data.shape[0]): 
        phi_sum = 0 
        phi_x_list = []
        
        for j in range(0,k):
            phi_x = pi_k[j] * y_list[j][i]
            phi_sum = phi_sum + phi_x
            phi_x_list.append(phi_x)
        
        phi_x_i = phi_x_list/phi_sum
        
        #print phi_x_i
        phi_k.append(phi_x_i)
        
        #print phi_x_list
        #print phi_x_list/phi_sum
    phi_k = np.array(phi_k)
    #print "phi_k:" , phi_k
    #print y_list
    
    ## Caculate Objective Fn
    ob_fn = 0
    for i in range(0,x_data.shape[0]):
        sum_term = 0
        for j in range(0,k):
            prod_term = pi_k[j]*y_list[j][i]
            sum_term = sum_term + prod_term 
        log_sum = math.log(sum_term)
        ob_fn = ob_fn + log_sum   
    
    #print "objective fn: " , ob_fn
        
    
    
    return phi_k, ob_fn
        
        
        


def m_step(x_data,phi_k,k):
    pi_k = []
    mu_k = []
    sig_k = []
    n = x_data.shape[0]   
    nk = np.sum(phi_k,axis=0)
    pi_k = nk/n
    
    #print "nk:", nk
    #print "pi_k:", pi_k

    for i in range(0,k): 
        sum_phi_xi = np.zeros(10)
        for j in range(0,x_data.shape[0]):
            phi_xi = phi_k[j][i]*x_data[j,0:10]
            #print "phi_xi:", phi_xi
            sum_phi_xi = sum_phi_xi + phi_xi
            #print "sum_phi_x", sum_phi_xi
        mu_x = np.true_divide(sum_phi_xi, nk[i])
        #mu_x = sum_phi_xi/nk
        #print "mu_x" ,mu_x
        mu_k.append(mu_x)
    #print "mu_k:", mu_k
    
    for i in range(0,k):
        sum_term = np.zeros((10,10))
        
        for j in range(0,x_data.shape[0]):   #*** change later
            xi = x_data[j,0:10]
            xi_u = xi - mu_k[i]
            
            xi_u = xi_u.reshape((1,10))  # need to change 10 to shape[1]
            xi_ut= xi_u.reshape((-1, 1))
            #print "xi_u:", xi_u
            #xi_u_t = np.transpose(xi_u)
            #print "xi_ut:", xi_ut
            xi_dot = np.dot(xi_ut, xi_u)
            #print "size of xi_u", xi_u.shape
            #print "size of xi_u_t", xi_u_t.shape
    
            #print "xi_dot", xi_dot
            
            phi_xi_dot = phi_k[j][i] * xi_dot
            sum_term = sum_term + phi_xi_dot
        
        sig_term = np.true_divide(sum_term, nk[i])
        sig_k.append(sig_term)
        
    #print np.array(sig_k).shape
    
    return pi_k, mu_k, sig_k    
        


   

def run_em(class_data, mu_k, sigma_k, pi_k,k):
    
    local_mu_k = mu_k
    local_sigma_k = sigma_k
    local_pi_k = pi_k
    
    obj_fn_list = []
    #mu_list = []
    #sigma_list = []
    #pi_list = []
    
    for i in range(0,30):    
            
        phi_k, obj_fn = e_step(class_data, local_mu_k, local_sigma_k, local_pi_k,k) 
        obj_fn_list.append(obj_fn)
        #mu_list.append(local_mu_k)
        #sigma_list.append(local_sigma_k)
        #pi_list.append(local_pi_k)
        
        #print "local_pi_k" , local_pi_k
        local_pi_k, local_mu_k, local_sigma_k =  m_step(class_data,phi_k,k)
        
     
    return obj_fn_list, local_pi_k, local_mu_k, local_sigma_k

def run_ten_em(class_n,k):
    
    all_obj_fn = []
    all_mu = []
    all_sigma = []
    all_pi =[]
    
    for i in range(0,10): 
        
        #print "Iteration Number: ", i
        
        global_mu_k, global_sigma_k, global_pi_k = get_initial_class_parameters(k, mean, sigma) 
        obj_fn_list,pi_run, mu_run, sigma_run  = run_em(class_n,global_mu_k, global_sigma_k, global_pi_k,k)
        
        all_obj_fn.append(obj_fn_list)
        all_mu.append(mu_run)
        all_sigma.append(sigma_run)
        all_pi.append(pi_run)
    
    #print "all_obj_fn", all_obj_fn
    my_max_ob = []
    for ob in all_obj_fn:
        ob_list = ob 
        #print ob
        max_ob_index = ob.index(max(ob))
        #print "max ob index:", max_ob_index
        my_max_ob.append(ob_list[max_ob_index])
        
    #print my_max_ob    
    best_ob_run = my_max_ob.index(max(my_max_ob))
    best_ob = my_max_ob[best_ob_run]
    #print "best_ob" , best_ob
    #print "best_ob_run", best_ob_run
    #print "best mu", all_mu[best_ob_run]
    #print "best_pi:", all_pi[best_ob_run]
    #print "best_sigma:", all_sigma[best_ob_run]
    
    return all_pi[best_ob_run], all_mu[best_ob_run], all_sigma[best_ob_run], all_obj_fn

#=============================================================
# Plot Objective Fn
#=============================================================
def plot_objective_fn(obj_fn_list, class_n):
    obj_fn_arr = np.array(obj_fn_list)
    
    idx  = np.arange(5, 30, 1)
    colors = ["red","blue","olive","purple","brown","green","magenta","orange", "teal","pink"]
    plt.xlabel("Iterations")
    plt.ylabel("Log Marginal Objective Fn ")
    #plt.xticks(idx)
    #plt.xticks(np.arange(0,100, step=10))
    #plt.title('K Means Objective Function k=%i for 20 iterations' %k)
    plt.title(" Log Marginal objectuve Fn for iterations 2 to 100 for 10 runs for Class-%i " %class_n)
    for i in range(0,obj_fn_arr.shape[0]) : 
        l = i+1
        plt.plot(idx, obj_fn_arr[i,np.array(idx-1)],  color = colors[i], label = "Run %i" %l)   
   
    plt.legend(loc='best') 
    
    plt.show() 
#================================
    

def print_confusion_matrix(conf_mat, k_clust):
    
    p1 = conf_mat[0,0]
    p2 = conf_mat[0,1]
    p3 = conf_mat[1,0]
    p4 = conf_mat[1,1]
    
    print " "
    print " "
    print "   -------------------------------------------------------------"
    print "  Confusion Matrix for  Bayes Classifier ", k_clust , "GMM Model"
    print "   -------------------------------------------------------------"
    print "                               "
    print "             Predicted Y       "
    print " " 
    print "             0            1  "
    print "        --------------------------  "
    print " T     |            |            | "
    print " R   0 |    ",p1,"   |    ",p2,"    | "
    print " U     |            |            | "
    print " E     |-------------------------|   "
    print "       |            |            | "
    print " Y   1 |    ",p3,"     |    ",p4,"   | "
    print "       |            |            | "
    print "        -------------------------  "
    
    accuracy = (conf_mat[0,0] + conf_mat[1,1]) / float(np.sum(conf_mat)) 
        
    print "Accuracy of Prediction is " , accuracy*100, "%"
    #print "Accuracy of Prediction is " , accuracy
    

#========================================================
# bayes Classifer
#========================================================



    
def run_bayes(pi_0, mu_0, sig_0, pi_1, mu_1, sig_1,test_data, k):
    
    y_list_0 = []
    y_list_1 = []
    
    for i in range(0,k):
        y_0 = multivariate_normal.pdf(test_data[:,0:10], mu_0[i], sig_0[i], allow_singular = True )
        y_list_0.append(y_0)
        
        y_1 = multivariate_normal.pdf(test_data[:,0:10], mu_1[i], sig_1[i], allow_singular = True)
        y_list_1.append(y_1)
    
    pred_y = []
    mat_conf = np.matrix([[0,0], [0,0]])
    
    for i in range(0,test_data.shape[0]):
        sum_y0 = 0
        sum_y1 = 0
        
        for j in range(0,k):
            val_0 = y_list_0[j][i]*pi_0[j]
            sum_y0 = sum_y0 + val_0
            
            val_1 = y_list_1[j][i]*pi_1[j]
            sum_y1 = sum_y1 + val_1
            
        #sum_y0_log = math.log(sum_y0)
        #sum_y1_log = math.log(sum_y1)    
       
        
        
        if (sum_y1-sum_y0) > 0 : 
            y_new = 1 
        else :
            y_new = 0

        pred_y.append(y_new)
        
        if (test_data[i,10] == 0) and (y_new == 0):
            mat_conf[0,0] = mat_conf[0,0] + 1
        if (test_data[i,10] == 0) and (y_new == 1):   
            mat_conf[0,1] = mat_conf[0,1] + 1
        if (test_data[i,10] == 1) and (y_new == 0): 
            mat_conf[1,0] = mat_conf[1,0] + 1
        if (test_data[i,10] == 1) and (y_new == 1): 
            mat_conf[1,1] = mat_conf[1,1] + 1 


    print "mat_conf", mat_conf
    print_confusion_matrix(mat_conf,k)

#################################################################
# Problem 2a
# Find inidvidual class paramters and plot marginal objective fn
#################################################################
print " --------------------------------------------"
print " ######### Problem 2a #######################"
print " --------------------------------------------"
k_clusters = 3
class_0_pi, class_0_mu, class_0_sigma, all_obj_fn_0 = run_ten_em(class_0,k_clusters)    
class_1_pi, class_1_mu, class_1_sigma, all_obj_fn_1 = run_ten_em(class_1,k_clusters) 

plot_objective_fn(all_obj_fn_0, 0)
plot_objective_fn(all_obj_fn_1, 1)

###############################################################
# Problem 2b 
###############################################################
print " --------------------------------------------"
print " ######### Problem 2a #######################"
print " --------------------------------------------"

# GMM for k3
print " ---------- 3 Gaussian Mixture Model for Bayes Classifier -------"
run_bayes(class_0_pi, class_0_mu, class_0_sigma, class_1_pi, class_1_mu, class_1_sigma, test_data, k_clusters)


########GMM 1 ############
print " ---------- 1 Gaussian Mixture Model for Bayes Classifier -------"
k_clusters = 1
class_0_pi, class_0_mu, class_0_sigma, all_obj_fn_0 = run_ten_em(class_0,k_clusters)    
class_1_pi, class_1_mu, class_1_sigma, all_obj_fn_1 = run_ten_em(class_1,k_clusters) 
run_bayes(class_0_pi, class_0_mu, class_0_sigma, class_1_pi, class_1_mu, class_1_sigma, test_data, k_clusters)

########## GMM 2 ##############
print " ---------- 2 Gaussian Mixture Model for Bayes Classifier -------"
k_clusters = 2
class_0_pi, class_0_mu, class_0_sigma, all_obj_fn_0 = run_ten_em(class_0,k_clusters)    
class_1_pi, class_1_mu, class_1_sigma, all_obj_fn_1 = run_ten_em(class_1,k_clusters) 

run_bayes(class_0_pi, class_0_mu, class_0_sigma, class_1_pi, class_1_mu, class_1_sigma, test_data, k_clusters)

########GMM 4 ############
print " ---------- 4 Gaussian Mixture Model for Bayes Classifier -------"
k_clusters = 4
class_0_pi, class_0_mu, class_0_sigma, all_obj_fn_0 = run_ten_em(class_0,k_clusters)    
class_1_pi, class_1_mu, class_1_sigma, all_obj_fn_1 = run_ten_em(class_1,k_clusters) 

run_bayes(class_0_pi, class_0_mu, class_0_sigma, class_1_pi, class_1_mu, class_1_sigma, test_data, k_clusters)

 
#####################################################################
## Problem 3
#####################################################################
print " --------------------------------------------"
print " ######### Problem 3 #######################"
print " --------------------------------------------"


# Data reading from csv files          
ratings = np.genfromtxt('hw3-data/Prob3_ratings.csv', delimiter=',')
print "Shape of ratings : " , ratings.shape

test_data = np.genfromtxt('hw3-data/Prob3_ratings_test.csv', delimiter=',')
print "Shape of ratings test data : " , test_data.shape


f = open('hw3-data/Prob3_movies.txt', 'r')
x = f.read().splitlines()

f.close()

movies_list = []

for i in x:
    #print i
    movies_list.append(i)
  
movies = np.array(movies_list)

print "Movies Number :", len(movies)
#print "Shape of movies data : " , movies.shape


#==============================================================================
# Do some pandas for easy data transformation code
#==============================================================================
ratings_df = pd.DataFrame(ratings, columns=['user_id','movie_id','rating'])
movies_df = pd.DataFrame(movies, columns=['movie_name'])

#print ratings_df
#movie_id = movies_df.index

#print movie_id



#ratings_df = ratings_df.rename(columns={'o': 'newName1', 'oldName2': 'newName2'})
movies_df['movie_id'] = movies_df.index
ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
ratings_df['user_id'] = ratings_df['user_id'].astype(int)

#print movies_df.head()

#print ratings_df.head()
#--------------------------------------------------------------------------------
# convert ratings_df to a matrix of form Mij 
#--------------------------------------------------------------------------------
#my_Mij_df = ratings_df.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
#ratings_df['movie_id'] = ratings_df['movie_id'].astype('category')
cols = np.arange(1,1683,1)
#my_Mij_df = ratings_df.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
my_Mij_df = ratings_df.pivot(index = 'user_id', columns ='movie_id', values = 'rating').reindex(cols, axis=1)

#my_Mij_df = ratings_df.pivot('user_id', 'movie_id', 'rating')
print "Shape of my_Mij:", my_Mij_df.shape
#df['year']=df['year'].astype(int)

#--------------------------------------------------------------------------------
# Using some operations to map movie names to ratings table 
#--------------------------------------------------------------------------------
all_df = pd.merge(ratings_df,movies_df, on='movie_id')
#all_df = pd.merge(ratings_df,movies_df, on='movie_id', how='right')
print " all_df head"
#print  all_df.tail(100)


all_df_rows = all_df.shape
print " Total rows in all_df:" , all_df_rows

#--------------------------------------------------------------------------------
# Calculate n1 and n2
#--------------------------------------------------------------------------------
#n1 = all_df.user_id.nunique()
n1 = my_Mij_df.shape[0]
print " No of Users n1:" , n1

# Not using this value for n2, as we have 1682 movies in all
# 1675 movies are rated , but total movies is 1682
#n2 = all_df.movie_id.nunique()
n2 = len(movies)
#n2 = my_Mij_df.shape[1]
print " No of Movies n2:" , n2

#Mij_df = all_df.iloc[:,0:3]
#print "Mij : ", Mij_df.head()
#--------------------------------------------------------------------------------
# Convert to Numpy Array
#--------------------------------------------------------------------------------
Mij_arr = np.array(my_Mij_df)
print " Shape of Mij_arr", Mij_arr.shape


#sum_Mij = np.sum(Mij_arr[:,2])
#sum_Mij = np.sum(Mij_arr)
#sum_Mij = np.nansum(Mij_arr)
sum_Mij = ratings_df['rating'].sum(axis=0)
print "sum_Mij:", sum_Mij

#==============================================================================
# Given paramters
#==============================================================================
d = 10
lamda = 1
sigma_sq = 0.25

lamda_inv = 1/lamda

I_matrix = np.identity(d)
#print "I_matrix : ", I_matrix

Lamda_inv_I = lamda_inv * I_matrix
#print "Lamnda_inv_I :", Lamda_inv_I

#==============================================================================
# Calculate mean and Sigma to generate vj and ui, i.e u_mat, v_mat
#==============================================================================
mean = np.zeros(d)
#print "mean: ", mean

sigma = Lamda_inv_I

#v_mat = np.random.multivariate_normal(mean, sigma, n2).T 
#print "Shape of V matrix", v_mat.shape

#u_mat = np.random.multivariate_normal(mean, sigma, n1)
#print "Shape of U matrix", u_mat.shape


#v_random_normal = np.random.multivariate_normal(mean, sigma, 1000) 
# =============================================================================
# movies_id = []
# for row in movies_df.index: 
#     movies_id.append(row)
# 
# movies_id_df = pd.DataFrame(movies_id)
#     
# movies_df <- pd.merge(movies_id_df,movies_df)
# 
# =============================================================================
#movies_df.index.name = 'movie_id'
#movies_df.reset_index(inplace=True)
#movie_id = pd.rownames(movies_df)
#movies_df <- pd.cbind(movie_d,movies_df)

#print movies_df[1]

#class_0 = df[df[10] == 0]
#class_1 = df[df[10] == 1]

#u_mat = np.full((n1,10),np.nan)



lamda_term = lamda * sigma_sq * I_matrix
#print lamda_term

def calculate_ui(u_mat,v_mat):
    my_u_mat = u_mat
    
    # Calculate all v_vt dot products
    list_vj_vj_t = []
    
    for i in range(0,n2):
        vj = v_mat[:,i].reshape((-1, 1))
        vj_t = vj.reshape((1,-1))
     
        vj_vj_t = np.dot(vj, vj_t)
        list_vj_vj_t.append(vj_vj_t)
    
    #print "calculated vj_vj"
    
    for i in range(0, u_mat.shape[0]):
        
        u_idx = np.where(~np.isnan(Mij_arr[i, :]))
        u_idx = list(u_idx[0])
        
        
        sum_Mij_vj = np.zeros(10)
        sum_vj_vj_t = np.full((10,10),0)
        
        for j in u_idx:
            #print u_idx
            #print list_vj_vj_t[i]
            sum_vj_vj_t =np.add(sum_vj_vj_t, list_vj_vj_t[j])
            sum_Mij_vj = np.add(sum_Mij_vj , Mij_arr[i,j] * v_mat[:,j])
            
        #print "finished sum "
        
        ui = np.dot( np.linalg.inv(np.add(lamda_term,sum_vj_vj_t)), sum_Mij_vj)
        
        # update u_mat
        ui_t = ui.reshape((1,-1))
        my_u_mat[i] = ui_t
        
    return my_u_mat
    
    #print " u Step done"


def calculate_vj(u_mat,v_mat):
    
    my_v_mat = v_mat
    
    # Calculate all v_vt dot products
    list_ui_ui_t = []
    
    for i in range(0,n1):
        ui = u_mat[i,:].reshape((-1, 1))
        ui_t = ui.reshape((1,-1))
     
        ui_ui_t = np.dot(ui, ui_t)
        list_ui_ui_t.append(ui_ui_t)
    
    #print "calculated ui_ui_t"
    
    for i in range(0, v_mat.shape[1]):
        
        v_idx = np.where(~np.isnan(Mij_arr[:,i]))
        v_idx = list(v_idx[0])
        
        
        sum_Mij_ui = 0 #np.zeros(10)
        sum_ui_ui_t = np.full((10,10),0)
        
        for j in v_idx:
            #print u_idx
            #print list_vj_vj_t[i]
            sum_ui_ui_t =np.add(sum_ui_ui_t, list_ui_ui_t[j])
            #print Mij_arr[j,i].shape
            #print u_mat.shape, u_mat[j].shape
            ui = u_mat[j,:]
            
            ui= ui.reshape((-1, 1))
            sum_Mij_ui = np.add(sum_Mij_ui , Mij_arr[j,i] * ui)
            
        #print "finished sum "
        
        vj = np.dot( np.linalg.inv(np.add(lamda_term,sum_ui_ui_t)), sum_Mij_ui)
       
    
        my_v_mat[:,i] = vj[:,0]
        
    return my_v_mat
   
    #print "v Step done"

#============================================================================================
# Function to find neighbors
#============================================================================================
def getKey_dist(item):
    return item[1]
 
def find_nearest_movies(my_movie_name,best_u_mat, best_v_mat):
    
    #predict_mat = np.matmul(best_u_mat, best_v_mat)
    movie_idx = movies_df[movies_df['movie_name'].str.contains(my_movie_name)].movie_id.values[0]
    movie_id_num = movie_idx + 1
    print "Movie Index is ", movie_idx
    
    
    dist_list = []
    #v_col = best_v_mat[:,movie_idx]
    #my_v = 
    
    #j = 1
    for i in range(0,best_v_mat.shape[1]):
        dist = np.linalg.norm(best_v_mat[:,movie_idx] - best_v_mat[:,i])
        if (i != movie_idx) :
            dist_list.append([i,dist])
            
    
    
    #print dist_list
    dist_list = sorted(dist_list, key=getKey_dist)
    dist_list = list(dist_list)
    
    #print "Dist List", dist_list
    
    my_neighbor = []

    for i in range(0,10):
        movie_col_num = dist_list[i][0] 
        this_movie_name = movies_df.iloc[movie_col_num].movie_name
        this_movie_rating = dist_list[i][1]
        my_neighbor.append([this_movie_name,this_movie_rating])
        #my_neighbor.append([movies_df.iloc[dist_list[i][0]+1][0], dist_list[i][1]])

    my_neighbor_df = pd.DataFrame(my_neighbor, columns=["Movie Name", "Distance"])
    
    print "Top Ten Closest Movies to :", my_movie_name, "are:"
    print "------------------------------------------------"
    print my_neighbor_df
    




def predict_rating(test_data, u_mat, v_mat) :
    
    predict_mat = np.matmul(u_mat,v_mat)
    #print "shape of predict map", predict_mat.shape
    #print predict_mat
    #print test_data
    sum_sq = 0 
    for i in range(0,test_data.shape[0]):
        user_id = int(test_data[i,0]) - 1
        movie_id = int(test_data[i,1]) - 1
        predict_rating = predict_mat[user_id,movie_id]
        #print user_id, movie_id, predict_rating, test_data[i,2]
        sum_sq = sum_sq +  (( float(test_data[i,2]) - float(predict_rating) ) ** 2 )
    
    #print "sum_sq:", sum_sq
        
    RMSE = math.sqrt(float(sum_sq)/test_data.shape[0])
    
    return RMSE
    

    
def calculate_objective_fn(u_mat,v_mat):
    predict_mat = np.matmul(u_mat,v_mat)
    #print "Shape of predict_map:", predict_mat.shape
    sum_err = 0
    for i in range(0,len(ratings_df)):
        user_id = ratings_df.loc[i,'user_id']-1
        movie_id = ratings_df.loc[i,'movie_id']-1
        predict_rating = predict_mat[user_id,movie_id]
        #print "predict rating:", predict_rating
        #print "cal value", np.matmul(u_mat[user_id,:].reshape(1, 10), v_mat[:, movie_id].reshape(10, 1))
        #sum_err = sum_err + ( (ratings_df.loc[i,'rating'] - predict_rating)**2 )
        sum_err = sum_err +  (ratings_df.loc[i,'rating'] - predict_rating)**2 / ( 2* sigma_sq)
        #sum_err = sum_err + (ratings_df.loc[i, 'rating'] - np.matmul(u_mat[user_id,:].reshape(1, 10), v_mat[:, movie_id].reshape(10, 1))[0][0])**2 / (2*sigma_sq)
        #sum_err = sum_err + (ratings_df.loc[i, 'rating'] - np.matmul(u_mat[user_id,:].reshape(1, 10), v_mat[:, movie_id].reshape(10, 1)))**2 / (2*sigma_sq) 
    #obj_fn_1 = sum_err / ( 2 * sigma_sq)
    obj_fn_1 = sum_err 
    obj_fn_2 = (np.sum(np.square(np.linalg.norm(u_mat, axis =1)))) * lamda * 0.5
    obj_fn_3 = (np.sum(np.square(np.linalg.norm(v_mat, axis =0)))) * lamda * 0.5
    
    obj_fn = -( obj_fn_1 + obj_fn_2 + obj_fn_3)
    
    #find_nearest_movies("Star Wars", u_mat, v_mat)
    
    return obj_fn


#======================================================================================
#Function to plot Log Likelihood for iterations 2 to 100 for 10 runs
#======================================================================================
 

def plot_mf_likelihood(ljl_list):
    ljl_arr = np.array(ljl_list)
    idx  = np.arange(2, 100, 1)
    colors = ["red","blue","olive","purple","brown","green","magenta","orange", "teal","pink"]
    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood Value ")
    #plt.xticks(idx)
    #plt.xticks(np.arange(0,100, step=10))
    plt.title(" Log Joint Likelihood for iterations 2 to 100 for 10 runs")
    for i in range(0,ljl_arr.shape[0]) : 
        l = i+1
        plt.plot(idx, ljl_arr[i,np.array(idx-1)],  color = colors[i], label = "Run %i" %l)   
   
    plt.legend(loc='best') 
    
    plt.show()   
    
    

def run_mf(u_mat, v_mat):
    
    
    #log_sum_Mij = math.log(sum_Mij)
    
    ljl_list = []
    
    for i in range(1,101):
        print "Iteration Number ----------------: ", i
        new_u_mat = calculate_ui(u_mat,v_mat)  
        
        new_v_mat = calculate_vj(new_u_mat,v_mat) 
        
        u_mat = new_u_mat
        v_mat = new_v_mat
        
        
        #print "u_mat:" , u_mat[0,:]
        
        ui_pdf = multivariate_normal.pdf(u_mat, mean, sigma, allow_singular = True)
        sum_log_ui = np.sum(np.log(ui_pdf))
        
        v_mat_t = np.transpose(v_mat)
        vj_pdf = multivariate_normal.pdf(v_mat_t, mean, sigma, allow_singular = True)
        #log_sum_ui = math.log(np.prod(ui_pdf))
        sum_log_vj = np.sum(np.log(vj_pdf))
        
        log_sum_Mij = np.nansum(stats.norm.logpdf(Mij_arr, np.dot(u_mat,v_mat), sigma_sq))
        
        log_joint_likelihood = log_sum_Mij + sum_log_ui + sum_log_vj
            
        
        
        ljl_list.append(log_joint_likelihood)
        
    #print "log likelihood" , ljl_list
    
    obj_fn = calculate_objective_fn(new_u_mat,new_v_mat)
    RMSE = predict_rating(test_data,new_u_mat,new_v_mat)
    
    #print "RMSE: " , RMSE
    #print "Log Likelihood: ", log_joint_likelihood
    return RMSE, obj_fn, ljl_list, new_u_mat, new_v_mat


#==================================================================================
# Function to sort RMSE and Objective Fn 
#==================================================================================
def getKey(item):
    return item[0]

    
def sort_RMSE_Obj_fn(output_list):
    
    pair_list = []
    
    for item in output_list:
        pair_list.append([item[0], float(item[1])])
    
    print pair_list
    
    new_list = sorted(pair_list, key=getKey, reverse= True)
    out_df = pd.DataFrame(list(new_list),  columns=['Objecive Function','RMSE'])
    print out_df
    
    best_run_idx = pair_list.index(new_list[0])
    
    return best_run_idx



    

#================================================================================
# Problem 3 Main call 
#================================================================================

def problem_3_run():
    
    RMSE_list = []
    OBJ_FN_list = []
    ljl_list_n = []
    my_output_list = []
    
    for num in range(1,11): ##########################################
        print "Run Number ----------------------------------------: ", num
        # Re-generate ui and vj before each run 
        u_mat = np.random.multivariate_normal(mean, sigma, n1)
        v_mat = np.random.multivariate_normal(mean, sigma, n2).T 
    
        RMSE, obj_fn, ljl_list,u_mat_op, v_mat_op = run_mf(u_mat, v_mat)
        RMSE_list.append(RMSE)
        OBJ_FN_list.append(obj_fn)
        ljl_list_n.append(ljl_list)
        my_output_list.append((obj_fn,RMSE,u_mat_op,v_mat_op))
     
     
    
    plot_mf_likelihood(ljl_list_n) # #############uncomment later
    
    best_run_idx = sort_RMSE_Obj_fn(my_output_list) ######
    
    print "Best Run Index is :", best_run_idx
    
    best_u_mat = my_output_list[best_run_idx][2]
    best_v_mat = my_output_list[best_run_idx][3]
    
    print "Shape of best_u_mat", best_u_mat.shape
    print "Shape of best_v_mat", best_v_mat.shape
    #find_nearest_movies("Star Wars", u_mat_op, v_mat_op)
    find_nearest_movies("Star Wars", best_u_mat, best_v_mat)
    find_nearest_movies("My Fair Lady", best_u_mat, best_v_mat)
    find_nearest_movies("GoodFellas", best_u_mat, best_v_mat)
    
    print RMSE_list
    print OBJ_FN_list
    
    #return my_output_list ############################

problem_3_run()   ####################
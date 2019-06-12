
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:01:57 2019

@author: anita
"""

import numpy as np

from sklearn import preprocessing
from math import e

import math

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import poisson

#plt.rcParams["figure.figsize"] = (10,8)


####################################################################
# Read Data from CSV files
# Combine X and Y to one Matrix
# Shuffle Data 
####################################################################

x_data = np.genfromtxt('X.csv', delimiter=',')

#print x_data
x_shape = x_data.shape
print " Shape of X matrix " , x_shape

## y = 1 -> Spam , else 0 
y_data = np.genfromtxt('Y.csv')

y_shape = y_data.shape

#print y_data
print " Shape of Y Data" , y_shape

all_data = np.hstack(( x_data, y_data[:,None],))

print "Shape of all data", all_data.shape


#---- Shuffle Data 

#print all_data

new_data = all_data

np.random.shuffle(new_data)

#print new_data
#print "Shape of new Data" , new_data.shape

########################################################################
# Problem 2a : Implement Naives Bayes Classification 
########################################################################

#----------------------------------------------------------------------
# Separate Training and Test Data , 9:1 Ratio 
# Training 4140, Test 460 : Total 4600
#----------------------------------------------------------------------
def get_data(data, part):
    #print part
    #print "data is", data
    low = part*460
    high = (part*460) + 460
    test_data = data[low:high ,]
    
    #print "shape of test data", test_data.shape
    if (part == 0 ):
        train_data = data[460:4600,]
    else :
        train_data = np.vstack((data[0:low,], data[high:4600,]))       
    #print "test data is ",  test_data
    #print "shape of train_data", train_data.shape
    return test_data, train_data
    


#----------------------------------------------------------------------
# Calculate pi for Y training data 
#----------------------------------------------------------------------
def cal_pi(train_data):
    #pi_val = 6
    pi_val = np.sum(train_data[:,54])/len(train_data[:,54])
    #print "pi_val is ", pi_val
    return pi_val

# ap pi = cal_pi(train_data)


#----------------------------------------------------------------------
# Calculate Lambda for each Class , Y =0 and Y =1 
#----------------------------------------------------------------------
def cal_lamda(train_data): 
    
    lamda_0 = []
    lamda_1 = []
    
    for i in range (0 , 54) :
        #print "sum of train data", np.sum(train_data[:,0])
        #print "len of train data", len(train_data[:,0])

        l_1 = (np.sum(train_data[:,i]*train_data[:,54])+ 1)/np.sum(train_data[:,54])
        #print l_0
        lamda_1.append(l_1)

        l_0 = (np.sum(train_data[:,i]) - np.sum(train_data[:,i]*train_data[:,54])+ 1)/(len(train_data[:,54])-np.sum(train_data[:,54]))
        #print l_0
        lamda_0.append(l_0)
        #print "Nr", (np.sum(train_data[:,1]) - np.sum(train_data[:,1]*train_data[:,0])+ 1)
        #print "Dr" , (len(train_data[:,0])- np.sum(train_data[:,0]))
    
    #print "len of lambda_0", len(lamda_0)
    return lamda_0, lamda_1


#cal_lamda(train_data)


#----------------------------------------------------------------------
# Predict Y for each test data 
# Given pi, lamda 0, lamda1 

#----------------------------------------------------------------------

def predict_y(test_data, pi, l0, l1):
    
    #print "pi is ", pi
    
    
    #print "l0", l0
    #print "l1", l1
    
    pred_y = []
    
    mat_conf = np.matrix([[0,0], [0,0]])
    
    
    for n in range (0,460): 
        
        #print "Test Data Number : ------------------", n
        posterior_y0 = math.log(1-pi)
        posterior_y1 = math.log(pi)
        for i in range (0,54):
            #print "Iteration Number", i
            #xi_y0 = poisson.cdf(test_data[n,i], l0[i])
            #xi_y0 = (math.log(( (e ** -(l0[i])) * (l0[i] ** test_data[n,i]) ) )) - (math.log(math.factorial(test_data[n,i])))
            Dr = sum(math.log(ii) for ii in range(1, int(test_data[n,i])))
            Nr_y0= (-(l0[i])) + (test_data[n,i] * (math.log(l0[i])))
            xi_y0 = Nr_y0 - Dr
            posterior_y0 = posterior_y0 + xi_y0
            #print "xi_y0", xi_y0
            #print "pos_y0", posterior_y0
            #xi_y1 = poisson.cdf(test_data[n,i], l1[i])
            #xi_y1 = (math.log(( (e ** -(l1[i])) * (l1[i] ** test_data[n,i]) ))) - (math.log( math.factorial(test_data[n,i])))
            Nr_y1 = (-(l1[i])) + (test_data[n,i] * (math.log(l1[i])))
            xi_y1 = Nr_y1 - Dr
            #print "x1_y1", xi_y1
            posterior_y1 = posterior_y1 + xi_y1
            #print "pos_y1", posterior_y1
            
        if (posterior_y1-posterior_y0) > 0 : 
            y_new = 1 
        else :
            y_new = 0
    
        pred_y.append(y_new)
        
        if (test_data[n,54] == 0) and (y_new == 0):
            mat_conf[0,0] = mat_conf[0,0] + 1
        if (test_data[n,54] == 0) and (y_new == 1):   
            mat_conf[0,1] = mat_conf[0,1] + 1
        if (test_data[n,54] == 1) and (y_new == 0): 
            mat_conf[1,0] = mat_conf[1,0] + 1
        if (test_data[n,54] == 1) and (y_new == 1): 
            mat_conf[1,1] = mat_conf[1,1] + 1 
            
                
    
    #print "pred_y" , pred_y
    #print "y test data", test_data[:,54]
    #print "confusion matrix" , mat_conf
    
    return mat_conf
    
#----------------------------------------------------------------------
# Main function to call naives classifier
#----------------------------------------------------------------------    

def main_run(new_data): 
    # Call this 10 times , 10 partitions 
    my_matrix = np.matrix([[0,0],[0,0]])
    for i in range(0,10):
        
         print " Naive Bayes Classifer : Training Set :", i   
        
         lamda_0_all = []
         lamda_1_all = []
         test_data, train_data = get_data(new_data,i)
         
         pi = cal_pi(train_data)
         
         l0,l1 = cal_lamda(train_data)
         
         lamda_0_all.append(l0)
         lamda_1_all.append(l1)
         
         confusion_mat = predict_y(test_data, pi, l0, l1)
         my_matrix = my_matrix + confusion_mat
    
    lamda_0_final = np.mean(lamda_0_all, axis=0)
    lamda_1_final = np.mean(lamda_1_all, axis=0)
    #print "lamda_0", lamda_0_final
    #print "lamda_1", lamda_1_final
    
    print "final confusion matrix", my_matrix
    return lamda_0_final, lamda_1_final, my_matrix
    
    
# Call Main Naives Function   
lam_0, lam_1, naive_mat =  main_run(new_data)



def print_naive_matrix(naive_mat):
    
    p1 = naive_mat[0,0]
    p2 = naive_mat[0,1]
    p3 = naive_mat[1,0]
    p4 = naive_mat[1,1]
    
    print " "
    print " "
    print "   ------------------------------------------"
    print "  Confusion Matrix for Naive Bayes Classifier"
    print "   ------------------------------------------"
    print "                               "
    print "             Predicted Y       "
    print " " 
    print "             0            1  "
    print "        --------------------------  "
    print " T     |            |            | "
    print " R   0 |    ",p1,"  |    ",p2,"   | "
    print " U     |            |            | "
    print " E     |-------------------------|   "
    print "       |            |            | "
    print " Y   1 |    ",p3,"   |    ",p4,"  | "
    print "       |            |            | "
    print "        -------------------------  "
    
    accuracy = (naive_mat[0,0] + naive_mat[1,1]) / float(np.sum(naive_mat)) 
        
    #print "Accuracy of Prediction is " , accuracy*100, "%"
    print "Accuracy of Prediction is " , accuracy

print_naive_matrix(naive_mat)


########################################################################
# Problem 2b : Plot Stem plot for each lambda vs Dimensions
########################################################################

#--------------------------------------------------------------------------
# Stem Plot for Lamda 
#--------------------------------------------------------------------------
#plt.close('all')
def plot_stem(lam_0, lam_1): 
    f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
    markerline, stemlines, baseline = ax1.stem(range(1, 55), lam_0, '-.')
    plt.setp(baseline, 'color', 'g', 'linewidth', 2)
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(stemlines, 'color', 'b','alpha', 0.5)
    
    markerline, stemlines, baseline = ax1.stem(range(1, 55), lam_1, '--')
    plt.setp(baseline, 'color', 'g', 'linewidth', 2)
    plt.setp(markerline, 'markerfacecolor', 'r')
    plt.setp(stemlines, 'color', 'r', 'alpha', 0.5)
    
    ax1.set_xlabel("Dimensions")
    
    ax1.set_ylabel(r"$\lambda$ value")
    
    plt.legend(["Class 0", "Class 1"], loc='best')
    plt.title("Plot for Lamda Parameters for y = 0 (Blue) and y = 1 (Red) vs. Dimensions")
    plt.show()    


plot_stem(lam_0, lam_1)



########################################################################
# Problem 2c : KNN Classifer 
########################################################################


def find_neighbors(train_data, test_data): 
     
    all_pred_y = []
    mat_conf = np.matrix([[0,0], [0,0]])
    right = np.zeros(20)
    wrong = np.zeros(20)
    for i in range(0,test_data.shape[0]) : 
    #for i in range(0,5) :
        #print "i", i
        dist_list = []
        for j in range(0,train_data.shape[0]):
            my_dist = np.sum(np.absolute(test_data[i,0:54]- train_data[j,0:54]))
            #print "my_dist", my_dist
            dist_list.append(my_dist)
        #print "dist_list len", len(dist_list)
        arg_sorted = np.argsort(dist_list)
        #print "arg sorted", arg_sorted
        #print "list" , dist_list
        
        k_pred_y = []
        
        
        
        for k in range(1,21):
            #print "k : ", k
        
            top_arg = arg_sorted[0:k]
        
            #print "top_arg", top_arg
        
            count_y0 = 0
            count_y1 = 0 
            for m in range(0,k):
                #print "m", m
                if train_data[arg_sorted[m],54] == 0:
                    count_y0 = count_y0 + 1;
                if train_data[arg_sorted[m],54] == 1:
                    count_y1 = count_y1 + 1;
                #print "count y0" , count_y0
                #print "count y1", count_y1
        
            if (count_y0 > count_y1) :
                new_y = 0
            if ( count_y0 < count_y1) :
                new_y = 1
            if ( count_y0 == count_y1): 
                new_y = np.random.randint(0,2)
                
            if (new_y == test_data[i,54]) :
                right[k-1] = right[k-1] + 1
            else :
                wrong[k-1] = wrong[k-1] + 1
        
            #print "new_y" , new_y
    
            #k_pred_y.append(new_y) # only if you need pred y
            #print "k_pred_y" , k_pred_y
        #all_pred_y.append(k_pred_y)# only if you need pred y
    #print "Pred y", all_pred_y
    
    # array_pred = np.array(all_pred_y) # only if you need pred y
    #print " right" , right
    #print "wrong" , wrong
    
    acc = []
    for i in range(0,20):
        acc_k = right[i] / (right[i] + wrong[i])
        acc.append(acc_k)
        
    #print "accuracy", acc
    return acc   
    #print " shape of array_pred", array_pred.shape
    
# =============================================================================
#             
#             if (test_data[i,54] == 0) and (new_y == 0):
#                 mat_conf[0,0] = mat_conf[0,0] + 1
#             if (test_data[i,54] == 0) and (new_y == 1):   
#                 mat_conf[0,1] = mat_conf[0,1] + 1
#             if (test_data[i,54] == 1) and (new_y == 0): 
#                 mat_conf[1,0] = mat_conf[1,0] + 1
#             if (test_data[i,54] == 1) and (new_y == 1): 
#                 mat_conf[1,1] = mat_conf[1,1] + 1 
#     
#     #print "pred_y", pred_y
#     #print "mat conf", mat_conf
#     #print "all ", np.sum(mat_conf)
#     accuracy = (mat_conf[0,0] + mat_conf[1,1]) / float(np.sum(mat_conf))
#     
#     #print "accuracy", accuracy
#     return accuracy
# =============================================================================
        
         
def run_knn(new_data):
    # list of accuracy list for 10 iterations
    #all_accuracy = []
    
    accur_list = []
    for i in range(0,10):
        print " KNN Classifer : Training Set ", i
       
        test_data, train_data = get_data(new_data,i)
        
        
        accur = find_neighbors(train_data, test_data)
        
        accur_list.append(accur)
        
    
    all_accuracy = np.array(accur_list)
    
    
    #print "accur list" , accur_list
    #print "all_accuracy" , all_accuracy
    
    accuracy_final = np.mean(all_accuracy, axis=0)
    
    #print "Accuracy Final", accuracy_final
#        for k in range(1,5):
#            accur = find_neighbors(train_data, test_data, k)
#            print "K is", k, "Accuracy is", accur
#            accur_list.append(accur)
#        #print accur_list
#        
#        all_accuracy.append(accur_list)
#        print " Valiue of i is ", i, "Accuracy List is :", all_accuracy
#    
#    accuracy_final = np.mean(all_accuracy, axis=0)
#    
#    print "Accuracy Final", accuracy_final
    return accuracy_final


accuracy_final = run_knn(new_data)


acc_final = {}
for i in range(0,len(accuracy_final)) :
    acc_final[i+1] = accuracy_final[i]

# ** print "Final Accuracy", acc_final
# ------------------------------------------------------
# Plot Accuracy vs K for KNN Classifer
# ------------------------------------------------------ 
    
def plot_knn(acc_final):
    plt.figure(figsize=(15, 6))
    plt.plot(acc_final.keys(), acc_final.values(), marker="o")
    plt.xticks(acc_final.keys())
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title(" KNN Classifier Accuracy vs. K-value ")
    plt.show()   

plot_knn(acc_final)  
    

########################################################################
# Problem 2d : Logistic Regression 
########################################################################
    
# ------------------------------------------------------     
## To caculate sigmoid of a function
# ------------------------------------------------------   
def cal_sigmoid(yi,xi,w):
    
    Nr =  e ** ( yi * np.dot( xi, w))
    Dr = 1 + Nr
    sigmoid_val = Nr/Dr
    
    return sigmoid_val


# ------------------------------------------------------  
#  Calculate sigmoid term in a loop for each xi, yi, w 
# ------------------------------------------------------ 

def cal_sigterm(train_data, w):
    x = train_data[:,0:55]
    y = train_data[:,55]
    sum_term = 0
    objective_term = 0
    
    for i in range(0, train_data.shape[0]):
        xi = x[i]
        yi = y[i]
        sigmoid_val = cal_sigmoid(yi,xi,w)
        sum_term = sum_term + ( (1 - sigmoid_val) * yi * xi)
        objective_term = objective_term + math.log(sigmoid_val)
    
    return sum_term, objective_term
    
# ------------------------------------------------------  
#    Logistic regression function
# ------------------------------------------------------    
def perform_logistic(train_data):
    
    ones_col = np.ones((train_data.shape[0], 1))
    # Add ones col at begining of train data 
    train_data = np.hstack((ones_col, train_data))
    #print " now shape of train data " , train_data.shape
    
    w = np.zeros(train_data.shape[1]-1)
    step = 0.01/4600
    objective_fn = []
    for i in range(1,1001):
        sigterm, objective_term = cal_sigterm(train_data,w) 
        wt = w + (step * sigterm)
        objective_fn.append(objective_term)
        w = wt 
    
    #print objective_fn
    return objective_fn
    
    
# ------------------------------------------------------  
# Main program to call logistic regression 
# ------------------------------------------------------       
    
def run_logistic(new_data):
    
    obj_fn_all =[]
    np.place(new_data[:,54], new_data[:,54] == 0, -1)
    for i in range(0,10):
         print " Logistic Regression Training Set :", i+1

         test_data, train_data = get_data(new_data,i)
         obj = perform_logistic(train_data)
         obj_fn_all.append(obj)
    return obj_fn_all
         
objective_fn = run_logistic(new_data)  

# ------------------------------------------------------  
# Plot Objective Function for each iteration, each fold 1-10
# ------------------------------------------------------ 
#print objective_fn
def plot_logistic(objective_fn):    
    #import matplotlib.pyplot as plt
    plt.plot(objective_fn[0], label = "Fold 1")
    plt.plot(objective_fn[1], label = "Fold 2")
    plt.plot(objective_fn[2], label = "Fold 3")
    plt.plot(objective_fn[3], label = "Fold 4")
    plt.plot(objective_fn[4], label = "Fold 5")
    plt.plot(objective_fn[5], label = "Fold 6")
    plt.plot(objective_fn[6], label = "Fold 7")
    plt.plot(objective_fn[7], label = "Fold 8")
    plt.plot(objective_fn[8], label = "Fold 9")
    plt.plot(objective_fn[9], label = "Fold 10")
    plt.legend(loc='best')
    plt.ylabel('Objective Trainning Function ( log )')
    plt.xlabel('Iterations')
    plt.title(" Logistic Regression : Objective Training Function for each cross validation fold")
    plt.show()


plot_logistic(objective_fn)
    
    
#============================================================  
# Problem 2e and 2f
# Logistic Regression with Newton Approximation
#============================================================ 


def calculate_newton_parameters(train_data,w):
    
    x = train_data[:,0:55]
    y = train_data[:,55]
    
    val_L1 = np.zeros(len(w))
    val_L2 = np.zeros(len(w))
    val_obj_fn = 0
    m= 10^-6
    
    for i in range(0, train_data.shape[0]):
        xi = x[i]
        yi = y[i]
        sigmoid_val = cal_sigmoid(yi,xi,w)
        val_L1 = val_L1 + ( ( 1 - sigmoid_val ) * yi * xi) 
        val_L2 = val_L2 + ( sigmoid_val * ( 1 - sigmoid_val) * np.outer(xi, xi))
        val_obj_fn = val_obj_fn + math.log(sigmoid_val)
    
    #KInv = linalg.inv(k + numpy.eye(k.shape[1])*m)
    val_L2 = np.linalg.inv(-(val_L2) + np.eye(len(w))*m)
    #val_L2 = np.linalg.inv(-val_L2)
    
    return val_L1, val_L2, val_obj_fn

    
def compute_newton(train_data): 
    
    ones_col = np.ones((train_data.shape[0], 1))
    # Add ones col at begining of train data 
    train_data = np.hstack((ones_col, train_data))
    #print " now shape of train data " , train_data.shape
    
    w = np.zeros(train_data.shape[1]-1)
    objective_fn = []
    
    for i in range(1,101):
        
        #L1 = calculate_L1(train_data,w)
        #L2 = calculate_L2(train_data,w)
        #obj_fn = calculate_objective_fn(train_data,w) 
        
        L1, L2, obj_fn = calculate_newton_parameters(train_data,w)
        objective_fn.append(obj_fn)
        
        wt = w - np.dot(L1,L2)
        w = wt
         
    
    #print objective_fn
    return objective_fn, w
    
    
def test_newton(test_data,w):
    
    ones_col = np.ones((test_data.shape[0], 1))
    # Add ones col at begining of test data 
    test_data = np.hstack((ones_col, test_data))
    #print " now shape of test data " , test_data.shape
    
    x = test_data[:,0:55]
    y = test_data[:,55]
    
    mat_conf = np.matrix([[0,0],[0,0]])
    
    
    for i in range(0, test_data.shape[0]):
        
        xi = x[i]
        yi = y[i]
        sigmoid_val = cal_sigmoid(1,xi,w) # just using 1 instead of y
        
        if (sigmoid_val > 0.5) :
            pred_y = 1
        else:
            pred_y = -1
            
    
        if (y[i] == -1) and (pred_y == -1):
            mat_conf[0,0] = mat_conf[0,0] + 1
        if (y[i] == -1) and (pred_y == 1):   
            mat_conf[0,1] = mat_conf[0,1] + 1
        if (y[i] == 1) and (pred_y == -1): 
            mat_conf[1,0] = mat_conf[1,0] + 1
        if (y[i] == 1) and (pred_y == 1): 
            mat_conf[1,1] = mat_conf[1,1] + 1 
    
    #print "pred_y", pred_y
    #print "mat conf", mat_conf
    #print "all ", np.sum(mat_conf)
    #accuracy = (mat_conf[0,0] + mat_conf[1,1]) / float(np.sum(mat_conf))
    
    #print "accuracy", accuracy
    #return accuracy  
    return mat_conf
        
    
    
    
   
def run_newton(new_data):
    
    #np.random.shuffle(new_data)
    
    # Convert all y data 0 to -1
    np.place(new_data[:,54], new_data[:,54] == 0, -1)
    
    # List of objective function values 
    new_obj_fn_all =[]
    
    
    my_mat = np.matrix([[0,0],[0,0]])
    
    for i in range(0,10):
        
        print "Newtown Approximation : Training Set :" , i+1
                
        test_data, train_data = get_data(new_data,i)
        obj, w = compute_newton(train_data)
         
        new_obj_fn_all.append(obj)
         
        conf_mat = test_newton(test_data,w)
        my_mat = my_mat + conf_mat
    
    
    #print my_mat
    
    #print new_obj_fn_all
    return new_obj_fn_all, my_mat

# Call Logistic Regression using Newton Approximation
    
new_obj_fn_all, my_mat = run_newton(new_data)


def plot_newton_objective(objective_fn):
    
    plt.plot(objective_fn[0], label = "Fold 1")
    plt.plot(objective_fn[1], label = "Fold 2")
    plt.plot(objective_fn[2], label = "Fold 3")
    plt.plot(objective_fn[3], label = "Fold 4")
    plt.plot(objective_fn[4], label = "Fold 5")
    plt.plot(objective_fn[5], label = "Fold 6")
    plt.plot(objective_fn[6], label = "Fold 7")
    plt.plot(objective_fn[7], label = "Fold 8")
    plt.plot(objective_fn[8], label = "Fold 9")
    plt.plot(objective_fn[9], label = "Fold 10")
    plt.legend(loc='best')
    plt.ylabel('Objective Trainning Function ( log )')
    plt.xlabel('Iterations')
    plt.title(" Newton Approximation: Objective Training Function for each cross validation fold")
    plt.show()
    
   
plot_newton_objective(new_obj_fn_all)

#print my_mat
#print my_mat.shape

def print_confusion_matrix(my_mat):
    m1 = my_mat[0,0]
    m2 = my_mat[0,1]
    m3 = my_mat[1,0]
    m4 = my_mat[1,1]
    
    print " "
    print " "
    print "       ---------------------------"
    print "       Confusion Matrix for Newton"
    print "       ---------------------------"
    print "                               "
    print "             Predicted Y       "
    print " " 
    print "             0            1  "
    print "        --------------------------  "
    print " T     |            |            | "
    print " R   0 |    ",m1,"  |    ",m2,"   | "
    print " U     |            |            | "
    print " E     |-------------------------|   "
    print "       |            |            | "
    print " Y   1 |    ",m3,"   |    ",m4,"  | "
    print "       |            |            | "
    print "        -------------------------  "
    
    accuracy = (my_mat[0,0] + my_mat[1,1]) / float(np.sum(my_mat)) 
        
    print "Accuracy of Prediction is " , accuracy

print_confusion_matrix(my_mat)
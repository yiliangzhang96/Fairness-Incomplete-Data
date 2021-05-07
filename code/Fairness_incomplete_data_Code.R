# useful package 
library(ipw)
library(ranger)
require(ggplot2)
library(randomForest)
library(e1071)
library(xgboost)
library(Rlab)

# To reproduce the results in the paper:

## Synthetica data experiments
# Fairness guarantee on complete data domain using upper bound
# result mentioned in the context of Section 4.1
Simulation_0()

# Justification of convergence rate
# Figure 1-a
Simulation_1()

# Effect of different weights# Effect of sample imbalance
# Figure 1-b
Simulation_2()

# Effect of sample imbalance
# Figure 1-c
for(i in c(1,3,5,7,9)){
  Simulation_3(imbalance_ratio = i)
}


# Effect of sensitive graups' domains
# Figure 1-d
for(i in c(0.5,1,2,4,8)){
  Simulation_4(distance_parameter = i)
}

## Real data experiments
## Table 1
# COMPAS
Realdata_compas(type = 'mcar')
Realdata_compas(type = 'mar')
Realdata_compas(type = 'mnar')
# ADNI
Realdata_adni(type = 'mcar')
Realdata_adni(type = 'mar')
Realdata_adni(type = 'mnar')


## Figure 2
# COMPAS
Realdata_compas_2(type = 'mcar')
Realdata_compas_2(type = 'mar')
Realdata_compas_2(type = 'mnar')
# ADNI
Realdata_adni_2(type = 'mcar')
Realdata_adni_2(type = 'mar')
Realdata_adni_2(type = 'mnar')





###########  simulation 0: transferred fairness vs classification  ############

# use lienar SVM as the classifier


Simulation_0 = function(Iter = 50){

Col = 10
unfairness_ipw_train = matrix(0, nrow=1, ncol=40)
# sample size
n = 100000
n_0 = 50000
n_1 = 50000

for(j in 1:Iter){
    
    # data generation
    A = c(rep(0,n_0), rep(1,n_1))
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = rbern(n, exp( X%*%t(t(beta)) )/(1 + exp(X%*%t(t(beta))) ) )
    
    # missing data mechanism
    missing_index = runif(n)
    Threshold = 1/(1 + exp(0.5*(A-0.5)))
    missing = 1*(missing_index > Threshold)
    
    # test data
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = rbern(n, exp( X_test%*%t(t(beta)) )/(1 + exp(X_test%*%t(t(beta))) ) )

    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0)]
    A_CC = A[which(missing == 0)]

    
    # propensity score model
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    weight <- ipwpoint(exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw)
    
    w = weight$ipw.weights
    w = w/(sum(w[which(missing == 0)])/length(w[which(missing == 0)]))
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:length(w_true)){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    
    
    # IPW
    Z = data.frame(X_CC, y_CC)
    rf_ipw <- svm(y_CC ~ ., data = Z, type = 'C-classification',kernel = "linear", cost = 1, scale = FALSE)
    
    pred_train = as.numeric(predict(rf_ipw, data = Z))-1
    MSE_0_ipw_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_ipw_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = as.numeric(predict(rf_ipw, data = data.frame(X_test, y_test)))-1
    MSE_0_ipw = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_ipw = sum(A_test*abs(pred - y_test))/sum(A_test)

    unfairness_ipw_train[j] = abs(MSE_0_ipw_train - MSE_1_ipw_train) 
    
}
print('fairness estimation on the training set')
print(unfairness_ipw_train)
}


###########  simulation 1: transferred fairness vs sample size  ############

Simulation_1 = function(Iter = 200){

Col = 10
unfairness_cc = matrix(0, nrow=11, ncol=Iter)
unfairness_cc_train = matrix(0, nrow=11, ncol=Iter)
unfairness_cc_test = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw_train = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw_test = matrix(0, nrow=11, ncol=Iter)
lower_bound = matrix(0, nrow=11, ncol=Iter)

for(i in 1:11){
  
  lambda = 1
  k = 1 
  # sample size
  n = round((10^(3+(i-1)*0.2)))
  n_0 = round(n/2)
  n_1 = n - n_0
  
  for(j in 1:Iter){
    
    # generate data
    A = c(rep(0,n_0), rep(1,n_1))
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    
    # missing mechanism used in this experiment:
    missing_index = runif(n)
    Threshold = 1/(1 + exp(k - 1*apply(X[,1:5],1,mean)))
    missing = 1*(missing_index > Threshold)
    
    # test data
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    
    # complete cases
    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0),]
    A_CC = A[which(missing == 0)]

    # propensity score model
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(exposure = miss, family = "binomial", link = "logit", 
      numerator = ~ 1, denominator = ~ ., data = D_0)
    weight_1 <- ipwpoint(exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = D_1)
    
    w_0 = weight_0$ipw.weights
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 = weight_1$ipw.weights
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    w = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    Threshold[which(Threshold > 1)] = 1; Threshold[which(Threshold < 0)] = 0.01
    
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:length(w_true)){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }

    # IPW
    Z = data.frame(X_CC, y_CC)
    rf_ipw <- ranger(y_CC ~ ., data = Z, case.weights = 1) #w_true[which(missing == 0)]
    
    pred_train = predict(rf_ipw, data = Z)$predictions
    MSE_0_ipw_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_ipw_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_ipw, data = data.frame(X_test, y_test))$predictions
    MSE_0_ipw = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_ipw = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    N_0 = sum(which(missing == 0) < n_0+1)
    N_1 = sum(which(missing == 0) >= n_0+1)
    
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    unfairness_ipw_train[i,j] = abs(MSE_0_ipw_train - MSE_1_ipw_train) 
    unfairness_ipw_test[i,j] = abs(MSE_0_ipw - MSE_1_ipw)
    
    #### calculate lower bound: ####
    missing_index = runif(100000)
    Threshold_test = 1/(1 + exp(k - 1*apply(X_test[,1:5],1,mean)))
    missing = 1*(missing_index > Threshold_test)
    w_true = 1/Threshold_test
    denom_0 = (sum(w_true[which(missing == 0 & A_test == 0)])/length(w_true[which(missing == 0 & A_test == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A_test == 1)])/length(w_true[which(missing == 0 & A_test == 1)]))
    for(m in 1:length(w_true)){
      if(A_test[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    X_CC = X_test[which(missing == 0),]
    y_CC = y_test[which(missing == 0),]
    A_CC = A_test[which(missing == 0)]
    Z = data.frame(X_CC, y_CC)
    
    pred_test = predict(rf_ipw, data = Z)$predictions
    sigma_0 = var((w_true[which(missing == 0)]*abs(pred_test - y_CC))[which(A_CC == 0)])
    sigma_1 = var((w_true[which(missing == 0)]*abs(pred_test - y_CC))[which(A_CC == 1)])
    lower_bound[i,j] = (sigma_0/(1*N_0) + sigma_1/(1*N_1))^0.5 / 24
  }
}

# 0.95-quantile
nintycover_up <- function(x){
  up = sort(x)[190]
  return(up)
}
# 0.05-quantile
nintycover_low <- function(x){
  low = sort(x)[11]
  return(low)
}


print('Fairness estimation bias using IPW with true weights:')
apply(unfairness_ipw,1,mean)
# sd of bias
apply(unfairness_ipw,1,sd)
# 0.95-quantile of bias
apply(unfairness_ipw,1,nintycover_up)
# 0.05-quantile of bias
apply(unfairness_ipw,1,nintycover_low)

print('Lower bound:')
apply(lower_bound,1,mean)
apply(lower_bound,1,sd)

}



###########  simulation 2: transferred fairness vs propensity score model  ############

Simulation_2 = function(Iter = 100){

Col = 10
unfairness_cc = matrix(0, nrow=11, ncol=Iter)
unfairness_true = matrix(0, nrow=11, ncol=Iter)
unfairness_logistic = matrix(0, nrow=11, ncol=Iter)
unfairness_logistic_2 = matrix(0, nrow=11, ncol=Iter)
unfairness_rf = matrix(0, nrow=11, ncol=Iter)
unfairness_svm = matrix(0, nrow=11, ncol=Iter)
unfairness_xgb = matrix(0, nrow=11, ncol=Iter)


for(i in 1:11){
  
  n_1 = round((10^(2+(i-1)*0.1))/0.3)
  n_0 = round(3*n_1/7)
  n = n_0 + n_1
  
  # default 100 iterations
  for(j in 1:Iter){
    
    # generate data
    A = c(rep(0,n_0), rep(1,n_1))
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)

    # missing data mechanism
    missing_index = runif(n)
    Threshold = 1/(1 + exp(-3 + 1*(apply((X[,1:5])^3,1,mean))))
    missing = 1*(missing_index > Threshold)
    
    # test data
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    
    # complete cases
    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0),]
    A_CC = A[which(missing == 0)]
    
    # fit random forest and check transferred fairness
    Z = data.frame(X_CC, y_CC)
    rf_cc<- ranger(y_CC ~ ., data = Z)
    
    pred_train = predict(rf_cc, data = Z)$predictions
    MSE_0_cc_train = sum((1-A_CC)*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_cc_train = sum(A_CC*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_cc, data = data.frame(X_test, y_test))$predictions
    MSE_0_cc = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_cc = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # true weight
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:length(w_true)){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
   
    ## correctly specified logistic regression
    data_ipw_2 = data.frame((X[,1:5])^3, 1-missing)
    colnames(data_ipw_2) = c('V1','V2','V3','V4','V5','miss')

    D_0 = data_ipw_2[which(A==0),]
    D_1 = data_ipw_2[which(A==1),]
    weight_0 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = D_0)
    weight_1 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = D_1)
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw_2[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw_2[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w_logistic_2 = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w_logistic_2[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_logistic_2[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    ## Incorrectly specified logistic regression
    data_ipw = data.frame(X[,1:5], 1-missing)
    colnames(data_ipw) = c('V1','V2','V3','V4','V5','miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = D_0)
    weight_1 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = D_1)
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w_logistic = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w_logistic[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_logistic[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    rf_0 <- randomForest(
      miss ~ .,
      data = D_0
    )
    rf_1 <- randomForest(
      miss ~ .,
      data = D_1
    )
    
    w_0 <- 1/(rf_0$predicted)
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 <- 1/(rf_1$predicted)
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w_rf = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w_rf[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_rf[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # svm
    svm_0 = svm(miss ~ ., data = data_ipw[which(A==0),], kernel = "radial", cost = 10, scale = FALSE)
    svm_1 = svm(miss ~ ., data = data_ipw[which(A==1),], kernel = "radial", cost = 10, scale = FALSE)
    
    w_0 <- 1/(svm_0$fitted)
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 <- 1/(svm_1$fitted)
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w_svm = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w_svm[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_svm[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # XGboosting
    xgb_0 = xgboost(data = matrix(as.numeric(as.matrix(data_ipw[which(A==0),1:5])),nrow = n_0), label = as.matrix(data_ipw$miss[which(A==0)]), max_depth = 10, eta = 1, nthread = 2, nrounds = 3,"binary:logistic")
    xgb_1 = xgboost(data = matrix(as.numeric(as.matrix(data_ipw[which(A==1),1:5])),nrow = n_1), label = as.matrix(data_ipw$miss[which(A==1)]), max_depth = 10, eta = 1, nthread = 2, nrounds = 3,"binary:logistic")
    
    w_0 <- 1/(predict(xgb_0, matrix(as.numeric(as.matrix(data_ipw[which(A==0),1:5])),nrow = n_0)))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 <- 1/(predict(xgb_1, matrix(as.numeric(as.matrix(data_ipw[which(A==1),1:5])),nrow = n_1)))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w_xgb = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w_xgb[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_xgb[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # IPW
    Z = data.frame(X_CC, y_CC)
    # true
    rf_true <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    pred_train = predict(rf_true, data = Z)$predictions
    MSE_0_true_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_true_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_true, data = data.frame(X_test, y_test))$predictions
    MSE_0_true = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_true = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # logistic_incorrect
    rf_logistic <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    pred_train = predict(rf_logistic, data = Z)$predictions
    MSE_0_logistic_train = sum((1-A_CC)*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_logistic_train = sum(A_CC*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_logistic, data = data.frame(X_test, y_test))$predictions
    MSE_0_logistic = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_logistic = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # logistic_correct
    rf_logistic <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    pred_train = predict(rf_logistic, data = Z)$predictions
    MSE_0_logistic_2_train = sum((1-A_CC)*w_logistic_2[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_logistic_2_train = sum(A_CC*w_logistic_2[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_logistic, data = data.frame(X_test, y_test))$predictions
    MSE_0_logistic_2 = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_logistic_2 = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # rf
    rf_rf <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    pred_train = predict(rf_rf, data = Z)$predictions
    MSE_0_rf_train = sum((1-A_CC)*w_rf[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_rf_train = sum(A_CC*w_rf[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_rf, data = data.frame(X_test, y_test))$predictions
    MSE_0_rf = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_rf = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # svm
    rf_svm <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    pred_train = predict(rf_svm, data = Z)$predictions
    MSE_0_svm_train = sum((1-A_CC)*w_svm[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_svm_train = sum(A_CC*w_svm[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_svm, data = data.frame(X_test, y_test))$predictions
    MSE_0_svm = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_svm = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # xgb
    rf_xgb <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    pred_train = predict(rf_xgb, data = Z)$predictions
    MSE_0_xgb_train = sum((1-A_CC)*w_xgb[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_xgb_train = sum(A_CC*w_xgb[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_xgb, data = data.frame(X_test, y_test))$predictions
    MSE_0_xgb = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_xgb = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_true[i,j] = abs( abs(MSE_0_true - MSE_1_true) - abs(MSE_0_true_train - MSE_1_true_train) )
    unfairness_logistic[i,j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    unfairness_logistic_2[i,j] = abs( abs(MSE_0_logistic_2 - MSE_1_logistic_2) - abs(MSE_0_logistic_2_train - MSE_1_logistic_2_train) )
    unfairness_rf[i,j] = abs( abs(MSE_0_rf - MSE_1_rf) - abs(MSE_0_rf_train - MSE_1_rf_train) )
    unfairness_svm[i,j] = abs( abs(MSE_0_svm - MSE_1_svm) - abs(MSE_0_svm_train - MSE_1_svm_train) )
    unfairness_xgb[i,j] = abs( abs(MSE_0_xgb - MSE_1_xgb) - abs(MSE_0_xgb_train - MSE_1_xgb_train) )
    
  }
}

nintycover_up <- function(x){
  up = sort(x)[round(0.95*Iter)]
  return(up)
}

nintycover_low <- function(x){
  low = sort(x)[round(0.05*Iter)]
  return(low)
}

print('Fairness estimation bias using unweighted estimator:')
apply(unfairness_cc,1,mean)
apply(unfairness_cc,1,sd)
apply(unfairness_cc,1,nintycover_up)
apply(unfairness_cc,1,nintycover_low)

print('Fairness estimation bias using true weight estimator:')
apply(unfairness_true,1,mean)
apply(unfairness_true,1,sd)
apply(unfairness_true,1,nintycover_up)
apply(unfairness_true,1,nintycover_low)

print('Fairness estimation bias using incorrectly specified estimator:')
apply(unfairness_logistic,1,mean)
apply(unfairness_logistic,1,sd)
apply(unfairness_logistic,1,nintycover_up)
apply(unfairness_logistic,1,nintycover_low)

print('Fairness estimation bias using correctly specified estimator:')
apply(unfairness_logistic_2,1,mean)
apply(unfairness_logistic_2,1,sd)
apply(unfairness_logistic_2,1,nintycover_up)
apply(unfairness_logistic_2,1,nintycover_low)

print('Fairness estimation bias using random forest estimator:')
apply(unfairness_rf,1,mean)
apply(unfairness_rf,1,sd)
apply(unfairness_rf,1,nintycover_up)
apply(unfairness_rf,1,nintycover_low)

print('Fairness estimation bias using SVM estimator:')
apply(unfairness_svm,1,mean)
apply(unfairness_svm,1,sd)
apply(unfairness_svm,1,nintycover_up)
apply(unfairness_svm,1,nintycover_low)

print('Fairness estimation bias using XGBoost estimator:')
apply(unfairness_xgb,1,mean)
apply(unfairness_xgb,1,sd)
apply(unfairness_xgb,1,nintycover_up)
apply(unfairness_xgb,1,nintycover_low)

}








###########  simulation 3: effect of sample imbalance  ############

# change imbalance ratio from 1 to 9
Simulation_3 = function(imbalance_ratio = 1, Iter = 100){

Col = 10
unfairness_cc = matrix(0, nrow=11, ncol=Iter)
unfairness_cc_train = matrix(0, nrow=11, ncol=Iter)
unfairness_cc_test = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw_train = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw_test = matrix(0, nrow=11, ncol=Iter)

for(i in 1:11){
  
  
  lambda = imbalance_ratio #1 3 5 7 9
  n = round(10^(3+(i-1)*0.2))
  n_0 = round(lambda*n/(1+lambda))
  n_1 = n - n_0
  
  for(j in 1:Iter){
    
    # generate data
    A = c(rep(0,n_0), rep(1,n_1))
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    
    # missing data mechanism
    missing_index = runif(n)
    Threshold = 1/(1 + exp(1 - 1*apply(X[,1:5],1,mean)))
    missing = 1*(missing_index > Threshold)
    
    # test
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    
    # complete cases
    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0),]
    A_CC = A[which(missing == 0)]

    # propensity score model
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = data_ipw[which(A==1),])
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # IPW
    Z = data.frame(X_CC, y_CC)
    rf_ipw <- ranger(y_CC ~ ., data = Z, case.weights = 1) #w[which(missing == 0)]
    
    pred_train = predict(rf_ipw, data = Z)$predictions
    MSE_0_ipw_train = sum((1-A_CC)*w[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_ipw_train = sum(A_CC*w[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_ipw, data = data.frame(X_test, y_test))$predictions
    MSE_0_ipw = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_ipw = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    unfairness_ipw_train[i,j] = abs(MSE_0_ipw_train - MSE_1_ipw_train) 
    unfairness_ipw_test[i,j] = abs(MSE_0_ipw - MSE_1_ipw)
    
  }
  
}

nintycover_up <- function(x){
  up = sort(x)[round(0.95*Iter)]
  return(up)
}

nintycover_low <- function(x){
  low = sort(x)[round(0.05*Iter)]
  return(low)
}

print('Fairness estimation bias using correctly specified logistic estimator:')
apply(unfairness_ipw,1,mean)
apply(unfairness_ipw_train,1,mean)
apply(unfairness_ipw_test,1,mean)
apply(unfairness_ipw,1,sd)
apply(unfairness_ipw,1,nintycover_up)
apply(unfairness_ipw,1,nintycover_low)
}


###########  simulation 4: CCA when domains are different  ##############

Simulation_4 = function(distance_parameter = 0.5, Iter = 100){
  
Col = 10
unfairness_cc = matrix(0, nrow=11, ncol=Iter)
unfairness_ipw = matrix(0, nrow=11, ncol=Iter)

for(i in 1:11){
  
  lambda = 1 
  n = round(10^(3+(i-1)*0.2))
  n_0 = round(n/5)
  n_1 = n - n_0
  M = distance_parameter 
  
  for(j in 1:Iter){
    
    # generate data
    A = c(rep(0,n_0), rep(1,n_1))
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-M) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+M)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    
    # missing data mechanism
    missing_index = runif(n)
    Threshold = 1/(1 + exp(4*(A-0.5)))
    missing = 1*(missing_index > Threshold)
    
    # test data
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-M) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+M)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    
    # complete cases
    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0),]
    A_CC = A[which(missing == 0)]
    
    # propensity score model
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    w_true = 1/Threshold
    w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint(exposure = miss,family = "binomial",link = "logit",
      numerator = ~ 1,denominator = ~ ., data = data_ipw[which(A==1),])
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    
    w = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:length(Threshold)){
      if(A[m]==0){
        w[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # IPW
    Z = data.frame(X_CC, y_CC)
    rf_ipw <- ranger(y_CC ~ ., data = Z, case.weights = 1) #w[which(missing == 0)]
    
    pred_train = predict(rf_ipw, data = Z)$predictions
    MSE_0_ipw_train = sum((1-A_CC)*w[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_ipw_train = sum(A_CC*w[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    pred = predict(rf_ipw, data = data.frame(X_test, y_test))$predictions
    MSE_0_ipw = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_ipw = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    
  }
  
}

print(sum(which(missing == 0) < n_0+1))
print(sum(which(missing == 0) >= n_0+1))

nintycover_up <- function(x){
  up = sort(x)[round(0.95*Iter)]
  return(up)
}
nintycover_low <- function(x){
  low = sort(x)[round(0.05*Iter)]
  return(low)
}

print('Fairness estimation bias using correctly specified logistic estimator:')
apply(unfairness_ipw,1,mean)
apply(unfairness_ipw_train,1,mean)
apply(unfairness_ipw_test,1,mean)
apply(unfairness_ipw,1,sd)
apply(unfairness_ipw,1,nintycover_up)
apply(unfairness_ipw,1,nintycover_low)

}




########## real data ############

####### experiment on different weight estimators #######

#### recidivism

Realdata_compas = function(type = 'mcar', Iter = 100){

load("compas_data.RData")
compas_gender = compas_data[,1]
compas_race = compas_data[,3]
compas_data_imp = scale(compas_data[,-c(1,3,10,12)])
compas_data_imp[,10] = 1*(compas_data_imp[,10] > 0)


# training set
set.seed(816)
seed = sample(10000,5000)

unfairness_cc = matrix(0, nrow=1, ncol=Iter)
unfairness_true = matrix(0, nrow=1, ncol=Iter)
unfairness_logistic = matrix(0, nrow=1, ncol=Iter)
unfairness_rf = matrix(0, nrow=1, ncol=Iter)
unfairness_svm = matrix(0, nrow=1, ncol=Iter)
unfairness_xgb = matrix(0, nrow=1, ncol=Iter)

n = 4000
for(j in 1:Iter){
    
    set.seed(seed[j])
    S = sample(nrow(compas_data_imp))
    training_index = S[1:4000]
    test_index = S[4001:nrow(compas_data_imp)]
    
    training_data = compas_data_imp[training_index,]
    test_data = compas_data_imp[test_index,]

    
    missing_index = runif(n)
    K = 5
    if(type == 'mcar'){Threshold = rep(0.8,n)}
    if(type == 'mar'){Threshold = 1/(1 + exp(-3 - 2*training_data[,1:K]%*% c(1,1,1,1,1)))}
    if(type == 'mnar'){Threshold = 1/(1 + exp(0 + 2*training_data[,10] + 2*(training_data[,9])))}
    missing = 1*(missing_index > Threshold)
    observe_index = which(missing==0)
    
    ### full data's MSE
    training_cc = training_data[which(missing==0),]
    A = compas_gender[training_index]
    n_0 = length(which(A==0))
    n_1 = length(which(A==1))
    A_CC = compas_gender[training_index[observe_index]]
    y_CC = training_cc[,10]
    A_test = compas_gender[test_index]
    y_test = test_data[,10]
      
    Z = data.frame(training_cc)
    colnames(Z)[10] = 'y_CC'
    
    # true weight
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:n){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    
    # logistic
    data_ipw = data.frame(training_data[,1:K],1-missing)
    colnames(data_ipw) = c('V1','V2','V3','V4','V5','miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ ., data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ ., data = data_ipw[which(A==1),])
    
    w_0 = weight_0$ipw.weights
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 = weight_1$ipw.weights
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_logistic = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_logistic[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_logistic[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }

    # IPW
    # true weights
    rf_true <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_true[which(missing == 0)]
    pred_train = as.numeric(predict(rf_true, data = Z)$predictions)-1
    MSE_0_true_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_true_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_true, data = test_data)$predictions)-1
    MSE_0_true = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_true = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # CC
    rf_cc<- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1)
    pred_train = as.numeric(predict(rf_cc, data = Z)$predictions)-1
    MSE_0_cc_train = sum((1-A_CC)*1*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_cc_train = sum(A_CC*1*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_cc, data = test_data)$predictions)-1
    MSE_0_cc = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_cc = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # logistic
    rf_logistic <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_logistic[which(missing == 0)]
    pred_train = as.numeric(predict(rf_logistic, data = Z)$predictions)-1
    MSE_0_logistic_train =sum((1-A_CC)*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_logistic_train = sum(A_CC*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_logistic, data = test_data)$predictions)-1
    MSE_0_logistic = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_logistic = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # rf
    rf_rf <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_rf[which(missing == 0)]
    pred_train = as.numeric(predict(rf_rf, data = Z)$predictions)-1
    MSE_0_rf_train = sum((1-A_CC)*w_rf[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_rf_train = sum(A_CC*w_rf[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_rf, data = test_data)$predictions)-1
    MSE_0_rf = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_rf = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # svm
    rf_svm <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_svm[which(missing == 0)]
    pred_train = as.numeric(predict(rf_svm, data = Z)$predictions)-1
    MSE_0_svm_train = sum((1-A_CC)*w_svm[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_svm_train = sum(A_CC*w_svm[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_svm, data = test_data)$predictions)-1
    MSE_0_svm = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_svm = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # xgb
    rf_xgb <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_xgb[which(missing == 0)]
    pred_train = as.numeric(predict(rf_xgb, data = Z)$predictions)-1
    MSE_0_xgb_train = sum((1-A_CC)*w_xgb[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_xgb_train = sum(A_CC*w_xgb[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_xgb, data = test_data)$predictions)-1
    MSE_0_xgb = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_xgb = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    unfairness_cc[j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_true[j] = abs( abs(MSE_0_true - MSE_1_true) - abs(MSE_0_true_train - MSE_1_true_train) )
    unfairness_logistic[j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    unfairness_rf[j] = abs( abs(MSE_0_rf - MSE_1_rf) - abs(MSE_0_rf_train - MSE_1_rf_train) )
    unfairness_svm[j] = abs( abs(MSE_0_svm - MSE_1_svm) - abs(MSE_0_svm_train - MSE_1_svm_train) )
    unfairness_xgb[j] = abs( abs(MSE_0_xgb - MSE_1_xgb) - abs(MSE_0_xgb_train - MSE_1_xgb_train) )
  
}

print('Fairness estimation bias using different estimators:')
print('unweighted:')
mean(unfairness_cc[1,1:Iter])
print('true weight:')
mean(unfairness_true[1,1:Iter])
print('correctly specified logistic:')
mean(unfairness_logistic[1,1:Iter])
print('random forest:')
mean(unfairness_rf[1,1:Iter])
print('SVM:')
mean(unfairness_svm[1,1:Iter])
print('XGBoost:')
mean(unfairness_xgb[1,1:Iter])

print('standard deviation:')
print('unweighted:')
mean(unfairness_cc[1,1:Iter])
print('true weight:')
mean(unfairness_true[1,1:Iter])
print('correctly specified logistic:')
mean(unfairness_logistic[1,1:Iter])
print('random forest:')
mean(unfairness_rf[1,1:Iter])
print('SVM:')
mean(unfairness_svm[1,1:Iter])
print('XGBoost:')
mean(unfairness_xgb[1,1:Iter])
}



#### ADNI
Realdata_adni = function(type = 'mcar', Iter = 100){
  
load("adni_imp.RData")
adni = sa_imp 
adni_gender = 2 - adni[,1002] # 0: female; 1: male
adni_race = 2 - adni[,1004]
adni_data_imp = as.matrix(adni[,c(1:1001)]) # ,1003 for pred


# training set
set.seed(816)
seed = sample(10000,5000)

unfairness_cc = matrix(0, nrow=1, ncol=Iter)
unfairness_true = matrix(0, nrow=1, ncol=Iter)
unfairness_logistic = matrix(0, nrow=1, ncol=Iter)
unfairness_rf = matrix(0, nrow=1, ncol=Iter)
unfairness_svm = matrix(0, nrow=1, ncol=Iter)
unfairness_xgb = matrix(0, nrow=1, ncol=Iter)

for(j in 1:Iter){
    
    set.seed(seed[2*j])
    
    S = sample(nrow(adni_data_imp))
    training_index = S[1:500]
    test_index = S[501:nrow(adni_data_imp)]
    ratio = 1
    n = length(training_index)
    
    training_data = adni_data_imp[training_index,]
    test_data = adni_data_imp[test_index,]
    training_gender = adni_gender[training_index]
    
    missing_index = runif(n)
    if(type == 'mcar'){Threshold = rep(0.8,n)}
    if(type == 'mar'){Threshold = 1/(1 + exp(2 + 2*apply(training_data[,1:10],1,mean)))}
    if(type == 'mnar'){Threshold = 1/(1 + exp(2 + 2*apply(training_data[,101:110],1,mean)))}
    missing = 1*(missing_index > Threshold)
    observe_index = which(missing==0)
    
    ### full data's MSE
    training_cc = training_data[which(missing==0),]
    
    A = training_gender
    n_0 = length(which(A==0))
    n_1 = length(which(A==1))
    A_CC = adni_gender[training_index[observe_index]]
    y_CC = training_cc[,1001]
    A_test = adni_gender[test_index]
    y_test = test_data[,1001]
    
    Z = data.frame(training_cc)
    colnames(Z)[1001] = 'y_CC'
    rf_cc<- ranger((y_CC) ~ ., data = Z)
    
    pred_train = as.numeric(predict(rf_cc, data = Z)$predictions)
    MSE_0_cc_train = sum((1-A_CC)*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_cc_train = sum(A_CC*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_cc, data = data.frame(test_data))$predictions)
    MSE_0_cc = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_cc = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # true weight
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:n){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }

    # logistic
    K = 100
    data_ipw = data.frame(training_data[,1:K],1-missing)
    varname <- 'V'
    n_K <- K + 1
    names(data_ipw)[1:ncol(data_ipw)] <- unlist(mapply(function(x,y) paste(x, seq(1,y), sep="_"), varname, n_K))
    colnames(data_ipw)[n_K] = 'miss'
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(exposure = miss,family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw[which(A==1),])
    
    w_0 = weight_0$ipw.weights
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 = weight_1$ipw.weights
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_logistic = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_logistic[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_logistic[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    ## random forest
    rf_0 <- randomForest(
      miss ~ .,
      data = data_ipw[which(A==0),]
    )
    rf_1 <- randomForest(
      miss ~ .,
      data = data_ipw[which(A==1),]
    )
    w_0 <- 1/(rf_0$predicted)
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 <- 1/(rf_1$predicted)
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_rf = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_rf[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_rf[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # svm
    svm_0 = svm(miss ~ ., data = data_ipw[which(A==0),], kernel = "radial", cost = 10, scale = FALSE)
    svm_1 = svm(miss ~ ., data = data_ipw[which(A==1),], kernel = "radial", cost = 10, scale = FALSE)
    
    w_0 <- 1/(svm_0$fitted)
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 <- 1/(svm_1$fitted)
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_svm = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_svm[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_svm[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    #xgb
    xgb_0 = xgboost(data = matrix(as.numeric(as.matrix(data_ipw[which(A==0),1:K])),nrow = n_0), label = as.matrix(data_ipw$miss[which(A==0)]), max_depth = 10, eta = 1, nthread = 2, nrounds = 10,"binary:logistic")
    xgb_1 = xgboost(data = matrix(as.numeric(as.matrix(data_ipw[which(A==1),1:K])),nrow = n_1), label = as.matrix(data_ipw$miss[which(A==1)]), max_depth = 10, eta = 1, nthread = 2, nrounds = 10,"binary:logistic")
    
    w_0 <- 1/(predict(xgb_0, matrix(as.numeric(as.matrix(data_ipw[which(A==0),1:K])),nrow = n_0)))
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 <- 1/(predict(xgb_1, matrix(as.numeric(as.matrix(data_ipw[which(A==1),1:K])),nrow = n_1)))
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_xgb = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_xgb[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_xgb[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # IPW
    # true weights
    rf_true <- ranger((y_CC) ~ ., data = Z, case.weights = 1) #w_true[which(missing == 0)]
    pred_train = as.numeric(predict(rf_true, data = Z)$predictions)
    MSE_0_true_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_true_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_true, data = data.frame(test_data))$predictions)
    MSE_0_true = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_true = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # logistic
    rf_logistic <- ranger((y_CC) ~ ., data = Z, case.weights = 1) #w_logistic[which(missing == 0)]
    pred_train = as.numeric(predict(rf_logistic, data = Z)$predictions)
    MSE_0_logistic_train =sum((1-A_CC)*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_logistic_train = sum(A_CC*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_logistic, data = data.frame(test_data))$predictions)
    MSE_0_logistic = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_logistic = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # rf
    rf_rf <- ranger((y_CC) ~ ., data = Z, case.weights = 1) #w_rf[which(missing == 0)]
    pred_train = as.numeric(predict(rf_rf, data = Z)$predictions)
    MSE_0_rf_train = sum((1-A_CC)*w_rf[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_rf_train = sum(A_CC*w_rf[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_rf, data = data.frame(test_data))$predictions)
    MSE_0_rf = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_rf = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # svm
    rf_svm <- ranger((y_CC) ~ ., data = Z, case.weights = 1) #w_svm[which(missing == 0)]
    pred_train = as.numeric(predict(rf_svm, data = Z)$predictions)
    MSE_0_svm_train = sum((1-A_CC)*w_svm[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_svm_train = sum(A_CC*w_svm[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_svm, data = data.frame(test_data))$predictions)
    MSE_0_svm = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_svm = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # xgb
    rf_xgb <- ranger((y_CC) ~ ., data = Z, case.weights = 1) #w_xgb[which(missing == 0)]
    pred_train = as.numeric(predict(rf_xgb, data = Z)$predictions)
    MSE_0_xgb_train = sum((1-A_CC)*w_xgb[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_xgb_train = sum(A_CC*w_xgb[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_xgb, data = data.frame(test_data))$predictions)
    MSE_0_xgb = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_xgb = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    unfairness_cc[j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_true[j] = abs( abs(MSE_0_true - MSE_1_true) - abs(MSE_0_true_train - MSE_1_true_train) )
    unfairness_logistic[j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    unfairness_rf[j] = abs( abs(MSE_0_rf - MSE_1_rf) - abs(MSE_0_rf_train - MSE_1_rf_train) )
    unfairness_svm[j] = abs( abs(MSE_0_svm - MSE_1_svm) - abs(MSE_0_svm_train - MSE_1_svm_train) )
    unfairness_xgb[j] = abs( abs(MSE_0_xgb - MSE_1_xgb) - abs(MSE_0_xgb_train - MSE_1_xgb_train) )
    
    
}

print('Fairness estimation bias using different estimators:')
print('unweighted:')
mean(unfairness_cc[1,1:Iter])
print('true weight:')
mean(unfairness_true[1,1:Iter])
print('correctly specified logistic:')
mean(unfairness_logistic[1,1:Iter])
print('random forest:')
mean(unfairness_rf[1,1:Iter])
print('SVM:')
mean(unfairness_svm[1,1:Iter])
print('XGBoost:')
mean(unfairness_xgb[1,1:Iter])

print('standard deviation:')
print('unweighted:')
mean(unfairness_cc[1,1:Iter])
print('true weight:')
mean(unfairness_true[1,1:Iter])
print('correctly specified logistic:')
mean(unfairness_logistic[1,1:Iter])
print('random forest:')
mean(unfairness_rf[1,1:Iter])
print('SVM:')
mean(unfairness_svm[1,1:Iter])
print('XGBoost:')
mean(unfairness_xgb[1,1:Iter])
  
}



####### experiment on the effect of sample imbalance #######

#### recidivism
Realdata_compas_2 = function(type = 'mcar', Iter = 500){

load("compas_data.RData")
compas_gender = compas_data[,1]
compas_race = compas_data[,3]
compas_data_imp = scale(compas_data[,-c(1,3,10,12)])
compas_data_imp[,10] = 1*(compas_data_imp[,10] > 0)


# different sample size?
# training/test split?

# training set
set.seed(816)
seed = sample(10000,5000)

unfairness_cc = matrix(0, nrow=5, ncol=Iter)
unfairness_true = matrix(0, nrow=5, ncol=Iter)
unfairness_logistic = matrix(0, nrow=5, ncol=Iter)
unfairness_rf = matrix(0, nrow=5, ncol=Iter)
unfairness_svm = matrix(0, nrow=5, ncol=Iter)
unfairness_xgb = matrix(0, nrow=5, ncol=Iter)


for(i in 1:5){
  
  lambda = 1
  alpha = 3*lambda/7 
  
  n_1 = round((10^(2+(i-1)*0.1))/0.5)
  n_0 = round(alpha*n_1)
  n = n_0 + n_1
  rate = c(0.1,1/8,1/6,1/4,1/2)
  
  for(j in 1:Iter){
    
    set.seed(seed[j])
    S = sample(nrow(compas_data_imp))
    training_index_old = S[1:4000]
    test_index = S[4001:nrow(compas_data_imp)]
    n_minority = min(table(compas_gender[training_index_old]))
    
    ratio = rate[i]
    S_new = sample(c(sample(which(compas_gender[training_index_old] == 1), round(ratio*n_minority)),sample(which(compas_gender[training_index_old] == 0), round((1-ratio)*n_minority))))
    training_index = training_index_old[S_new]
    n = length(training_index)
    
    test_data = compas_data_imp[test_index,]
    training_data = compas_data_imp[training_index,]
    training_gender = compas_gender[training_index]
    
    
    missing_index = runif(n)
    K = 5
    if(type == 'mcar'){Threshold = rep(0.8,n)}
    if(type == 'mar'){Threshold = 1/(1 + exp(-3 - 2*training_data[,1:K]%*% c(1,1,1,1,1)))}
    if(type == 'mnar'){Threshold = 1/(1 + exp(0 + 2*training_data[,10] + 2*(training_data[,9])))}
    missing = 1*(missing_index > Threshold)
    observe_index = which(missing==0)
    
    ### full data's MSE
    training_cc = training_data[which(missing==0),]
    
    A = compas_gender[training_index]
    n_0 = length(which(A==0))
    n_1 = length(which(A==1))
    A_CC = compas_gender[training_index[observe_index]]
    y_CC = training_cc[,10]
    A_test = compas_gender[test_index]
    y_test = test_data[,10]
    
    Z = data.frame(training_cc)
    colnames(Z)[10] = 'y_CC'
    
    
    # logistic
    data_ipw = data.frame(training_data[,1:K],1-missing)
    colnames(data_ipw) = c('V1','V2','V3','V4','V5','miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw[which(A==1),])
    
    w_0 = weight_0$ipw.weights
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 = weight_1$ipw.weights
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_logistic = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_logistic[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_logistic[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
    
    # IPW
    # logistic
    rf_logistic <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_logistic[which(missing == 0)]
    pred_train = as.numeric(predict(rf_logistic, data = Z)$predictions)-1
    MSE_0_logistic_train =sum((1-A_CC)*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_logistic_train = sum(A_CC*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_logistic, data = test_data)$predictions)-1
    MSE_0_logistic = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_logistic = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    unfairness_logistic[i,j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    
  }
}
print('fairness estimation bias:')
apply(unfairness_logistic[,1:Iter],mean,MARGIN = 1)
print('Standard deviation:')
apply(unfairness_logistic[,1:Iter],sd,MARGIN = 1)

}




#### ADNI

Realdata_adni_2 = function(type = 'mcar', Iter = 1000){
  
load("adni_imp.RData")
adni = sa_imp 
adni_gender = 2 - adni[,1002] # 0: female; 1: male
adni_race = 2 - adni[,1004]
adni_data_imp = as.matrix(adni[,c(1:1001)]) # ,1003 for pred

# training set
set.seed(816)
seed = sample(10000,5000)


unfairness_cc = matrix(0, nrow=5, ncol=Iter)
unfairness_true = matrix(0, nrow=5, ncol=Iter)
unfairness_logistic = matrix(0, nrow=5, ncol=Iter)
unfairness_rf = matrix(0, nrow=5, ncol=Iter)
unfairness_svm = matrix(0, nrow=5, ncol=Iter)
unfairness_xgb = matrix(0, nrow=5, ncol=Iter)



for(i in 1:5){
  
  n = 200
  rate = c(0.1,1/8,1/6,1/4,1/2)
  
  for(j in 1:Iter){
    
    set.seed(seed[2*j])
    S = sample(nrow(adni_data_imp))
    training_index_old = S[1:300]
    test_index = S[301:nrow(adni_data_imp)]
    n_minority = min(table(adni_gender[training_index_old]))
    
    ratio = rate[i]
    S_new = sample(c(sample(which(adni_gender[training_index_old] == 0), round(ratio*n_minority)), sample(which(adni_gender[training_index_old] == 1), round((1-ratio)*n_minority))))
    training_index = training_index_old[S_new]
    n = length(training_index)
    training_data = adni_data_imp[training_index,]
    test_data = adni_data_imp[test_index,]
    training_gender = adni_gender[training_index]
    
    missing_index = runif(n)
    if(type == 'mcar'){Threshold = rep(0.8,n)}
    if(type == 'mar'){Threshold = 1/(1 + exp(2 + 2*apply(training_data[,1:10],1,mean)))}
    if(type == 'mnar'){Threshold = 1/(1 + exp(2 + 2*apply(training_data[,101:110],1,mean)))}
    missing = 1*(missing_index > Threshold)
    observe_index = which(missing==0)
    
    ### full data's MSE
    training_cc = training_data[which(missing==0),]
    
    A = training_gender
    n_0 = length(which(A==0))
    n_1 = length(which(A==1))
    
    A_CC = adni_gender[training_index[observe_index]]
    y_CC = training_cc[,1001]
    A_test = adni_gender[test_index]
    y_test = test_data[,1001]
    
    Z = data.frame(training_cc)
    colnames(Z)[1001] = 'y_CC'
    
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    
    # logistic
    K = 100
    data_ipw = data.frame(training_data[,1:K],1-missing)
    varname <- 'V'
    n_K <- K + 1
    names(data_ipw)[1:ncol(data_ipw)] <- unlist(mapply(function(x,y) paste(x, seq(1,y), sep="_"), varname, n_K))
    colnames(data_ipw)[n_K] = 'miss'
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint( exposure = miss, family = "binomial", link = "logit",
      numerator = ~ 1, denominator = ~ .,  data = data_ipw[which(A==1),])
    
    w_0 = weight_0$ipw.weights
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    w_1 = weight_1$ipw.weights
    w_1 = w_1/(sum(w_1[which(D_1[,K+1]==1)])/length(w_1[which(D_1[,K+1]==1)]))
    
    w_logistic = c()
    index_0 = 1
    index_1 = 1
    for(m in 1:n){
      if(A[m]==0){
        w_logistic[m] = w_0[index_0]
        index_0 = index_0+1
      }
      else{
        w_logistic[m] = w_1[index_1]
        index_1 = index_1+1
      }
    }
  
    # IPW
    # logistic
    rf_logistic <- ranger((y_CC) ~ ., data = Z, case.weights = 1) #w_logistic[which(missing == 0)]
    pred_train = as.numeric(predict(rf_logistic, data = Z)$predictions)
    MSE_0_logistic_train =sum((1-A_CC)*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_logistic_train = sum(A_CC*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    pred = as.numeric(predict(rf_logistic, data = data.frame(test_data))$predictions)
    MSE_0_logistic = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_logistic = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    unfairness_logistic[i,j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    
  }
  
}

print('fairness estimation bias:')
apply(unfairness_logistic[,1:Iter],mean,MARGIN = 1)
print('Standard deviation:')
apply(unfairness_logistic[,1:Iter],sd,MARGIN = 1)
}

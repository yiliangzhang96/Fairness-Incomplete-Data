# simulations for transferred fairness


# generate data
library(ipw)
library(ranger)
require(ggplot2)
library(randomForest)
library(e1071)
library(xgboost)



###########  simulation 0: transferred fairness vs classification  ############

# use lienar SVM as the classifier

Col = 10

unfairness_cc = matrix(0, nrow=11, ncol=40)
unfairness_cc_train = matrix(0, nrow=11, ncol=40)
unfairness_cc_test = matrix(0, nrow=11, ncol=40)
unfairness_ipw = matrix(0, nrow=11, ncol=40)
unfairness_ipw_train = matrix(0, nrow=11, ncol=40)
unfairness_ipw_test = matrix(0, nrow=11, ncol=40)
lower_bound = matrix(0, nrow=11, ncol=40)


for(i in 1:1){
  
  # missingness: 0.1/0.9
  # 1 = 0.1/0.9
  
  # lambda, alpha / 1-alpha = 0.1*lambda / 0.9+0.1*lambda
  lambda = 1
  k = 5 #3, 5
  #alpha = lambda/4.1 #13 # 3*lambda/7 #(0.1*lambda)/(0.9+0.1*lambda)
  
  n = 100000
  n_0 = 50000
  n_1 = 50000
  
  #n_1 = round((10^(2+(i-1)*0.2))/0.12) #0.12
  #n_0 = round(alpha*n_1)
  #n = n_0 + n_1
  
  for(j in 1:10){
    
    # X is mixed gaussian
    A = c(rep(0,n_0), rep(1,n_1))
    #B = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    #C = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = rbern(n, exp( X%*%t(t(beta)) )/(1 + exp(X%*%t(t(beta))) ) )
    #y = abs(sin(6*(X%*%t(t(beta)))^2 + epsilon))
    #y = 1/(1 + exp(5*(1-(X%*%t(t(beta)))^2) + epsilon/5))
    #A = 1*(apply(abs(X[,1:5]),1,mean) > apply(abs(X[,6:10]),1,mean))
    #A = 1*(X[,10]>1.5)
    
    # the last 5 column are missing with 2-pattern mechanism
    
    missing_index = runif(n)
    Threshold = 1/(1 + exp(0.5*(A-0.5)))
    #Threshold = 1/(1 + exp(k - 1*apply(X[,1:5],1,mean)))
    #Threshold = 0.5 + 0.2 * apply(X[,1:5],1,mean)
    #Threshold =  abs(X[,10])
    missing = 1*(missing_index > Threshold)
    
    # testing
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = rbern(n, exp( X_test%*%t(t(beta)) )/(1 + exp(X_test%*%t(t(beta))) ) )
    #y_test = 1/(1 + exp(5*(1-(X_test%*%t(t(beta)))^2) + rnorm(100000)/5)) 
    #A_test = 1*(apply(abs(X_test[,1:5]),1,mean) > apply(abs(X_test[,6:10]),1,mean))
    #A_test = 1*(X_test[,10]>0)
    
    # CCA
    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0)]
    A_CC = A[which(missing == 0)]
    
    # fit random forest and check transferred fairness
    Z = data.frame(X_CC, y_CC)
    #rf_cc<- ranger(y_CC ~ ., classification = T, data = Z)
    rf_cc <- svm(y_CC ~ ., data = Z, type = 'C-classification',kernel = "linear", cost = 1, scale = FALSE)
    
    
    pred_train = as.numeric(predict(rf_cc, data = Z))-1
    MSE_0_cc_train = sum((1-A_CC)*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_cc_train = sum(A_CC*abs(pred_train - y_CC))/sum(A_CC)
    
    pred = as.numeric(predict(rf_cc, data = data.frame(X_test, y_test)))-1
    MSE_0_cc = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_cc = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    print(c(MSE_0_cc_train, MSE_1_cc_train))
    print(c(MSE_0_cc, MSE_1_cc))
    
    # propensity score model
    
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    weight <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw)
    
    w = weight$ipw.weights
    w = w/(sum(w[which(missing == 0)])/length(w[which(missing == 0)]))
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    w_true = 1/Threshold
    w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    
    
    
    # IPW
    Z = data.frame(X_CC, y_CC)
    #rf_ipw <- ranger(y_CC ~ ., classification = T, data = Z, case.weights = 1) #w_true[which(missing == 0)]
    rf_ipw <- svm(y_CC ~ ., data = Z, type = 'C-classification',kernel = "linear", cost = 1, scale = FALSE)
    
    
    pred_train = as.numeric(predict(rf_ipw, data = Z))-1
    MSE_0_ipw_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_ipw_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    
    pred = as.numeric(predict(rf_ipw, data = data.frame(X_test, y_test)))-1
    MSE_0_ipw = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_ipw = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    #N_0 = sum(which(missing == 0) < n_0+1)
    #N_1 = sum(which(missing == 0) >= n_0+1)
    #sigma_0 = var((w_true[which(missing == 0)]*abs(pred_train - y_CC))[which(A_CC == 0)])
    #sigma_1 = var((w_true[which(missing == 0)]*abs(pred_train - y_CC))[which(A_CC == 1)])
    #print(N_0)
    #print(N_1)
    
    print(c(MSE_0_ipw_train, MSE_1_ipw_train))
    print(c(MSE_0_ipw, MSE_1_ipw))
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_cc_train[i,j] = abs(MSE_0_cc_train - MSE_1_cc_train) 
    unfairness_cc_test[i,j] = abs(MSE_0_cc - MSE_1_cc)
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    unfairness_ipw_train[i,j] = abs(MSE_0_ipw_train - MSE_1_ipw_train) 
    unfairness_ipw_test[i,j] = abs(MSE_0_ipw - MSE_1_ipw)
    
    print(sum(1 - A_CC))
    print(sum(A_CC))
  }
  
}





###########  simulation 1: transferred fairness vs sample size  ############

Col = 10

unfairness_cc = matrix(0, nrow=11, ncol=200)
unfairness_cc_train = matrix(0, nrow=11, ncol=200)
unfairness_cc_test = matrix(0, nrow=11, ncol=200)
unfairness_ipw = matrix(0, nrow=11, ncol=200)
unfairness_ipw_train = matrix(0, nrow=11, ncol=200)
unfairness_ipw_test = matrix(0, nrow=11, ncol=200)
lower_bound = matrix(0, nrow=11, ncol=200)
#Sigma_0 = matrix(0, nrow=11, ncol=200)
#Sigma_1 = matrix(0, nrow=11, ncol=200)



for(i in 1:1){
  
  # missingness: 0.1/0.9
  # 1 = 0.1/0.9
  
  # lambda, alpha / 1-alpha = 0.1*lambda / 0.9+0.1*lambda
  lambda = 1
  k = 1 #3, 5
  #alpha = lambda/4.1 #13 # 3*lambda/7 #(0.1*lambda)/(0.9+0.1*lambda)
  
  n = round((10^(3+(i-1)*0.2)))   #0.2
  n_0 = round(n/2)
  n_1 = n - n_0
  
  #n_1 = round((10^(2+(i-1)*0.2))/0.12) #0.12
  #n_0 = round(alpha*n_1)
  #n = n_0 + n_1
  
  for(j in 1:2){
    
    # X is mixed gaussian
    A = c(rep(0,n_0), rep(1,n_1))
    #B = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    #C = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    #y = abs(sin(6*(X%*%t(t(beta)))^2 + epsilon))
    #y = 1/(1 + exp(5*(1-(X%*%t(t(beta)))^2) + epsilon/5))
    #A = 1*(apply(abs(X[,1:5]),1,mean) > apply(abs(X[,6:10]),1,mean))
    #A = 1*(X[,10]>1.5)
    
    # the last 5 column are missing with 2-pattern mechanism
    
    missing_index = runif(n)
    Threshold = 1/(1 + exp(k - 1*apply(X[,1:5],1,mean)))
    #Threshold = 0.5 + 0.2 * apply(X[,1:5],1,mean)
    #Threshold =  abs(X[,10])
    missing = 1*(missing_index > Threshold)
    
    # testing
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    #y_test = 1/(1 + exp(5*(1-(X_test%*%t(t(beta)))^2) + rnorm(100000)/5)) 
    #A_test = 1*(apply(abs(X_test[,1:5]),1,mean) > apply(abs(X_test[,6:10]),1,mean))
    #A_test = 1*(X_test[,10]>0)
    
    # CCA
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
    
    print(c(MSE_0_cc_train, MSE_1_cc_train))
    print(c(MSE_0_cc, MSE_1_cc))
    
    # propensity score model
    
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = D_0)
    
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = D_1)
    
    w_0 = weight_0$ipw.weights
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    
    w_1 = weight_1$ipw.weights
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    # merge to w
    
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
    
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    
    w_true = 1/Threshold
    #w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:length(w_true)){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    
    #w_true = w_true/(sum(w_true[which(missing == 0 && A == 0)])/length(w_true[which(missing == 0)]))
    
    
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
    #sigma_0 = var((w_true[which(missing == 0)]*abs(pred_train - y_CC))[which(A_CC == 0)])
    #sigma_1 = var((w_true[which(missing == 0)]*abs(pred_train - y_CC))[which(A_CC == 1)])
    #print(N_0)
    #print(N_1)
    
    print(c(MSE_0_ipw_train, MSE_1_ipw_train))
    print(c(MSE_0_ipw, MSE_1_ipw))
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_cc_train[i,j] = abs(MSE_0_cc_train - MSE_1_cc_train) 
    unfairness_cc_test[i,j] = abs(MSE_0_cc - MSE_1_cc)
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    unfairness_ipw_train[i,j] = abs(MSE_0_ipw_train - MSE_1_ipw_train) 
    unfairness_ipw_test[i,j] = abs(MSE_0_ipw - MSE_1_ipw)
    
    #### calculate lower bound: ####
    
    missing_index = runif(100000)
    Threshold_test = 1/(1 + exp(k - 1*apply(X_test[,1:5],1,mean)))
    missing = 1*(missing_index > Threshold_test)
    w_true = 1/Threshold_test
    #w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
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
    #MSE_0_ipw_test = sum((1-A_CC)*w_true[which(missing == 0)]*(pred_test - y_CC)^2)/sum(1 - A_CC)
    #MSE_1_ipw_test = sum(A_CC*w_true[which(missing == 0)]*(pred_test - y_CC)^2)/sum(A_CC)
    
    #N_0 = sum(which(missing == 0) <= n_0) #sum(which(missing == 0) < 50000)
    #N_1 = sum(which(missing == 0) > n_0) #sum(which(missing == 0) >= 50000)
    sigma_0 = var((w_true[which(missing == 0)]*abs(pred_test - y_CC))[which(A_CC == 0)])
    sigma_1 = var((w_true[which(missing == 0)]*abs(pred_test - y_CC))[which(A_CC == 1)])
    
    
    #Sigma_0[i,j] = sigma_0
    #Sigma_1[i,j] = sigma_1
    lower_bound[i,j] = (sigma_0/(1*N_0) + sigma_1/(1*N_1))^0.5 / 24
    
  }
}


print(sum(which(missing == 0) < n_0+1))
print(sum(which(missing == 0) >= n_0+1))


nintycover_up <- function(x){
  up = sort(x)[190]
  return(up)
}


nintycover_low <- function(x){
  low = sort(x)[11]
  return(low)
}

print('cc:')
apply(unfairness_cc,1,mean)
apply(unfairness_cc_train,1,mean)
apply(unfairness_cc_test,1,mean)
apply(unfairness_cc,1,sd)
apply(unfairness_cc,1,nintycover_up)
apply(unfairness_cc,1,nintycover_low)


print('ipw:')
apply(unfairness_ipw,1,mean)
apply(unfairness_ipw_train,1,mean)
apply(unfairness_ipw_test,1,mean)
apply(unfairness_ipw,1,sd)
apply(unfairness_ipw,1,nintycover_up)
apply(unfairness_ipw,1,nintycover_low)

#mean(unfairness_ipw[11,1:10])
#mean(lower_bound[11,1:10])

print('LB:')
apply(lower_bound,1,mean)
apply(lower_bound,1,sd)





# plot

PLOT = FALSE

epoch = seq(3,5,length.out = 11)

a_1 = log(c(0.64697518, 0.37001050, 0.32435127, 0.27449113, 0.22259672, 0.14911764, 0.13738978, 0.10119541, 0.10165951, 0.07504841, 0.07414515))/log(10)
upr_a_1 = log(c(1.6922977, 0.9339650, 0.8843042, 0.6506126, 0.5018826, 0.3469372, 0.3385335, 0.2303208, 0.2184305, 0.1469285, 0.1475173))/log(10)
lwr_a_1 = log(c(0.031645252, 0.037657142, 0.008922848, 0.011729117, 0.010998032, 0.036022100, 0.001655677, 0.013333654, 0.022070455, 0.012365848, 0.002501084))/log(10)

lb_a_1 = log(c(0.085196538, 0.060542862, 0.045502235, 0.032785149, 0.024281402, 0.017722691, 0.013171325, 0.009731604, 0.007156458, 0.005290472, 0.003881727))/log(10)

a_2 = log(c(0.52041703, 0.40197058, 0.31811287, 0.28159614, 0.19916717, 0.16618034, 0.14937053, 0.10229134, 0.09421206, 0.08227956, 0.07113836))/log(10)
upr_a_2 = log(c(1.6922977, 0.9339650, 0.8843042, 0.6506126, 0.5018826, 0.3469372, 0.3385335, 0.2303208, 0.2184305, 0.1469285, 0.1475173))/log(10)
lwr_a_2 = log(c(0.052537481, 0.042321204, 0.031010325, 0.022059119, 0.012707756, 0.014623594, 0.016229936, 0.012619039, 0.008688827, 0.010158232, 0.007373208))/log(10)

lb_a_2 = log(c(0.085196538, 0.060542862, 0.045502235, 0.032785149, 0.024281402, 0.017722691, 0.013171325, 0.009731604, 0.007156458, 0.005290472, 0.003881727))/log(10)


a_3 = log(c(19.456610, 15.769656, 13.517369, 10.772533, 10.108115 , 7.920448 , 6.739823, 5.549925 , 4.481484,  3.793864,  3.094829))
upr_a_3 = log(c(30.057868, 21.647154, 19.006570, 14.518360, 12.782274,  9.764989,  8.184639, 6.975948,  5.369472,  4.350427,  3.478778))
lwr_a_3 = log(c(8.835933, 10.362062,  8.849643,  8.007693,  7.695630,  5.301883 , 5.398585, 4.426971,  3.835406,  3.106338 , 2.592057))

a_4 = log(c(22.759418, 21.298306, 16.859866, 14.520576, 11.676912, 10.026820 , 7.892496, 6.693010,  5.595175,  4.678083,  3.812950))
upr_a_4 = log(c(32.564544, 27.684185, 22.976905, 18.071926, 14.383730, 11.889813,  9.240028, 7.887040,  6.301626,  5.338650,  4.204753))
lwr_a_4 = log(c(14.444362, 16.261089, 12.516046, 10.849751,  8.667233,  7.635550,  6.255691, 5.713058,  4.922015,  3.935741,  3.383264))

a_5 = log(c(25.726718, 22.778521, 19.214834, 15.462957, 13.731010, 10.932193,  9.300017, 7.668642,  6.412065,  5.219049,  4.416464))
upr_a_5 = log(c(33.161073, 31.159738, 23.646929, 20.430577, 15.799052, 12.700304, 10.577175, 8.756554,  7.445108,  5.649539,  4.893585))
lwr_a_5 = log(c(18.187977, 15.906324, 15.222947, 12.318759, 11.544354,  8.377073, 8.041419, 6.365448,  5.513882 , 4.675852,  3.893852))



b = log(c(67.480712, 62.811315, 46.447439, 37.625635, 31.154704, 26.070394, 21.901551, 17.315131, 14.260627, 11.296347,  8.895828))
upr_b = log(c(131.147469, 104.411247,  73.579621,  61.800014,  42.395069,  35.145717, 25.459421,  19.848760,  15.364889,  12.078054,   9.443137))
lwr_b = log(c(4.727696, 17.230152, 11.529076, 18.241311, 17.758189, 19.913519, 16.755316, 14.341901, 12.910361, 10.354802,  8.460403))

if(PLOT){

  # expression('n'['max']*'/n'['min']*' = 1')
  data = data.frame(epoch,a_1,lb_a_1,a_2,lwr_a_2,'lwr_a_1'= lwr_a_1,'upr_a_1'= upr_a_1, 'lwr_a_2'= lwr_a_2,'upr_a_2'= upr_a_2, stringsAsFactors = F)
  (plot <- ggplot(data, aes(epoch, a_1))+
      #geom_point(data, mapping = aes(epoch, a_1, color = "k = 3"),size = 2, shape = 17)+ 
      geom_point(data, mapping = aes(epoch, a_2, color = "Fairness difference"),size = 2, shape = 17)+ 
      #geom_point(data, mapping = aes(epoch, a_2, color = "Lower bound"),size = 2, shape = 12)+ 
      #geom_point(data, mapping = aes(epoch, a_3, color = "n_max/n_min = 5"),size = 2, shape = 13)+ 
      #geom_point(data, mapping = aes(epoch, a_4, color = "n_max/n_min = 7"),size = 2, shape = 14)+ 
      #geom_point(data, mapping = aes(epoch, a_5, color = "n_max/n_min = 9"),size = 2, shape = 16)+ 
      #geom_point(data, mapping = aes(epoch, b, color = "CCA"),size = 2, shape = 15)+
      #geom_line(data, mapping = aes(epoch, a_1, color = "k = 3"))+
      #geom_line(data, mapping = aes(epoch, lb_a_1, color = "Lower bound"))+
      geom_line(data, mapping = aes(epoch, a_2, color = "Fairness difference"), size = 1)+
      geom_line(data, mapping = aes(epoch, lb_a_2, color = "Lower bound"), size = 1.5)+
      #geom_line(data, mapping = aes(epoch, a_5, color = "n_max/n_min = 9"))+
      #geom_line(data, mapping = aes(epoch, b, color = "CCA"))+
      #scale_color_manual(values=c('red',"#fb9a99"))+
      scale_color_manual(values=c('#33a02c',"#b2df8a"))+
      #geom_ribbon(data=data,aes(ymin=lwr_a_1,ymax=upr_a_1),alpha=0.1,fill = "#fb9a99")+
      geom_ribbon(data=data,aes(ymin=lwr_a_2,ymax=upr_a_2),alpha=0.2,fill = "#b2df8a")+
      #geom_ribbon(data=data,aes(ymin=lwr_a_2,ymax=upr_a_2),alpha=0.0,fill = '#1f78b4')+
      #geom_ribbon(data=data,aes(ymin=lwr_a_3,ymax=upr_a_3),alpha=0.0,fill = '#b2df8a')+
      #geom_ribbon(data=data,aes(ymin=lwr_a_4,ymax=upr_a_4),alpha=0.0,fill = '#33a02c')+
      #geom_ribbon(data=data,aes(ymin=lwr_a_5,ymax=upr_a_5),alpha=0.0,fill = '#fb9a99')+
      #geom_ribbon(data=data,aes(ymin=lwr_b,ymax=upr_b),alpha=0.2,fill = 'black')+
      #geom_hline(yintercept=0.914, linetype="dashed", color = "black")+
      #annotate("text", x = 18, y = 0.92, label = "91.4%")+
      coord_cartesian(ylim = c(-3, 0.5))+
      labs(color="",
           x= expression("log"['10']*" n"['min']), y = expression('log'['10']*' Bias') #expression('log'['10']*' |'*Delta*' - '*widehat(Delta)*'|')
           #, y= expression('log |'*Delta*' - '*widehat(Delta)*'|')
           )+
      #ggtitle("Justification of bounds")+ #with different sample imbalance
      theme(legend.position= c(0.85,0.82),axis.text=element_text(size=12),panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"),plot.title = element_text(hjust = 0.5,size = 14),legend.text = element_text(size = 7)))
  
}



# +0.2

a_1 = log(c(34.146690, 30.938097, 26.036715, 23.199961, 20.601245, 17.036921, 14.177089, 12.064271, 10.012948,  8.647221,  7.210629))
upr_a_1 = log(c(45.963396, 39.898675, 34.588164, 29.248617, 24.582946, 21.531796, 16.191414, 13.024239, 10.879340,  9.456153,  7.890636))
lwr_a_1 = log(c(21.597893, 19.750207, 18.619151, 16.420467, 17.362689, 14.312186, 12.117661, 11.038719,  8.934940,  7.742766,  6.539853))

a_2 = log(c(28.275773, 23.964857, 21.632575, 17.665368, 15.113213, 12.476313, 10.935179, 9.263838,  7.785187,  6.580118,  5.500591))
upr_a_2 = log(c(36.995554, 28.781157, 26.855926, 21.380120, 18.085347, 14.344999, 12.804484, 10.847895,  8.342990,  7.128675,  5.889213))
lwr_a_2 = log(c(13.499309, 18.322432, 16.135699, 13.693835, 13.052980, 10.238247,  9.485332, 8.496673,  7.162111,  6.065271,  5.084364))

a_3 = log(c(21.223827, 17.149698, 14.976195, 13.369660, 11.686977,  9.696556,  8.203882, 6.991169,  5.886124,  4.959206,  4.209499))
upr_a_3 = log(c(27.413751, 21.731770, 17.981498, 16.455525, 13.412496, 10.758002,  9.278318, 7.789154,  6.368131,  5.330383,  4.513999))
lwr_a_3 = log(c(15.001147, 13.192837,  9.214767, 10.371923,  9.787943,  8.381641,  7.193801, 5.928792,  5.303227,  4.563946,  3.893399))

a_4 = log(c(14.683807, 12.768994, 11.556229, 9.716301,  8.477094,  7.127180,  6.123697, 5.178223 , 4.356179 , 3.757037 , 3.159909))
upr_a_4 = log(c(20.240369, 16.536762, 13.415671, 11.212248 , 9.735023 , 7.980206 , 6.721673, 5.693633 , 4.896842 , 4.149252 , 3.389003))
lwr_a_4 = log(c(8.972988, 9.191921, 9.862814, 7.062207, 6.077725, 6.079935, 5.119478, 4.701061, 3.760582, 3.218139, 2.886461))

a_5 = log(c(9.564266, 8.382383, 7.586789, 6.763787, 6.242054, 5.387398, 4.746358, 3.961931, 3.386022, 2.866895, 2.415184))
upr_a_5 = log(c(13.491025, 11.649717, 10.139834,  8.041631,  7.080033,  6.176584,  5.145783, 4.320835,  3.701234,  3.127964,  2.628493))
lwr_a_5 = log(c(5.027620, 4.163040, 4.875221, 4.486690, 5.291557, 4.691149, 4.382963, 3.540929, 3.052629, 2.616750, 2.168078))


# linear - true
a_1 = log(c(9.639733, 8.958265, 6.410026, 5.892860, 4.768200, 3.952951, 2.956317, 2.538113, 2.054120, 1.763876, 1.569601))
upr_a_1 = log(c(45.963396, 39.898675, 34.588164, 29.248617, 24.582946, 21.531796, 16.191414, 13.024239, 10.879340,  9.456153,  7.890636))
lwr_a_1 = log(c(21.597893, 19.750207, 18.619151, 16.420467, 17.362689, 14.312186, 12.117661, 11.038719,  8.934940,  7.742766,  6.539853))

a_2 = log(c(7.562774, 6.764218, 5.687884, 4.739969, 4.094645, 3.035983, 2.728124, 1.982544, 1.792940, 1.516041, 1.141738))
upr_a_2 = log(c(36.995554, 28.781157, 26.855926, 21.380120, 18.085347, 14.344999, 12.804484, 10.847895,  8.342990,  7.128675,  5.889213))
lwr_a_2 = log(c(13.499309, 18.322432, 16.135699, 13.693835, 13.052980, 10.238247,  9.485332, 8.496673,  7.162111,  6.065271,  5.084364))

a_3 = log(c(17.887092, 15.086026, 12.115677, 10.376465,  8.858238,  7.357704,  5.906625, 4.772245,  4.075021,  3.233386,  2.735922))
upr_a_3 = log(c(27.413751, 21.731770, 17.981498, 16.455525, 13.412496, 10.758002,  9.278318, 7.789154,  6.368131,  5.330383,  4.513999))
lwr_a_3 = log(c(15.001147, 13.192837,  9.214767, 10.371923,  9.787943,  8.381641,  7.193801, 5.928792,  5.303227,  4.563946,  3.893399))

a_4 = log(c(22.812233, 19.555073, 16.349243, 13.367821, 10.757100,  9.380292,  7.682968, 6.403882 , 5.218288,  4.310699,  3.561632))
upr_a_4 = log(c(20.240369, 16.536762, 13.415671, 11.212248 , 9.735023 , 7.980206 , 6.721673, 5.693633 , 4.896842 , 4.149252 , 3.389003))
lwr_a_4 = log(c(8.972988, 9.191921, 9.862814, 7.062207, 6.077725, 6.079935, 5.119478, 4.701061, 3.760582, 3.218139, 2.886461))

a_5 = log(c(25.078724, 21.741573, 18.522040, 15.524567, 12.861336, 10.550814,  8.840666, 7.353201,  6.176136,  4.981964,  4.192017))
upr_a_5 = log(c(13.491025, 11.649717, 10.139834,  8.041631,  7.080033,  6.176584,  5.145783, 4.320835,  3.701234,  3.127964,  2.628493))
lwr_a_5 = log(c(5.027620, 4.163040, 4.875221, 4.486690, 5.291557, 4.691149, 4.382963, 3.540929, 3.052629, 2.616750, 2.168078))



# linear
a_1 = log(c(5.1595728, 4.5600593, 3.4908766, 2.5459677, 1.9805733, 1.4664640, 1.2993473, 0.9005362, 0.7409871, 0.5863408, 0.4399264))
upr_a_1 = log(c(45.963396, 39.898675, 34.588164, 29.248617, 24.582946, 21.531796, 16.191414, 13.024239, 10.879340,  9.456153,  7.890636))
lwr_a_1 = log(c(21.597893, 19.750207, 18.619151, 16.420467, 17.362689, 14.312186, 12.117661, 11.038719,  8.934940,  7.742766,  6.539853))

a_2 = log(c(10.867057,  9.946026,  7.976688,  6.834164,  5.353211,  4.292979,  3.730260, 3.032966,  2.626187,  2.065673,  1.821772))
upr_a_2 = log(c(36.995554, 28.781157, 26.855926, 21.380120, 18.085347, 14.344999, 12.804484, 10.847895,  8.342990,  7.128675,  5.889213))
lwr_a_2 = log(c(13.499309, 18.322432, 16.135699, 13.693835, 13.052980, 10.238247,  9.485332, 8.496673,  7.162111,  6.065271,  5.084364))

a_3 = log(c(20.722488, 17.565588, 13.765457, 11.682747,  9.374822,  8.007877,  6.614843, 5.374156,  4.565416,  3.800383,  3.131231))
upr_a_3 = log(c(27.413751, 21.731770, 17.981498, 16.455525, 13.412496, 10.758002,  9.278318, 7.789154,  6.368131,  5.330383,  4.513999))
lwr_a_3 = log(c(15.001147, 13.192837,  9.214767, 10.371923,  9.787943,  8.381641,  7.193801, 5.928792,  5.303227,  4.563946,  3.893399))

a_4 = log(c(23.345327, 20.727723, 17.343437, 14.142584, 11.803304, 10.098256,  8.222633, 6.781827,  5.662427,  4.721707,  3.946223))
upr_a_4 = log(c(20.240369, 16.536762, 13.415671, 11.212248 , 9.735023 , 7.980206 , 6.721673, 5.693633 , 4.896842 , 4.149252 , 3.389003))
lwr_a_4 = log(c(8.972988, 9.191921, 9.862814, 7.062207, 6.077725, 6.079935, 5.119478, 4.701061, 3.760582, 3.218139, 2.886461))

a_5 = log(c(26.953637, 23.016347, 19.173404, 16.597578, 13.371131, 11.053464,  9.278464, 7.613533,  6.344037,  5.272593,  4.370690))
upr_a_5 = log(c(13.491025, 11.649717, 10.139834,  8.041631,  7.080033,  6.176584,  5.145783, 4.320835,  3.701234,  3.127964,  2.628493))
lwr_a_5 = log(c(5.027620, 4.163040, 4.875221, 4.486690, 5.291557, 4.691149, 4.382963, 3.540929, 3.052629, 2.616750, 2.168078))




# logistic - true
a_1 = log(c(30.408019, 26.872097, 21.739791, 18.689319, 14.602823, 12.864890, 10.507210, 8.826101,  7.309406,  6.435280,  5.401755))
upr_a_1 = log(c(45.963396, 39.898675, 34.588164, 29.248617, 24.582946, 21.531796, 16.191414, 13.024239, 10.879340,  9.456153,  7.890636))
lwr_a_1 = log(c(21.597893, 19.750207, 18.619151, 16.420467, 17.362689, 14.312186, 12.117661, 11.038719,  8.934940,  7.742766,  6.539853))

a_2 = log(c(11.863358, 10.458495,  9.085102,  7.604982,  6.679642,  5.991151,  5.009325, 4.468757,  3.981844,  3.535438,  3.091797))
upr_a_2 = log(c(36.995554, 28.781157, 26.855926, 21.380120, 18.085347, 14.344999, 12.804484, 10.847895,  8.342990,  7.128675,  5.889213))
lwr_a_2 = log(c(13.499309, 18.322432, 16.135699, 13.693835, 13.052980, 10.238247,  9.485332, 8.496673,  7.162111,  6.065271,  5.084364))

a_3 = log(c(5.7491437, 4.4112678, 3.1812435, 2.0176850, 1.7754460, 1.2413668, 0.7529428, 0.6461570, 0.5142498, 0.2975216, 0.2523155))
upr_a_3 = log(c(27.413751, 21.731770, 17.981498, 16.455525, 13.412496, 10.758002,  9.278318, 7.789154,  6.368131,  5.330383,  4.513999))
lwr_a_3 = log(c(15.001147, 13.192837,  9.214767, 10.371923,  9.787943,  8.381641,  7.193801, 5.928792,  5.303227,  4.563946,  3.893399))

a_4 = log(c(12.150548,  8.740784,  6.966197,  5.967663,  4.866421,  3.768882,  3.295220, 2.676551,  2.081746,  1.773656,  1.281582))
upr_a_4 = log(c(20.240369, 16.536762, 13.415671, 11.212248 , 9.735023 , 7.980206 , 6.721673, 5.693633 , 4.896842 , 4.149252 , 3.389003))
lwr_a_4 = log(c(8.972988, 9.191921, 9.862814, 7.062207, 6.077725, 6.079935, 5.119478, 4.701061, 3.760582, 3.218139, 2.886461))

a_5 = log(c(16.637112, 13.367014, 10.862385,  8.775448,  7.699000,  5.989279,  4.942867, 4.133590,  3.248508,  2.647209,  2.144467))
upr_a_5 = log(c(13.491025, 11.649717, 10.139834,  8.041631,  7.080033,  6.176584,  5.145783, 4.320835,  3.701234,  3.127964,  2.628493))
lwr_a_5 = log(c(5.027620, 4.163040, 4.875221, 4.486690, 5.291557, 4.691149, 4.382963, 3.540929, 3.052629, 2.616750, 2.168078))


# logistic
a_1 = log(c(7.521236, 6.513735, 5.855265, 4.480503, 3.515277, 2.887980, 2.712908, 2.175850, 2.029223, 1.625459, 1.393791))
upr_a_1 = log(c(45.963396, 39.898675, 34.588164, 29.248617, 24.582946, 21.531796, 16.191414, 13.024239, 10.879340,  9.456153,  7.890636))
lwr_a_1 = log(c(21.597893, 19.750207, 18.619151, 16.420467, 17.362689, 14.312186, 12.117661, 11.038719,  8.934940,  7.742766,  6.539853))

a_2 = log(c(7.4276447, 5.4397550, 4.1394819, 2.9421735, 2.7407782, 1.8004912, 1.3533648, 0.8567149, 0.7758452, 0.5610944, 0.3832534))
upr_a_2 = log(c(36.995554, 28.781157, 26.855926, 21.380120, 18.085347, 14.344999, 12.804484, 10.847895,  8.342990,  7.128675,  5.889213))
lwr_a_2 = log(c(13.499309, 18.322432, 16.135699, 13.693835, 13.052980, 10.238247,  9.485332, 8.496673,  7.162111,  6.065271,  5.084364))

a_3 = log(c(14.810730, 12.755749,  9.824852,  8.130171,  6.622142,  5.739297,  4.905051, 4.030008,  3.271210,  2.677333,  2.277649))
upr_a_3 = log(c(27.413751, 21.731770, 17.981498, 16.455525, 13.412496, 10.758002,  9.278318, 7.789154,  6.368131,  5.330383,  4.513999))
lwr_a_3 = log(c(15.001147, 13.192837,  9.214767, 10.371923,  9.787943,  8.381641,  7.193801, 5.928792,  5.303227,  4.563946,  3.893399))

a_4 = log(c(19.747507, 16.419143, 13.488511, 11.369476,  9.819394,  8.143942,  6.689358, 5.500113,  4.558785,  3.725397,  3.102043))
upr_a_4 = log(c(20.240369, 16.536762, 13.415671, 11.212248 , 9.735023 , 7.980206 , 6.721673, 5.693633 , 4.896842 , 4.149252 , 3.389003))
lwr_a_4 = log(c(8.972988, 9.191921, 9.862814, 7.062207, 6.077725, 6.079935, 5.119478, 4.701061, 3.760582, 3.218139, 2.886461))

a_5 = log(c(22.149754, 19.750204, 15.566917, 13.677565, 11.624458,  9.291232,  7.526209, 6.338509,  5.276280,  4.420989,  3.597338))
upr_a_5 = log(c(13.491025, 11.649717, 10.139834,  8.041631,  7.080033,  6.176584,  5.145783, 4.320835,  3.701234,  3.127964,  2.628493))
lwr_a_5 = log(c(5.027620, 4.163040, 4.875221, 4.486690, 5.291557, 4.691149, 4.382963, 3.540929, 3.052629, 2.616750, 2.168078))





###########  simulation 2: transferred fairness vs propensity score model  ############


Col = 10

unfairness_cc = matrix(0, nrow=11, ncol=100)
unfairness_true = matrix(0, nrow=11, ncol=100)
unfairness_logistic = matrix(0, nrow=11, ncol=100)
unfairness_logistic_2 = matrix(0, nrow=11, ncol=100)
unfairness_rf = matrix(0, nrow=11, ncol=100)
unfairness_svm = matrix(0, nrow=11, ncol=100)
unfairness_xgb = matrix(0, nrow=11, ncol=100)

we_cc = matrix(0, nrow=11, ncol=100)
we_true = matrix(0, nrow=11, ncol=100)
we_logistic = matrix(0, nrow=11, ncol=100)
we_logistic_2 = matrix(0, nrow=11, ncol=100)
we_rf = matrix(0, nrow=11, ncol=100)
we_svm = matrix(0, nrow=11, ncol=100)
we_xgb = matrix(0, nrow=11, ncol=100)


for(i in 1:1){
  
  # let n_0 = n_1
  n_1 = round((10^(2+(i-1)*0.1))/0.3)
  n_0 = round(3*n_1/7)
  n = n_0 + n_1
  
  for(j in 1:2){
    
    # X is mixed gaussian
    A = c(rep(0,n_0), rep(1,n_1))
    #B = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    #C = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    #X[,6:10] = (X[,6:10] + X[,1:5])/sqrt(2)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    #y = abs(sin((X%*%t(t(beta)))^2 + epsilon/5))
    #y = 1/(1 + exp(5*(1-(X%*%t(t(beta)))^2) + epsilon/5))
    #A = 1*(apply(abs(X[,1:5]),1,mean) > apply(abs(X[,6:10]),1,mean))
    #A = 1*(X[,10]>1.5)
    
    # the last 5 column are missing with 2-pattern mechanism
    #1*(X[,1:5] > 0.5)
    
    missing_index = runif(n)
    Threshold = 1/(1 + exp(-3 + 1*(apply((X[,1:5])^3,1,mean))))
    #Threshold = 1/(1 + exp(1 - 2*(apply((X[,1:5]+1)^4,1,mean))))
    #Threshold = 1/(1 + exp(1 - 2*((1 + X[,1]) * apply(X[,1:5],1,mean))))
    #Threshold =  abs(X[,10])
    missing = 1*(missing_index > Threshold)
    
    # testing
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    #X_test[,6:10] = (X_test[,6:10] + X_test[,1:5])/sqrt(2)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    #A_test = 1*(apply(abs(X_test[,1:5]),1,mean) > apply(abs(X_test[,6:10]),1,mean))
    #A_test = 1*(X_test[,10]>0)
    
    # CCA
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
    #w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:length(w_true)){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    #w_true = 1/Threshold
    #for(m in 1:length(w_true)){
    #  w_true = w_true/(sum(w_true[which(missing == 0 & A = A[m])])/length(w_true[which(missing == 0 & A = A[m])]))
    #}
    we_cc[i,j] = mean((w_true-1)^2)
    
    
    
    ## correctly specified logit
    #data_ipw_2 = data.frame(X[,1]*X[,2]*X[,3]*X[,4]*X[,5], X[,1:5], 1-missing)
    data_ipw_2 = data.frame((X[,1:5])^3, 1-missing)
    colnames(data_ipw_2) = c('V1','V2','V3','V4','V5','miss')
    
    #weight_2 <- ipwpoint(
    #  exposure = miss,
    #  family = "binomial",
    #  link = "logit",
    #  numerator = ~ 1,
    #  denominator = ~ ., 
    #  data = data_ipw_2)
    
    ##w_logistic_2 = weight_2$ipw.weights
    #w_logistic_2 = (1+exp(-weight_2$den.mod[[1]][1] - as.matrix(data_ipw_2[,1:5]) %*% as.matrix(weight_2$den.mod[[1]][2:6])))
    #w_logistic_2 = w_logistic_2/(sum(w_logistic_2[which(missing == 0)])/length(w_logistic_2[which(missing == 0)]))
    #we_logistic_2[i,j] = mean((w_true-w_logistic_2)^2)
    

    
    D_0 = data_ipw_2[which(A==0),]
    D_1 = data_ipw_2[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = D_0)
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = D_1)
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw_2[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw_2[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    # merge to w
    
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
    
    
    
    # logistic
    data_ipw = data.frame(X[,1:5], 1-missing)
    colnames(data_ipw) = c('V1','V2','V3','V4','V5','miss')
    
    #weight <- ipwpoint(
    #  exposure = miss,
    #  family = "binomial",
    #  link = "logit",
    #  numerator = ~ 1,
    #  denominator = ~ ., 
    #  data = abs(data_ipw))
    
    ##w_logistic = weight$ipw.weights
    #w_logistic = (1+exp(-weight$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight$den.mod[[1]][2:6])))
    #w_logistic = w_logistic/(sum(w_logistic[which(missing == 0)])/length(w_logistic[which(missing == 0)]))
    #we_logistic[i,j] = mean((w_true-w_logistic)^2)
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = D_0)
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = D_1)
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    # merge to w
    
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
    
    
    
    ## random forest
    #rf <- randomForest(
    #  miss ~ .,
    #  data = data_ipw
    #)
    #w_rf <- 1/(rf$predicted)
    #w_rf[which(w_rf > 100)] = 100
    #w_rf[which(w_rf < 0.01)] = 0.01
    #w_rf = w_rf/(sum(w_rf[which(missing == 0)])/length(w_rf[which(missing == 0)]))
    
    
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
    we_rf[i,j] = mean((w_true-w_rf)^2)
    
    #if(mean((w_true-w_rf)^2) > 10000){
    #  break
    #  print((sum(w_rf[which(missing == 0)])/length(w_rf[which(missing == 0)])))
    #  print(max(w_true))
    #  print(max(w_rf))
    #}
    
    # svm
    #svm = svm(miss ~ ., data = data_ipw, kernel = "radial", cost = 10, scale = FALSE)
    #w_svm <- 1/(svm$fitted)
    #w_svm[which(w_svm > 100)] = 100
    #w_svm[which(w_svm < 0.01)] = 0.01
    #w_svm = w_svm/(sum(w_svm[which(missing == 0)])/length(w_svm[which(missing == 0)]))
    #we_svm[i,j] = mean((w_true-w_svm)^2)
    
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
    we_svm[i,j] = mean((w_true-w_svm)^2)
    
    
    
    # XGboosting
    #xgb <- xgboost(data = matrix(as.numeric(as.matrix(data_ipw[,1:5])),nrow = n), label = as.matrix(data_ipw$miss), max_depth = 10, eta = 1, nthread = 2, nrounds = 3,"binary:logistic")
    #w_xgb <- 1/(predict(xgb, matrix(as.numeric(as.matrix(data_ipw[,1:5])),nrow = n)))
    #w_xgb[which(w_xgb > 100)] = 100
    #w_xgb[which(w_xgb < 0.01)] = 0.01
    #w_xgb = w_xgb/(sum(w_xgb[which(missing == 0)])/length(w_xgb[which(missing == 0)]))
    #we_xgb[i,j] = mean((w_true-w_xgb)^2)
    
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
    we_xgb[i,j] = mean((w_true-w_xgb)^2)
    
    
    
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
    
    # logistic
    rf_logistic <- ranger(y_CC ~ ., data = Z, case.weights = 1)
    
    pred_train = predict(rf_logistic, data = Z)$predictions
    MSE_0_logistic_train = sum((1-A_CC)*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_logistic_train = sum(A_CC*w_logistic[which(missing == 0)]*abs(pred_train - y_CC))/sum(A_CC)
    
    pred = predict(rf_logistic, data = data.frame(X_test, y_test))$predictions
    MSE_0_logistic = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_logistic = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    # logistic_2
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
    
    
    
    print(sum(which(missing == 0) < n_0+1))
    print(sum(which(missing == 0) >= n_0+1))
    
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_true[i,j] = abs( abs(MSE_0_true - MSE_1_true) - abs(MSE_0_true_train - MSE_1_true_train) )
    unfairness_logistic[i,j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    unfairness_logistic_2[i,j] = abs( abs(MSE_0_logistic_2 - MSE_1_logistic_2) - abs(MSE_0_logistic_2_train - MSE_1_logistic_2_train) )
    unfairness_rf[i,j] = abs( abs(MSE_0_rf - MSE_1_rf) - abs(MSE_0_rf_train - MSE_1_rf_train) )
    unfairness_svm[i,j] = abs( abs(MSE_0_svm - MSE_1_svm) - abs(MSE_0_svm_train - MSE_1_svm_train) )
    unfairness_xgb[i,j] = abs( abs(MSE_0_xgb - MSE_1_xgb) - abs(MSE_0_xgb_train - MSE_1_xgb_train) )
    
  }
}

print(sum(which(missing == 0) < n_0+1))
print(sum(which(missing == 0) >= n_0+1))

nintycover_up <- function(x){
  up = sort(x)[95]
  return(up)
}

nintycover_low <- function(x){
  low = sort(x)[6]
  return(low)
}

print('cc:')
apply(unfairness_cc,1,mean)
apply(unfairness_cc,1,sd)
apply(unfairness_cc,1,nintycover_up)
apply(unfairness_cc,1,nintycover_low)

print('true:')
apply(unfairness_true,1,mean)
apply(unfairness_true,1,sd)
apply(unfairness_true,1,nintycover_up)
apply(unfairness_true,1,nintycover_low)

print('logistic:')
apply(unfairness_logistic,1,mean)
apply(unfairness_logistic,1,sd)
apply(unfairness_logistic,1,nintycover_up)
apply(unfairness_logistic,1,nintycover_low)



print('logistic_2:')
apply(unfairness_logistic_2,1,mean)
apply(unfairness_logistic_2,1,sd)
apply(unfairness_logistic_2,1,nintycover_up)
apply(unfairness_logistic_2,1,nintycover_low)



print('rf:')
apply(unfairness_rf,1,mean)
apply(unfairness_rf,1,sd)
apply(unfairness_rf,1,nintycover_up)
apply(unfairness_rf,1,nintycover_low)


print('svm:')
apply(unfairness_svm,1,mean)
apply(unfairness_svm,1,sd)
apply(unfairness_svm,1,nintycover_up)
apply(unfairness_svm,1,nintycover_low)


print('xgb:')
apply(unfairness_xgb,1,mean)
apply(unfairness_xgb,1,sd)
apply(unfairness_xgb,1,nintycover_up)
apply(unfairness_xgb,1,nintycover_low)


apply(we_cc,1,mean)
apply(we_logistic,1,mean)
apply(we_logistic_2,1,mean)
apply(we_rf,1,mean)
apply(we_svm,1,mean)
apply(we_xgb,1,mean)






# plot

PLOT = FALSE

epoch = seq(2.7,3.7,length.out = 11)

a_1 = log(c(0.7059968, 0.6809577, 0.6579272, 0.6619428, 0.6173124, 0.5772237, 0.5636061, 0.5225703, 0.4992953, 0.4976452, 0.4675314))/(log(10))
upr_a_1 = log(c(72.48522, 61.52518, 57.93687, 56.28220, 52.75719, 50.09074, 43.52766, 41.72784, 31.28183, 29.98080, 27.99507))
lwr_a_1 = log(c(7.823412, 11.682837,  5.077613, 11.406248, 13.406470,  7.252252, 13.430611, 14.589470, 13.482124, 11.687027, 13.215550))

a_2 = log(c(0.33102210, 0.31439929, 0.28743832, 0.26068126, 0.22694654, 0.18723539, 0.16410088, 0.14503833, 0.13048584, 0.10579618, 0.09404877))/(log(10))
upr_a_2 = log(c(53.02950, 48.02452, 40.13991, 46.27366, 36.73458, 38.43635, 25.80111, 28.52783, 20.94729, 18.16833, 17.06797))
lwr_a_2 = log(c(2.4727579, 5.0144548, 1.5447426, 6.5169961, 3.2143061, 2.0976128, 2.2729125, 2.2950593, 3.8117641, 0.8249885, 2.8302670))

a_3 = log(c(0.5425409, 0.5202433, 0.5053872, 0.4883772, 0.4662399, 0.4312914, 0.4212015, 0.3848392, 0.3687369, 0.3742689, 0.3480729))/(log(10))
upr_a_3 = log(c(50.27956, 47.94873, 44.93054, 43.92273, 35.56307, 39.97922, 29.46152, 26.45893, 19.67960, 17.27654, 16.03470))
lwr_a_3 = log(c(3.4562187, 4.9035897, 1.4429926, 2.1827835, 3.8815763, 2.2822505, 1.2743563, 2.2179986, 2.7297301, 0.9924345, 2.6037512))

a_4 = log(c(0.35524553, 0.30842060, 0.28582187, 0.25573951, 0.24576987, 0.20284237, 0.17658014, 0.15225268, 0.12838249, 0.10227010, 0.09531036))/(log(10))
upr_a_4 = log(c(219.08811, 199.91245, 148.64987, 160.85542, 109.95266,  96.75443, 104.96755, 90.38365,  77.47091,  79.70534,  69.19788))
lwr_a_4 = log(c(50.38507, 53.91970, 46.76332, 56.23302, 63.85971, 56.66809, 54.02957, 52.04395, 56.61413, 52.01585, 48.58413))

a_5 = log(c(0.34934165, 0.32274162, 0.28779405, 0.26324306, 0.24855224, 0.19354208, 0.17036844, 0.14541657, 0.12070052, 0.09950525, 0.09229841))/(log(10))
upr_a_5 = log(c(231.91775, 163.66869, 191.12165, 138.90176, 111.51234, 115.84999,  98.80405, 93.88906,  99.70461,  94.54270,  81.60279))
lwr_a_5 = log(c(62.36478, 63.95609, 54.10460, 63.83764, 60.98497, 59.81483, 61.88678, 62.42938, 56.55635, 58.06949, 56.90947))

a_6 = log(c(0.6162411, 0.5732753, 0.5533419, 0.5513584 ,0.5237490, 0.4751764, 0.4549465, 0.4277667, 0.4085651, 0.4209032, 0.4011798))/(log(10))
upr_a_6 = log(c(231.91775, 163.66869, 191.12165, 138.90176, 111.51234, 115.84999,  98.80405, 93.88906,  99.70461,  94.54270,  81.60279))
lwr_a_6 = log(c(62.36478, 63.95609, 54.10460, 63.83764, 60.98497, 59.81483, 61.88678, 62.42938, 56.55635, 58.06949, 56.90947))


b = log(c(0.6930309, 0.6568415, 0.6277096, 0.6088576, 0.5761324, 0.5224942, 0.5030001, 0.4537817, 0.4165385, 0.3970182, 0.3595048))/(log(10))
upr_b = log(c(79.67134, 55.96888, 60.50155, 58.67109, 54.28017, 58.16584, 56.17252, 60.81402, 66.86630, 59.65105, 55.35201))
lwr_b = log(c(8.779674,  6.155561, 13.276236,  9.917463, 22.259108, 21.553371, 20.505383, 25.195179, 26.834475, 28.123884, 29.715004))



if(PLOT){
  
  
  data = data.frame(epoch,a_1,a_2,a_3,a_4,a_5,a_6,b,'lwr_a_1'= lwr_a_1,'upr_a_1'= upr_a_1, 'lwr_a_2'= lwr_a_2,'upr_a_2'= upr_a_2, 'lwr_a_3'= lwr_a_3,'upr_a_3'= upr_a_3, 'lwr_a_4'= lwr_a_4,'upr_a_4'= upr_a_4, 'lwr_a_5'= lwr_a_5,'upr_a_5'= upr_a_5,'lwr_a_6'= lwr_a_6,'upr_a_6'= upr_a_6,'lwr_b'= lwr_b,'upr_b'= upr_b,stringsAsFactors = F)
  (plot <- ggplot(data, aes(epoch, a_1))+
      geom_point(data, mapping = aes(epoch, a_1, color = "Unweighted"),size = 2, shape = 17)+ 
      geom_point(data, mapping = aes(epoch, a_2, color = "True"),size = 2, shape = 12)+ 
      geom_point(data, mapping = aes(epoch, a_3, color = "Logit-incorrect"),size = 2, shape = 13)+ 
      geom_point(data, mapping = aes(epoch, a_4, color = "Logit-correct"),size = 2, shape = 13)+ 
      geom_point(data, mapping = aes(epoch, a_5, color = "RF"),size = 2, shape = 14)+ 
      geom_point(data, mapping = aes(epoch, a_6, color = "SVM"),size = 2, shape = 16)+ 
      geom_point(data, mapping = aes(epoch, b, color = "XGB"),size = 2, shape = 15)+
      geom_line(data, mapping = aes(epoch, a_1, color = "Unweighted"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_2, color = "True"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_3, color = "Logit-incorrect"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_4, color = "Logit-correct"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_5, color = "RF"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_6, color = "SVM"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, b, color = "XGB"),size = 0.8)+
      scale_color_manual(values=c('#a6cee3', "#1f78b4", "#33a02c", "#b2df8a", "#fb9a99", '#e31a1c','#fdbf6f'))+
      geom_ribbon(data=data,aes(ymin=lwr_a_1,ymax=upr_a_1),alpha=0.0,fill = '#a6cee3')+
      geom_ribbon(data=data,aes(ymin=lwr_a_2,ymax=upr_a_2),alpha=0.0,fill = '#e31a1c')+
      geom_ribbon(data=data,aes(ymin=lwr_a_3,ymax=upr_a_3),alpha=0.0,fill = '#fb9a99')+
      geom_ribbon(data=data,aes(ymin=lwr_a_4,ymax=upr_a_4),alpha=0.0,fill = '#33a02c')+
      geom_ribbon(data=data,aes(ymin=lwr_a_5,ymax=upr_a_5),alpha=0.0,fill = '#b2df8a')+
      geom_ribbon(data=data,aes(ymin=lwr_a_5,ymax=upr_a_6),alpha=0.0,fill = '#fdbf6f')+
      geom_ribbon(data=data,aes(ymin=lwr_b,ymax=upr_b),alpha=0.0,fill = '#1f78b4')+
      #geom_hline(yintercept=0.914, linetype="dashed", color = "black")+
      #annotate("text", x = 18, y = 0.92, label = "91.4%")+
      coord_cartesian(ylim = c(-1.2, 0.5))+
      labs(color="", x= expression("log"['10']*" n"), y = expression('log'['10']*' Bias') #expression('log'['10']*' |'*Delta*' - '*widehat(Delta)*'|')
           #x= expression("log n"),
           #y= expression('log |'*Delta*' - '*widehat(Delta)*'|')
           )+
      #ggtitle(expression("Weight estimation MSE with different "*hat(omega)))+
      theme(legend.position= c(0.85,0.8),axis.text=element_text(size=12),panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"),plot.title = element_text(hjust = 0.5,size = 14),legend.title = element_text(size = 9), legend.text = element_text(size = 7)))
  
}

# weights

a_1 = log(c(1.682557 ,1.704178 ,1.606904, 1.590078, 1.605647 ,1.613720 ,1.566800, 1.558047, 1.613466, 1.622980 ,1.622770))/(log(10))
upr_a_1 = log(c(72.48522, 61.52518, 57.93687, 56.28220, 52.75719, 50.09074, 43.52766, 41.72784, 31.28183, 29.98080, 27.99507))
lwr_a_1 = log(c(7.823412, 11.682837,  5.077613, 11.406248, 13.406470,  7.252252, 13.430611, 14.589470, 13.482124, 11.687027, 13.215550))

a_2 = log(c(1.685457, 1.704040 ,1.604578, 1.592153, 1.605172 ,1.613614, 1.566122 ,1.558835, 1.612722 ,1.622241, 1.622508))/(log(10))
upr_a_2 = log(c(53.02950, 48.02452, 40.13991, 46.27366, 36.73458, 38.43635, 25.80111, 28.52783, 20.94729, 18.16833, 17.06797))
lwr_a_2 = log(c(2.4727579, 5.0144548, 1.5447426, 6.5169961, 3.2143061, 2.0976128, 2.2729125, 2.2950593, 3.8117641, 0.8249885, 2.8302670))

a_3 = log(c(0.12907955, 0.32764576, 0.17101433, 0.08699264, 0.07401455, 0.02862978, 0.01825177, 0.02006380, 0.01103121, 0.01120234, 0.01228811))/(log(10))
upr_a_3 = log(c(50.27956, 47.94873, 44.93054, 43.92273, 35.56307, 39.97922, 29.46152, 26.45893, 19.67960, 17.27654, 16.03470))
lwr_a_3 = log(c(3.4562187, 4.9035897, 1.4429926, 2.1827835, 3.8815763, 2.2822505, 1.2743563, 2.2179986, 2.7297301, 0.9924345, 2.6037512))

a_4 = log(c(0.8357512, 0.7617110, 0.7459388 ,0.6548172, 0.6236550, 0.6116204, 0.5640114, 0.5050466, 0.4916762, 0.5284401, 0.5377864))/(log(10))
upr_a_4 = log(c(219.08811, 199.91245, 148.64987, 160.85542, 109.95266,  96.75443, 104.96755, 90.38365,  77.47091,  79.70534,  69.19788))
lwr_a_4 = log(c(50.38507, 53.91970, 46.76332, 56.23302, 63.85971, 56.66809, 54.02957, 52.04395, 56.61413, 52.01585, 48.58413))

a_5 = log(c(14.14926, 14.65110, 15.88119, 14.85778, 14.90645, 14.06969, 14.88116, 15.01233, 14.75905, 14.66846, 13.88822))/(log(10))
upr_a_5 = log(c(231.91775, 163.66869, 191.12165, 138.90176, 111.51234, 115.84999,  98.80405, 93.88906,  99.70461,  94.54270,  81.60279))
lwr_a_5 = log(c(62.36478, 63.95609, 54.10460, 63.83764, 60.98497, 59.81483, 61.88678, 62.42938, 56.55635, 58.06949, 56.90947))

a_6 = log(c(665.55205, 586.25628, 543.39004 ,322.34078, 274.12567, 260.85679, 171.32460, 126.08104, 100.78240, 82.93427,  53.56319))/(log(10))
upr_a_6 = log(c(231.91775, 163.66869, 191.12165, 138.90176, 111.51234, 115.84999,  98.80405, 93.88906,  99.70461,  94.54270,  81.60279))
lwr_a_6 = log(c(62.36478, 63.95609, 54.10460, 63.83764, 60.98497, 59.81483, 61.88678, 62.42938, 56.55635, 58.06949, 56.90947))


b = log(c(12.409734 ,11.156089, 11.576919,  9.341760 , 9.353469, 10.236030,  8.587973, 8.045398,  7.014965,  6.623984,  6.620397))/(log(10))
upr_b = log(c(79.67134, 55.96888, 60.50155, 58.67109, 54.28017, 58.16584, 56.17252, 60.81402, 66.86630, 59.65105, 55.35201))
lwr_b = log(c(8.779674,  6.155561, 13.276236,  9.917463, 22.259108, 21.553371, 20.505383, 25.195179, 26.834475, 28.123884, 29.715004))






###########  simulation 3: effect of sample imbalance  ############

Col = 10

unfairness_cc = matrix(0, nrow=11, ncol=200)
unfairness_cc_train = matrix(0, nrow=11, ncol=200)
unfairness_cc_test = matrix(0, nrow=11, ncol=200)
unfairness_ipw = matrix(0, nrow=11, ncol=200)
unfairness_ipw_train = matrix(0, nrow=11, ncol=200)
unfairness_ipw_test = matrix(0, nrow=11, ncol=200)

for(i in 1:11){
  
  
  # missingness: 0.1/0.9
  # 1 = 0.1/0.9
  
  # lambda, alpha / 1-alpha = 0.1*lambda / 0.9+0.1*lambda
  #lambda = 1
  #alpha = lambda/4.1 #13 # 3*lambda/7 #(0.1*lambda)/(0.9+0.1*lambda)
  
  #n_1 = round((10^(2+(i-1)*0.2))/0.12) #0.12
  #n_0 = round(alpha*n_1)
  #n = n_0 + n_1
  
  
  lambda = 1 #1 3 5 7 9
  n = round(10^(3+(i-1)*0.2))
  n_0 = round(lambda*n/(1+lambda))
  n_1 = n - n_0
  
  for(j in 1:200){
    
    # X is mixed gaussian
    A = c(rep(0,n_0), rep(1,n_1))
    #B = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    #C = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-1) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+1)
    #beta = c(1,1,1,1,1,10,10,10,10,10)/10
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    #y = 1/(1 + exp(5*(1-(X%*%t(t(beta)))^2) + epsilon/5))
    #y = (X%*%t(t(beta)))^2 + epsilon/2
    #A = 1*(apply(abs(X[,1:5]),1,mean) > apply(abs(X[,6:10]),1,mean))
    #A = 1*(X[,10]>1.5)
    
    # the last 5 column are missing with 2-pattern mechanism
    
    missing_index = runif(n)
    Threshold = 1/(1 + exp(1 - 1*apply(X[,1:5],1,mean)))
    #Threshold = 0.5 + 0.2 * apply(X[,1:5],1,mean)
    #Threshold =  abs(X[,10])
    missing = 1*(missing_index > Threshold)
    
    # testing
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-1) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+1)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    #y_test = (X_test%*%t(t(beta)))^2 + rnorm(100000)/2
    #A_test = 1*(apply(abs(X_test[,1:5]),1,mean) > apply(abs(X_test[,6:10]),1,mean))
    #A_test = 1*(X_test[,10]>0)
    
    # CCA
    X_CC = X[which(missing == 0),]
    y_CC = y[which(missing == 0),]
    A_CC = A[which(missing == 0)]
    
    # fit random forest and check transferred fairness
    Z = data.frame(X_CC, y_CC)
    rf_cc<- ranger(y_CC ~ ., data = Z)
    
    pred_train = predict(rf_cc, data = Z)$predictions
    #pred_train[pred_train>1] = 1
    #pred_train[pred_train<0] = 0
    MSE_0_cc_train = sum((1-A_CC)*abs(pred_train - y_CC))/sum(1 - A_CC)
    MSE_1_cc_train = sum(A_CC*abs(pred_train - y_CC))/sum(A_CC)
    
    pred = predict(rf_cc, data = data.frame(X_test, y_test))$predictions
    MSE_0_cc = sum((1-A_test)*abs(pred - y_test))/sum(1 - A_test)
    MSE_1_cc = sum(A_test*abs(pred - y_test))/sum(A_test)
    
    print(c(MSE_0_cc_train, MSE_1_cc_train))
    print(c(MSE_0_cc, MSE_1_cc))
    
    # propensity score model
    
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    #weight <- ipwpoint(
    #  exposure = miss,
    #  family = "binomial",
    #  link = "logit",
    #  numerator = ~ 1,
    #  denominator = ~ ., 
    #  data = data_ipw)
    
    ##w = weight$ipw.weights
    #w = (1+exp(-weight$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight$den.mod[[1]][2:6])))
    #w = w/(sum(w[which(missing == 0)])/length(w[which(missing == 0)]))
    #Threshold[which(Threshold > 1)] = 1
    #Threshold[which(Threshold < 0)] = 0.01
    #w_true = 1/Threshold
    #w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==1),])
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    # merge to w
    
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
    
    print(sum(which(missing == 0) < n_0+1))
    print(sum(which(missing == 0) >= n_0+1))
    
    print(c(MSE_0_ipw_train, MSE_1_ipw_train))
    print(c(MSE_0_ipw, MSE_1_ipw))
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_cc_train[i,j] = abs(MSE_0_cc_train - MSE_1_cc_train) 
    unfairness_cc_test[i,j] = abs(MSE_0_cc - MSE_1_cc)
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    unfairness_ipw_train[i,j] = abs(MSE_0_ipw_train - MSE_1_ipw_train) 
    unfairness_ipw_test[i,j] = abs(MSE_0_ipw - MSE_1_ipw)
    
  }
  
}

print(sum(which(missing == 0) < n_0+1))
print(sum(which(missing == 0) >= n_0+1))

nintycover_up <- function(x){
  up = sort(x)[190]
  return(up)
}


nintycover_low <- function(x){
  low = sort(x)[11]
  return(low)
}

print('cc:')
apply(unfairness_cc,1,mean)
apply(unfairness_cc_train,1,mean)
apply(unfairness_cc_test,1,mean)
apply(unfairness_cc,1,sd)
apply(unfairness_cc,1,nintycover_up)
apply(unfairness_cc,1,nintycover_low)


print('ipw:')
apply(unfairness_ipw,1,mean)
apply(unfairness_ipw_train,1,mean)
apply(unfairness_ipw_test,1,mean)
apply(unfairness_ipw,1,sd)
apply(unfairness_ipw,1,nintycover_up)
apply(unfairness_ipw,1,nintycover_low)






# plot

PLOT = FALSE

epoch = seq(3,5,length.out = 11)


a_1 = log(c(0.9432525, 0.8895239, 0.7932530, 0.7386128, 0.6723830, 0.6078001, 0.5605029,
            0.4993775, 0.4442940, 0.4010261, 0.3538143))/(log(10))
upr_a_1 = log(c(15.029935, 10.807682,  8.185903 , 5.269213,  5.766114 , 3.134290 , 2.648462, 1.709895,  1.453334,  1.184222 , 1.053767))
lwr_a_1 = log(c(0.85146016, 0.22052057, 0.44007206, 0.66038537, 0.23232218, 0.20299654, 0.07501393, 0.10665649, 0.08435149, 0.05531432, 0.02617201))

a_2 = log(c(1.5892535, 1.5326247, 1.4024828, 1.3056739, 1.1754619, 1.1151052, 0.9972386,
            0.8982654, 0.8103588, 0.7281033, 0.6419015))/(log(10))
upr_a_2 = log(c(18.343392, 15.938511 ,11.618273, 11.005465,  8.653228 , 7.296526,  5.164994, 4.556073,  3.307630 , 2.965694,  2.264291))
lwr_a_2 = log(c(1.543517, 2.883210, 5.194162, 3.326425, 1.982859, 2.419667, 2.256115, 2.389251, 1.731922, 1.336899, 1.258668))

a_3 = log(c(1.8314059, 1.8022352, 1.7814981, 1.6312748, 1.4941845, 1.3464703, 1.2398914,
            1.1240800, 0.9951526, 0.8925909, 0.7866654))/(log(10))
upr_a_3 = log(c(30.057868, 21.647154, 19.006570, 14.518360, 12.782274,  9.764989,  8.184639, 6.975948,  5.369472,  4.350427,  3.478778))
lwr_a_3 = log(c(8.835933, 10.362062,  8.849643,  8.007693,  7.695630,  5.301883 , 5.398585, 4.426971,  3.835406,  3.106338 , 2.592057))

a_4 = log(c(2.0636901, 1.9737087, 1.8881385, 1.7563959, 1.6704965, 1.5635669, 1.3895495,
            1.2876168, 1.1408125, 1.0169687, 0.9023562))/(log(10))
upr_a_4 = log(c(32.564544, 27.684185, 22.976905, 18.071926, 14.383730, 11.889813,  9.240028, 7.887040,  6.301626,  5.338650,  4.204753))
lwr_a_4 = log(c(14.444362, 16.261089, 12.516046, 10.849751,  8.667233,  7.635550,  6.255691, 5.713058,  4.922015,  3.935741,  3.383264))

a_5 = log(c(2.289121, 2.157638, 2.092825, 2.004089, 1.834056, 1.647969, 1.540264, 1.401974,
           1.257175, 1.140227, 1.007892))/(log(10))
upr_a_5 = log(c(33.161073, 31.159738, 23.646929, 20.430577, 15.799052, 12.700304, 10.577175, 8.756554,  7.445108,  5.649539,  4.893585))
lwr_a_5 = log(c(18.187977, 15.906324, 15.222947, 12.318759, 11.544354,  8.377073, 8.041419, 6.365448,  5.513882 , 4.675852,  3.893852))



b = log(c(67.480712, 62.811315, 46.447439, 37.625635, 31.154704, 26.070394, 21.901551, 17.315131, 14.260627, 11.296347,  8.895828))
upr_b = log(c(131.147469, 104.411247,  73.579621,  61.800014,  42.395069,  35.145717, 25.459421,  19.848760,  15.364889,  12.078054,   9.443137))
lwr_b = log(c(4.727696, 17.230152, 11.529076, 18.241311, 17.758189, 19.913519, 16.755316, 14.341901, 12.910361, 10.354802,  8.460403))

if(PLOT){
  
  # expression('n'['max']*'/n'['min']*' = 1')
  data = data.frame(epoch,a_1,a_2,a_3,a_4,a_5,b,'lwr_a_1'= lwr_a_1,'upr_a_1'= upr_a_1, 'lwr_a_2'= lwr_a_2,'upr_a_2'= upr_a_2, 'lwr_a_3'= lwr_a_3,'upr_a_3'= upr_a_3, 'lwr_a_4'= lwr_a_4,'upr_a_4'= upr_a_4, 'lwr_a_5'= lwr_a_5,'upr_a_5'= upr_a_5,'lwr_b'= lwr_b,'upr_b'= upr_b,stringsAsFactors = F)
  (plot <- ggplot(data, aes(epoch, a_1))+
      geom_point(data, mapping = aes(epoch, a_1, color = "ratio = 1"),size = 2, shape = 17)+ 
      geom_point(data, mapping = aes(epoch, a_2, color = "ratio = 3"),size = 2, shape = 12)+ 
      geom_point(data, mapping = aes(epoch, a_3, color = "ratio = 5"),size = 2, shape = 13)+ 
      geom_point(data, mapping = aes(epoch, a_4, color = "ratio = 7"),size = 2, shape = 14)+ 
      geom_point(data, mapping = aes(epoch, a_5, color = "ratio = 9"),size = 2, shape = 16)+ 
      #geom_point(data, mapping = aes(epoch, b, color = "CCA"),size = 2, shape = 15)+
      geom_line(data, mapping = aes(epoch, a_1, color = "ratio = 1"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_2, color = "ratio = 3"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_3, color = "ratio = 5"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_4, color = "ratio = 7"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_5, color = "ratio = 9"),size = 0.8)+
      #geom_line(data, mapping = aes(epoch, b, color = "CCA"))+
      scale_color_manual(values=c('#a6cee3', "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99"))+
      geom_ribbon(data=data,aes(ymin=lwr_a_1,ymax=upr_a_1),alpha=0.0,fill = '#a6cee3')+
      geom_ribbon(data=data,aes(ymin=lwr_a_2,ymax=upr_a_2),alpha=0.0,fill = '#1f78b4')+
      geom_ribbon(data=data,aes(ymin=lwr_a_3,ymax=upr_a_3),alpha=0.0,fill = '#b2df8a')+
      geom_ribbon(data=data,aes(ymin=lwr_a_4,ymax=upr_a_4),alpha=0.0,fill = '#33a02c')+
      geom_ribbon(data=data,aes(ymin=lwr_a_5,ymax=upr_a_5),alpha=0.0,fill = '#fb9a99')+
      #geom_ribbon(data=data,aes(ymin=lwr_b,ymax=upr_b),alpha=0.2,fill = 'black')+
      #geom_hline(yintercept=0.914, linetype="dashed", color = "black")+
      #annotate("text", x = 18, y = 0.92, label = "91.4%")+
      coord_cartesian(ylim = c(-0.5, 0.8))+
      labs(color="",x= expression("log"['10']*" n"), y = expression('log'['10']*' Bias') #expression('log'['10']*' |'*Delta*' - '*widehat(Delta)*'|')
           #x= expression("log n"),
           #y= expression('log |'*Delta*' - '*widehat(Delta)*'|')
      )+
      #ggtitle("Fairness guarantees with different sample imbalance")+ #with different sample imbalance
      theme(legend.position= c(0.85,0.8),axis.text=element_text(size=12),panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"),plot.title = element_text(hjust = 0.5,size = 14),legend.text = element_text(size = 7)))
  
}






###########  simulation 4: CCA when domains are different  ##############



Col = 10

unfairness_cc = matrix(0, nrow=11, ncol=200)
unfairness_ipw = matrix(0, nrow=11, ncol=200)

for(i in 1:11){
  
  
  # missingness: 0.1/0.9
  # 1 = 0.1/0.9
  
  # lambda, alpha / 1-alpha = 0.1*lambda / 0.9+0.1*lambda
  
  lambda = 1 
  n = round(10^(3+(i-1)*0.2))
  n_0 = round(2*n/(2+8.4))
  n_1 = n - n_0
  M = 0.5 #0.5 - 8
  
  for(j in 1:200){
    
    # X is mixed gaussian
    A = c(rep(0,n_0), rep(1,n_1))
    #B = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    #C = matrix(rnorm(n*Col,sd = 0.5),nrow = n)
    X = A*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)-M) + (1 - A)*(matrix(rnorm(n*Col,sd = 0.5),nrow = n)+M)
    beta = c(0.1,0.1,0.1,0.1,0.1,1,1,1,1,1)
    epsilon = rnorm(n)
    y = ((X%*%t(t(beta)))^2 + epsilon)
    #y = abs(sin((X%*%t(t(beta)))^2 + epsilon/5))
    #y = 1/(1 + exp(5*(1-(X%*%t(t(beta)))^2) + epsilon/5))
    #A = 1*(apply(abs(X[,1:5]),1,mean) > apply(abs(X[,6:10]),1,mean))
    #A = 1*(X[,10]>1.5)
    
    # the last 5 column are missing with 2-pattern mechanism
    
    missing_index = runif(n)
    #Threshold = 0.5 + 0.2 * apply(X[,1:5],1,mean)
    #Threshold = 1/(1 + exp(1 - 1*apply(X[,1:5],1,mean)))
    Threshold = 1/(1 + exp(4*(A-0.5)))
    #Threshold =  abs(X[,10])
    missing = 1*(missing_index > Threshold)
    
    # testing
    A_test = c(rep(0,50000), rep(1,50000))
    X_test = A_test*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)-M) + (1 - A_test)*(matrix(rnorm(100000*Col,sd = 0.5),nrow = 100000)+M)
    y_test = ((X_test%*%t(t(beta)))^2 + rnorm(100000))
    #y_test = 1/(1 + exp(5*(1-(X_test%*%t(t(beta)))^2) + rnorm(100000)/5))
    #A_test = 1*(apply(abs(X_test[,1:5]),1,mean) > apply(abs(X_test[,6:10]),1,mean))
    #A_test = 1*(X_test[,10]>0)
    
    # CCA
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
    
    print(c(MSE_0_cc_train, MSE_1_cc_train))
    print(c(MSE_0_cc, MSE_1_cc))
    
    # propensity score model
    
    # first 5 features are fully observed
    data_ipw = data.frame(X[,1:5],1-missing)
    colnames(data_ipw) = c(1:5,'miss')
    
    #weight <- ipwpoint(
    #  exposure = miss,
    #  family = "binomial",
    #  link = "logit",
    #  numerator = ~ 1,
    #  denominator = ~ ., 
    #  data = data_ipw)
    
    ##w = weight$ipw.weights
    #w = (1+exp(-weight$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight$den.mod[[1]][2:6])))
    #w = w/(sum(w[which(missing == 0)])/length(w[which(missing == 0)]))
    Threshold[which(Threshold > 1)] = 1
    Threshold[which(Threshold < 0)] = 0.01
    w_true = 1/Threshold
    w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==1),])
    
    w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_0$den.mod[[1]][2:6])))
    w_0 = w_0/(sum(w_0[which(D_0[,6]==1)])/length(w_0[which(D_0[,6]==1)]))
    
    w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:5]) %*% as.matrix(weight_1$den.mod[[1]][2:6])))
    w_1 = w_1/(sum(w_1[which(D_1[,6]==1)])/length(w_1[which(D_1[,6]==1)]))
    # merge to w
    
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
    
    print(sum(which(missing == 0) < n_0+1))
    print(sum(which(missing == 0) >= n_0+1))
    
    print(c(MSE_0_ipw_train, MSE_1_ipw_train))
    print(c(MSE_0_ipw, MSE_1_ipw))
    
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_ipw[i,j] = abs( abs(MSE_0_ipw - MSE_1_ipw) - abs(MSE_0_ipw_train - MSE_1_ipw_train) )
    
  }
  
}

print(sum(which(missing == 0) < n_0+1))
print(sum(which(missing == 0) >= n_0+1))

nintycover_up <- function(x){
  up = sort(x)[190]
  return(up)
}


nintycover_low <- function(x){
  low = sort(x)[11]
  return(low)
}

print('cc:')
apply(unfairness_cc,1,mean)
apply(unfairness_cc,1,sd)
apply(unfairness_cc,1,nintycover_up)
apply(unfairness_cc,1,nintycover_low)


print('ipw:')
apply(unfairness_ipw,1,mean)
apply(unfairness_ipw_train,1,mean)
apply(unfairness_ipw_test,1,mean)
apply(unfairness_ipw,1,sd)
apply(unfairness_ipw,1,nintycover_up)
apply(unfairness_ipw,1,nintycover_low)



# more missing data mechanisms?

# plot

PLOT = FALSE

epoch = seq(3,5,length.out = 11)

a_1 = log(c(0.20654621, 0.17542459, 0.16644587, 0.14299468, 0.13283142, 0.11738761,
            0.09354560, 0.09023772, 0.07569167, 0.06674080, 0.05807883))/(log(10))
upr_a_1 = log(c(4.7213045, 4.3913369, 3.3721303, 2.6951603, 2.3416215, 1.4288000, 1.0071951, 0.6573790, 0.4765861, 0.5514151, 0.4675769))
lwr_a_1 = log(c(0.07173156, 0.12717608, 0.22461785, 0.10799507, 0.05298796, 0.04783707, 0.02986662, 0.03641023, 0.01311635, 0.01473280, 0.00978960))

a_2 = log(c(0.4484641, 0.4033655, 0.3706931, 0.2882214, 0.2727075, 0.2383334, 0.2358465,
            0.1961162, 0.1792396, 0.1662794, 0.1432615))/(log(10))
upr_a_2 = log(c(25.024519, 24.798615, 24.298795, 17.031712, 13.434120, 12.181482,  8.039063, 6.732297,  6.069337,  4.448441,  3.629685))
lwr_a_2 = log(c(1.6714353, 0.3450367, 1.9730771, 1.7747307, 0.9977454, 2.5198891, 2.2968844, 1.7261820, 1.8965007, 1.7420321, 1.6549542))

a_3 = log(c(0.9495432, 0.7778759, 0.7260999, 0.5962137, 0.5538820, 0.5099053, 0.4584664,
            0.4120102, 0.3702675, 0.3404227, 0.3052626))/(log(10))
upr_a_3 = log(c(106.17876,  90.08212 , 70.58378 , 56.84606 , 55.64849 , 39.80154 , 31.73869, 27.69657 , 20.43282,  17.24226 , 14.58357))
lwr_a_3 = log(c(14.642980,  5.554315, 11.949506,  5.918086, 13.106791, 15.984914, 17.938678, 13.742332, 11.665893, 11.036419,  9.571227))

a_4 = log(c(1.7743519, 1.4949945, 1.5107806, 1.2320504, 1.1152589, 1.0438272, 0.9481494,
            0.8182929, 0.7737598, 0.6960076, 0.6215488))/(log(10))
upr_a_4 = log(c(309.55728, 286.31388, 179.47133, 154.70058, 135.64111, 123.33239, 103.58888, 70.08969,  58.47472,  46.51242,  37.65839))
lwr_a_4 = log(c(75.87796, 63.12973, 36.90449, 50.07692, 44.68293 ,53.61936, 44.94044, 44.22664, 37.55720, 31.36677, 26.83217))

a_5 = log(c(3.853373, 3.120740, 2.984569, 2.554073, 2.247068, 2.085917, 1.936126, 1.686563,
            1.548474, 1.384178, 1.239721))/(log(10))
upr_a_5 = log(c(580.93569, 496.14698, 410.22000, 352.51504, 294.10144, 216.33768, 195.48548, 152.95218, 125.35079, 109.76746,  86.74441))
lwr_a_5 = log(c(104.41741,  71.06706,  73.68478, 100.24693, 104.33253, 120.72234, 101.08246, 90.08091,  87.98390,  77.86632,  60.84619))



b = log(c(724.2590, 555.4987, 548.9432, 441.1496, 396.8760, 318.9032, 271.1221, 232.5219, 208.7822, 170.2965, 140.1213))
upr_b = log(c(1135.9950,  970.6052,  849.0792,  736.4744,  609.0714,  455.3771,  368.1343, 308.5954,  279.2565 , 197.6475,  160.6990))
lwr_b = log(c(285.9466, 222.5215, 243.6220, 206.3424, 240.5767, 147.1186, 197.2681, 169.5590, 150.1482, 142.2851, 118.8849))

if(PLOT){
  
  # expression('n'['max']*'/n'['min']*' = 1')
  data = data.frame(epoch,a_1,a_2,a_3,a_4,a_5,b,'lwr_a_1'= lwr_a_1,'upr_a_1'= upr_a_1, 'lwr_a_2'= lwr_a_2,'upr_a_2'= upr_a_2, 'lwr_a_3'= lwr_a_3,'upr_a_3'= upr_a_3, 'lwr_a_4'= lwr_a_4,'upr_a_4'= upr_a_4, 'lwr_a_5'= lwr_a_5,'upr_a_5'= upr_a_5,'lwr_b'= lwr_b,'upr_b'= upr_b,stringsAsFactors = F)
  (plot <- ggplot(data, aes(epoch, a_1))+
      geom_point(data, mapping = aes(epoch, a_1, color = "M = 0.5"),size = 2, shape = 17)+ 
      geom_point(data, mapping = aes(epoch, a_2, color = "M = 1"),size = 2, shape = 12)+ 
      geom_point(data, mapping = aes(epoch, a_3, color = "M = 2"),size = 2, shape = 13)+ 
      geom_point(data, mapping = aes(epoch, a_4, color = "M = 4"),size = 2, shape = 14)+ 
      geom_point(data, mapping = aes(epoch, a_5, color = "M = 8"),size = 2, shape = 16)+ 
      #geom_point(data, mapping = aes(epoch, b, color = "M = 3"),size = 2, shape = 15)+
      geom_line(data, mapping = aes(epoch, a_1, color = "M = 0.5"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_2, color = "M = 1"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_3, color = "M = 2"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_4, color = "M = 4"),size = 0.8)+
      geom_line(data, mapping = aes(epoch, a_5, color = "M = 8"),size = 0.8)+
      #geom_line(data, mapping = aes(epoch, b, color = "M = 3"))+
      scale_color_manual(values=c('#a6cee3', "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"))+
      geom_ribbon(data=data,aes(ymin=lwr_a_1,ymax=upr_a_1),alpha=0.0,fill = '#a6cee3')+
      geom_ribbon(data=data,aes(ymin=lwr_a_2,ymax=upr_a_2),alpha=0.0,fill = '#1f78b4')+
      geom_ribbon(data=data,aes(ymin=lwr_a_3,ymax=upr_a_3),alpha=0.0,fill = '#b2df8a')+
      geom_ribbon(data=data,aes(ymin=lwr_a_4,ymax=upr_a_4),alpha=0.0,fill = '#33a02c')+
      geom_ribbon(data=data,aes(ymin=lwr_a_5,ymax=upr_a_5),alpha=0.0,fill = '#fb9a99')+
      #geom_ribbon(data=data,aes(ymin=lwr_b,ymax=upr_b),alpha=0.0,fill = '#e31a1c')+
      #geom_hline(yintercept=0.914, linetype="dashed", color = "black")+
      #annotate("text", x = 18, y = 0.92, label = "91.4%")+
      coord_cartesian(ylim = c(-1.5, 1.5))+
      labs(color="", x= expression("log"['10']*" n"), y = expression('log'['10']*' Bias') #expression('log'['10']*' |'*Delta*' - '*widehat(Delta)*'|')
           #x= expression("log n"),
           #y= expression('log |'*Delta*' - '*widehat(Delta)*'|')
           )+
      #ggtitle(expression("Fairness guarantees with different d"['TV']))+
      theme(legend.position= c(0.85,0.82),axis.text=element_text(size=12),panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"),plot.title = element_text(hjust = 0.5,size = 14),legend.title = element_text(size = 9), legend.text = element_text(size = 7)))
  
}






########## real data ############


#### recidivism

load("compas_data.RData")
load("compas.RData")

compas_gender = compas[,1]
compas_gender = 1*(compas_gender == 'Male')
compas_race = compas[,3]
compas_race = 1*(compas_race == 'White')
compas_data_imp = scale(compas_data[,-c(1,3,10,12)])
compas_data_imp[,10] = 1*(compas_data_imp[,10] > 0)


# different sample size?
# training/test split?

# training set
set.seed(816)
seed = sample(10000,5000)


unfairness_cc = matrix(0, nrow=11, ncol=200)
unfairness_true = matrix(0, nrow=11, ncol=200)
unfairness_logistic = matrix(0, nrow=11, ncol=200)
unfairness_rf = matrix(0, nrow=11, ncol=200)
unfairness_svm = matrix(0, nrow=11, ncol=200)
unfairness_xgb = matrix(0, nrow=11, ncol=200)


for(i in 1:1){
  

  lambda = 1
  alpha = 3*lambda/7 
  
  n_1 = round((10^(2+(i-1)*0.1))/0.5)
  n_0 = round(alpha*n_1)
  n = n_0 + n_1
  n = 4000
  
  for(j in 1:40){
    
    set.seed(seed[j])
    S = sample(nrow(compas_data_imp))
    training_index = S[1:4000]
    test_index = S[4001:nrow(compas_data_imp)]
    
    training_data = compas_data_imp[training_index,]
    test_data = compas_data_imp[test_index,]
    test_data[,10] = 1*(test_data[,10]>0)
    
    # sample n training data
    S = sample(n)[1:n]
    training_index = training_index[S]
    training_data = compas_data_imp[training_index,]
    training_gender = compas_gender[training_index]
    
    missing_index = runif(n)
    K = 5
    Threshold = 1/(1 + exp(1 - 1*training_data[,1:K]%*% c(1,-1,1,-1,1)))
    #1/(1 + exp(0 + 2*training_data[,1:K]%*% c(1,1,1,1,1)))
    #1/(1 + exp(0 + 2*training_data[,10] + 2*(training_data[,9])))
    #1/(1 + exp(-2 - 1*training_data[,1:8]%*% c(1,1,1,1,1,1,1,1)))
    # rep(0.5,n)
    #1/(1 + exp(0 + 2*training_data[,10]+2*(training_data[,6]*training_data[,7]*training_data[,8]*training_data[,9])))
    #Threshold =  abs(X[,10])
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
    
    
    ## true weight
    #Threshold[which(Threshold > 1)] = 1
    #Threshold[which(Threshold < 0)] = 0.01
    #w_true = 1/Threshold
    #w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    #w_true = rep(1,4000)
    
    ## logistic
    #data_ipw = data.frame(training_data[,1:8],1-missing)
    #colnames(data_ipw) = c('V1','V2','V3','V4','V5','V6','V7','V8','miss')
    
    #weight <- ipwpoint(
    #  exposure = miss,
    #  family = "binomial",
    #  link = "logit",
    #  numerator = ~ 1,
    #  denominator = ~ ., 
    #  data = data_ipw)
    
    #w_logistic = (1+exp(-weight$den.mod[[1]][1] - as.matrix(data_ipw[,1:8]) %*% as.matrix(weight$den.mod[[1]][2:9])))
    #w_logistic = w_logistic/(sum(w_logistic[which(missing == 0)])/length(w_logistic[which(missing == 0)]))
    
    ## random forest
    #rf <- randomForest(
    #  miss ~ .,
    #  data = data_ipw
    #)
    #w_rf <- 1/(rf$predicted)
    #w_rf = w_rf/(sum(w_rf[which(missing == 0)])/length(w_rf[which(missing == 0)]))
    
    ## svm
    #svm = svm(miss ~ ., data = data_ipw, kernel = "radial", cost = 10, scale = FALSE)
    #w_svm <- 1/(svm$fitted)
    #w_svm = w_svm/(sum(w_svm[which(missing == 0)])/length(w_svm[which(missing == 0)]))
    
    
    ## XGboosting
    #xgb <- xgboost(data = matrix(as.numeric(as.matrix(data_ipw[,1:8])),nrow = n), label = as.matrix(data_ipw$miss), max_depth = 10, eta = 1, nthread = 2, nrounds = 10,"binary:logistic")
    #w_xgb <- 1/(predict(xgb, matrix(as.numeric(as.matrix(data_ipw[,1:8])),nrow = n)))
    #w_xgb = w_xgb/(sum(w_xgb[which(missing == 0)])/length(w_xgb[which(missing == 0)]))
    
    
    # true weight
    #Threshold[which(Threshold > 1)] = 1
    #Threshold[which(Threshold < 0)] = 0.01
    
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:n){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    #w_true = rep(1,4000)
    
    # logistic
    data_ipw = data.frame(training_data[,1:K],1-missing)
    colnames(data_ipw) = c('V1','V2','V3','V4','V5','miss')
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==1),])
    
    w_0 = weight_0$ipw.weights
    #w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:K]) %*% as.matrix(weight_0$den.mod[[1]][2:(K+1)])))
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    
    w_1 = weight_1$ipw.weights
    #w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:K]) %*% as.matrix(weight_1$den.mod[[1]][2:(K+1)])))
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
    xgb_0 = xgboost(data = matrix(as.numeric(as.matrix(data_ipw[which(A==0),1:K])),nrow = n_0), label = as.matrix(data_ipw$miss[which(A==0)]), max_depth = 10, eta = 1, nthread = 2, nrounds = 3,"binary:logistic")
    xgb_1 = xgboost(data = matrix(as.numeric(as.matrix(data_ipw[which(A==1),1:K])),nrow = n_1), label = as.matrix(data_ipw$miss[which(A==1)]), max_depth = 10, eta = 1, nthread = 2, nrounds = 3,"binary:logistic")
    
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
    
    # true
    rf_true <- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1) #w_true[which(missing == 0)]
    
    pred_train = as.numeric(predict(rf_true, data = Z)$predictions)-1
    MSE_0_true_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_true_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    
    pred = as.numeric(predict(rf_true, data = test_data)$predictions)-1
    MSE_0_true = sum((1-A_test)*abs(pred - y_test))/sum((1 - A_test))
    MSE_1_true = sum(A_test*abs(pred - y_test))/sum((A_test))
    
    # CC
    rf_cc<- ranger(as.factor(y_CC) ~ ., data = Z, case.weights = 1)
    
    # include (1-y_CC) / (1-y_test) when considering equal opportunity in each MSE
    pred_train = as.numeric(predict(rf_cc, data = Z)$predictions)-1
    MSE_0_cc_train = sum((1-A_CC)*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((1 - A_CC))
    MSE_1_cc_train = sum(A_CC*w_true[which(missing == 0)]*abs(pred_train - y_CC))/sum((A_CC))
    
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
    
    
    
    print(sum(which(missing == 0) < n_0+1))
    print(sum(which(missing == 0) >= n_0+1))
    
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_true[i,j] = abs( abs(MSE_0_true - MSE_1_true) - abs(MSE_0_true_train - MSE_1_true_train) )
    unfairness_logistic[i,j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    unfairness_rf[i,j] = abs( abs(MSE_0_rf - MSE_1_rf) - abs(MSE_0_rf_train - MSE_1_rf_train) )
    unfairness_svm[i,j] = abs( abs(MSE_0_svm - MSE_1_svm) - abs(MSE_0_svm_train - MSE_1_svm_train) )
    unfairness_xgb[i,j] = abs( abs(MSE_0_xgb - MSE_1_xgb) - abs(MSE_0_xgb_train - MSE_1_xgb_train) )
    
    
  }
  
}


mean(unfairness_cc[1,1:40])
mean(unfairness_true[1,1:40])
mean(unfairness_logistic[1,1:40])
mean(unfairness_rf[1,1:40])
mean(unfairness_svm[1,1:40])
mean(unfairness_xgb[1,1:40])

sd(unfairness_cc[1,1:40])
sd(unfairness_true[1,1:40])
sd(unfairness_logistic[1,1:40])
sd(unfairness_rf[1,1:40])
sd(unfairness_svm[1,1:40])
sd(unfairness_xgb[1,1:40])


mean(unfairness_cc[1,1:20])
mean(unfairness_true[1,1:20])
mean(unfairness_logistic[1,1:20])
mean(unfairness_rf[1,1:20])
mean(unfairness_svm[1,1:20])
mean(unfairness_xgb[1,1:20])


sd(unfairness_cc[1,1:20])
sd(unfairness_true[1,1:20])
sd(unfairness_logistic[1,1:20])
sd(unfairness_rf[1,1:20])
sd(unfairness_svm[1,1:20])
sd(unfairness_xgb[1,1:20])



mean(unfairness_cc[1,1:100])
mean(unfairness_true[1,1:100])
mean(unfairness_logistic[1,1:100])
mean(unfairness_rf[1,1:100])
mean(unfairness_svm[1,1:100])
mean(unfairness_xgb[1,1:100])

sd(unfairness_cc[1,1:100])
sd(unfairness_true[1,1:100])
sd(unfairness_logistic[1,1:100])
sd(unfairness_rf[1,1:100])
sd(unfairness_svm[1,1:100])
sd(unfairness_xgb[1,1:100])


# different propensity score model






#### ADNI

load("adni_imp.RData")
#load("adni_pred.RData")

# define adni data
adni = sa_imp #sa_pred #sa_imp

#adni_data = cbind(scale(adni[,1:1000]),adni[,1001],adni[,1003])
adni_gender = 2 - adni[,1002]
adni_race = 2 - adni[,1004]
#gender = rep(adni_gender,100)
adni_data_imp = as.matrix(adni[,c(1:1001)]) # ,1003 for pred


# training set
set.seed(816)
seed = sample(10000,5000)


unfairness_cc = matrix(0, nrow=11, ncol=100)
unfairness_true = matrix(0, nrow=11, ncol=100)
unfairness_logistic = matrix(0, nrow=11, ncol=100)
unfairness_rf = matrix(0, nrow=11, ncol=100)
unfairness_svm = matrix(0, nrow=11, ncol=100)
unfairness_xgb = matrix(0, nrow=11, ncol=100)




for(i in 1:1){
  
  n = 500
  
  for(j in 1:100){
    
    set.seed(seed[j])
    S = sample(nrow(adni_data_imp))
    training_index = S[1:500]
    test_index = S[501:nrow(adni_data_imp)]
    
    training_data = adni_data_imp[training_index,]
    test_data = adni_data_imp[test_index,]
    #test_data[,1001] = 1*(test_data[,1001]>0)
    
    # sample n training data
    S = sample(500)[1:n]
    training_index = training_index[S]
    training_data = adni_data_imp[training_index,]
    training_gender = adni_gender[training_index]
    
    missing_index = runif(n)
    Threshold = 1/(1 + exp(-2 + 5*apply(training_data[,1:50],1,mean)))
    #1/(1 + exp(-2 + 5*apply(training_data[,1:50],1,mean)))
    #1/(1 + exp(-2 + 5*apply(training_data[,101:150],1,mean)))
    #rep(0.5,n)#1/(1 + exp(5 - 10*training_data[,10] + 20*training_data[,6]*training_data[,7]*training_data[,8]*(training_data[,9])))
    #Threshold =  abs(X[,10])
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
    #w_true = 1/Threshold
    #w_true = w_true/(sum(w_true[which(missing == 0)])/length(w_true[which(missing == 0)]))
    w_true = 1/Threshold
    denom_0 = (sum(w_true[which(missing == 0 & A == 0)])/length(w_true[which(missing == 0 & A == 0)]))
    denom_1 = (sum(w_true[which(missing == 0 & A == 1)])/length(w_true[which(missing == 0 & A == 1)]))
    for(m in 1:n){
      if(A[m]==0){denom = denom_0}
      else{denom = denom_1}
      w_true[m] = w_true[m]/denom
    }
    #w_true = rep(1,500)
    
    
    # logistic
    
    #w_logistic = weight$ipw.weights
    #w_logistic = (1+exp(-weight$den.mod[[1]][1] - as.matrix(data_ipw[,1:K]) %*% as.matrix(weight$den.mod[[1]][2:K + 1])))
    #w_logistic = w_logistic/(sum(w_logistic[which(missing == 0)])/length(w_logistic[which(missing == 0)]))
    
    
    K = 100
    data_ipw = data.frame(training_data[,1:K],1-missing)
    varname <- 'V'
    n_K <- K + 1
    names(data_ipw)[1:ncol(data_ipw)] <- unlist(mapply(function(x,y) paste(x, seq(1,y), sep="_"), varname, n_K))
    colnames(data_ipw)[n_K] = 'miss'
    
    D_0 = data_ipw[which(A==0),]
    D_1 = data_ipw[which(A==1),]
    weight_0 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==0),])
    weight_1 <- ipwpoint(
      exposure = miss,
      family = "binomial",
      link = "logit",
      numerator = ~ 1,
      denominator = ~ ., 
      data = data_ipw[which(A==1),])
    
    w_0 = weight_0$ipw.weights
    #w_0 = (1+exp(-weight_0$den.mod[[1]][1] - as.matrix(data_ipw[,1:K]) %*% as.matrix(weight_0$den.mod[[1]][2:(K+1)])))
    w_0 = w_0/(sum(w_0[which(D_0[,K+1]==1)])/length(w_0[which(D_0[,K+1]==1)]))
    
    w_1 = weight_1$ipw.weights
    #w_1 = (1+exp(-weight_1$den.mod[[1]][1] - as.matrix(data_ipw[,1:K]) %*% as.matrix(weight_1$den.mod[[1]][2:(K+1)])))
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
    #rf <- randomForest(
    #  miss ~ .,
    #  data = data_ipw
    #)
    #w_rf <- 1/(rf$predicted)
    #w_rf = w_rf/(sum(w_rf[which(missing == 0)])/length(w_rf[which(missing == 0)]))
    
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
    #svm = svm(miss ~ ., data = data_ipw, kernel = "radial", cost = 10, scale = FALSE)
    #w_svm <- 1/(svm$fitted)
    #w_svm = w_svm/(sum(w_svm[which(missing == 0)])/length(w_svm[which(missing == 0)]))
    
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
    #xgb <- xgboost(data = matrix(as.numeric(as.matrix(data_ipw[,1:K])),nrow = n), label = as.matrix(data_ipw$miss), max_depth = 10, eta = 1, nthread = 2, nrounds = 10,"binary:logistic")
    #w_xgb <- 1/(predict(xgb, matrix(as.numeric(as.matrix(data_ipw[,1:K])),nrow = n)))
    #w_xgb = w_xgb/(sum(w_xgb[which(missing == 0)])/length(w_xgb[which(missing == 0)]))
    
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
    
    # true
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
    
    
    
    print(sum(which(missing == 0) < n_0+1))
    print(sum(which(missing == 0) >= n_0+1))
    
    
    unfairness_cc[i,j] = abs( abs(MSE_0_cc - MSE_1_cc) - abs(MSE_0_cc_train - MSE_1_cc_train) )
    unfairness_true[i,j] = abs( abs(MSE_0_true - MSE_1_true) - abs(MSE_0_true_train - MSE_1_true_train) )
    unfairness_logistic[i,j] = abs( abs(MSE_0_logistic - MSE_1_logistic) - abs(MSE_0_logistic_train - MSE_1_logistic_train) )
    unfairness_rf[i,j] = abs( abs(MSE_0_rf - MSE_1_rf) - abs(MSE_0_rf_train - MSE_1_rf_train) )
    unfairness_svm[i,j] = abs( abs(MSE_0_svm - MSE_1_svm) - abs(MSE_0_svm_train - MSE_1_svm_train) )
    unfairness_xgb[i,j] = abs( abs(MSE_0_xgb - MSE_1_xgb) - abs(MSE_0_xgb_train - MSE_1_xgb_train) )
    
    
  }
  
}


mean(unfairness_cc[1,1:40])
mean(unfairness_true[1,1:40])
mean(unfairness_logistic[1,1:40])
mean(unfairness_rf[1,1:40])
mean(unfairness_svm[1,1:40])
mean(unfairness_xgb[1,1:40])

sd(unfairness_cc[1,1:40])
sd(unfairness_true[1,1:40])
sd(unfairness_logistic[1,1:40])
sd(unfairness_rf[1,1:40])
sd(unfairness_svm[1,1:40])
sd(unfairness_xgb[1,1:40])


mean(unfairness_cc[1,1:20])
mean(unfairness_true[1,1:20])
mean(unfairness_logistic[1,1:20])
mean(unfairness_rf[1,1:20])
mean(unfairness_svm[1,1:20])
mean(unfairness_xgb[1,1:20])

sd(unfairness_cc[1,1:20])
sd(unfairness_true[1,1:20])
sd(unfairness_logistic[1,1:20])
sd(unfairness_rf[1,1:20])
sd(unfairness_svm[1,1:20])
sd(unfairness_xgb[1,1:20])



mean(unfairness_cc[1,1:100])
mean(unfairness_true[1,1:100])
mean(unfairness_logistic[1,1:100])
mean(unfairness_rf[1,1:100])
mean(unfairness_svm[1,1:100])
mean(unfairness_xgb[1,1:100])

sd(unfairness_cc[1,1:100])
sd(unfairness_true[1,1:100])
sd(unfairness_logistic[1,1:100])
sd(unfairness_rf[1,1:100])
sd(unfairness_svm[1,1:100])
sd(unfairness_xgb[1,1:100])






##### calculating the theoretical results


g = function(x){
  #y = 2*(x^2 + 2*5*x/3 + 2*(5/3)^2)*exp(-3*x/5)
  y = 2*(x^2)*exp(-3*x/5) + 4*exp(-0.5*x^2)
  return(y)
}

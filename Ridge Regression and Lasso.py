from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pylab

price=[]
#reading file
infile = open("house.txt")
var = infile.read().splitlines()
infile.close()
# creating matrices for variables and prices seperately
for i in range (len(var)):
    var[i] =var[i].split(" ")
    price.append(var[i][0])
    var[i].remove(var[i][0])
    var[i] = [j.split(":")[1] for j in var[i]]
    var[i] = map(float, var[i])
price = map(float, price)
price = np.array(price)
var = np.array(var)

# creating lasso and ridge model for the data
Rmodel = linear_model.Ridge(alpha = (1000./len(price)), fit_intercept=False)
Ridge = Rmodel.fit(var, price)
print "if lambda = 1000 the Ridge coefs are:\n ",Ridge.coef_,"\n"

lmodel = linear_model.Lasso(alpha = (1000./len(price)), fit_intercept=False)
lasso = lmodel.fit(var, price)
print "if lambda = 1000 the lasso coefs are: \n",lasso.coef_,"\n"

# creating training and testing datasets
price_Trn = price[:399]
price_Tst = price[400:]
var_Trn = var[:399,]
var_Tst = var[400:,]
# lists to remember all errors corresponding to each lambda
Ridge_error_Trn=[]
lasso_error_Trn=[]
Ridge_error_Tst=[]
lasso_error_Tst=[]
# running lasso and Ridge for training and testing datasets and different lambda plus remembering errors.
for lmda in [0,0.1,1,10,100,1000]:
    R_Trn = linear_model.Ridge(alpha = float(lmda)/len(price_Trn), fit_intercept=False)
    Ridge_Trn = R_Trn.fit(var_Trn, price_Trn)
    Ridge_predTrn = np.dot(var_Trn, np.asmatrix(Ridge_Trn.coef_).T)
    R_RMSETrn = mean_squared_error(price_Trn, Ridge_predTrn)**0.5
    Ridge_error_Trn.append(R_RMSETrn)
    
    l_Trn = linear_model.Lasso(alpha = float(lmda)/len(price_Trn), fit_intercept=False)
    lasso_Trn = l_Trn.fit(var_Trn, price_Trn)
    lasso_predTrn = np.dot(var_Trn, np.asmatrix(lasso_Trn.coef_).T)
    l_RMSETrn = mean_squared_error(price_Trn, lasso_predTrn)**0.5
    lasso_error_Trn.append(l_RMSETrn)

    Ridge_predTst = np.dot(var_Tst, np.asmatrix(Ridge_Trn.coef_).T)
    R_RMSETst = mean_squared_error(price_Tst, Ridge_predTst)**0.5
    Ridge_error_Tst.append(R_RMSETst)

    lasso_predTst = np.dot(var_Tst, np.asmatrix(lasso_Trn.coef_).T)
    l_RMSETst = mean_squared_error(price_Tst, lasso_predTst)**0.5
    lasso_error_Tst.append(l_RMSETst)

print "Ridge errors on train set:\n ",Ridge_error_Trn,"\n"
print "lasso errors on train set: \n",lasso_error_Trn,"\n"
print "Ridge errors on test set:\n ",Ridge_error_Tst,"\n"
print "lasso errors on test set:\n ",lasso_error_Tst,"\n"

pylab.figure(1)
pylab.plot([0,0.1,1,10,100,1000],Ridge_error_Trn)
pylab.xlabel("lambda")
pylab.ylabel("Ridge_error_Trn")

pylab.figure(2)
pylab.plot([0,0.1,1,10,100,1000],lasso_error_Trn)
pylab.xlabel("lambda")
pylab.ylabel("lasso_error_Trn")

pylab.figure(3)
pylab.plot([0,0.1,1,10,100,1000],Ridge_error_Tst)
pylab.xlabel("lambda")
pylab.ylabel("Ridge_error_Tst")

pylab.figure(4)
pylab.plot([0,0.1,1,10,100,1000],lasso_error_Tst)
pylab.xlabel("lambda")
pylab.ylabel("lasso_error_Tst")

pylab.show()

# running Cross Validation for Ridge and lasso and extracting the best fitted lambda/alpha
R_Trn = linear_model.RidgeCV(fit_intercept=False, cv=5)
Ridge_Trn = R_Trn.fit(var_Trn, price_Trn)
Ridge_Trn.alpha=Ridge_Trn.alpha_
print "The best alpha for Ridge_train is: \n",Ridge_Trn.alpha,"\n"

l_Trn = linear_model.LassoCV(fit_intercept=False, cv=5)
lasso_Trn = l_Trn.fit(var_Trn, price_Trn)
lasso_Trn.alpha=lasso_Trn.alpha_
print "The best alpha for lasso_train is: \n",lasso_Trn.alpha,"\n"

# extracting the w using the best fitted lambda for both Ridge and lasso
R_Trn = linear_model.Ridge(alpha = Ridge_Trn.alpha, fit_intercept=False)
Ridge_Trn = R_Trn.fit(var_Trn, price_Trn)
print "The best fitted coefs for Ridge_train is: \n",Ridge_Trn.coef_,"\n"
    
l_Trn = linear_model.Lasso(alpha = lasso_Trn.alpha, fit_intercept=False)
lasso_Trn = l_Trn.fit(var_Trn, price_Trn)
print "The best fitted coefs for lasso_train is: \n",lasso_Trn.coef_,"\n"

# fitting obtained model to test dataset and mesuring errors
R_Tst = linear_model.Ridge(alpha =  Ridge_Trn.alpha, fit_intercept=False)
Ridge_Tst = R_Tst.fit(var_Tst, price_Tst)
Ridge_predTst = np.dot(var_Tst, np.asmatrix(Ridge_Trn.coef_).T)
R_RMSETst = mean_squared_error(price_Tst, Ridge_predTst)**0.5
print "The errors of Ridge on testset: \n",(R_RMSETst),"\n"

l_Tst = linear_model.Lasso(alpha = lasso_Trn.alpha/len(price_Tst), fit_intercept=False)
lasso_Tst = l_Tst.fit(var_Tst, price_Tst)
lasso_predTst = np.dot(var_Tst, np.asmatrix(lasso_Trn.coef_).T)
l_RMSETst = mean_squared_error(price_Tst, lasso_predTst)**0.5
print "The errors of lasso on testset: \n",(l_RMSETst),"\n"

# with Original data

R = linear_model.RidgeCV(fit_intercept=False, cv=5)
Ridge = R.fit(var, price)
Ridge.alpha=Ridge.alpha_
print "The Ridge alpha for original data set: \n",Ridge.alpha,"\n"

l = linear_model.LassoCV(fit_intercept=False, cv=5)
lasso = l.fit(var, price)
lasso.alpha=lasso.alpha_
print "The lasso alpha for original dataset: \n",lasso.alpha,"\n"

R = linear_model.Ridge(alpha = Ridge.alpha/len(price), fit_intercept=False)
Ridge = R.fit(var, price)
print "The fitted Ridge model coefs on original dataset: \n",Ridge.coef_,"\n"
    
l = linear_model.Lasso(alpha = lasso.alpha/len(price), fit_intercept=False)
lasso = l.fit(var, price)
print "The fitted lasso model coefs on original dataset: \n",lasso.coef_,"\n"

Ridge_pred = np.dot(var, np.asmatrix(Ridge.coef_).T)
R_RMSE = mean_squared_error(price, Ridge_pred)**0.5
print "The errors of Ridge on original dataset: \n",(R_RMSE),"\n"

lasso_pred = np.dot(var, np.asmatrix(lasso.coef_).T)
l_RMSE = mean_squared_error(price, lasso_pred)**0.5
print "The errors of lasso on original dataset: \n",(l_RMSE),"\n"


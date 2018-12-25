% download the lasso codes here http://www.cs.ubc.ca/~schmidtm/Software/lasso.html
% put the codes in the same folder as this script
%add path 
addpath(genpath('lasso'))

%load data
load housing_scale.mat
X= data.X; % X is n*d data matrix
y=data.y; % y is n*1 output vector
[n,d]=size(X);

% set the value for lambda
lambda=10;

% Ridge Regression
% since this is a small data, we are going to compute the exact solution 
w_rg= (X'*X + lambda*eye(d))\(X'*y);
  
   
% Lasso
% We are going to use the matlab code written by Mark Schmidt 
% the code is available here http://www.cs.ubc.ca/~schmidtm/Software/lasso.html
% Under the Download Section, follow the link to download the codes
w_la = LassoBlockCoordinate(X, y, 2*lambda,'verbose', 0);


% compare the performance of regression for different values of lambda
% take the first 400 examples as training data and remaining 106 examples as validation data 
% for each value of lambda in [0.1, 1, 10, 100, 1000, 10000] run the Ridge regression and Lasso code above
% to get a model and then compute the root mean square error on the training and on the testing data 
% plot the error curves for both the training error and testing error vs  different values of lambda

% cross-validation 
% use 400 examples as training, follow the 10-fold cross-validation procedure to select the best lambda for both ridge regression 
% and lasso. 
% Then train the model on the 400 examples using the selected lambda and compute the averaged error on the testing data


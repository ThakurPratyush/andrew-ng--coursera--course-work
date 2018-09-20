function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); %number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
prediction = X * theta ;
sq_error = (prediction - y) .^ 2;
error = sum(sq_error);
reg_error = theta(2:end,1) .^ 2;
reg = sum(reg_error);
J = ( 1/(2 * m) ) * error + (lambda / (2*m) ) * reg;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
grad1 = zeros(size(theta));
grad2 = zeros(size(theta));
value  = (X' * (prediction - y));
grad1 = (1/m) * value;
grad2 = grad1(2:end,1) + (lambda/m) * theta(2:end,1);
grad = [grad1(1);grad2];













% =========================================================================

grad = grad(:);

end

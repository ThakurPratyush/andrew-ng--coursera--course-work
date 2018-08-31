function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
k = length(theta);
grad = zeros(size(theta));
prediction = sigmoid(X * theta);
log_err1 = -(y .* log(prediction)) - ((1-y) .* log(1 - prediction));
log_err2 = theta(2:k,1).^2;
J = 1/m * sum(log_err1) + (lambda/(2*m)) * sum(log_err2);

gard1 = zeros(k,1);
grad = zeros(k,1);
grad1 = (1/m) * (X' * (prediction - y));
grad2 = grad1(2:k,1);
temp = grad1(1);
grad1 = grad1(2:k,1);
grad2 = grad1 + (lambda/m) * theta(2:k,1);
grad = [temp;grad2]
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end

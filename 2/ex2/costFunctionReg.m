function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
k = length(theta);
z = X * theta;
prediction = sigmoid(z);
log_cost1 = - (y .* log(prediction)) - ((1 - y) .* log(1 - prediction)) ;
log_cost2 = theta(2:k,1).^2 ;

J =  (1/m) * sum(log_cost1)  + (lambda/(2 * m)) * sum(log_cost2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
k = length(theta);
grad1 = zeros(k,1);
grad = zeros(k,1);
grad1 = 1/m *( X' * (prediction - y));
grad2 = grad1(2:k,1);
temp = grad1(1);
grad1 = grad1(2:k,1);
grad2 =  grad1 + (lambda/m) * theta(2:k,1);
grad = [temp;grad2];

% =============================================================

end

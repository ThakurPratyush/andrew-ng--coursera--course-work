function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 
m = size(X, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 5000 * 1

% Add ones to the X data matrix
X = [ones(m, 1) X]; % 5000 * 401


p1 = sigmoid(all_theta * X') ;
[~,p] = max(p1,[],1);

%predict = sigmoid(X*all_theta'); % 5000 * 401 by 401*10
%[~, p] = max(predict, [], 2);


end
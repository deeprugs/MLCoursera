function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

term1 = y.*log(sigmoid(X*theta));

term2 = (1-y).*log(1-sigmoid(X*theta));

J= (-1)*(1/m) * sum(term1+term2);

hyp = sigmoid(X*theta);

% The gradient is calculated as follows. 
% 1) Hypothesis - y will give the error vector mean[1:100] for eg.
% 2) For each feature j, x[j] denotes the vector of training samples.
%    Now calculate sum (x[i,j] * mean[i]) for all the i samples of the feature x[j] (x[j] is the vector). 
grad = (1/m)*X'*(hyp-y);

% =============================================================

end

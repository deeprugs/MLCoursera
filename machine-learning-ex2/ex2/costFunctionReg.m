function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

a  = (size(theta)) % 28 x 1
b= size(X) % 118x28
pause

nbrows=size(theta,1)
h = sigmoid(X*theta);
cost_func_term1 = ((-y)'*log(h) - (1-y)'*log(1-h))/m

theta1 = [0 ; theta(2:size(theta), :)];
cost_func_term2 = lambda*(theta1'*theta1)/(2*m);

J=cost_func_term1+cost_func_term2;

grad = (X'*(h - y)+lambda*theta1)/m;

% =============================================================

end

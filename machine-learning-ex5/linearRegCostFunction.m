function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

h = (X * theta);
errors = h - y;
sse = errors' * errors;
reg = (lambda / (2 * m)) * (theta(2:end)' * theta(2:end));
J = ((1 / (2 * m)) * sse) + reg;

grad = (X' * (h - y) ./ m);

grad_reg = (lambda / m) .* theta;
grad(2:end) = grad(2:end) + grad_reg(2:end);

end

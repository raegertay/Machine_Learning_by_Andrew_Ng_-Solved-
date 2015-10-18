function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%===Section 1.2 Regularized linear regression cost function===%
% X = m x 2 ; Theta = 2 x 1
h = X * theta; % h = m x 1

J = (0.5/m) * (sum((h-y).^2)) + (0.5*lambda/m) * sum((theta(2:end)).^2);

%===Regularized linear regression gradient===%
grad(1) = (1/m) * sum((h-y) .* X(:,1));

grad(2:end) = (1/m) * (X(:,2:end))' * (h-y) + (lambda/m)*theta(2:end);





% =========================================================================

grad = grad(:);

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%===Section 1.3 Feedforward and cost function===%
% a1 = m x (input_layer_size + 1)
a1 = [ones(m,1) X];

% Theta1 = hidden_layer_size x (input_layer_size + 1)
% a2 = m x (hidden_layer_size + 1)
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m,1) a2];

% Theta2 = num_labels x (hidden_layer_size + 1)
% h = a3 = m x num_labels
h = sigmoid(a2 * Theta2');

for k = 1:num_labels
    y_k = (y == k);
    J = J + (1/m) * sum(-y_k .* log(h(:,k)) - (1-y_k) .* log(1-h(:,k)));
end

%===Section 1.4 Regulariszed cost function===%
% R = Regularization term
T1 = Theta1(:,2:end);
T2 = Theta2(:,2:end);
R = (lambda/(2*m)) * (sum(sum(T1.^2)) + sum(sum(T2.^2)));
J = J + R;

%===Section 2.3 Backpropagation===%
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
   % Step 1 
   a1 = [1 X(t,:)];
   a1 = a1(:); % (input_layer size + 1) x 1
   %Theta1 = hidden_layer_size x (input_layer_size + 1)
   z2 = Theta1 * a1; % hidden_layer_size x 1
   a2 = [1; sigmoid(z2)]; % (hidden_layer_size + 1) x 1
   % Theta2 = num_labels x (hidden_layer_size + 1)
   z3 = Theta2 * a2; % num_labels x 1
   a3 = sigmoid(z3);
   
   % Step 2
   y_k = (1:num_labels)';
   delta3 = a3 - (y_k == y(t)); % num_labels x 1
   
   % Step 3
   % Theta2 = num_labels x (hidden_layer_size + 1)
   delta2 = Theta2(:, 2:end)' * delta3 .* sigmoidGradient(z2); % hidden_layer_size x 1
   
   % Step 4
   % a2 = (hidden_layer_size + 1) x 1
   Delta2 = Delta2 + delta3 * a2'; % num_labels x (hidden_layer_size + 1)
   % a1 = (input_layer_size + 1) x 1
   Delta1 = Delta1 + delta2 * a1'; % hidden_layer_size x (input_layer_size + 1)    
end

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

%===Section 2.5 Regularized Neural Networks===%
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

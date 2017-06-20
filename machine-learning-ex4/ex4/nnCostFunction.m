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


%%%______________Part One______________________

%disp(size(Theta1)) %(25 x 401)
%disp(size(Theta2)) %(10 x 26)
%disp(size(X))  %5000 x 400
%disp(size(y))  %5000 x 1

x_sz = size(X,1);
X = [ones(x_sz,1) X]; %add a +1 term to ensure that the vectors of activations for layers a(1) has bias unit. 

%map y -> y_vals,  
%y_vals will be the binary vector of 1's and 0's to be used with the neural network cost function.
%y_vals = eye(num_labels) (y,:);
y_vals = (1:num_labels)==y;

%printf("a1 hidden layer calc: ")

a1 = sigmoid(X * Theta1'); % 5000 x 25
a1 = [ones(x_sz,1) a1];

%disp(size(a1))
%printf("\nh layer calc: ")

h = sigmoid(a1*Theta2');
%disp(size(a1))

J = sum((y_vals.*log(h)) + ((1-y_vals).*log(1-h)),2); % This is non regularized.
J = (-1/m)*sum(J);


% Take the sum of all the Theta^2
temp_theta1_sq =  sum((Theta1(:, 2:end)).^2);
temp_theta2_sq =  sum((Theta2(:, 2:end)).^2);

%disp(size(temp_theta1_sq))
%disp(size(temp_theta2_sq))

regularization_term = (lambda/(2*m))*(sum(temp_theta1_sq) + sum(temp_theta2_sq));

J += regularization_term;

%%%______________Part two______________________

% Already X has bias term. 

total_delta_1 = 0.0;
total_delta_2 = 0.0;


for t=1:m % For each sample t in X, 

%Step1:Perform a feedforward pass (Figure 2), computing the activations (z(2); a(2); z(3); a(3))for layers 2 and 3.
    a1 = X(t,:)';
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)]; %add a +1 term to ensure that the vectors of activations for layers a(2) also include the bias unit.
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

	h = a3;

%step2: Since y is in label form, convert into Logical array column
    y_bool  = ((1:num_labels)'==y(t));
	delta_3 = h  - y_bool; %difference from actual output.
	
%step 3: Hidden layer calculation of the delta. 
	delta_2 = Theta2(:, 2:end)'*delta_3.*sigmoidGradient(z2);

%Step 4: Accumulate the gradient.
	total_delta_2 = total_delta_2 + (delta_3.*a2') ;
    total_delta_1 = total_delta_1 + (delta_2.*a1') ;
	
endfor

	
%Step 5: Obtain the normalized gradient. 
Theta1_grad = (1 / m) * total_delta_1;
Theta2_grad = (1 / m) * total_delta_2;

% -------------------------------------------------------------

% Part 3 - Regularized Gradient
%
Theta1_grad(:, 2:end) += ((lambda / m) * (Theta1(:, 2:end)));
Theta2_grad(:, 2:end) += ((lambda / m) * (Theta2(:, 2:end)));
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%disp(size(grad));
end

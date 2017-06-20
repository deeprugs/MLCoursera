function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%disp(size(X))       % 211 x 2 (x1,x2)
%disp(size(y))       % 211 x 1
%disp(size(Xval))    % 200 x 2 
%disp(size(yval))    % 200 x 1

C_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
maxErr = Inf;

disp(size(C_arr))
disp(size(sigma_arr))

for C_index = 1:length(C_arr)
   for sigma_index = 1:length(sigma_arr)
     model = svmTrain(X, y, C_arr(C_index), @(x1, x2) gaussianKernel(x1, x2, sigma_arr(sigma_index) )); 
     predictions =  svmPredict(model, Xval);
     predictionErr = mean(double(predictions ~= yval));
     if predictionErr < maxErr
       maxErr = predictionErr;
       C = C_arr(C_index);
       sigma = sigma_arr(sigma_index);
     end
   end
 end
 
 disp(C_index)
 disp(sigma_index)
 disp(C)
 disp(sigma)
   
 









% =========================================================================

end

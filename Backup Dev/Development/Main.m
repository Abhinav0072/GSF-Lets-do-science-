%% parameters of neural network
window=3;                   % sub-section in which we will cut data
input_l_size=18.*3;
layers=1;                   % number of hidden layers
hidden_l_size=10;            % size of hidden layer
output_l_size=1;            % number of output labels
lambda=1;                  % regularization parameters

%% loading training data




%% random initialization & finding initial cost
fprintf('\nInitializing Neural Network Parameters ...\n')

Theta1=randInitializeWeights(input_l_size,hidden_l_size);
Theta2=randInitializeWeights(hidden_l_size,output_l_size);

initial_nn_params=[Theta1(:);Theta2(:)];
[J,~]=calculate_cost(raw_X,raw_y,input_l_size,layers,hidden_l_size,output_l_size,initial_nn_params,lambda);

fprintf('\n %d \n Program paused. Press enter to continue.\n',J);
pause;

%% TRAINING NEURAL NETWORK (Implementing Back propagation)
fprintf('\nTraining Neural Network... \n')

options=optimset('MaxIter',400);
costFunction=@(p)calculate_cost(raw_X,raw_y,input_l_size,layers,hidden_l_size,output_l_size,p,lambda);  % making a function handle.
                                            
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);   % Using advanced optimization method to minimize cost.

fprintf('\n parameters trained with back propagation \n');
pause;
%% Prediction




%% parameters of neural network
window=3;                   % sub-section in which we will cut data
input_l_size=18.*3;
layers=1;                   % number of hidden layers
hidden_l_size=10;           % size of hidden layer
output_l_size=2;            % number of output labels
lambda=1;                   % regularization parameters
confidence_percentage=90;   % percentage confidence i nbound given by prob
cut_off=.7;                % what percentage of length is acceptable for comparison

%% loading training data
raw_data=load('traindata.txt');
raw_X=raw_data(:,1:end-1);
raw_y=raw_data(:,end);



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



%% CROSS vALIDATION
valid_matrix=load('validdata.txt');                              
[J,~]=calculate_cost(valid_matrix(:,1:end-1),valid_matrix(:,end),input_l_size,layers,hidden_l_size,output_l_size,initial_nn_params,lambda);
fprintf('\n Cost in cross validation %d \n',J);


%% Prediction
test_matrix=load('testdata.txt');                              
data_prob=load('data_prob.txt');
[J,~]=calculate_cost(test_matrix(:,1:end-1),test_matrix(:,end),input_l_size,layers,hidden_l_size,output_l_size,initial_nn_params,lambda);
[~,prediction]=predict_letter(test_matrix(:,1:end-1),input_l_size,layers,hidden_l_size,output_l_size,nn_params,lambda);
fprintf('\n Cost in testing %d \n',J);

true_pos=0;
false_pos=0;
false_neg=0;
for i=1:size(test_matrix,1),
     if(and(test_matrix(i,end)==1,prediction(i,end)==1))
         true_pos=true_pos+1;
     elseif(and(test_matrix(i,end)==2,prediction(i,end)==1))
         false_pos=false_pos+1;
     elseif(and(test_matrix(i,end)==1,prediction(i,end)==2))
         false_neg=false_neg+1;
     end
end

precision=true_pos./(true_pos+false_pos);
recall=true_pos./(true_pos+false_neg);
F_score=2.*(precision.*recall)./(precision+recall);

fprintf('\n precision=%d , recall=%d , f_score=:%d \n',precision,recall,F_score);

        
     
     
     

   



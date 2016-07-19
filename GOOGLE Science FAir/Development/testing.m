
window=3;                   % sub-section in which we will cut data
input_l_size=18.*3;
layers=2;                   % number of hidden layers
hidden_l_size=10;           % size of hidden layer
output_l_size=2;            % number of output labels
lambda=1;                   % regularization parameters
confidence_percentage=60;   % percentage confidence i nbound given by prob
cut_off=.7;  
%% Prediction
test_matrix=load('testdata.txt');                              
data_prob=load('data_prob.txt');
[J,~]=calculate_cost(test_matrix(:,1:end-1),test_matrix(:,end),input_l_size,layers,hidden_l_size,output_l_size,initial_nn_params,lambda);
[~,prediction]=predict_letter(test_matrix(:,1:end-1),input_l_size,layers,hidden_l_size,output_l_size,nn_params,lambda);
fprintf('\n Cost in testing %d \n',J);
   
    
    




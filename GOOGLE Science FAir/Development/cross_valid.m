
window=3;                   % sub-section in which we will cut data
input_l_size=18.*3;
layers=2;                   % number of hidden layers
hidden_l_size=10;           % size of hidden layer
output_l_size=2;            % number of output labels
lambda=10;                   % regularization parameters
confidence_percentage=60;   % percentage confidence i nbound given by prob
cut_off=.7;  
%% Prediction
valid_matrix=load('validdata.txt');                              
[J,~]=calculate_cost(valid_matrix(:,1:end-1),valid_matrix(:,end),input_l_size,layers,hidden_l_size,output_l_size,initial_nn_params,lambda);
fprintf('\n Cost in cross validation %d \n',J);
   
    
    




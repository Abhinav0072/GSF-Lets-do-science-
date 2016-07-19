function [strength,num_label]=predict_letter(feature,input_l_size,k_layer,hidden_l_size,output_l_size,NN_param,lambda)
%% Modifying the data set for use.(C)
m=size(feature,1);
X=[ones(m,1),feature];             % appending ones column.

Theta1=reshape(NN_param(1:hidden_l_size*(input_l_size+1)),hidden_l_size,(input_l_size+1));    
Theta_hidden=zeros((k_layer-1)*hidden_l_size,hidden_l_size+1);
for i=1:k_layer,
    if(i==k_layer),
        Theta_k=reshape(NN_param(end-output_l_size*(hidden_l_size+1)+1:end),output_l_size,(hidden_l_size+1));
        break
    
    else 
        n=(input_l_size+1)*hidden_l_size+(i-1)*hidden_l_size*(hidden_l_size+1)+1;
        theta_temp=reshape(NN_param(n:n+hidden_l_size*(hidden_l_size+1)-1),hidden_l_size,(hidden_l_size+1));
        n=(i-1)*hidden_l_size+1;
        Theta_hidden(n:n+hidden_l_size-1,:)=theta_temp;
    end
end

%% Foreward Propagation ....(C)
z2=X*(Theta1');
a2=[ones(m,1),sigmoid(z2)];
x_temp=a2;
z_rest=zeros((k_layer-1)*m,hidden_l_size);
a_rest=zeros((k_layer-1)*m,hidden_l_size+1);
for i=1:k_layer-1,                                           % storing rest of z from 2nd hidden layer to last, in one matrix in a column.
    n=(i-1)*hidden_l_size+1;
    z_temp=x_temp*(Theta_hidden(n:n+hidden_l_size-1,:)');
    z_rest((i-1)*m+1:(i-1)*m+1+m-1,:)=z_temp;
    a_temp=[ones(m,1),sigmoid(z_temp)];
    a_rest((i-1)*m+1:(i-1)*m+1+m-1,:)=a_temp;
    x_temp=a_temp;
end
z_last=x_temp*(Theta_k');
h=sigmoid(z_last);

%% Choosing one with max strength
[strength,num_label]=max(h,[],2);

end
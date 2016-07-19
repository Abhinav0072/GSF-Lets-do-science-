function J=calculate_cost(raw_X,raw_y,input_l_size,k_layer,hidden_l_size,output_l_size,NN_param,lambda)
%% Modifying the data set for use.
m=size(raw_X,1);
X=[ones(m,1),raw_X];             % appending ones column.


logical_y=zeros(m,output_l_size);
for i=1:m,
    logical_y(i,raw_y(i))=1;    % creating the logical array of output according to class.
end
                                % nn param are found first by finding matrix
                                % then taking their (:).
                                % reshaping to extract the parameters for
                                % differnet layers(generalized)
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
                                              % eliminating the parameter
                                              % which need not be regulized
                                                   
Theta1_reg=Theta1(:,2:end);
Theta_hidden_reg=Theta_hidden(:,2:end);
Theta_k_reg=Theta_k(:,2:end);
Theta_reg=[Theta1_reg(:);Theta_hidden_reg(:);Theta_k_reg(:)];


%% Foreward Propagation ....
z2=X*(Theta1');
a2=[ones(m,1),sigmoid(z2)];
x_temp=a2;
a_rest=zeros((k_layer-1)*m,hidden_l_size+1);
a_temp=a2;
for i=1:k_layer-1,                                           % storing rest of z from 2nd hidden layer to last, in one matrix in a column.
    n=(i-1)*hidden_l_size+1;
    z_temp=x_temp*(Theta_hidden(n:n+hidden_l_size-1,:)');
    a_temp=[ones(m,1),sigmoid(z_temp)];
    a_rest((i-1)*m+1:(i-1)*m+1+m-1,:)=a_temp;
    x_temp=a_temp;
end
z_last=a_temp*(Theta_k');
h=sigmoid(z_last);

%% Finding error using Cost Function.
J_unreg=sum(sum(logical_y.*log(h)+(1-logical_y).*log(1-h),2),1)./(-1*m);
J_reg=(lambda./(2*m)).*sum((Theta_reg).^2,1);
J=J_unreg+J_reg;

%% Finding Gradient using Backpropagation algorithm ..

  

end
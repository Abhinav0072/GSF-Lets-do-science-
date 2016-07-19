
input_l_size=3;
k_layer=2;
hidden_l_size=2;
output_l_size=2;
NN_param=[1;5;2;6;3;7;4;8;9;12;10;13;11;14;15;16;17;18;19;20];



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
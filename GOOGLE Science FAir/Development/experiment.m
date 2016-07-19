%% Prediction
data_matrix=load('Experimentdata.txt');                               % load new data here
data_prob=load('data_prob.txt');

length_matrix=data_prob(:,1);                                   % from experiment population data(corresponding length and acceleration)
net_acceleration_matrix=data_prob(:,2);
avg_length= mean(data_prob(:,1));                               % from experiment
nn_params=load('nn_params.txt');



[~,feature_prob]=extract_juice(data_matrix(1:floor(avg_length*cut_off),:),window);
total_length=size(data_matrix,1);
[lower_b,upper_b]=give_conditional_probability(length_matrix,net_acceleration_matrix,feature_prob(1,2),confidence_percentage);

temp=total_length;
if(and(total_length<=upper_b,total_length>lower_b))
    upper_b=total_length;
elseif(total_length<=lower_b)
    upper_b=total_length;
    lower_b=upper_b;
end
flag=0;
new_start=0;

while (temp>=cut_off*avg_length),
    flag=1;
    index_temp=floor(upper_b-lower_b+1);
    prediction_matrix=zeros(index_temp,2);
    pivot=1;
    for i=floor(lower_b):floor(upper_b),
        [feature_NN,~]=extract_juice(data_matrix(1+new_start:i,:),window);
        [prediction_matrix(pivot,1),prediction_matrix(pivot,2)]=predict_letter(feature_NN,input_l_size,layers,hidden_l_size,output_l_size,nn_params,lambda);  
        pivot=pivot+1;
    end
    [m,index]=max(prediction_matrix(:,1),[],1);
    num_label=prediction_matrix(index,2);
    fprintf(' The predicted output is class : %d with strength: %d \n',num_label,m);
    
    
    new_start=floor(lower_b+index-1);
    if(new_start+avg_length*cut_off>total_length)
        break;
    end
    
    [~,feature_prob]=extract_juice(data_matrix((new_start+1):(new_start+floor(avg_length*cut_off)),:),window);
    [lower_b,upper_b]=give_conditional_probability(length_matrix,net_acceleration_matrix,feature_prob(1,2),confidence_percentage);
    lower_b=lower_b+new_start;
    upper_b=upper_b+new_start;
    
    if(and(total_length<=upper_b,total_length>lower_b))
        upper_b=total_length;
        temp=avg_length;
    elseif(and(total_length<=lower_b,(total_length-new_start)>=cut_off*avg_length))
        upper_b=total_length;
        lower_b=upper_b;
        temp=avg_length;
    elseif((total_length-new_start)<cut_off*avg_length)
        temp=0;
    else
        temp=avg_length;
           
    end
    
       
end


if(flag==0)
    fprintf('\n NO CHARACTER IS WRITTEN \n');
end


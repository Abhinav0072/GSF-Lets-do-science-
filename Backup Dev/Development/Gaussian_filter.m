function filtered_data=Gaussian_filter(data_matrix,arc_l,index,filter_window,cut_off_frequency)
m=filter_window;
dist_from_k=arc_l-ones(length(arc_l),1).*arc_l(index,1);          % measuring the distance(arclength from point k
%% filterig parameters of the point in whole range without window.
temp=exp((-1.*(dist_from_k.^2))./(2.*cut_off_frequency));
numerator_matrix=data_matrix(:,:).*[temp,temp,temp];
denominator_matrix=exp((-1.*(dist_from_k.^2))./(2.*cut_off_frequency));

%% filtered point with a fixed window and cut off frequency.
if(index<=m),
    if(index+m>size(data_matrix)),
        filtered_data=(sum(numerator_matrix(1:end,:),1))./(sum(denominator_matrix(1:end,:),1)); 
    else
        filtered_data=(sum(numerator_matrix(1:(index+filter_window),:),1))./(sum(denominator_matrix(1:index+filter_window,:),1));   
    end
elseif(and(index>m,index<(length(data_matrix)-filter_window+1))),
    filtered_data=(sum(numerator_matrix(index-filter_window:index+filter_window,:),1))./(sum(denominator_matrix(index-filter_window:index+filter_window,:),1));
else
    filtered_data=(sum(numerator_matrix(index-filter_window:end,:),1))./(sum(denominator_matrix(index-filter_window:end,:),1));
end

end
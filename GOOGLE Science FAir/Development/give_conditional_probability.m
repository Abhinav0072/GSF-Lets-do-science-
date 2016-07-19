function [lower_bound,upper_bound]=give_conditional_probability(length,net_acceleration,observed_acceleration,confidence_percentage)
%% finding sample mean,corelation for conditional distribution
sample_mean_length=mean(length);
sample_mean_acceleration=mean(net_acceleration);
corelation_matrix=corrcoef(net_acceleration,length);

%% mean and sigma fro conditional distribution
corelation=(corelation_matrix(1,2));
sigma_acceleration=std(net_acceleration);
sigma_length=std(length);

distribution_mean=sample_mean_length+corelation.*(sigma_length./sigma_acceleration).*(observed_acceleration-sample_mean_acceleration);
distribution_sigma=sqrt(sigma_length.^2.*(1-corelation.^2));

%% finding confidence interval
alpha_inv=(100-(100-confidence_percentage)./2)./100;
z_alphaby2=norminv(alpha_inv,0,1);

lower_bound=distribution_mean-(z_alphaby2.*distribution_sigma);
upper_bound=distribution_mean+(z_alphaby2.*distribution_sigma);
end
function[len,avg_x,std_x,avg_y,std_y,avg_z,std_z,net_a,net_std]=feature_extract(x,cal)
len=size(x,1);
cal_acc=zeros(len,3);
for i=1:len,
    cal_acc(i,:)=x(i,4:6);
end
avg_acc=mean(cal_acc,1);            %taking sample mean
avg_x=avg_acc(1);
avg_y=avg_acc(2);
avg_z=avg_acc(3);
[std_acc]=std(cal_acc,0,1);           % taking sample andard deviation
std_x=std_acc(1);
std_y=std_acc(2);
std_z=std_acc(3);
net_a=mean(sqrt(sum(cal_acc.^2,2)),1);          %avg of root mean square of accln component over time.
net_std=sqrt(std_x.^2+std_y.^2+std_z.^2);
end
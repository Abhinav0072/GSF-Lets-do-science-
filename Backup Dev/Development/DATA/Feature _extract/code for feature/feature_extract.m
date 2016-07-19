function[timetakelen,avg_x,std_x,med_x,avg_y,std_y,med_y,avg_z,std_z,med_z,net_a]=feature_extract(x,cal)
len=length(x);
timetaken=x(len,1);
cal_acc=zeros(len,3);
for i=1:len,
    cal_acc(i,:)=x(i,2:4);
end
avg_acc=mean(cal_acc,1);            %taking sample mean
avg_x=avg_acc(1);
avg_y=avg_acc(2);
avg_z=avg_acc(3);
[std_acc]=std(cal_acc,0,1);           % taking sample andard deviation
std_x=std_acc(1);
std_y=std_acc(2);
std_z=std_acc(3);
med_acc=median(cal_acc,1);          % taking median
med_x=med_acc(1);
med_y=med_acc(2);
med_z=med_acc(3);
net_a=mean(sqrt(sum(cal_acc.^2,2)),1);          %avg of root mean square of accln component over time.
end
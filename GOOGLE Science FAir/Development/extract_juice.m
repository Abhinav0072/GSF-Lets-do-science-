function [feature_NN,feature_prob]=extract_juice(data_matrix,window)
n=size(data_matrix,1);
psi=data_matrix(:,1);
theta=data_matrix(:,2);
phi=data_matrix(:,3);
force=data_matrix(:,7);

[curvature,torsion]=feature_extract(psi,theta,phi);
feature=[psi,theta,phi,curvature,torsion,force];

size_per_window=floor(n./window);
size_of_first=n-((window-1).*size_per_window);

feature_NN=zeros(1,18.*window);

for i=1:window,
    if(i==1),
        sumtemp=sum(feature(1:size_of_first,:),1);
        avgtemp=mean(feature(1:size_of_first,:),1);
        count=zeros(1,6);
        for j=1:6,
            temp=avgtemp(1,j);
            for k=1:size_of_first,
                if(feature(k,j)>=temp)
                    count(1,j)=count(1,j)+1;
                end
            end
        end
        feature_NN(1,1:18)=[sumtemp,avgtemp,count];
        index_next=size_of_first+1;
                   
    else
        sumtemp=sum(feature(index_next:index_next+size_per_window-1,:),1);
        avgtemp=mean(feature(index_next:index_next+size_per_window-1,:),1);
        count=zeros(1,6);
        for j=1:6,
            temp=avgtemp(1,j);
            for k=1:size_per_window,
                if (feature(k,j)>=temp),
                    count(1,j)=count(1,j)+1;
                end
            end
        end
        feature_NN(1,18*(i-1)+1:18*i)=[sumtemp,avgtemp,count];
        index_next=index_next+size_per_window;
    end      
end


len=n;
x=data_matrix;
cal_acc=zeros(len,3);
cal_acc(:,:)=x(:,4:6);
avg_acc=mean(cal_acc,1);            %taking sample mean
std_acc=std(cal_acc,0,1);           % taking sample andard deviation
net_a=mean(sqrt(sum(cal_acc.^2,2)),1);          %avg of root mean square of accln component over time.
net_std=sqrt(sum(std_acc.^2,2));

feature_prob=[len,net_a,net_std,avg_acc,std_acc];

end
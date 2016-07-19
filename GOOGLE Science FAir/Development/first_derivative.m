function f_dash=first_derivative(f,parameter)
n=size(f,1);
f_dash=zeros(size(f));
for index=1:length(f),
    if(index==1),
        f_dash(1,:)=(f(2,:)-f(1,:))./(parameter(2)-parameter(1));
    elseif (index==length(f)),
        f_dash(n,:)=(f(n,:)-f(n-1,:))./(parameter(n)-parameter(n-1));
    else
        f_dash(index,:)=(f(index+1,:)-f(index-1,:))./(2.*(parameter(index+1)-parameter(index-1)));
    end
end
end
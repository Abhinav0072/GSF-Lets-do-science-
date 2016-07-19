function f_double_dash=second_derivative(f,parameter)
n=size(f,1);
f_double_dash=zeros(size(f));
for index=1:n,
    if(index==1),
        f_double_dash(1,:)=(f(3,:)-2.*f(2,:)+f(1,:))./(parameter(3)-parameter(1)).^2;
    elseif(index==n),
        f_double_dash(n,:)=((f(n,:)-2.*f(n-1,:)+f(n-2,:))./(parameter(n)-parameter(n-2)).^2);
    else
        f_double_dash(index,:)=(f(index+1,:)-2.*f(index,:)+f(index-1,:))./(parameter(index+1)-parameter(index-1)).^2;
    end
end
end
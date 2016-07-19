function norm=quaternion_norm(q)
norm=zeros(size(q,1),1);
norm(:,1)=sqrt(q(:,1).^2+q(:,2).^2+q(:,3).^2+q(:,4).^2);
end
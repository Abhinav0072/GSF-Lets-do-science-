function cross=quaternion_cross(q,r)
t=zeros(size(q,1),4);
t(:,1)=r(:,1).*q(:,1)-r(:,2).*q(:,2)-r(:,3).*q(:,3)-r(:,4).*q(:,4);
t(:,2)=r(:,1).*q(:,2)+r(:,2).*q(:,1)-r(:,3).*q(:,4)+r(:,4).*q(:,3);
t(:,3)=r(:,1).*q(:,3)+r(:,2).*q(:,4)+r(:,3).*q(:,1)-r(:,4).*q(:,2);
t(:,4)=r(:,1).*q(:,4)-r(:,2).*q(:,3)+r(:,3).*q(:,2)+r(:,4).*q(:,1);
cross=t;
end
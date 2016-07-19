function [curvature,torsion]=feature_extract(psi,theta,phi)
n=length(psi);                          %length of character
orientation=[psi,theta,phi];
%% converting euler to unit quaternions
q_x=zeros(n,4);                        
q_y=zeros(n,4);
q_z=zeros(n,4);
q=zeros(n,4);
for i=1:n,
    q_x(i,:)=[cos(psi(i)./2),sin(psi(i)./2),0,0];
    q_y(i,:)=[cos(theta(i)./2),0,sin(theta(i)./2),0];
    q_z(i,:)=[cos(phi(i)./2),0,0,sin(phi(i)./2)];
    q(i,:)=quaternion_cross((quaternion_cross(q_z(i,:),q_y(i,:))),q_x(i,:));
end
%% getting cumulated arc length for parametrization
arc_l=zeros(n,1);
temp=0;
for i=2:n,
    temp=temp+quaternion_norm(q(i,:)-q(i-1,:));
    arc_l(i,:)=temp;
end
%% filtering the orientation matrix(euler angle data) using sliding gaussian filter window.
filtered_orientation=zeros(n,3);
window=5;               % setting up window for filtering (change it).
sigma=1;                % setting up cut off frequency for filtering (change it).
for i=1:n,
    filtered_orientation(i,:)=Gaussian_filter(orientation,arc_l,i,window,sigma);
end

%% finding curvature at each point
tangent=first_derivative(filtered_orientation,arc_l);           % T=dr/ds
normal=second_derivative(filtered_orientation,arc_l);           % N=(dT/ds)/|curvature|

curvature=quaternion_norm([normal,zeros(size(normal,1))]);                              % curvature=|dT/ds|

normal=normal./[curvature,curvature,curvature];
binormal=cross(tangent,normal);

torsion=-1.*sum(first_derivative(binormal,arc_l).*normal,2);    % torsion= -1*(dB/ds).N  

end
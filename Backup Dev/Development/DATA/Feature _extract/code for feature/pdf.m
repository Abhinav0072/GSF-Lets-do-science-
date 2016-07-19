
x1=feature(:,2);
x2=feature(:,12);
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)]);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-3 3 -3 3 0 .4])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
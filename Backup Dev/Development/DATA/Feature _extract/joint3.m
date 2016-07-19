
 X = mvnrnd([0,0],[1 .6; .6 .5], 10000);

 nbins = 50;
 marg = range(X)/20;
 minx = min(X,[],1);
 maxx = max(X,[],1);
 x1edges = linspace(minx(1)-marg(1),maxx(1)+marg(1),nbins+1);
 x2edges = linspace(minx(2)-marg(2),maxx(2)+marg(2),nbins+1);
 [dum,x1bins] = histc(X(:,1),x1edges);
 [dum,x2bins] = histc(X(:,2),x2edges);
 Z = full(sparse(x1bins,x2bins,1,nbins,nbins));

 x1ctrs = x1edges(1:end-1) + diff(x1edges)/2;
 x2ctrs = x2edges(1:end-1) + diff(x2edges)/2;
 mesh(x1ctrs,x2ctrs,Z);


 function f = ksdensity2d(x,gridx1,gridx2,bw)
 % KSDENSITY2D Compute kernel density estimate in 2D.
 % F = KSDENSITY2D(X,GRIDX,GRIDX2,BW) computes a nonparametric estimate of
 % the probability density function of the sample in the N-by-2 matrix X.
 % F is the vector of density values evaluated at the points in the grid
 % defined by the vectors GRIDX1 and GRIDX2. The estimate is based on a
 % normal kernel function, using a window parameter (bandwidth) that is a
 % function of the number of points in X.
 [n,p] = size(x);
 m1 = length(gridx1);
 m2 = length(gridx2);

 % Choose bandwidths optimally for Gaussian kernel
 if nargin < 4 || isempty(bw)
      sig1 = median(abs(gridx1-median(gridx1))) / 0.6745;
      if sig1 <= 0, sig1 = max(gridx1) - min(gridx1); end
      if sig1 > 0
          bw(1) = sig1 * (1/n)^(1/6);
      else
          bw(1) = 1;
      end
      sig2 = median(abs(gridx2-median(gridx2))) / 0.6745;
      if sig2 <= 0, sig2 = max(gridx2) - min(gridx2); end
      if sig2 > 0
          bw(2) = sig2 * (1/n)^(1/6);
      else
          bw(2) = 1;
      end
 end

 % Compute the kernel density estimate
 [gridx2,gridx1] = meshgrid(gridx2,gridx1);
 x1 = repmat(gridx1, [1,1,n]);
 x2 = repmat(gridx2, [1,1,n]);
 mu1(1,1,:) = x(:,1); mu1 = repmat(mu1,[m1,m2,1]);
 mu2(1,1,:) = x(:,2); mu2 = repmat(mu2,[m1,m2,1]);
 f = sum((normpdf(x1,mu1,bw(1)) .* normpdf(x2,mu2,bw(2))), 3) / n;

 % Plot the estimate
 surf(gridx1,gridx2,f);
 hold on;
 plot3(x(:,1),x(:,2),zeros(n,1),'bo');
 hold off;
 view(-37.50,30);


 function testksdensity2d
 gridx1 = 0:.05:1;
 gridx2 = 5:.1:10;
 X = [0+.5*rand(20,1) 5+2.5*rand(20,1);
      .75+.25*rand(10,1) 8.75+1.25*rand(10,1)];
 ksdensity2d(X,gridx1,gridx2); 
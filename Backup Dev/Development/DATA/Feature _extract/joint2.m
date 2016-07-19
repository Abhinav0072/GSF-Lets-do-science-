%// Data (example):
X = randn(1,1e5); % random variables.
Y = randn(1,1e5);

x_axis = -3:.2:3; % Define edges of bins for x axis. Column vector
y_axis = -3:.2:3; % Same for y axis

%// Compute and plot pdf
figure
histogram2(X, Y, x_axis, y_axis, 'Normalization', 'pdf')

%// Compute and plot cdf
figure
histogram2(X, Y, x_axis, y_axis, 'Normalization', 'cdf')
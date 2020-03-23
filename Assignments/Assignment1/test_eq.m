tspan = [0, 10];
n = tspan(2) / 0.01; % 0.025, 0.05, 0.1
lambda = 1; % 0, 1, 20
y0 = 2; % 2,10

[T_test, Y_test] = TestFunc(tspan, n, y0, [lambdas]);


function [t, y] = TestFunc(tspan, n, y0, param)
    dt = (tspan(2) - tspan(1)) / n;
    ny = size(y0,1);
    y = zeros(ny,n+1);
    t = zeros(1,n+1);
    
    y(:,1) = y0;
    t(:,1) = tspan(1);
    for k=1:n
        t(:,k+1) = t(:,k) + dt;
        y(:,k+1) = exp(param(1)*t(:,k+1));
    end
    
    y=y';
    t=t';
end
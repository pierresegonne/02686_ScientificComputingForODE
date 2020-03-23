tspan = [0, 10];
n = tspan(2) / 0.025; % 0.025, 0.05, 0.1
lambda = 1; % 0, 1, 20
y0 = 2; % 2,10
[TG2, G2] = GFunc(tspan, n, y0, [lambda]);
[TG10, G10] = GFunc(tspan, n, 10, [lambda]);
n = tspan(2) / 0.1; % 0.025, 0.05, 0.1
[TG2b, G2b] = GFunc(tspan, n, y0, [lambda]);


function [tgout, gout] = GFunc(tspan, n, y0, param)
    dt = (tspan(2) - tspan(1)) / n;
    ny = size(y0,1);
    gout = zeros(ny,n+1);
    tgout = zeros(1,n+1);
    
    gout(:,1) = y0;
    tgout(:,1) = tspan(1);
    for k=1:n
        tgout(:,k+1) = tgout(:,k) + dt;
        gout(:,k+1) = y0*(tgout(:,k+1)*tgout(:,k+1)+1)*(tgout(:,k+1)*tgout(:,k+1)+1);
    end
end
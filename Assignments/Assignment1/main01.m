
tspan = [0, 10];
n = (tspan(2) - tspan(1)) / 0.01; % 0.025, 0.05, 0.1
lambda = 1; % 0, 1, 20
y0 = 2; % 2,10
[T, Y] = LMsolver(@FunJac, tspan, n, y0, {lambda});
%[T_test_num, Y_test_num] = LMsolver( @TestJac, tspan, n, y0, {lambda});

y0s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 10, 10, 10];
lambdas = [0, 0, 0, 1, 1, 1, 10, 10, 10, 0, 0, 0, 1, 1, 1, 10, 10, 10];
ks = [0.025, 0.05, 0.1, 0.025, 0.05, 0.1, 0.025, 0.05, 0.1, 0.025, 0.05, 0.1, 0.025, 0.05, 0.1,  0.025, 0.05, 0.1];

ys = {};
disp(ys);
for i=1:18
    n = fix((tspan(2) - tspan(1)) / ks(i));
    [T, Y] = LMsolver(@FunJac, tspan, n, y0s(i), {lambdas(i)});
    ys{end+1} = Y;
end

function [tout,yout] = LMsolver( func, tspan, n, y0, param)
% func : name of the function for computing the right hand side.
% tspan : [start, end] times.
% n : number of steps to take starting form tspan(1) to finishing at tspan(2)
% y0 : initial solution
% param : a vector array of parameters that may be transferred and used in func.
% tout : vector of time values corresponding to steps taken.
% yout : vector of solution values corresponding to steps taken.
    tol = 1.0e-8;
    maxiters = 100;

    dt = (tspan(2) - tspan(1)) / n;
    ny = size(y0,1);
    yout = zeros(ny,n+1);
    tout = zeros(1,n+1);
    
    yout(:,1) = y0;
    tout(:,1) = tspan(1);
    for k=1:n
        [fk,Jfk] = feval(func, tout(:,k), yout(:,k), param{:});
        tout(:,k+1) = tout(:,k) + dt;
        yinit = yout(:,k) + dt*fk; % explicit form
        yout(:,k+1) = TrapezoidalUpdate(func, tout(:,k), yout(:,k), dt, yinit, tol, maxiters, param);
    end
    
    % Prettify
    yout = yout';
    tout = tout';
end

function [y] = TrapezoidalUpdate(func, tk, yk, dt, yinit, tol, maxiters, param)
    k = 0;
    y = yinit;
    [fk,Jfk] = feval(func, tk, yk, param{:}); % in k
    t = tk + dt;
    [fk1,Jfk1] = feval(func, t, y, param{:}); % in k + 1
    R = y - (dt/2)*(fk1 + fk) - yk;
    I = eye(size(yk,1));
    while ((k < maxiters) && (norm(R,'inf') > tol))
        k = k + 1;
        M = I - dt*Jfk1;
        dy = M\R;
        y = y - dy;
        [fk1,Jfk1] = feval(func, t, y, param{:});
        R = y - (dt/2)*(fk1 + fk) - yk;
    end
end

function [f, Jf] = FunJac(t, y, param)
    f = 4*t*sqrt(y) - param(1)*(y - (1+t*t)*(1+t*t));
    Jf = 2*t/sqrt(y) - param(1);
end


function [f, Jf] = TestJac(t, y, param)
    f = exp(param(1)*t);
    Jf = 0;
end
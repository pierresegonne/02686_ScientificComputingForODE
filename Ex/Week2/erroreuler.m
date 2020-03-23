% Using the f1 function, for which we have an analytical solution.
N = 1000;
x0 = 1;
ta = 0;
tb = 10;
lambda = 0.002;
delta = zeros(1,N);

for n=1:N
    [X,T] = EulerExplicitFixedStepSize(@f1, ta, tb, n-1, x0, lambda);
    delta(n) = trueX(n) - X(end);
end

function [X,T] = EulerExplicitFixedStepSize(f,ta,tb,N,x0,varargin)
    dt = (tb-ta)/N;
    nx = size(x0,1);
    X = zeros(nx,N+1);
    T = zeros(1,N+1);
    
    % Initialization
    X(:,1) = x0;
    T(:,1) = ta;

    % Iterative computing
    for k=1:N
        ftx = feval(f, T(k), X(:,k), varargin{:});
        X(:,k+1) = X(:,k) + dt*ftx;
        T(:,k+1) = T(:,k) + dt;
    end
    
    X = X';
    T = T';
    
end


function Xdot = f1(t,X,lambda)
    Xdot = lambda * X;
end
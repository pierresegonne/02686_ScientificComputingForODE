x0 = 1;
ta = 0;
tb = 100;
N = 5000;
lambda = 2;
%Van Der Pol
mu = 10;
x0 = [2.0; 0.0];
[X,T] = EulerExplicitFixedStepSize(@VanDerPol, ta, tb, N, x0, [mu]);

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

function Xdot = f2(t,X)
    Xdot = cos(t)*X;
end

function xdot = VanDerPol(t,x,mu)

xdot=zeros(2,1);
xdot(1) = x(2);
xdot(2) = mu*(1-x(1)*x(1))*x(2) - x(1);
end
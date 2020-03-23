mu = 10;
lambda=0.01;
ta = 0;
tb = 100;
N = 15000;
x0 = [2.0; 0.0];
[X,Xexp,T] = EulerImplicitFixedStepSize(@VanderPolfunjac, ta, tb, N, x0, [mu]);
%[X,Xexp,T] = EulerImplicitFixedStepSize(@Testfunjac, ta, tb, N, x0, [lambda]);


function [X,Xexp,T] = EulerImplicitFixedStepSize(FJac,ta,tb,N,xa,varargin)
    dt = (tb-ta)/N;
    nx = size(xa,1);
    X = zeros(nx,N+1);
    Xexp = zeros(nx,N+1);
    T = zeros(1,N+1);
    
    tol = 1.0e-8;
    maxiters = 100;
    
    X(:,1) = xa;
    Xexp(:,1) = xa;
    T(:,1) = ta;
    for k=1:N
        %[ftx,~] = feval(FJac, T(k), X(:,k), varargin{:});
        ftx = feval(FJac, T(k), X(:,k), varargin{:});
        [ftxexp,~] = feval(FJac, T(k), Xexp(:,k), varargin{:});
        T(:,k+1) = T(:,k) + dt;
        xinit = X(:,k) + dt*ftx; % explicit form
        X(:,k+1) = NewtonsMethodODE(FJac, T(:,k), X(:,k), dt, xinit, tol, maxiters, varargin{:});
        Xexp(:,k+1) = Xexp(:,k) + dt*ftxexp;
    end
    
    X = X';
    Xexp = Xexp';
    T = T';
end

function x = NewtonsMethodODE(FJac, tk, xk, dt, xinit, tol, maxiters, varargin)
    k = 0;
    x = xinit;
    t = tk + dt;
    [f,Jf] = feval(FJac, t, x, varargin{:});
    R = x - dt*f - xk;
    I = eye(size(xk,1));
    while ((k < maxiters) && (norm(R,'inf') > tol))
        k = k + 1;
        M = I - dt*Jf;
        dx = M\R;
        x = x - dx;
        [f,Jf] = feval(FJac, t, x, varargin{:});
        R = x - dt*f - xk;
    end
end

function [f,Jac] = VanderPolfunjac(t,x,mu)
    f = zeros(2,1);
    f(1) = x(2);
    f(2) = mu*(1-x(1)*x(1))*x(2)-x(1);
    Jac = zeros(2,2);
    Jac(1,2) = 1;
    Jac(2,1) = -2*mu*x(1)*x(2)-1;
    Jac(2,2) = mu*(1-x(1)*x(1));
end

function [f,Jac] = Testfunjac(t,x,lambda)
    f = zeros(2,1);
    f(1) = lambda*x(1);
    f(2) = 0;
    Jac = zeros(2,2);
    Jac(1,1) = lambda;
end
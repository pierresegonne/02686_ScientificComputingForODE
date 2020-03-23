mu = 10;
ta = 0;
tb = 100;
N = 15000;
xa = [2.0; 0.0];
[X,T] = ImplicitEulerFixedStepSize(@VanderPolFunJac, ta, tb, N, xa, [mu]);

function [X,T] = ImplicitEulerFixedStepSize(funJac,ta,tb,N,xa,varargin)

    % Compute step size and allocate memory
    dt = (tb-ta)/N;
    nx = size(xa,1);
    X = zeros(nx,N+1);
    T = zeros(1,N+1);

    tol = 1.0e-8;
    maxit = 100;

    % Eulers Implicit Method
    X(:,1) = xa;
    T(:,1) = ta;
    for k=1:N
        f = feval(funJac,T(k),X(:,k),varargin{:});
        T(:,k+1) = T(:,k) + dt;
        xinit = X(:,k) + dt*f;
        X(:,k+1) = NewtonsMethodODE(funJac,T(:,k), X(:,k), dt, xinit, tol, maxit, varargin{:});
    end

    % Form a nice table for the result
    T = T';
    X = X';
end

function x = NewtonsMethodODE(FunJac, tk, xk, dt, xinit, tol, maxit, varargin)
    k = 0;
    x = xinit;
    t = tk + dt;
    [f,J] = feval(FunJac,t,x,varargin{:});
    R = x - dt*f - xk;
    I = eye(size(xk,1));
    while( (k < maxit) && (norm(R,'inf') > tol) )
        k = k+1;
        dRdx = I - dt*J;
        dx = dRdx\R;
        x = x - dx;
        [f,J] = feval(FunJac,t,x,varargin{:});
        R = x - dt*f - xk;
    end
end

function [f,Jac] = VanderPolFunJac(t,x,mu)
    f = zeros(2,1);
    f(1) = x(2);
    f(2) = mu*(1-x(1)*x(1))*x(2)-x(1);
    Jac = zeros(2,2);
    Jac(1,2) = 1;
    Jac(2,1) = -2*mu*x(1)*x(2)-1;
    Jac(2,2) = mu*(1-x(1)*x(1));
end
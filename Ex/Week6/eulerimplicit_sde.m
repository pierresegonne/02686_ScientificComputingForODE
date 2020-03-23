Ns = 10;
N = 1000;
T = 10;

[W, Tw, dW] = ScalarStdWienerProcess(T, N, Ns, 100);
X = zeros(size(W));
for s=1:Ns
    X(s,:,:) = SDEEulerImplicitExplicit(@GeometricBrownianMotionF, @GeometricBrownianMotionG, Tw, 1, W(s,:), [0.15,0.15]);
end

function X = SDEEulerImplicitExplicit(f, g, T, x0, W, varargin)
    
    N = size(T,2) - 1;
    nX = size(x0,1);
    X = zeros(nX, N+1);
    
    tol = 1e-8;
    maxiters = 100;
    
    X(:,1) = x0;
    [fval,~] = feval(f, T(1), X(:,1), varargin{:});
    for k=1:N
        dt = T(k+1)-T(k);
        dW = W(:,k+1) - W(:,k);
        gval = feval(g, T(k), X(:,k), varargin{:});
        psi = X(:,k) + gval*dW;
        xinit = psi + fval*dt;
        [X(:,k+1), fval] = SDENewtonSolver(f, T(k+1), dt, psi, xinit, tol, maxiters, varargin{:});
    end
    
    X = X';
end

function [x, fval] = SDENewtonSolver(fJ, t, dt, psi, xinit, tol, maxiters, varargin)
    k = 0;
    x = xinit;
    I = eye(length(xinit));
    
    [fval, J] = feval(fJ, t, x, varargin{:});
    R = x - fval*dt - psi;
    
    while ((k <= maxiters) && (norm(R,'inf') > tol))
        k = k + 1;
        dRdx = I - J*dt;
        mdx = dRdx\R;
        x = x - mdx;
        [fval, J] = feval(fJ, t,x,varargin{:});
        R = x - fval*dt - psi;
    end
end

function [W, Tw, dW] = ScalarStdWienerProcess(T, N, Ns, seed)
    if nargin == 4
        rng(seed);
    end
    dt = T/N;
    dW = sqrt(dt)*randn(Ns,N);
    W = [zeros(Ns,1) cumsum(dW,2)];
    Tw = 0:dt:T;
end

function [f, J] = GeometricBrownianMotionF(t, x, params)
    f = params(1)*x;
    J = [params(1)];
end

function g = GeometricBrownianMotionG(t, x, params)
    g = params(2)*x;
end
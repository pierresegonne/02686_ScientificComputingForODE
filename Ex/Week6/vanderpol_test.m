mu = 3;
sigma = 1.0;
x0 = [0.5; 0.5];
p = [mu; sigma];

tf = 5*mu;
nw = 1;
N = 1000;
Ns = 5;
seed = 100;

[W,T,~]=StdWienerProcess(tf,N,nw,Ns,seed);
X = zeros(Ns,N+1,length(x0));
for s=1:Ns
    X(s,:,:) = SDEEulerExplicitExplicit(@VanderpolDrift, @VanderPolDiffusion1, Tw, x0, W(:,:,s), p);
end

Xd = SDEEulerExplicitExplicit(@VanderpolDrift, @VanderPolDiffusion1, Tw, x0, W(:,:,s), [mu; 0]);
Xdi = SDEEulerImplicitExplicit(@VanderpolDrift, @VanderPolDiffusion1, Tw, x0, W(:,:,s), [mu; 0]);

function X = SDEEulerExplicitExplicit(f, g, T, x0, W, varargin)
    N = size(T,2) - 1;
    nX = size(x0,1);
    X = zeros(nX, N+1);
    
    X(:,1) = x0;
    for k=1:N
        dt = T(k+1)-T(k);
        dW = W(:,k+1) - W(:,k);
        fval = feval(f, T(k), X(:,k), varargin{:});
        gval = feval(g, T(k), X(:,k), varargin{:});
        X(:,k+1) = X(:,k) + fval*dt + gval*dW;
    end
    
    X = X';
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


function [W,Tw,dW] = StdWienerProcess(T,N,nW,Ns,seed)
    if nargin == 4
        rng(seed)
    end
    dt = T/N;
    dW = sqrt(dt)*randn(nW,N,Ns);
    W = [zeros(nW,1,Ns) cumsum(dW,2)];
    Tw = 0:dt:T;
end

function [fval, J] = VanderpolDrift(t, x, p)
    mu = p(1);
    tmp = mu*(1.0-x(1)*x(1));
    fval = zeros(2,1);
    fval(1,1) = x(2);
    fval(2,1) = tmp*x(2)-x(1);
    
    if nargout > 1
        J = [0 1; -2*mu*x(1)*x(2)-1.0 tmp];
    end
end

function g = VanderPolDiffusion1(t, x, p)
    sigma = p(2);
    g = [0.0; sigma];
end

function g = VanderPolDiffusion2(t, x, p)
    sigma = p(2);
    g = [0.0; sigma*(1.0+x(1)*x(1))];
end
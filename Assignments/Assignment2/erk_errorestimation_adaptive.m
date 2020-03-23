
mu = 100;
x0 = [2.0; 0.0];
tspan= [0; 5*mu];
options = odeset('Jacobian',@JacVanDerPol,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[Tode,Xode]=ode15s(@VanDerPol, tspan, x0, options, mu);
solver = ERKSolverErrorEstimationParameters('ERK4C');
[T,X,E] = ERKSolverErrorEstimationAdaptive(@VanDerPol, tspan, x0, 1, solver,1.0e-6, 1.0e-6, [mu]);


function [T,X,E] = ERKSolverErrorEstimationAdaptive(f, tspan, x0, h0, solver, abstol, reltol, varargin)

    % Error Controlling
    epstol = 0.8;
    facmax = 5;
    facmin = 0.1;

    % Size parameters
    t0 = tspan(1);
    tf = tspan(end);
    
    % Initial parameters
    h = h0;
    t = t0;
    x = x0;
    
    % Storage
    T = t;
    X = x';
    E = zeros(1,1);
    
    while t < tf
        if (t + h > tf)
            h = tf - t;
        end
        
        acceptStep = false;
        while ~acceptStep
            % Original step
            [x1 , ~, ~] = RKStep(f, t, x, h, solver, varargin{:});
            
            % Mid steps
            [x_hat12, t_hat12, ~] = RKStep(f, t, x, h/2, solver, varargin{:});
            [x_hat1, ~, e_est] = RKStep(f, t_hat12, x_hat12, h/2, solver, varargin{:});
            
            % Error
            e = abs(x1 - x_hat1);
            r = max(e ./ max(abstol, abs(x_hat1)*reltol));
            
            acceptStep = (r <= 1.0);
            if acceptStep
                t = t + h;
                x = x_hat1;
                
                T = [T;t];
                X = [X;x'];
                E = [E;e_est];
            end
            h =  max(facmin, min((epstol/r)^(1/5), facmax))*h;
        end
    end
end

function [x1, t1, e1] = RKStep(f, t, x, h, solver, varargin)
    % Solver Parameters
    s  = solver.stages;     % Number of stages in ERK method
    AT = solver.AT;         % Transpose of A-matrix in Butcher tableau
    b  = solver.b;          % b-vector in Butcher tableau
    c  = solver.c;          % c-vector in Butcher tableau
    d  = solver.d;

    % Parameters related to constant step size
    hAT = h*AT;
    hb  = h*b;
    hc  = h*c;
    hd  = h*d;
    
    % Size parameter
    nx = length(x);
    
    % Allocate memory
    T  = zeros(1,s);
    X  = zeros(nx,s);
    F  = zeros(nx,s);
    
    % Stage 1
    T(1)   = t;
    X(:,1) = x;
    F(:,1) = feval(f, T(1), X(:,1), varargin{:});
    
    % Stage 2,3,...,s
    T(2:s) = t + hc(2:s);
    for i=2:s
        X(:,i) = x + F(:,1:i-1) * hAT(1:i-1,i);
        F(:,i) = feval(f, T(i), X(:,i), varargin{:});
    end

    % Next step
    t1 = t + h;
    x1 = x + F*hb;
    e1 = F*hd;
    
    
end

function xdot = VanDerPol(t,x,mu)

xdot=zeros(2,1);
xdot(1) = x(2);
xdot(2) = mu*(1-x(1)*x(1))*x(2) - x(1);
end

function Jac = JacVanDerPol(t,x,mu)

Jac = zeros(2,2);
Jac(1,2) = 1;
Jac(2,1) = -2*mu*x(1)*x(2)-1;
Jac(2,2) = mu*(1-x(1)*x(1));
end
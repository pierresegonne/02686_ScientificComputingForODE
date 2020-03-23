mu = 10;
x0 = [2.0; 0.0];
tspan= [0; 5*mu];
options = odeset('Jacobian',@JacVanDerPol,'RelTol',1.0e-6,'AbsTol',1.0e-6);
[Tode,Xode]=ode15s(@VanDerPol, tspan, x0, options, mu);
[X,T] = RK4AdaptiveStep(@VanDerPol, tspan, x0, 1, 1.0e-6, 1.0e-6, [mu]);


function [X,T] = RK4AdaptiveStep(f, tspan, x0, h0, abstol, reltol, varargin)
    % Error Controlling
    epstol = 0.8;
    facmax = 5;
    facmin = 0.1;
    
    % Integration Interval
    t0 = tspan(1);
    tf = tspan(2);
    
    % Initial Conditions
    h = h0;
    t = t0;
    x = x0;
    
    % Storage
    T = t;
    X = x';
    
    while t < tf
        if (t+h>tf)
            h = tf - t;
        end
        
        acceptStep = false;
        while ~acceptStep
            % Original step
            [x1,~] = RK4Step(f, t, x, h, varargin{:});
            
            % Smaller steps
            [x_hat12,t_hat12] = RK4Step(f, t, x, h/2, varargin{:});
            [x_hat1,~] = RK4Step(f, t_hat12, x_hat12, h/2, varargin{:});
            
            % Error
            e = abs(x1 - x_hat1);
            r = max(e ./ max(abstol, abs(x_hat1)*reltol));
            
            acceptStep = (r <= 1.0);
            if acceptStep
                t = t + h;
                x = x_hat1;
                
                T = [T;t];
                X = [X;x'];
            end
            h =  max(facmin, min((epstol/r)^(1/5), facmax))*h;
        end
        
    end
end

function [x1,t1] = RK4Step(f, t, x, h, varargin)
    h2 = h/2;
    alpha = 1/6;
    beta = 1/3;
    
    % 1
    T1 = t;
    X1 = x;
    F1 = feval(f, T1, X1, varargin{:});
    
    % 2
    T2 = t + h2;
    X2 = x + h2*F1;
    F2 = feval(f, T2, X2, varargin{:});
    
    % 3
    T3 = t + h2;
    X3 = x + h2*F2;
    F3 = feval(f, T3, X3, varargin{:});
    
    % 4
    T4 = t + h;
    X4 = x + h*F3;
    F4 = feval(f, T4, X4, varargin{:});
    
    t1 = t + h;
    x1 = x + h*(alpha*F1 + beta*F2 + beta*F3 + alpha*F4);
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
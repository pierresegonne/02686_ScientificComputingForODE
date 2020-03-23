% Recorded params
u = 0.05;
t_incubation = 5.1;
t_infectious = 3.3;
R0 = 2.4;

% model params
alpha = 1/t_incubation;
gamma = 1/t_infectious;
beta = R0 * gamma;

% PID Params
Kp = 0;
Kd = 0;
Ki = 0.01;
umin = 0.01;
umax = 0.8;
ubar = 0;
percentage_hospitalized = 8700 / 580000;% out of the infected, 
%percentage_hospital_beds = 2.5 / 1000;% In fraction of available beds, only 2.5 for 1000 people
percentage_hospital_beds = 1 / 1000;
ybar = percentage_hospital_beds / percentage_hospitalized; % 0.167
Ipid = 0;

% Based on data on 16-03-2020
number_death = 4;
death_rate_dk = 0.003;
nbr_infected = (number_death/death_rate_dk) * 8; % Source https://medium.com/@tomaspueyo/coronavirus-act-today-or-people-will-die-f4d3d9cd99ca
population = 5800000;
i0 = nbr_infected / population;
X0 = [1-i0, 0, i0, 0]';
tspan = [0,200];
params = [u, beta, alpha, gamma];
options = odeset('Jacobian',@SEIRJac,'RelTol',1.0e-6,'AbsTol',1.0e-6);

% Loop with per day evolution
X = reshape(X0,4,1);
T = [0];
U = [u];
for k=1:tspan(end)
    y = X(3,end);
    if k == 1
        yprev = y;
    else
        yprev = X(3,end-1);
    end
    [u, Ipid] = PIDcontroller(y,Ipid,yprev,ybar,ubar,Kp,Ki,Kd,1,umin,umax);
    params(1) = u;
    [Tode,Xode]=ode15s(@SEIR, [k-1, k], X(:,end), options, params);
    % add newly simulated values
    T = [T;Tode];
    X = [X';Xode]';
    U = [U;u];
end

% Add line to not go over for hospital overload.
O = ybar * ones(size(X,2));


function [u,Ipid] = PIDcontroller(y,Ipid,yprev,ybar,ubar,Kp,Ki,Kd,Ts,umin,umax)
    e = ybar - y;
    P = Kp * e;
    D = (Kd/Ts)*(yprev - y);
    v = ubar + P + Ipid + D;
    u = max(umin,min(umax,v));
    Ipid = Ipid + (Ki*Ts)*e;
end

function Xdot = SEIR(t,X,params)
    % Params
    u = params(1);
    beta = params(2);
    alpha = params(3);
    gamma = params(4);
    
    % Derivative
    Xdot = zeros(4,1);
    Xdot(1) = -(1-u)*beta*X(1)*X(3);
    Xdot(2) = (1-u)*beta*X(1)*X(3) - alpha*X(2);
    Xdot(3) = alpha*X(2) - gamma*X(3);
    Xdot(4) = gamma*X(3);
end

function J = SEIRJac(t,X,params)
    % Params
    u = params(1);
    beta = params(2);
    alpha = params(3);
    gamma = params(4);
    
    % Jacobian
    J = zeros(4,4);
    J(1,1) = -(1-u)*beta*X(3);
    J(1,3) = -(1-u)*beta*X(1);
    J(2,1) = (1-u)*beta*X(3);
    J(2,2) = -alpha;
    J(2,3) = (1-u)*beta*X(1);
    J(3,2) = alpha;
    J(3,3) = -gamma;
    J(4,3) = gamma;
end
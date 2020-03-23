k12 = 0.066;
ka1 = 0.006;
ka2 = 0.06;
ka3 = 0.03;
ke = 0.138;
thauD = 40;
thauS = 55;
AG = 0.8;
SI1 = 5.12e-3;
SI2 = 8.2e-4;
SI3 = 5.2e-2;
kb1 = SI1*ka1;
kb2 = SI2*ka2;
kb3 = SI3*ka3;
BW = 70;
MwG = 180.1577;
VI = 0.12*BW;
VG = 0.16*BW;
EGP0 = 0.0161*BW;
F01 = 0.0097*BW;

par = struct('k12', k12, 'ka1', ka1, 'ka2', ka2, 'ka3', ka3, 'ke', ke, 'thauD', thauD, 'thauS', thauS, 'AG', AG, 'kb1', kb1, 'kb2', kb2, 'kb3', kb3, 'BW', BW, 'MwG', MwG, 'VI', VI, 'VG', VG, 'EGP0', EGP0, 'F01', F01);
x0 = zeros(10,1);
T = [0, 300, 600, 900, 1200];
U = [1100, 1100, 1100, 1100];
D = [25, 25, 25, 25, 25];
[Tx,G,I,X] = HovorkaModelSimulation(T,x0,U,D,par);


function [Tx,G,I,X]=HovorkaModelSimulation(T,x0,U,D,par)
% HOVORKAMODELSIMULATION Simulation using the Hovorka model
%
% Syntax: [Tx,G,I,X]=HovorkaModelSimulation(T,x0,U,D,par)
options = odeset('RelTol',1e-6,'AbsTol',1e-6);
nx = length(x0);
N = length(T);
Tx(1) = T(1);
X = x0';
for k=1:N-1
    x = X(end,:)';
    [Tk,Xk]=ode45(@HovorkaModel,[T(k) T(k+1)],x,options,U(:,k),D(:,k),par);
    X = [X; Xk];
    Tx = [Tx; Tk];
end

G = X(:,5)/par.VG;
I = X(:,7);
end

function xdot = HovorkaModel(t,x,u,d,par)
% Indices - variables
% 1 = D1
% 2 = D2
% 3 = s1
% 4 = s2
% 5 = Q1
% 6 = Q2 
% 7 = I
% 8 = x1
% 9 = x2
% 10 = x3

% F01c
G = x(5)/par.VG;
if G >= 4.5
    F01c = par.F01;
else
    F01c = par.F01*G/4.5;
end

% FR
if G >= 9
    FR = 0.003*(G-9)*par.VG;
else
    FR = 0;
end

UG = (1/par.thauD)*x(2);
UI = (1/par.thauS)*x(4);

xdot = zeros(10,1);
xdot(1) = par.AG*1000*d/par.MwG - (1/par.thauD)*x(1); % todo figure out d
xdot(2) = (1/par.thauD)*x(1) - (1/par.thauD)*x(2);
xdot(3) = u - (1/par.thauS)*x(3);
xdot(4) = (1/par.thauS)*x(3) - (par.thauS)*x(4);
xdot(5) = UG - F01c - FR - x(8)*x(5) + par.k12*x(6) + par.EGP0*(1-x(10)); %Q1
xdot(6) = x(8)*x(5) - par.k12*x(6) - x(9)*x(6);
xdot(7) = UI - par.ke*x(7); %I
xdot(8) = -par.ka1*x(8) + par.kb1*x(7);
xdot(9) = -par.ka2*x(9) + par.kb2*x(7);
xdot(10) = -par.ka3*x(10) + par.kb3*x(7);
end


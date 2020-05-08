% Copyright informations
% Teaching material for 02685 course at DTU
% Author: Andrea Capolei acap@dtu.dk


close all
clear all
clc

%% Solve the Van der Pol
mu = 100;
x0 = [2;0];
tspan = [0 5*mu];
h = 0.01;
options.abstol = 1e-6;
options.reltol = options.abstol;
options.ESDIRKTYPE = 'ESDIRK23' %'ESDIRK23' % 'ESDIRK12'
options.IsFixedStepsize = 1

% plotting parameters
linewidth = 2;
fontsize = 14;



tic
[Tout,Xout,~,info1]= ESDIRKintegrator (@vdpProbl,@vdpProblJac,tspan,x0,h,options,mu);
toc
% new figure
figure


%plot solution
subplot(2,1,1)
plot(Tout,Xout(:,1),'o-','linewidth',linewidth);
title(sprintf('mu = %d\nESDIRK23 function eval = %d',mu,info1.nFun),'FontSize',fontsize);
ylabel('x1','FontSize',fontsize)
xlabel('t','FontSize',fontsize)
grid on;
legend('ESDIRK23');

subplot(2,1,2)
plot(Tout,Xout(:,2),'o-','linewidth',linewidth)
ylabel('x2','FontSize',fontsize)
xlabel('t','FontSize',fontsize)
grid on;
legend('ESDIRK23');

% Zoom plot
figure();

box = [70 95 -2.5 -1.5];
subplot(2,1,1)
plot(Tout,Xout(:,1),'o-',[box(1) box(2) box(2) box(1) box(1)],[box(3) box(3) box(4) box(4) box(3)],'r-','linewidth',linewidth);
title('Close up of the step size control','FontSize',fontsize);
ylabel('x1','FontSize',fontsize)
xlabel('t','FontSize',fontsize)
grid on;
legend('ESDIRK23');

subplot(2,1,2)
plot(Tout,Xout(:,1),'o-','linewidth',linewidth);
ylabel('x1','FontSize',fontsize)
xlabel('t','FontSize',fontsize)
grid on;
legend('ESDIRK23');
axis(box);

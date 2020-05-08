% Copyright informations
% Teaching material for 02685 course at DTU
% Author: Andrea Capolei acap@dtu.dk


function [Tout,Xout,Eout,info] = ESDIRKintegrator(fun,jac,tspan,x0,h0, options, varargin) 
% ESDIRKSOLVER  Fixed or PI variable step size ESDIRK solver
%
%                           Solves ODE systems in the form dx/dt = f(t,x)
%                           with x(t0) = x0. 
%
% Syntax:
%  [Tout,Xout,Eout,info] = ESDIRKintegrator(fun,jac,tspan,x0,h0, options, varargin) 

abstol = options.abstol;
reltol = options.reltol;

% stepsize controller parameters
controllerParam.epsilon = 0.8;
controllerParam.tau = 0.1*controllerParam.epsilon;
controllerParam.hmax = 10; % Max increse pr step
controllerParam.hmin = 0.1;% Min increse pr step
%==========================================================================     
if options.IsFixedStepsize
    stepSizeControlleFunc = @fixedStepsizeController;
else
    stepSizeControlleFunc = @variableStepsizeController;
end

% load ESDIRK parameters
ESDIRKparam = ESDIRKparameter(options.ESDIRKTYPE);

stages = ESDIRKparam.stages;
AT = ESDIRKparam.AT;
c = ESDIRKparam.c;
b = ESDIRKparam.b;
bhat = ESDIRKparam.bhat;
d = ESDIRKparam.d;
gamma = ESDIRKparam.gamma;
%==========================================================================


% error and convergence controllers
epsilon = 0.8;
tau = 0.1*epsilon;


% Size parameters
t0  = tspan(1);          % Initial time
tf = tspan(end);        % Final time

nx = length(x0);        % System size (dim(x))

% Allocate memory
T  = zeros(1,stages);        % Stage T
X  = zeros(nx,stages);       % Stage X
F  = zeros(nx,stages);       % Stage F
I = eye(nx);            % Identity matrix

info = struct('nFun', 0, ... % Function evaluations
              'nJac',0);     % Jacobi evaluations          


tcurr = t0;
hcurr = h0;
xCurr = x0;
F(:,stages) =  feval(fun,T(1),xCurr,varargin{:});

nSteps = 1;
currNumOfDivergedSteps = 0;

maxFailsForStep = 20;
nItermaxInStages = 20;

Tout(1) = tcurr;
Xout(1,:) = xCurr';
while (tcurr<tf) && (currNumOfDivergedSteps< maxFailsForStep)
    if (tcurr+hcurr > tf)
        hcurr = tf-tcurr; 
    end
    
    
    % Compute the Jacobian and iteration matrix
    dfdx = feval(jac,tcurr,xCurr,varargin{:});
    M = I - hcurr*gamma*dfdx;
    [L,U] = lu(M);
    
    % Stage 1
    T(1)   = tcurr;
    X(:,1) = xCurr;
    F(:,1) = F(:,stages);
    
    %% ==== ESDIRK Step Integration ===========================================
    i1= 2;
    
    StepNotDiverged = true;
    while (i1 <= stages) && StepNotDiverged                     
        
        % ==== Inexact Newton method ===========================================
        psi = xCurr +  F(:,1:i1-1) * hcurr*AT(1:i1-1,i1);
        T(i1) = tcurr + hcurr*c(i1);
        X(:,i1) = xCurr + hcurr*c(i1)*F(:,1); % Initial Euler step guess
        F(:,i1) = feval(fun,T(i1),X(:,i1),varargin{:});

        R = X(:,i1) - hcurr*gamma*F(:,i1) - psi;
        
        rNewton = norm(R./(abstol + abs(X(:,i1)).*reltol),inf);

        nIterInStages = 0;
        rNewtonPrev = rNewton;
        while (rNewton > tau) && StepNotDiverged
            
            % next iteration 
            dX = U\(L\-R);
            X(:,i1) = X(:,i1) + dX;
            
            %= compute convergence ========================================
            F(:,i1) = feval(fun,T(i1),X(:,i1),varargin{:});
            R = X(:,i1) - hcurr*gamma*F(:,i1) - psi;
                        
            rNewton = norm(R./(abstol + abs(X(:,i1)).*reltol),inf);
            %=============================================================            
            nIterInStages = nIterInStages + 1;
           
            
            %= monitor convergence rate ===================================
            alpha = rNewton/rNewtonPrev;            
            newtonStepDiverged = (alpha >=1);
            %=============================================================            
            
            
            %= Diverging ==================================================
            AllowedNewtonIterationNumber = (nIterInStages < nItermaxInStages);                    
            StepNotDiverged = (AllowedNewtonIterationNumber) && ~newtonStepDiverged;
            %==============================================================

        end
            i1= i1+1;
        %==================================================================    
    end
    
    % Error estimation
    errEst = F*d*hcurr;
    rCurr = norm(errEst./(abstol + abs(X(:,stages)).*reltol),inf);
    %======================================================================
    
    % call stepsize controller
    stepsContr.hcurr = hcurr;
    stepsContr.rCurr = rCurr;
   [AcceptStep, stepsContr] = stepSizeControlleFunc(StepNotDiverged, (nSteps>1), stepsContr, controllerParam);
    hcurr = stepsContr.hcurr;
    
    if (AcceptStep)
                
        % Next step
        tcurr = T(stages);
        xCurr = X(:,stages);

        % Save output
        Tout(nSteps+1) = tcurr;
        Xout(nSteps+1,:) = xCurr';
        Eout(nSteps+1,:) = errEst';
        
        currNumOfDivergedSteps = 0;
        
        nSteps = nSteps+1;
    else
        currNumOfDivergedSteps = currNumOfDivergedSteps + 1;

    end
    %==== ESDIRK Step Integration (end) =======================================
    
end






function [AcceptStep, stepsContr] = fixedStepsizeController(StepNotDiverged, IsFirstStep, stepsContr, controllerParam)	
    epsilon= controllerParam.epsilon;
    hmax=  controllerParam.hmax;
    hmin=  controllerParam.hmin;
     	
    
    prevStepTemp = stepsContr.hcurr;
        
    if(StepNotDiverged)
        stepsContr.hcurr = stepsContr.hcurr;
        AcceptStep = true;       
    else
        stepsContr.hcurr = stepsContr.hcurr/2;
        AcceptStep = false;
    end
    
    
    % save previous values
    stepsContr.rPrev = stepsContr.rCurr;
    stepsContr.hPrev = prevStepTemp;
    
    
function [AcceptStep, stepsContr] = variableStepsizeController(StepNotDiverged, IsFirstStep, stepsContr, controllerParam)	
    epsilon= controllerParam.epsilon;
    hmax=  controllerParam.hmax;
    hmin=  controllerParam.hmin;
     	    
    
    rCurr= stepsContr.rCurr;
    hcurr = stepsContr.hcurr;
    if isfield(stepsContr, 'hPrev')
         rPrev = stepsContr.rPrev;
         hPrev = stepsContr.hPrev;
    end
    
    
    if(StepNotDiverged)
       % stepsContr.hcurr = stepsContr.hcurr;
        AcceptStep = (rCurr <= 1);
        
        if AcceptStep
            if IsFirstStep
                hch = (epsilon/rCurr)^(1/3)*hcurr;
                hcurr = min(max(hch, hcurr*hmin), hcurr*hmax);
            else            
              hch = (hcurr/hPrev)*((epsilon/rCurr)^(1/3))*((rPrev/rCurr)^(1/3))*hcurr;
               hcurr = min(max(hch, hcurr*hmin), hcurr*hmax);
            end
        else % rejected step
            hcurr = (epsilon/rCurr)^(1/3)*hcurr;
        end 
        
    else % not converged step
        
        stepsContr.hcurr = stepsContr.hcurr/2;
        AcceptStep = false;
    end
    
    
    % save previous values
    stepsContr.rPrev = stepsContr.rCurr;
    stepsContr.hPrev = stepsContr.hcurr;
    
    stepsContr.hcurr = hcurr;
    
    
    
    
 function ESDIRKparam = ESDIRKparameter(ESDIRKTYPE)

if strcmp(ESDIRKTYPE, 'ESDIRK23')
    % Solver Parameters for ESDIRK32
    stages = 3;                                      % Number of stages in ERK method
    gamma = 1-1/sqrt(2);
    a31 = (1-gamma)/2;
    AT = [0 gamma a31;0 gamma a31;0 0 gamma];   % Transpose of A-matrix in Butcher tableau
    c  = [0; 2*gamma; 1];                       % c-vector in Butcher tableau
    b  = AT(:,3);                               % b-vector in Butcher tableau
    bhat = [(6*gamma-1)/(12*gamma); ...
             1/(12*gamma*(1-2*gamma)); ...
            (1-3*gamma)/(3*(1-2*gamma))];
    d  = b-bhat;                                % d-vector in Butcher tableau
elseif strcmp(ESDIRKTYPE, 'ESDIRK12')
    gamma = 1;
    stages = 2;                                      % Number of stages in ERK method
    AT = [0 0;0 gamma];   % Transpose of A-matrix in Butcher tableau
    c  = [0; 1];                       % c-vector in Butcher tableau
    b  = AT(:,2);                               % b-vector in Butcher tableau
    bhat = [1/2; 1/2];
    d  = b-bhat;                                % d-vector in Butcher tableau
else
    error('not recognised ESDIRK type');
end


ESDIRKparam.stages = stages;
ESDIRKparam.AT = AT;
ESDIRKparam.gamma = gamma;
ESDIRKparam.c = c;
ESDIRKparam.b = b;
ESDIRKparam.bhat = bhat;
ESDIRKparam.d = d;



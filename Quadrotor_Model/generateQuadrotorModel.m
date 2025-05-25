function generateQuadrotorModel(params)
    
    IxxVal = params.Ixx;
    IyyVal = params.Iyy;
    IzzVal = params.Izz;
    
    kVal = params.k;
    lVal = params.l;
    mVal = params.m;
    bVal = params.b;
    gVal = params.g;
    
    % Create symbolic functions for time-dependent angles
    % roll angle
    % pitch angle
    % yaw angle
    syms roll(t) pitch(t) yaw(t)
    
    % Transformation matrix for angular velocities from inertial frame
    % to body frame
    W = [ 1,  0,        -sin(pitch);
          0,  cos(roll),  cos(pitch)*sin(roll);
          0, -sin(roll),  cos(pitch)*cos(roll) ];
    
    % Rotation matrix R_ZYX from body frame to inertial frame
    R = rotationMatrixEulerZYX(roll, pitch, yaw);
    
    % Create symbolic variables for diagonal elements of inertia matrix
    syms Ixx Iyy Izz
    
    % Jacobian that relates body frame to inertial frame velocities
    I = [Ixx, 0, 0; 0, Iyy, 0; 0, 0, Izz];
    J = W.'*I*W;
    
    % Coriolis matrix
    dJ_dt = diff(J);
    h_dot_J = [diff(roll,t), diff(pitch,t), diff(yaw,t)]*J;
    grad_temp_h = transpose(jacobian(h_dot_J,[roll pitch yaw]));
    C = dJ_dt - 1/2*grad_temp_h;
    C = subsStateVars(C,t);
    
    % Define fixed parameters and control input
    % k: lift constant
    % l: distance between rotor and center of mass
    % m: quadrotor mass
    % b: drag constant
    % g: gravity
    % ui: squared angular velocity of rotor i as control input
    syms k l m b g u1 u2 u3 u4
    
    % Torques in the direction of phi, theta, psi
    tau_beta = [l*k*(-u2+u4); l*k*(-u1+u3); b*(-u1+u2-u3+u4)];
    
    % Total thrust
    T = k*(u1+u2+u3+u4);
    
    % Create symbolic functions for time-dependent positions
    syms x(t) y(t) z(t)
    
    % Create state variables consisting of positions, angles,
    % and their derivatives
    state = [x; y; z; roll; pitch; yaw; diff(x,t); diff(y,t); ...
        diff(z,t); diff(roll,t); diff(pitch,t); diff(yaw,t)];
    state = subsStateVars(state,t);
    f = [ % Set time-derivative of the positions and angles
          state(7:12);
    
          % Equations for linear accelerations of the center of mass
          -g*[0;0;1] + R*[0;0;T]/m;
    
          % Euler–Lagrange equations for angular dynamics
          inv(J)*(tau_beta - C*state(10:12))
    ];
    
    f = subsStateVars(f,t);
    
    % Replace fixed parameters with given values here
    
    f = subs(f, [Ixx Iyy Izz k l m b g], ...
        [IxxVal IyyVal IzzVal kVal lVal mVal bVal gVal]);
    f = simplify(f);
    
    A = jacobian(f,state);
    control = [u1; u2; u3; u4];
    B = jacobian(f,control);
    
    % Create QuadrotorStateFcn.m with current state and control
    % vectors as inputs and the state time-derivative as outputs
    matlabFunction(f,'File','quadrotor', ...
        'Vars',{state,control});
    
    % Create QuadrotorStateJacobianFcn.m with current state and control
    % vectors as inputs and the Jacobians of the state time-derivative
    % as outputs
    matlabFunction(A,B,'File','quadrotor_grad', ...
        'Vars',{state,control});

end

function [Rz,Ry,Rx] = rotationMatrixEulerZYX(phi,theta,psi)
% Euler ZYX angles convention
    Rx = [ 1,           0,          0;
           0,           cos(phi),  -sin(phi);
           0,           sin(phi),   cos(phi) ];
    Ry = [ cos(theta),  0,          sin(theta);
           0,           1,          0;
          -sin(theta),  0,          cos(theta) ];
    Rz = [cos(psi),    -sin(psi),   0;
          sin(psi),     cos(psi),   0;
          0,            0,          1 ];
    if nargout == 3
        % Return rotation matrix per axes
        return;
    end
    % Return rotation matrix from body frame to inertial frame
    Rz = Rz*Ry*Rx;
end

function stateExpr = subsStateVars(timeExpr,var)
    if nargin == 1 
        var = sym("t");
    end
    repDiff = @(ex) subsStateVarsDiff(ex,var);
    stateExpr = mapSymType(timeExpr,"diff",repDiff);
    repFun = @(ex) subsStateVarsFun(ex,var);
    stateExpr = mapSymType(stateExpr,"symfunOf",var,repFun);
    stateExpr = formula(stateExpr);
end

function newVar = subsStateVarsFun(funExpr,var)
    name = symFunType(funExpr);
    name = replace(name,"_Var","");
    stateVar = "_" + char(var);
    newVar = sym(name + stateVar);
end

function newVar = subsStateVarsDiff(diffExpr,var)
    if nargin == 1 
      var = sym("t");
    end
    c = children(diffExpr);
    if ~isSymType(c{1},"symfunOf",var)
      % not f(t)
      newVar = diffExpr;
      return;
    end
    if ~any([c{2:end}] == var)
      % not derivative wrt t only
      newVar = diffExpr;
      return;
    end
    name = symFunType(c{1});
    name = replace(name,"_Var","");
    extension = "_" + join(repelem("d",numel(c)-1),"") + "ot";
    stateVar = "_" + char(var);
    newVar = sym(name + extension + stateVar);
end



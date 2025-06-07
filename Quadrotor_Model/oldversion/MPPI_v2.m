GRADIENTS=@quadrotor_grad;
DYNAMICS=@quadrotor;

nX = 12;%number of states
nU = 4;%number of inputs

% quadrotor w^2 to force/torque matrix
kf = 8.55*(1e-6)*91.61;
L = 0.17;
b = 1.6*(1e-2)*91.61;
m = 0.716;
g = 9.81;

A = [kf, kf, kf, kf; ...
    0, L*kf, 0, -L*kf; ...
    -L*kf, 0, L*kf, 0; ...
    b, -b, b, -b];


%initial conditions
x0= [0;0;0;0;0;0;0;0;0;0;0;0];
xd= [5;5;5;0;0;0;0;0;0;0;0;0];

Q = 1*diag([5, 5, 15, ...    % x, y, z (position)
         3,  3,  5, ...    % roll, pitch, yaw (orientation)
         0,  0,  0, ...   % vx, vy, vz (velocity)
         0,  0,  0]);  % wx, wy, wz (angular velocity)
R = 0*diag([8*1e-3,4,4,4]);

Qf = 1*diag([5, 5, 15, ...    % x, y, z (position)
         3,  3,  5, ...    % roll, pitch, yaw (orientation)
         0,  0,  0, ...   % vx, vy, vz (velocity)
         0,  0,  0]);  % wx, wy, wz (angular velocity)

% MPPI parameters
num_samples = 3000;
dt = 0.1;
time_steps = 2;
lambda = 100;
rho = 100;
var = diag([5, 5*1e-3, 5*1e-3, 5*1e-3]);

xtraj = [];
utraj = zeros(nU, time_steps);
utraj(1,:) = m*g;
optimal_u = [];
x  = x0;
xtraj(:,1) = x0;
Jc = 0;
optimal_u = [];

for iter = 1:300
    x
    J = [];
    for i = 1:num_samples
        du = sqrt(var) * randn(nU, time_steps)/(sqrt(rho*dt)); 
        dU{i} = du;
        u = utraj +  du;

        [~,Jk,xf] = sampleTrajectoryCosts(x,xd,u,Q, R, Qf, dt,time_steps,DYNAMICS);

        Ji = getCostToGo(Jk);
        J = [J;Ji];
    end

    minJ = min(J);
   
    for j = 1:time_steps
        ss = 0;
        su = 0;
        for k = 1:num_samples
            ss = ss + exp(-1/lambda*(J(k)-minJ));
            su = su + exp(-1/lambda*(J(k)-minJ))*dU{k}(:,j);
        end

        utraj(:,j) = utraj(:,j) + su/ss;
    end

    optimal_u = [optimal_u,utraj(:,1)];

    x = x + DYNAMICS(x,utraj(:,1))*dt;

    utraj = [utraj(:,2:end),m*g*ones(nU,1)];

end

%% HELPER FUNCTIONS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% roll out the trajectories and sample from the costs (FILL IN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J,Jk,x] = sampleTrajectoryCosts(x0,xd,u,Q,R,Qf,dt,N,DYNAMICS)
    x(:,1) = x0;
    t = 0;
    for k= 1:N
        Jk(k) = cost(x(:,k), u(:,k), xd, Q, R, dt); %cost at each time step
        x(:,k+1)= x(:,k) + DYNAMICS(x(:,k),u(:,k))*dt; %integrate the one step dynamics
        t = t+dt;
    end
    Jk(N+1) = final_cost(x(:,N+1), xd, Qf); %compute the final cost
    
    J = sum(Jk);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute running cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = cost(x,u,xd,Q,R,dt)

J = ((x-xd)'*Q*(x-xd)+1/2*u'*R*u)*dt;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute final cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = final_cost(x,xd,Qf)
J = (x-xd)'*Qf*(x-xd);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the cost-to-go for each time step from the instantaneous cost values Jk (FILL IN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = getCostToGo(Jk)
S = sum(Jk);
end


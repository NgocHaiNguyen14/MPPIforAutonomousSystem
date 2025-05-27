GRADIENTS=@quadrotor_grad;
DYNAMICS=@quadrotor;

T = 4; %time horizon
dt = 0.05;%time step
N = floor(T/dt);% number of time steps
nX = 12;%number of states
nU = 4;%number of inputs

%initial conditions
x0= [0;0;0;0;0;0;0;0;0;0;0;0];

xtraj=zeros(nX,N);%state-trajectory
utraj=zeros(nU,N-1);%input-trajectory
ktraj= zeros(nU,N);%gain-trajectory
Ktraj = zeros(nU,nX,N);%gain-trajectory
Q = 0*diag([...
    0.2, 0.2, 0.2, ...    % x, y, z (position)
     0.1,  0.1,  0.1, ...    % vx, vy, vz (velocity)
     0.2,  0.2,  0.2, ...   % roll, pitch, yaw (orientation)
     0.1,  0.1,  0.1]);  % wx, wy, wz (angular velocity)
Qf = 0.01*diag([...
    10, 10, 10, ...  % x, y, z
     1,  1,  1, ... % vx, vy, vz
     10,  10,  10, ... % roll, pitch, yaw
      1,   1,   1]); % wx, wy, wz
R = 0*diag([0.01, 0.01, 0.01, 0.01]); % penalize control effort (w1² to w4²)

xd = [1;10;5;0;0;0;0;0;0;0;0;0];%desired state

%call iterative lqr
disp("1. Solving iterative LQR")
[xtraj, utraj, ktraj, Ktraj] =iLQR_quadrotor(x0, xtraj, utraj, ktraj, Ktraj,N,dt, Q, R, Qf,xd, DYNAMICS, GRADIENTS);
disp("Solved iLQR")
disp("Start MPPI")

%Setup for MPPI
t = zeros(1,N);%time
x = zeros(nX,N);%state
optimal_u = [];%input

num_samples = 100;
lambda = 10;
var = 10;
rho = 100;

for i = 1:(N-1)
    disp("Optimizing step #: " + i)
    u = zeros(nU,N-1);%input
    J = zeros(num_samples,1);
    sampleU = zeros(nU, N-1, num_samples);
    for j = 1:num_samples
        u = utraj + du_rollouts(var, nU, N, rho, dt);
        u = max(u, 0);
        if ~isempty(optimal_u)
            for idx = 1 : size(optimal_u,2)
                u(:,idx) = optimal_u(:,idx);
            end
        end

        sampleU(:,:,j) = u;
        new_xtraj = rollouts(x0, xtraj, u, dt, DYNAMICS);
        rollout_cost = traj_cost(x0, xd, new_xtraj, u, N, Q, R, Qf, dt);
        J(j) = rollout_cost;
    end
    expS = exp(-J/lambda);
    weightsum = 0;
    weightinputsum = 0;
    for k = 1:num_samples
        weightsum = weightsum + expS(k);
        weightinputsum = weightinputsum + expS(k) * sampleU(:,:,k);
    end
    
    avginput = weightinputsum/weightsum;
    
    optimal_u = [optimal_u, avginput(:,i)];
   
end

x(:,1) = x0;

for k =1:N-1
    xdot = DYNAMICS(x(:,k),optimal_u(:,k));
    x(:,k+1)=x(:,k)+ xdot*dt;
    t(k+1) = t(k)+dt;
end 



%% ROLLOUTS delta u
function du = du_rollouts(var, nU, N, rho, dt)
    du = sqrt(var) * randn(nU, N-1)/(sqrt(rho) * sqrt(dt));
end

%% FORWARD DYNAMICS
function new_xtraj = rollouts(x0, xtraj, u, dt, DYNAMICS)
    new_xtraj = xtraj;
    new_xtraj(:,1) = x0;
    
    for i = 1:length(u)
        xdot = DYNAMICS(new_xtraj(:,i), u(:,i));
        new_xtraj(:,i+1) = new_xtraj(:,i) + xdot * dt;
    end
end

%% COST OF TRAJECTORY
function total_cost = traj_cost(x0, xd, traj, u, N, Q, R, Qf, dt)
    total_cost = 0;
    
    for i = 1:N-1
        total_cost = total_cost + cost(traj(:,i), u(:,i), Q, R, dt);
    end
    
    total_cost = total_cost + final_cost(traj(:,N), xd, Qf);
end

function J = cost(x, u, Q, R, dt)
    J = (x'*Q*x+u'*R*u)*dt;
end

function J = final_cost(x, xd, Qf)
    J = (x-xd)'*Qf*(x-xd);
end
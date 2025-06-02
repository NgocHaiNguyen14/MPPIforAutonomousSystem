GRADIENTS=@quadrotor_grad;
DYNAMICS=@quadrotor;

T = 4; %time horizon
dt = 0.05;%time step
N = floor(T/dt);% number of time steps
nX = 12;%number of states
nU = 4;%number of inputs

%initial conditions
x0= [0;0;0;0;0;0;0;0;0;0;0;0];

xref=zeros(nX,N);%state-trajectory
uref=zeros(nU,N-1);%input-trajectory
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
[xref, uref, kref, Kref] =iLQR_quadrotor(x0, xref, uref, ktraj, Ktraj,N,dt, Q, R, Qf,xd, DYNAMICS, GRADIENTS);

xout = xref;
%Output trajectory with noise
for i = 1:size(uref,2)
    xout(:,i+1) = xout(:,i) + dynamics_noise(xout(:,i),uref(:,i), DYNAMICS)*dt;
end

% plot_traj(xref, xout)

disp("Solved iLQR")
disp("2. Start MPPI Optimization")

%Setup for MPPI
t = zeros(1,N);%time
xtraj = zeros(nX,N);%state
optimal_u = [];%input

x = x0;
steps = 5;

for i = 1:N-1
    xtraj(:,i) = x;
    incr = i+steps;
    if incr > N
       incr = N;
       steps = steps - 1;
    end

    u = MPPI(Q,R,Qf,x, xref(:,i+steps), uref(:,i:i+steps-1), steps, dt, DYNAMICS);
    optimal_u(:,i) = u;

    x =  x + dynamics_noise(x, u, DYNAMICS)*dt;
end

xtraj(:,N) = x;


%% Helper Functions %%

function dxout = dynamics_noise(xin, u, DYNAMICS)
    noise_std = 2;

    % Initialize noise to zero
    noise = zeros(size(xin));

    % Add Gaussian noise only to the position states (first 3 rows)
    noise(1:3) = noise_std * randn(3, 1);

    % Compute next state with added noise on position
    dxout = DYNAMICS(xin, u) + noise;
end

function step_u = MPPI(Q, R, Qf, x, xd, uref, steps, dt, DYNAMICS)
    num_samples = 10000;
    lambda = 100;
    var = 20;
    rho = 10;
    nU = size(uref, 1);
    nX = size(x,1);

    xtraj = zeros(nX, steps);
    xtraj(:,1) = x;

    ss = [];
    su = [];

    for i = 1: num_samples
        du = du_rollouts(var, nU, steps, rho, dt);
        u = uref + du;
        new_xtraj = rollouts(x, xtraj, u, dt, DYNAMICS);
        rolloutCost = traj_cost(xd, new_xtraj, u, steps, Q, R, Qf, dt);
        ss(i) = exp(-1/lambda * rolloutCost);
        su(:,i) = ss(i) * u(:,1);
    end

    step_u = sum(su,2) / sum(ss);

end

%% ROLLOUTS delta u
function du = du_rollouts(var, nU, N, rho, dt)
    du = sqrt(var) * randn(nU, N)/(sqrt(rho) * sqrt(dt));
end

%% FORWARD DYNAMICS
function new_xtraj = rollouts(x0, xtraj, u, dt, DYNAMICS)
    new_xtraj = xtraj;
    new_xtraj(:,1) = x0;
    
    for i = 1:size(u,2)
        xdot = DYNAMICS(new_xtraj(:,i), u(:,i));
        new_xtraj(:,i+1) = new_xtraj(:,i) + xdot * dt;
    end
end

%% COST OF TRAJECTORY
function total_cost = traj_cost(xd, traj, u, N, Q, R, Qf, dt)
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


function plot_traj(xtraj, noisyTraj)
    % Extract positions
    pos = xtraj(1:3, :);
    pos_noisy = noisyTraj(1:3, :);

    figure;
    plot3(pos(1, :), pos(2, :), pos(3, :), 'b-', 'LineWidth', 2); hold on;
    plot3(pos_noisy(1, :), pos_noisy(2, :), pos_noisy(3, :), 'r--', 'LineWidth', 2);
    grid on; axis equal;
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    legend('Original Trajectory', 'Noisy Trajectory');
    title('3D Position Trajectory of Quadrotor');
end

DYNAMICS=@vtol1;

nX = 12;%number of states
nU = 3;%number of inputs

%initial conditions
x0= [0;0;2;0;0;0;0;0;0;0;0;0];
xd= [10;10;10;0;0;0;0;0;0;0;0;0];

% Initialization
num_samples = 1000;
N = 150;

utraj = zeros(nU, N-1);
utraj(1,:) = 5000;
uOpt = [];
xf = [];
dt = 0.02;
lambda = 10;
nu = 1000;
covu = diag([100,3e-2,3e-2]);

xtraj = zeros(nX, N);
R = lambda*inv(covu);

max_cost = 100;

x = x0;

%% Run MPPI Optimization
for iter = 1:200
    xf = [xf,x]; % Append the simulated trajectory
    Straj = zeros(1,num_samples); % Initialize cost of rollouts
    
    % Start the rollouts and compute rollout costs
    for k = 1:num_samples
        du = covu*randn(nU, N-1);
        dU{k} = du;
        xtraj = [];
        xtraj(:,1) = x;
        for t = 1:N-1
            u = utraj(:,t);
            xtraj(:,t+1) = xtraj(:,t) + DYNAMICS(xtraj(:,t), u+du(:,t))*dt;
            Straj(k) = Straj(k) + runningCost(xtraj(:,t), xd, R, u, du(:,t), nu);
        end
        Straj(k) = Straj(k) + finalCost(xtraj(:,N), xd);
    end

    minS = min(Straj) % Minimum rollout cost

    epsilon = 1e-6;
    Straj = min(Straj, max_cost * max(minS, epsilon));
    
    % Update the nominal inputs
    for t = 1:N-1
        ss = 0;
        su = 0;
        for k = 1:num_samples
            ss = ss + exp(-1/lambda*(Straj(k)-minS));
            su = su + exp(-1/lambda*(Straj(k)-minS))*dU{k}(:,t);
        end
        
        utraj(:,t) = utraj(:,t) + su/ss;
    end
    
    % Execute the utraj(0)
    x = x + DYNAMICS(x, utraj(:,1))*dt;
    uOpt = [uOpt, utraj(:,1)];
    
    % Shift the nominal inputs 
    for t = 2:N-1
        utraj(:,t-1) = utraj(:,t);
    end
    utraj(:,N-1) = [5000;0;0];
    
    x(1:3) %Current distance to target
    x(6)
end

%% Helper functions
function J = runningCost(x, xd, R, u, du, nu)
    Q = diag([2.5, 2.5, 20, 1, 1, 15, zeros(1, 6)]);
    qx = (x-xd)'*Q*(x-xd);
    J = qx + 1/2*u'*R*u + (1-1/nu)/2*du'*R*du + u'*R*du;
end

function J = finalCost(xT,xd)
    Qf = 20*diag([2.5, 2.5, 20, 1, 1, 15, zeros(1, 6)]);
    J = (xT-xd)'*Qf*(xT-xd);
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

% iLQR implementation for VTOL with quaternion-based dynamics
clear all;
close all;

% Define dynamics function
DYNAMICS = @vtol3_quaternion;

% State and control dimensions
nX = 13; % [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
nU = 4;  % [thrust1, thrust2, moment1, moment2]

% Initial and desired states
x0 = [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0]; % Start at origin, identity quaternion
xd = [100; 100; 50; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0]; % Target position, identity quaternion

% Simulation parameters
N = 150;          % Horizon length
dt = 0.02;        % Time step
maxIter = 400;    % Maximum outer iterations
maxILQRIter = 50; % Maximum iLQR iterations per step
lambda = 1e-4;    % Regularization for Quu
tol = 1e-4;       % Convergence tolerance

% Cost matrices
Q = diag([2.5, 2.5, 20, 0, 0, 0, 1, 1, 15, 1, 1, 15]); % For 12D error state
Qf = 20 * Q; % Final cost weight
R = diag([4, 4, 10000/3, 10000/3]); % Control cost, from lambda*inv(covu)

% Initialize variables
x = x0; % Current state
utraj = zeros(nU, N-1); % Initial control sequence
xf = []; % State history
uOpt = []; % Control history

% Main receding horizon loop
for iter = 1:maxIter
    % Initialize nominal trajectory from current state
    xtraj = zeros(nX, N);
    xtraj(:,1) = x;
    
    % Forward simulation with current controls
    for t = 1:N-1
        xtraj(:,t+1) = xtraj(:,t) + DYNAMICS(xtraj(:,t), utraj(:,t)) * dt;
        % Normalize quaternion
        q = xtraj(7:10,t+1);
        q = q / norm(q);
        if q(1) < 0
            q = -q;
        end
        xtraj(7:10,t+1) = q;
    end
    
    % Compute initial cost
    J = 0;
    for t = 1:N-1
        J = J + runningCost(xtraj(:,t), xd, Q, R, utraj(:,t));
    end
    J = J + finalCost(xtraj(:,N), xd, Qf);
    
    % iLQR optimization
    J_prev = inf;
    for iLQR_iter = 1:maxILQRIter
        % Backward pass
        Vx = finalCostGradient(xtraj(:,N), xd, Qf);
        Vxx = finalCostHessian(xtraj(:,N), xd, Qf);
        K = zeros(nU, nX, N-1); % Feedback gains
        k = zeros(nU, N-1);     % Feedforward terms
        
        for t = N-1:-1:1
            [A, B] = linearizeDynamics(DYNAMICS, xtraj(:,t), utraj(:,t), dt, 1e-6);
            [l, lx, lu, lxx, luu, lux] = runningCostDerivatives(xtraj(:,t), xd, Q, R, utraj(:,t));
            
            Qx = lx + A' * Vx;
            Qu = lu + B' * Vx;
            Qxx = lxx + A' * Vxx * A;
            Quu = luu + B' * Vxx * B + lambda * eye(nU); % Regularized
            Qux = lux + B' * Vxx * A;
            
            K(:,:,t) = -Quu \ Qux;
            k(:,t) = -Quu \ Qu;
            
            Vx = Qx + K(:,:,t)' * Qu + Qux' * k(:,t);
            Vxx = Qxx + K(:,:,t)' * Qux + Qux' * K(:,:,t);
            Vxx = (Vxx + Vxx') / 2; % Ensure symmetry
        end
        
        % Forward pass with line search
        alpha = 1;
        x_new = zeros(nX, N);
        x_new(:,1) = x;
        u_new = zeros(nU, N-1);
        J_new = 0;
        
        for ls = 1:10
            J_new = 0;
            for t = 1:N-1
                u_new(:,t) = utraj(:,t) + alpha * k(:,t) + K(:,:,t) * (x_new(:,t) - xtraj(:,t));
                % Clip controls
                u_new(1,t) = max(u_new(1,t), 0);
                u_new(2,t) = max(u_new(2,t), 0);
                u_new(3:4,t) = min(max(u_new(3:4,t), -15), 15);
                
                x_new(:,t+1) = x_new(:,t) + DYNAMICS(x_new(:,t), u_new(:,t)) * dt;
                q = x_new(7:10,t+1);
                q = q / norm(q);
                if q(1) < 0
                    q = -q;
                end
                x_new(7:10,t+1) = q;
                
                J_new = J_new + runningCost(x_new(:,t), xd, Q, R, u_new(:,t));
            end
            J_new = J_new + finalCost(x_new(:,N), xd, Qf);
            
            if J_new < J
                xtraj = x_new;
                utraj = u_new;
                break;
            end
            alpha = alpha * 0.5;
        end
        
        % Check convergence
        if abs(J - J_new) < tol
            break;
        end
        J = J_new;
    end
    
    % Apply first control and advance state
    u = utraj(:,1);
    x = x + DYNAMICS(x, u) * dt;
    q = x(7:10);
    q = q / norm(q);
    if q(1) < 0
        q = -q;
    end
    x(7:10) = q;
    
    % Store history
    xf = [xf, x];
    uOpt = [uOpt, u];
    
    % Shift control sequence
    utraj(:,1:end-1) = utraj(:,2:end);
    utraj(:,end) = zeros(nU,1);
    
    % Check if target reached
    dist = norm(x(1:3) - xd(1:3));
    fprintf('Iter %d: Dist to target = %.2f m, Cost = %.4f\n', iter, dist, J);
    if dist < 10
        fprintf('Target reached at iteration %d\n', iter);
        break;
    end
end

%% Visualization
time_steps = 0:dt:(size(xf,2)-1)*dt;
time_control = 0:dt:(size(uOpt,2)-1)*dt;

figure('Position', [100, 100, 1400, 900]);

% 3D Trajectory
subplot(3,3,1);
plot3(xf(1,:), xf(2,:), xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot3(x0(1), x0(2), x0(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(xd(1), xd(2), xd(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Trajectory');
legend('Path', 'Start', 'Target');

% X-Y Trajectory with Orientation
subplot(3,3,2);
plot(xf(1,:), xf(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd(1), xd(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
q = xf(7:10,end);
R = quat2rotm(q');
rect = [-1.5 -0.75; 1.5 -0.75; 1.5 0.75; -1.5 0.75; -1.5 -0.75];
rect_rot = (R(1:2,1:2) * rect')' + xf(1:2,end)';
plot(rect_rot(:,1), rect_rot(:,2), 'k-', 'LineWidth', 2);
grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('X-Y Trajectory with VTOL');
axis equal;

% Altitude
subplot(3,3,3);
plot(time_steps, xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot([0 time_steps(end)], [xd(3) xd(3)], 'r--');
grid on;
xlabel('Time (s)'); ylabel('Z (m)');
title('Altitude');

% Velocities
subplot(3,3,4);
plot(time_steps, xf(4:6,:), 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Velocities');
legend('vx', 'vy', 'vz');

% Quaternion
subplot(3,3,5);
plot(time_steps, xf(7:10,:), 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Quaternion');
title('Quaternion Components');
legend('qw', 'qx', 'qy', 'qz');

% Euler Angles
subplot(3,3,6);
euler = zeros(3, size(xf,2));
for i = 1:size(xf,2)
    [roll, pitch, yaw] = quat2angle(xf(7:10,i)', 'XYZ');
    euler(:,i) = [roll; pitch; yaw] * 180/pi;
end
plot(time_steps, euler, 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Angle (deg)');
title('Euler Angles');
legend('Roll', 'Pitch', 'Yaw');

% Angular Velocities
subplot(3,3,7);
plot(time_steps, xf(11:13,:), 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Ang Vel (rad/s)');
title('Angular Velocities');
legend('wx', 'wy', 'wz');

% Control Inputs
subplot(3,3,8);
plot(time_control, uOpt, 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Control');
title('Control Inputs');
legend('u1', 'u2', 'u3', 'u4');

% Quaternion Norm
subplot(3,3,9);
qnorm = sqrt(sum(xf(7:10,:).^2, 1));
plot(time_steps, qnorm, 'k-', 'LineWidth', 2);
hold on;
plot([0 time_steps(end)], [1 1], 'r--');
grid on;
xlabel('Time (s)'); ylabel('Norm');
title('Quaternion Norm');
ylim([0.99 1.01]);

sgtitle('iLQR VTOL Control');

%% Helper Functions
function J = runningCost(x, xd, Q, R, u)
    % Compute quaternion error: q_err = q_des_conj * q_curr
    q_des_conj = [xd(7), -xd(8), -xd(9), -xd(10)];  % Conjugate of desired quaternion
    q_err = quatmultiply(q_des_conj, x(7:10)');      % Quaternion product
    % State error (position, quaternion vector part, velocity)
    xerr = [x(1:6) - xd(1:6); 2*q_err(2:4)'; x(11:13) - xd(11:13)];
    % Compute cost
    J = xerr' * Q * xerr + u' * R * u;
end

function J = finalCost(x, xd, Qf)
    q_err = quatmultiply([xd(7) -xd(8:10)]', x(7:10)')';
    xerr = [x(1:6) - xd(1:6); 2*q_err(2:4); x(11:13) - xd(11:13)];
    J = xerr' * Qf * xerr;
end

function [l, lx, lu, lxx, luu, lux] = runningCostDerivatives(x, xd, Q, R, u)
    q_err = quatmultiply([xd(7) -xd(8:10)]', x(7:10)')';
    xerr = [x(1:6) - xd(1:6); 2*q_err(2:4); x(11:13) - xd(11:13)];
    l = xerr' * Q * xerr + u' * R * u;
    dxerr_dx = zeros(12,13);
    dxerr_dx(1:3,1:3) = eye(3);
    dxerr_dx(4:6,4:6) = eye(3);
    dxerr_dx(7:9,8:10) = 2*eye(3);
    dxerr_dx(10:12,11:13) = eye(3);
    lx = 2 * dxerr_dx' * Q * xerr;
    lxx = 2 * dxerr_dx' * Q * dxerr_dx;
    lu = 2 * R * u;
    luu = 2 * R;
    lux = zeros(nU, nX);
end

function Vx = finalCostGradient(x, xd, Qf)
    q_err = quatmultiply([xd(7) -xd(8:10)]', x(7:10)')';
    xerr = [x(1:6) - xd(1:6); 2*q_err(2:4); x(11:13) - xd(11:13)];
    dxerr_dx = zeros(12,13);
    dxerr_dx(1:3,1:3) = eye(3);
    dxerr_dx(4:6,4:6) = eye(3);
    dxerr_dx(7:9,8:10) = 2*eye(3);
    dxerr_dx(10:12,11:13) = eye(3);
    Vx = 2 * dxerr_dx' * Qf * xerr;
end

function Vxx = finalCostHessian(x, xd, Qf)
    dxerr_dx = zeros(12,13);
    dxerr_dx(1:3,1:3) = eye(3);
    dxerr_dx(4:6,4:6) = eye(3);
    dxerr_dx(7:9,8:10) = 2*eye(3);
    dxerr_dx(10:12,11:13) = eye(3);
    Vxx = 2 * dxerr_dx' * Qf * dxerr_dx;
end

function [A, B] = linearizeDynamics(dyn, x, u, dt, eps)
    nX = length(x);
    nU = length(u);
    f0 = dyn(x, u);
    A = zeros(nX, nX);
    B = zeros(nX, nU);
    for i = 1:nX
        xp = x; xp(i) = x(i) + eps;
        A(:,i) = (dyn(xp, u) - f0) / eps;
    end
    for i = 1:nU
        up = u; up(i) = u(i) + eps;
        B(:,i) = (dyn(x, up) - f0) / eps;
    end
    A = eye(nX) + dt * A;
    B = dt * B;
end

function q = quatmultiply(q1, q2)
    w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
    w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);
    q = [w1*w2 - x1*x2 - y1*y2 - z1*z2;
         w1*x2 + x1*w2 + y1*z2 - z1*y2;
         w1*y2 - x1*z2 + y1*w2 + z1*x2;
         w1*z2 + x1*y2 - y1*x2 + z1*w2];
end

function R = quat2rotm(q)
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1-2*(y^2+z^2) 2*(x*y-w*z) 2*(x*z+w*y);
         2*(x*y+w*z) 1-2*(x^2+z^2) 2*(y*z-w*x);
         2*(x*z-w*y) 2*(y*z+w*x) 1-2*(x^2+y^2)];
end
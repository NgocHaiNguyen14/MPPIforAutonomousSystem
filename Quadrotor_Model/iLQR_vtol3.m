DYNAMICS = @vtol3_quaternion;

nX = 13; % number of states (quaternion representation)
nU = 4;  % number of inputs

% Initial conditions (quaternion representation)
% [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
x0 = [0;0;0;0;0;0;1;0;0;0;0;0;0];  % quaternion starts as [1,0,0,0] (identity)
xd = [100;70;50;0;0;0;1;0;0;0;0;0;0];  % desired state with identity quaternion

% iLQR Parameters
N = 150;           % Time horizon
dt = 0.02;         % Time step
max_iter = 50;     % Maximum iLQR iterations
tol = 1e-3;        % Convergence tolerance (relaxed)

% Cost function weights (reduced for better numerical behavior)
Q = diag([1, 1, 5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]); 
Qf = 10 * Q;       % Final cost weights  
R = diag([0.1, 0.1, 1, 1]); % Control cost weights (reduced)

% Initialize control sequence with reasonable hover thrust
u_seq = zeros(nU, N-1);
% Initial guess: hover thrust to counteract gravity approximately
hover_thrust = 9.81 * 2.23 / 2; % mg/2 for each thruster
u_seq(1,:) = hover_thrust * ones(1, N-1);
u_seq(2,:) = hover_thrust * ones(1, N-1);
x_seq = zeros(nX, N);   % State trajectory
x_seq(:,1) = x0;

% Forward rollout with initial control sequence
for t = 1:N-1
    x_seq(:,t+1) = x_seq(:,t) + DYNAMICS(x_seq(:,t), u_seq(:,t)) * dt;
    
    % Normalize quaternion
    q = x_seq(7:10, t+1);
    q_norm = norm(q);
    if q_norm > 1e-6
        x_seq(7:10, t+1) = q / q_norm;
    else
        x_seq(7:10, t+1) = [1; 0; 0; 0];
    end
    
    % Ensure positive scalar part
    if x_seq(7, t+1) < 0
        x_seq(7:10, t+1) = -x_seq(7:10, t+1);
    end
end

% Storage for results
cost_history = [];
x_trajectory = [];

fprintf('Starting iLQR optimization...\n');

%% iLQR Main Loop
for iter = 1:max_iter
    % Compute current total cost
    total_cost = computeTotalCost(x_seq, u_seq, xd, Q, Qf, R, N);
    cost_history(iter) = total_cost;
    
    if mod(iter, 10) == 1
        fprintf('Iteration %d: Cost = %.6f\n', iter, total_cost);
    end
    
    %% Backward Pass - Compute derivatives and gains
    % Initialize value function approximation at final time
    V = computeFinalCost(x_seq(:,N), xd, Qf);
    Vx = computeFinalCostGradient(x_seq(:,N), xd, Qf);
    Vxx = Qf;
    
    % Storage for gains
    K = zeros(nU, nX, N-1);  % Feedback gains
    k = zeros(nU, N-1);      % Feedforward gains
    
    % Backward pass
    for t = N-1:-1:1
        % Linearize dynamics around current trajectory
        [A, B] = linearizeDynamics(x_seq(:,t), u_seq(:,t), dt, DYNAMICS);
        
        % Compute cost derivatives
        [l, lx, lu, lxx, luu, lux] = computeCostDerivatives(x_seq(:,t), u_seq(:,t), xd, Q, R);
        
        % Compute Q-function derivatives
        Qx = lx + A' * Vx;
        Qu = lu + B' * Vx;
        Qxx = lxx + A' * Vxx * A;
        Quu = luu + B' * Vxx * B;
        Qux = lux + B' * Vxx * A;
        
        % Add regularization to Quu for numerical stability
        reg = max(1e-4, 1e-2 / max(1, iter/10)); % Adaptive regularization
        Quu_reg = Quu + reg * eye(nU);
        
        % Check if Quu is positive definite
        [L_chol, p] = chol(Quu_reg, 'lower');
        if p > 0
            % If not positive definite, add more regularization
            reg = max(reg * 10, 1e-2);
            Quu_reg = Quu + reg * eye(nU);
            [L_chol, p] = chol(Quu_reg, 'lower');
            if p > 0
                fprintf('Warning: Quu not positive definite at iteration %d, time %d\n', iter, t);
                L_chol = chol(Quu_reg + 1e-1 * eye(nU), 'lower');
            end
        end
        
        % Compute gains using Cholesky decomposition for numerical stability
        k(:,t) = -L_chol' \ (L_chol \ Qu);
        K(:,:,t) = -L_chol' \ (L_chol \ Qux);
        
        % Update value function
        Vx = Qx + K(:,:,t)' * Quu * k(:,t) + K(:,:,t)' * Qu + Qux' * k(:,t);
        Vxx = Qxx + K(:,:,t)' * Quu * K(:,:,t) + K(:,:,t)' * Qux + Qux' * K(:,:,t);
        
        % Ensure Vxx remains symmetric
        Vxx = 0.5 * (Vxx + Vxx');
    end
    
    %% Forward Pass with Line Search
    alpha_candidates = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]; % More conservative line search
    best_cost = inf;
    best_x_new = [];
    best_u_new = [];
    improvement_found = false;
    
    for alpha = alpha_candidates
        % Initialize new trajectory
        x_new = zeros(nX, N);
        u_new = zeros(nU, N-1);
        x_new(:,1) = x0;
        
        % Forward rollout with line search
        valid_trajectory = true;
        for t = 1:N-1
            % Apply control update
            dx = x_new(:,t) - x_seq(:,t);
            u_new(:,t) = u_seq(:,t) + alpha * k(:,t) + K(:,:,t) * dx;
            
            % Clip control inputs
            u_new(1,t) = max(u_new(1,t), 0);    % Thrust constraints
            u_new(2,t) = max(u_new(2,t), 0);
            u_new(3,t) = max(min(u_new(3,t), 15), -15); % Moment constraints
            u_new(4,t) = max(min(u_new(4,t), 15), -15);
            
            % Propagate dynamics
            try
                x_next = x_new(:,t) + DYNAMICS(x_new(:,t), u_new(:,t)) * dt;
                
                % Check for valid state
                if any(~isreal(x_next)) || any(isnan(x_next)) || any(isinf(x_next))
                    valid_trajectory = false;
                    break;
                end
                
                x_new(:,t+1) = x_next;
                
                % Normalize quaternion
                q = x_new(7:10, t+1);
                q_norm = norm(q);
                if q_norm > 1e-6
                    x_new(7:10, t+1) = q / q_norm;
                else
                    x_new(7:10, t+1) = [1; 0; 0; 0];
                    valid_trajectory = false;
                    break;
                end
                
                % Ensure positive scalar part
                if x_new(7, t+1) < 0
                    x_new(7:10, t+1) = -x_new(7:10, t+1);
                end
                
            catch
                valid_trajectory = false;
                break;
            end
        end
        
        if valid_trajectory
            % Compute cost for this trajectory
            new_cost = computeTotalCost(x_new, u_new, xd, Q, Qf, R, N);
            
            if new_cost < best_cost
                best_cost = new_cost;
                best_x_new = x_new;
                best_u_new = u_new;
                improvement_found = true;
            end
        end
    end
    
    % Check if improvement was found
    if improvement_found && best_cost < total_cost
        % Accept the new trajectory
        cost_improvement = total_cost - best_cost;
        x_seq = best_x_new;
        u_seq = best_u_new;
        
        if mod(iter, 5) == 0
            fprintf('Iteration %d: Cost = %.2f, Improvement = %.2f\n', iter, best_cost, cost_improvement);
        end
        
        % Check for convergence
        if cost_improvement < tol
            fprintf('Converged at iteration %d with cost improvement %.2e\n', iter, cost_improvement);
            break;
        end
    else
        % Try to continue with smaller regularization if no improvement
        if iter < 10
            fprintf('No improvement at iteration %d, continuing with smaller step...\n', iter);
            continue; % Continue trying for first few iterations
        else
            fprintf('No improvement found at iteration %d, terminating\n', iter);
            break;
        end
    end
end

% Store final trajectory
x_trajectory = x_seq;
u_trajectory = u_seq;

fprintf('\niLQR optimization completed!\n');
fprintf('Final cost: %.6f\n', computeTotalCost(x_seq, u_seq, xd, Q, Qf, R, N));
fprintf('Final distance to target: %.2f m\n', norm(x_seq(1:3,end) - xd(1:3)));

%% Visualization (same as original)
figure('Position', [100, 100, 1400, 900]);

% Create subplots
subplot(3, 3, 1);
% 3D trajectory plot
plot3(x_trajectory(1,:), x_trajectory(2,:), x_trajectory(3,:), 'b-', 'LineWidth', 2);
hold on;
plot3(x0(1), x0(2), x0(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(xd(1), xd(2), xd(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot3(x_trajectory(1,end), x_trajectory(2,end), x_trajectory(3,end), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Trajectory');
legend('Trajectory', 'Start', 'Target', 'Final', 'Location', 'best');

% X-Y trajectory plot with VTOL rectangle
subplot(3, 3, 2);
plot(x_trajectory(1,:), x_trajectory(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd(1), xd(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Draw VTOL as rectangle at final position with orientation
vtol_width = 3;
vtol_height = 1.5;

% Get final quaternion and convert to rotation matrix for orientation
q_final = x_trajectory(7:10, end);
R_final = quat2rotm(q_final');

% Create rectangle vertices in body frame
rect_body = [-vtol_width/2, -vtol_height/2;
              vtol_width/2, -vtol_height/2;
              vtol_width/2,  vtol_height/2;
             -vtol_width/2,  vtol_height/2;
             -vtol_width/2, -vtol_height/2];

% Rotate rectangle to world frame
rect_world = zeros(size(rect_body));
for i = 1:size(rect_body, 1)
    point_3d = [rect_body(i,:), 0]'; % Add z=0
    rotated_point = R_final * point_3d;
    rect_world(i,:) = rotated_point(1:2)' + [x_trajectory(1,end), x_trajectory(2,end)];
end

plot(rect_world(:,1), rect_world(:,2), 'k-', 'LineWidth', 2);
fill(rect_world(:,1), rect_world(:,2), [0.3, 0.3, 0.8], 'FaceAlpha', 0.7);

% Add orientation arrow showing forward direction
arrow_length = 2;
forward_dir = R_final * [arrow_length; 0; 0]; % Forward direction in world frame
arrow_x = [x_trajectory(1,end), x_trajectory(1,end) + forward_dir(1)];
arrow_y = [x_trajectory(2,end), x_trajectory(2,end) + forward_dir(2)];
plot(arrow_x, arrow_y, 'r-', 'LineWidth', 3);
plot(arrow_x(2), arrow_y(2), 'r>', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('X-Y Trajectory with Oriented VTOL');
legend('Trajectory', 'Start', 'Target', 'VTOL', 'Direction', 'Location', 'best');
axis equal;

% Continue with remaining plots (altitude, velocities, quaternions, etc.)
% Altitude vs time
subplot(3, 3, 3);
time_steps = 0:dt:(N-1)*dt;
plot(time_steps, x_trajectory(3,:), 'b-', 'LineWidth', 2);
hold on;
plot([0, time_steps(end)], [xd(3), xd(3)], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Altitude (m)');
title('Altitude vs Time');
legend('Actual', 'Target', 'Location', 'best');

% Velocities
subplot(3, 3, 4);
plot(time_steps, x_trajectory(4,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'vx');
hold on;
plot(time_steps, x_trajectory(5,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'vy');
plot(time_steps, x_trajectory(6,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'vz');
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Linear Velocities');
legend('Location', 'best');

% Control inputs
subplot(3, 3, 5);
time_control = 0:dt:(N-2)*dt;
plot(time_control, u_trajectory(1,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'u1 (Thrust 1)');
hold on;
plot(time_control, u_trajectory(2,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'u2 (Thrust 2)');
plot(time_control, u_trajectory(3,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'u3 (Moment 1)');
plot(time_control, u_trajectory(4,:), 'm-', 'LineWidth', 1.5, 'DisplayName', 'u4 (Moment 2)');
grid on;
xlabel('Time (s)');
ylabel('Control Input');
title('Control Inputs vs Time');
legend('Location', 'best');

% Cost convergence
subplot(3, 3, 6);
plot(1:length(cost_history), cost_history, 'b-', 'LineWidth', 2);
grid on;
xlabel('iLQR Iteration');
ylabel('Total Cost');
title('Cost Convergence');

% Add overall title
sgtitle('iLQR VTOL3 Quaternion Control Results', 'FontSize', 16, 'FontWeight', 'bold');

%% Helper Functions

function total_cost = computeTotalCost(x_seq, u_seq, xd, Q, Qf, R, N)
    total_cost = 0;
    
    % Running costs
    for t = 1:N-1
        total_cost = total_cost + computeRunningCost(x_seq(:,t), u_seq(:,t), xd, Q, R);
    end
    
    % Final cost
    total_cost = total_cost + computeFinalCost(x_seq(:,N), xd, Qf);
end

function cost = computeRunningCost(x, u, xd, Q, R)
    % Compute quaternion error properly
    xerr = computeStateError(x, xd);
    cost = 0.5 * (xerr' * Q * xerr + u' * R * u);
end

function cost = computeFinalCost(x, xd, Qf)
    xerr = computeStateError(x, xd);
    cost = 0.5 * xerr' * Qf * xerr;
end

function grad = computeFinalCostGradient(x, xd, Qf)
    xerr = computeStateError(x, xd);
    
    % The gradient is simply Qf * xerr for the quadratic cost
    grad = Qf * xerr;
end

function xerr = computeStateError(x, xd)
    % Compute state error - return full 13-element error vector
    % For quaternion, we use the actual quaternion difference rather than rotation error
    
    % Position, velocity, and angular velocity errors (9 elements)
    pos_vel_omega_err = [x(1:6) - xd(1:6); x(11:13) - xd(11:13)];
    
    % Quaternion error (4 elements) - use direct difference for linearization
    % This is an approximation, but works well for small errors around the reference
    quat_err = x(7:10) - xd(7:10);
    
    % Construct full 13-element error vector
    xerr = [pos_vel_omega_err(1:6); quat_err; pos_vel_omega_err(7:9)];
end

function [l, lx, lu, lxx, luu, lux] = computeCostDerivatives(x, u, xd, Q, R)
    % Compute cost and its derivatives
    xerr = computeStateError(x, xd);
    
    % Cost
    l = 0.5 * (xerr' * Q * xerr + u' * R * u);
    
    % First derivatives
    lx = Q * xerr;
    lu = R * u;
    
    % Second derivatives
    lxx = Q;
    luu = R;
    lux = zeros(size(R, 1), size(Q, 2));
end

function [A, B] = linearizeDynamics(x, u, dt, dynamics_func)
    % Finite difference approximation for linearization
    eps = 1e-6;
    
    nX = length(x);
    nU = length(u);
    
    % Nominal dynamics
    f0 = dynamics_func(x, u);
    
    % Linearize w.r.t. state
    A = zeros(nX, nX);
    for i = 1:nX
        x_pert = x;
        x_pert(i) = x_pert(i) + eps;
        
        % Special handling for quaternion perturbation
        if i >= 7 && i <= 10
            % Normalize after perturbation
            q_pert = x_pert(7:10);
            x_pert(7:10) = q_pert / norm(q_pert);
        end
        
        f_pert = dynamics_func(x_pert, u);
        A(:, i) = (f_pert - f0) / eps;
    end
    
    % Linearize w.r.t. control
    B = zeros(nX, nU);
    for i = 1:nU
        u_pert = u;
        u_pert(i) = u_pert(i) + eps;
        f_pert = dynamics_func(x, u_pert);
        B(:, i) = (f_pert - f0) / eps;
    end
    
    % Discretize using first-order approximation
    A = eye(nX) + A * dt;
    B = B * dt;
end

function q = quatmultiply(q1, q2)
    % Quaternion multiplication
    % q1 and q2 are row vectors [w, x, y, z]
    w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
    w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);
    
    q = [w1*w2 - x1*x2 - y1*y2 - z1*z2, ...
         w1*x2 + x1*w2 + y1*z2 - z1*y2, ...
         w1*y2 - x1*z2 + y1*w2 + z1*x2, ...
         w1*z2 + x1*y2 - y1*x2 + z1*w2];
end
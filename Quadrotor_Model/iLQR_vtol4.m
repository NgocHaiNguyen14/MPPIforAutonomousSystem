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
max_iter = 400;    % Maximum iLQR iterations (same as MPPI)
tol = 1e-3;        % Convergence tolerance

% Cost function weights (matching MPPI final cost weights)
Q = diag([1, 1, 5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]); 
Qf_pos = 20 * diag([5, 5, 10]);        % Position weights (matching MPPI)
Qf_vel = 20 * diag([5, 5, 5]);         % Velocity weights (matching MPPI)
Qf_quat = 20 * diag([0, 5, 5, 0]);     % Quaternion weights (matching MPPI)
Qf_omega = 20 * diag([5, 5, 5]);       % Angular velocity weights (matching MPPI)
Qf = blkdiag(Qf_pos, Qf_vel, Qf_quat, Qf_omega);

R = diag([0.1, 0.1, 1, 1]); % Control cost weights

% Vertical landing constraints (same as MPPI)
position_tolerance = 10;        % meters
angle_tolerance = 0.15;         % radians (~8.6 degrees)
velocity_tolerance = 5;         % m/s for near-zero landing velocity
angular_velocity_tolerance = 0.5; % rad/s

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
xf = x_seq;  % Store trajectory for output (same variable name as MPPI)
uOpt = u_seq; % Store control inputs (same variable name as MPPI)

fprintf('=== iLQR VTOL Control with Quaternion Dynamics and Vertical Landing ===\n');
fprintf('Target position: [%.1f, %.1f, %.1f]\n', xd(1), xd(2), xd(3));
fprintf('Position tolerance: %.1f m\n', position_tolerance);
fprintf('Angle tolerance: %.2f rad (%.1f degrees)\n', angle_tolerance, rad2deg(angle_tolerance));
fprintf('Velocity tolerance: %.1f m/s\n', velocity_tolerance);
fprintf('Angular velocity tolerance: %.2f rad/s\n', angular_velocity_tolerance);
fprintf('\nStarting optimization...\n');

%% iLQR Main Loop
computation_start = tic;

for iter = 1:max_iter
    % Compute current total cost
    total_cost = computeTotalCost(x_seq, u_seq, xd, Q, Qf, R, N);
    cost_history(iter) = total_cost;
    
    if mod(iter, 50) == 1
        fprintf('Iteration %d: Cost = %.2f\n', iter, total_cost);
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
    alpha_candidates = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01];
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
        
        % Check convergence with position, angle, and altitude constraints (same as MPPI)
        current_pos = x_seq(1:3, end);
        current_vel = x_seq(4:6, end);
        current_quat = x_seq(7:10, end);
        current_angular_vel = x_seq(11:13, end);
        
        pos_dist = norm(current_pos - xd(1:3));
        vel_magnitude = norm(current_vel);
        angular_vel_magnitude = norm(current_angular_vel);
        
        % Convert quaternion to Euler angles for vertical landing check
        [roll, pitch, yaw] = quat2euler(current_quat);
        roll_error = abs(roll);
        pitch_error = abs(pitch);
        max_angle_error = max(roll_error, pitch_error);
        
        % Check altitude constraint: z should be >= z_expected
        altitude_ok = current_pos(3) >= xd(3);
        
        % Display current status every 20 iterations (same as MPPI)
        if mod(iter, 20) == 0
            fprintf('Iter %d: Pos dist=%.2f, Alt=%.2f(>=%.2f), Vel=%.2f, Ang err=%.3f rad (%.1f°), ω=%.2f\n', ...
                    iter, pos_dist, current_pos(3), xd(3), vel_magnitude, max_angle_error, rad2deg(max_angle_error), angular_vel_magnitude);
        end
        
        % Check if all landing criteria are satisfied
        position_ok = pos_dist < position_tolerance;
        altitude_ok = altitude_ok; % z >= z_expected
        angle_ok = max_angle_error < angle_tolerance;
        velocity_ok = vel_magnitude < velocity_tolerance;
        angular_velocity_ok = angular_vel_magnitude < angular_velocity_tolerance;
        
        if position_ok && altitude_ok && angle_ok && velocity_ok && angular_velocity_ok
            fprintf('\n=== SUCCESSFUL VERTICAL LANDING ===\n');
            fprintf('Target reached at iteration %d!\n', iter);
            fprintf('Final position distance: %.2f m\n', pos_dist);
            fprintf('Final altitude: %.2f m (target: %.2f m)\n', current_pos(3), xd(3));
            fprintf('Final velocity magnitude: %.2f m/s\n', vel_magnitude);
            fprintf('Final roll error: %.3f rad (%.1f degrees)\n', roll_error, rad2deg(roll_error));
            fprintf('Final pitch error: %.3f rad (%.1f degrees)\n', pitch_error, rad2deg(pitch_error));
            fprintf('Final angular velocity: %.2f rad/s\n', angular_vel_magnitude);
            break;
        end
        
        % Check for convergence with cost improvement
        if cost_improvement < tol
            if mod(iter, 5) == 0
                fprintf('Converged at iteration %d with cost improvement %.2e\n', iter, cost_improvement);
            end
            break;
        end
    else
        % Try to continue with smaller regularization if no improvement
        if iter < 10
            continue; % Continue trying for first few iterations
        else
            if mod(iter, 50) == 0
                fprintf('No improvement found at iteration %d\n', iter);
            end
            break;
        end
    end
end

computation_time = toc(computation_start);

% Store final trajectory (same variable names as MPPI)
xf = x_seq;
uOpt = u_seq;

%% Calculate Performance Metrics (exactly matching MPPI)
final_pos_error = norm(xf(1:3,end) - xd(1:3));
final_altitude = xf(3,end);
altitude_satisfied = final_altitude >= xd(3);
final_vel_magnitude = norm(xf(4:6,end));
final_quat = xf(7:10,end);
[final_roll, final_pitch, final_yaw] = quat2euler(final_quat);
final_angle_error = max(abs(final_roll), abs(final_pitch));
final_angular_vel = norm(xf(11:13,end));

% Control effort calculation
total_control_effort = 0;
for i = 1:size(uOpt, 2)
    total_control_effort = total_control_effort + norm(uOpt(:,i))^2 * dt;
end

% Path length calculation
path_length = 0;
for i = 2:size(xf, 2)
    path_length = path_length + norm(xf(1:3,i) - xf(1:3,i-1));
end

fprintf('\n=== PERFORMANCE METRICS ===\n');
fprintf('Computation time: %.3f seconds\n', computation_time);
fprintf('Number of iterations: %d\n', size(xf, 2));
fprintf('Final position error: %.3f m\n', final_pos_error);
fprintf('Final altitude: %.3f m (target: %.3f m, satisfied: %s)\n', final_altitude, xd(3), mat2str(altitude_satisfied));
fprintf('Final velocity magnitude: %.3f m/s\n', final_vel_magnitude);
fprintf('Final angle error: %.3f rad (%.1f degrees)\n', final_angle_error, rad2deg(final_angle_error));
fprintf('Final angular velocity: %.3f rad/s\n', final_angular_vel);
fprintf('Total control effort: %.3f\n', total_control_effort);
fprintf('Path length: %.3f m\n', path_length);
fprintf('Average speed: %.3f m/s\n', path_length / (size(xf, 2) * dt));

%% Visualization (exactly matching MPPI layout)
figure('Position', [100, 100, 1400, 900]);

% Create subplots
subplot(3, 3, 1);
% 3D trajectory plot
plot3(xf(1,:), xf(2,:), xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot3(x0(1), x0(2), x0(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(xd(1), xd(2), xd(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot3(xf(1,end), xf(2,end), xf(3,end), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Trajectory');
legend('Trajectory', 'Start', 'Target', 'Final', 'Location', 'best');

% X-Y trajectory plot with oriented VTOL rectangle
subplot(3, 3, 2);
plot(xf(1,:), xf(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd(1), xd(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Draw oriented VTOL rectangle at final position
vtol_width = 3;
vtol_height = 1.5;

% Get final quaternion and convert to rotation matrix for orientation
q_final = xf(7:10, end);
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
    rect_world(i,:) = rotated_point(1:2)' + [xf(1,end), xf(2,end)];
end

plot(rect_world(:,1), rect_world(:,2), 'k-', 'LineWidth', 2);
fill(rect_world(:,1), rect_world(:,2), [0.3, 0.3, 0.8], 'FaceAlpha', 0.7);

% Add orientation arrow showing forward direction
arrow_length = 2;
forward_dir = R_final * [arrow_length; 0; 0]; % Forward direction in world frame
arrow_x = [xf(1,end), xf(1,end) + forward_dir(1)];
arrow_y = [xf(2,end), xf(2,end) + forward_dir(2)];
plot(arrow_x, arrow_y, 'r-', 'LineWidth', 3);
plot(arrow_x(2), arrow_y(2), 'r>', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('X-Y Trajectory with Oriented VTOL');
legend('Trajectory', 'Start', 'Target', 'VTOL', 'Direction', 'Location', 'best');
axis equal;

% Altitude vs time
subplot(3, 3, 3);
time_steps = 0:dt:(size(xf,2)-1)*dt;
plot(time_steps, xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot([0, time_steps(end)], [xd(3), xd(3)], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Altitude (m)');
title('Altitude vs Time');
legend('Actual', 'Target', 'Location', 'best');

% Velocities
subplot(3, 3, 4);
plot(time_steps, xf(4,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'vx');
hold on;
plot(time_steps, xf(5,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'vy');
plot(time_steps, xf(6,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'vz');
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Linear Velocities');
legend('Location', 'best');

% Quaternions
subplot(3, 3, 5);
plot(time_steps, xf(7,:), 'k-', 'LineWidth', 1.5, 'DisplayName', 'qw');
hold on;
plot(time_steps, xf(8,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'qx');
plot(time_steps, xf(9,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'qy');
plot(time_steps, xf(10,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'qz');
grid on;
xlabel('Time (s)');
ylabel('Quaternion Components');
title('Quaternion Evolution');
legend('Location', 'best');

% Convert quaternions to Euler angles for visualization
euler_angles = zeros(3, size(xf, 2));
for i = 1:size(xf, 2)
    [euler_angles(1,i), euler_angles(2,i), euler_angles(3,i)] = quat2euler(xf(7:10,i));
end

% Euler Angles (converted from quaternions)
subplot(3, 3, 6);
plot(time_steps, rad2deg(euler_angles(1,:)), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Roll (φ)');
hold on;
plot(time_steps, rad2deg(euler_angles(2,:)), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Pitch (θ)');
plot(time_steps, rad2deg(euler_angles(3,:)), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Yaw (ψ)');
plot([0, time_steps(end)], [rad2deg(angle_tolerance), rad2deg(angle_tolerance)], 'r--', 'LineWidth', 1, 'DisplayName', 'Angle Limit');
plot([0, time_steps(end)], [-rad2deg(angle_tolerance), -rad2deg(angle_tolerance)], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Angle (degrees)');
title('Euler Angles (Vertical Landing)');
legend('Location', 'best');

% Control inputs
subplot(3, 3, 7);
time_control = 0:dt:(size(uOpt,2)-1)*dt;
plot(time_control, uOpt(1,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'u1 (Thrust 1)');
hold on;
plot(time_control, uOpt(2,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'u2 (Thrust 2)');
plot(time_control, uOpt(3,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'u3 (Moment 1)');
plot(time_control, uOpt(4,:), 'm-', 'LineWidth', 1.5, 'DisplayName', 'u4 (Moment 2)');
grid on;
xlabel('Time (s)');
ylabel('Control Input');
title('Control Inputs vs Time');
legend('Location', 'best');

% Angular velocities
subplot(3, 3, 8);
plot(time_steps, xf(11,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'ωx');
hold on;
plot(time_steps, xf(12,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'ωy');
plot(time_steps, xf(13,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'ωz');
plot([0, time_steps(end)], [angular_velocity_tolerance, angular_velocity_tolerance], 'r--', 'LineWidth', 1, 'DisplayName', 'ω Limit');
plot([0, time_steps(end)], [-angular_velocity_tolerance, -angular_velocity_tolerance], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
title('Angular Velocities');
legend('Location', 'best');

% Performance summary
subplot(3, 3, 9);
axis off;
text(0.1, 0.9, 'PERFORMANCE SUMMARY', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Computation Time: %.3f s', computation_time), 'FontSize', 10);
text(0.1, 0.7, sprintf('Iterations: %d', size(xf, 2)), 'FontSize', 10);
text(0.1, 0.6, sprintf('Final Position Error: %.3f m', final_pos_error), 'FontSize', 10);
text(0.1, 0.55, sprintf('Final Altitude: %.2f m (>= %.2f)', final_altitude, xd(3)), 'FontSize', 10);
text(0.1, 0.5, sprintf('Final Angle Error: %.1f°', rad2deg(final_angle_error)), 'FontSize', 10);
text(0.1, 0.4, sprintf('Final Velocity: %.3f m/s', final_vel_magnitude), 'FontSize', 10);
text(0.1, 0.3, sprintf('Control Effort: %.3f', total_control_effort), 'FontSize', 10);
text(0.1, 0.2, sprintf('Path Length: %.3f m', path_length), 'FontSize', 10);

% Add overall title
sgtitle('iLQR VTOL Control with Quaternion Dynamics and Vertical Landing', 'FontSize', 16, 'FontWeight', 'bold');

%% Helper Functions

function [roll, pitch, yaw] = quat2euler(q)
    % Convert quaternion to Euler angles (ZYX convention)
    % q = [w, x, y, z]
    w = q(1); x = q(2); y = q(3); z = q(4);
    
    % Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z);
    cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = atan2(sinr_cosp, cosr_cosp);
    
    % Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x);
    if abs(sinp) >= 1
        pitch = copysign(pi/2, sinp); % use 90 degrees if out of range
    else
        pitch = asin(sinp);
    end
    
    % Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y);
    cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = atan2(siny_cosp, cosy_cosp);
end

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
    % Enhanced final cost matching MPPI
    % Target state for vertical landing
    xd_vertical = xd;
    % For vertical landing: qw=1, qx=0, qy=0, qz can be current value
    xd_vertical(7) = 1;    % qw = 1
    xd_vertical(8) = 0;    % qx = 0 (no roll)
    xd_vertical(9) = 0;    % qy = 0 (no pitch)
    xd_vertical(10) = x(10); % qz = current qz (allow yaw freedom)
    
    % Normalize
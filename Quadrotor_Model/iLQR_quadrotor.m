DYNAMICS = @vtol4_iLQR;

nX = 13; % number of states (quaternion representation)
nU = 4;  % number of inputs

% Initial conditions (quaternion representation)
% [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
x0 = [0;0;0;0;0;0;0.7071;0;0;0.7071;0;0;0];  % quaternion starts as [0.7071,0,0,0.7071]
xd = [50;50;0;0;0;0;0.7071;0;0;0.7071;0;0;0];  % desired state with same quaternion

% Define phase targets
xd_phase1 = [0;0;50;0;0;0;x0(7:10);0;0;0];
xd_phase2 = [50;50;50;0;0;0;0.9239;0;0;0.3827;0;0;0];
xd_phase3 = xd;

% Initialization
N = 150;
utraj = zeros(nU, N-1);
uOpt = [];
xf = [];
dt = 0.02;
lambda = 10;
covu = diag([2.5,2.5,3e-2,3e-2]);

xtraj = zeros(nX, N);
R = lambda*inv(covu);

x = x0;

phase = 1;
xd_current = xd_phase1;

% Vertical landing constraints (used in phase 3 primarily)
position_tolerance_horizontal = 15;     % meters for hover phases, will adjust for landing
position_tolerance_vertical = 5;       % meters
angle_tolerance = 0.6;                 % radians (~45 degrees)
velocity_tolerance = 20;                % m/s
angular_velocity_tolerance = 10;         % rad/s

fprintf('=== iLQR VTOL Control with Quaternion Dynamics and Multi-Phase Trajectory ===\n');
fprintf('Phase 1: Fly to hover at [%.1f, %.1f, %.1f]\n', xd_phase1(1), xd_phase1(2), xd_phase1(3));
fprintf('Phase 2: Fly to hover at [%.1f, %.1f, %.1f]\n', xd_phase2(1), xd_phase2(2), xd_phase2(3));
fprintf('Phase 3: Vertical landing at [%.1f, %.1f, %.1f]\n', xd_phase3(1), xd_phase3(2), xd_phase3(3));
fprintf('Position tolerance (horizontal): %.1f m (hover phases), 20 m (landing)\n', position_tolerance_horizontal);
fprintf('Position tolerance (vertical): %.1f m\n', position_tolerance_vertical);
fprintf('Angle tolerance: %.2f rad (%.1f degrees)\n', angle_tolerance, rad2deg(angle_tolerance));
fprintf('Velocity tolerance: %.1f m/s\n', velocity_tolerance);
fprintf('Angular velocity tolerance: %.2f rad/s\n', angular_velocity_tolerance);
fprintf('\nStarting optimization...\n');

%% Run iLQR Optimization
computation_start = tic;

u_nom = zeros(nU, N-1);

for iter = 1:2000
    xf = [xf,x]; % Append the simulated trajectory
    
    [utraj, ~, minS] = ilqr_mpc(x, xd_current, u_nom, N, dt, DYNAMICS, phase);
    
    if mod(iter, 50) == 1
        fprintf('Iteration %d (Phase %d): Min cost = %.2f\n', iter, phase, minS);
    end
    
    % Execute the utraj(:,1)
    u = utraj(:,1);
    u = clippedu(u);
    x_next = x + DYNAMICS(x, u)*dt;
    
    % Normalize quaternion
    q = x_next(7:10);
    q_norm = norm(q);
    if q_norm > 1e-6
        x_next(7:10) = q / q_norm;
    else
        x_next(7:10) = [1; 0; 0; 0];
    end
    
    % Ensure positive scalar part
    if x_next(7) < 0
        x_next(7:10) = -x_next(7:10);
    end
    
    x = x_next;
    uOpt = [uOpt, u];
    
    % Shift the nominal inputs 
    u_nom = [utraj(:,2:end), zeros(nU,1)];
    
    % Check convergence with position, angle, and altitude constraints
    current_pos = x(1:3);
    current_vel = x(4:6);
    current_quat = x(7:10);
    current_angular_vel = x(11:13);
    
    pos_dist = norm(current_pos - xd_current(1:3));
    vel_magnitude = norm(current_vel);
    angular_vel_magnitude = norm(current_angular_vel);
    
    % Convert quaternion to Euler angles for vertical landing check
    [roll, pitch, yaw] = quat2euler(current_quat);
    roll_error = abs(roll);
    pitch_error = abs(pitch);
    max_angle_error = max(roll_error, pitch_error);
    
    % Phase-specific checks
    if phase < 3
        position_ok = pos_dist < position_tolerance_vertical; % Use vertical tol for all in hover phases
        altitude_ok = true; % No minimum altitude constraint in hover phases
        horizontal_tol = position_tolerance_horizontal; % For display
    else
        horizontal_tol = 20; % Larger horizontal tolerance in landing phase
        position_ok = norm(current_pos(1:2) - xd_current(1:2)) < horizontal_tol && ...
                      abs(current_pos(3) - xd_current(3)) < position_tolerance_vertical;
        altitude_ok = current_pos(3) >= xd_current(3);
    end
    
    angle_ok = max_angle_error < angle_tolerance;
    velocity_ok = vel_magnitude < velocity_tolerance;
    angular_velocity_ok = angular_vel_magnitude < angular_velocity_tolerance;
    
    % Display current status every 20 iterations
    if mod(iter, 20) == 0
        fprintf('Iter %d (Phase %d): Pos dist=%.2f, Alt=%.2f(%.2f), Vel=%.2f, Ang err=%.3f rad (%.1f°), ω=%.2f\n', ...
                iter, phase, pos_dist, current_pos(3), xd_current(3), vel_magnitude, max_angle_error, rad2deg(max_angle_error), angular_vel_magnitude);
    end
    
    % Check if criteria are satisfied
    if position_ok && altitude_ok && angle_ok && velocity_ok && angular_velocity_ok
        if phase == 1
            fprintf('\n=== PHASE 1 COMPLETED: REACHED FIRST HOVER POINT ===\n');
            fprintf('Switching to Phase 2 at iteration %d\n', iter);
            phase = 2;
            xd_current = xd_phase2;
            % Continue the loop without breaking
        elseif phase == 2
            fprintf('\n=== PHASE 2 COMPLETED: REACHED SECOND HOVER POINT ===\n');
            fprintf('Switching to Phase 3 at iteration %d\n', iter);
            phase = 3;
            xd_current = xd_phase3;
            % Continue the loop without breaking
        else
            fprintf('\n=== SUCCESSFUL VERTICAL LANDING ===\n');
            fprintf('Target reached at iteration %d!\n', iter);
            fprintf('Final position distance: %.2f m\n', pos_dist);
            fprintf('Final altitude: %.2f m (target: %.2f m)\n', current_pos(3), xd_current(3));
            fprintf('Final velocity magnitude: %.2f m/s\n', vel_magnitude);
            fprintf('Final roll error: %.3f rad (%.1f degrees)\n', roll_error, rad2deg(roll_error));
            fprintf('Final pitch error: %.3f rad (%.1f degrees)\n', pitch_error, rad2deg(pitch_error));
            fprintf('Final angular velocity: %.2f rad/s\n', angular_vel_magnitude);
            break;
        end
    end
    
    % Early termination messages for debugging
    if pos_dist < position_tolerance_vertical && ~altitude_ok
        if mod(iter, 50) == 0
            fprintf('Position reached but altitude issue: %.2f m (need %.2f m)\n', ...
                    current_pos(3), xd_current(3));
        end
    elseif pos_dist < position_tolerance_vertical && altitude_ok && ~angle_ok
        if mod(iter, 50) == 0
            fprintf('Position/altitude OK but angle error too large: %.3f rad (%.1f°)\n', ...
                    max_angle_error, rad2deg(max_angle_error));
        end
    end
end

computation_time = toc(computation_start);

%% Calculate Performance Metrics
final_pos_error = norm(xf(1:3,end) - xd_phase3(1:3));
final_altitude = xf(3,end);
altitude_satisfied = final_altitude >= xd_phase3(3);
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
fprintf('Final altitude: %.3f m (target: %.3f m, satisfied: %s)\n', final_altitude, xd_phase3(3), mat2str(altitude_satisfied));
fprintf('Final velocity magnitude: %.3f m/s\n', final_vel_magnitude);
fprintf('Final angle error: %.3f rad (%.1f degrees)\n', final_angle_error, rad2deg(final_angle_error));
fprintf('Final angular velocity: %.3f rad/s\n', final_angular_vel);
fprintf('Total control effort: %.3f\n', total_control_effort);
fprintf('Path length: %.3f m\n', path_length);
fprintf('Average speed: %.3f m/s\n', path_length / (size(xf, 2) * dt));

%% Visualization

% 3D trajectory plot
figure('Name', '3D Trajectory', 'Position', [100, 100, 700, 500]);
plot3(xf(1,:), xf(2,:), xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot3(x0(1), x0(2), x0(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(xd_phase1(1), xd_phase1(2), xd_phase1(3), 'co', 'MarkerSize', 10, 'MarkerFaceColor', 'c'); % Hover 1
plot3(xd_phase2(1), xd_phase2(2), xd_phase2(3), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm'); % Hover 2
plot3(xd_phase3(1), xd_phase3(2), xd_phase3(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Landing
plot3(xf(1,end), xf(2,end), xf(3,end), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Trajectory');
legend('Trajectory', 'Start', 'Hover 1', 'Hover 2', 'Landing Target', 'Final', 'Location', 'best');

% X-Y trajectory plot with oriented VTOL rectangle
figure('Name', 'X-Y Trajectory', 'Position', [100, 100, 700, 500]);
plot(xf(1,:), xf(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd_phase1(1), xd_phase1(2), 'co', 'MarkerSize', 10, 'MarkerFaceColor', 'c');
plot(xd_phase2(1), xd_phase2(2), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
plot(xd_phase3(1), xd_phase3(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

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
legend('Trajectory', 'Start', 'Hover 1', 'Hover 2', 'Landing Target', 'VTOL', 'Direction', 'Location', 'best');
axis equal;

% Altitude vs time
figure('Name', 'Altitude vs Time', 'Position', [100, 100, 700, 500]);
time_steps = 0:dt:(size(xf,2)-1)*dt;
plot(time_steps, xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot([0, time_steps(end)], [xd_phase1(3), xd_phase1(3)], 'c--', 'LineWidth', 1);
plot([0, time_steps(end)], [xd_phase2(3), xd_phase2(3)], 'm--', 'LineWidth', 1);
plot([0, time_steps(end)], [xd_phase3(3), xd_phase3(3)], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Altitude (m)');
title('Altitude vs Time');
legend('Actual', 'Hover 1 Target', 'Hover 2 Target', 'Landing Target', 'Location', 'best');

% Velocities
figure('Name', 'Linear Velocities', 'Position', [100, 100, 700, 500]);
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
figure('Name', 'Quaternion Evolution', 'Position', [100, 100, 700, 500]);
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
figure('Name', 'Euler Angles', 'Position', [100, 100, 700, 500]);
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
figure('Name', 'Control Inputs', 'Position', [100, 100, 700, 500]);
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
figure('Name', 'Angular Velocities', 'Position', [100, 100, 700, 500]);
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
figure('Name', 'Performance Summary', 'Position', [100, 100, 700, 500]);
axis off;
text(0.1, 0.9, 'PERFORMANCE SUMMARY', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Computation Time: %.3f s', computation_time), 'FontSize', 10);
text(0.1, 0.7, sprintf('Iterations: %d', size(xf, 2)), 'FontSize', 10);
text(0.1, 0.6, sprintf('Final Position Error: %.3f m', final_pos_error), 'FontSize', 10);
text(0.1, 0.55, sprintf('Final Altitude: %.2f m (>= %.2f)', final_altitude, xd_phase3(3)), 'FontSize', 10);
text(0.1, 0.5, sprintf('Final Angle Error: %.1f°', rad2deg(final_angle_error)), 'FontSize', 10);
text(0.1, 0.4, sprintf('Final Velocity: %.3f m/s', final_vel_magnitude), 'FontSize', 10);
text(0.1, 0.3, sprintf('Control Effort: %.3f', total_control_effort), 'FontSize', 10);
text(0.1, 0.2, sprintf('Path Length: %.3f m', path_length), 'FontSize', 10);

%% Helper Functions

function [u_traj, x_traj, total_cost] = ilqr_mpc(x0, xd, u_init, N, dt, dyn, phase)
    max_iter = 20;
    nX = 13;
    nU = 4;
    lambda = 10;
    covu = diag([2.5,2.5,3e-2,3e-2]);
    R = lambda * inv(covu);
    
    if phase < 3
        Q_pos = diag([2.5, 2.5, 2.5]);
        Q_vel = diag([0.1, 0.1, 0.1]);
        Q_quat = diag([0.5,5,5,0.5]);
        Q_omega = diag([0.1,0.1,0.1]);
        Qf_pos = 20 * diag([5,5,5]);
        Qf_vel = 20 * diag([1,1,1]);
        Qf_quat = 20 * diag([0,1,1,0]);
        Qf_omega = 20 * diag([1,1,1]);
    else
        Q_pos = diag([1,1,20]);
        Q_vel = diag([1,1,1]);
        Q_quat = diag([0.5,30,30,0.5]);
        Q_omega = diag([1,1,1]);
        Qf_pos = 20 * diag([1,1,10]);
        Qf_vel = 20 * diag([5,5,5]);
        Qf_quat = 20 * diag([0,10,10,0]);
        Qf_omega = 20 * diag([5,5,5]);
    end
    Q = blkdiag(Q_pos, Q_vel, Q_quat, Q_omega);
    Qf = blkdiag(Qf_pos, Qf_vel, Qf_quat, Qf_omega);
    
    u_traj = u_init;
    x_traj = zeros(nX, N);
    x_traj(:,1) = x0;
    for t = 1:N-1
        u = u_traj(:,t);
        x_next = x_traj(:,t) + dyn(x_traj(:,t), u) * dt;
        q = x_next(7:10);
        q_norm = norm(q);
        if q_norm > 1e-6
            q = q / q_norm;
        else
            q = [1; 0; 0; 0];
        end
        if q(1) < 0
            q = -q;
        end
        x_next(7:10) = q;
        x_traj(:,t+1) = x_next;
    end
    
    previous_cost = compute_trajectory_cost(x_traj, u_traj, Q, R, Qf, xd, phase, dt);
    
    for ii = 1:max_iter
        % Linearize dynamics
        A = zeros(nX, nX, N-1);
        B = zeros(nX, nU, N-1);
        eps = 1e-6;
        for t = 1:N-1
            x_t = x_traj(:,t);
            u_t = u_traj(:,t);
            f_nom = dyn(x_t, u_t) * dt;
            for j = 1:nX
                x_pert = x_t;
                x_pert(j) = x_pert(j) + eps;
                f_pert = dyn(x_pert, u_t) * dt;
                A(:,j,t) = (f_pert - f_nom) / eps;
            end
            A(:,:,t) = eye(nX) + A(:,:,t);
            for j = 1:nU
                u_pert = u_t;
                u_pert(j) = u_pert(j) + eps;
                f_pert = dyn(x_t, u_pert) * dt;
                B(:,j,t) = (f_pert - f_nom) / eps;
            end
        end
        
        % Backward pass
        Vx = 2 * (x_traj(:,N) - xd)' * Qf;  % 1 x nX
        Vxx = 2 * Qf;  % nX x nX
        
        if phase == 3
            if x_traj(3,N) < xd(3)
                delta_alt = xd(3) - x_traj(3,N);
                Vx(3) = Vx(3) - 2000 * delta_alt;
                Vxx(3,3) = Vxx(3,3) + 2000;
            end
        end
        
        k = zeros(nU, N-1);
        K = zeros(nU, nX, N-1);
        for t = (N-1):-1:1
            x_t = x_traj(:,t);
            u_t = u_traj(:,t);
            err = x_t - xd;
            cx = 2 * err' * Q;  % 1 x nX
            cxx = 2 * Q;  % nX x nX
            cu = 2 * u_t' * R;  % 1 x nU
            cuu = 2 * R;  % nU x nU
            cux = zeros(nU, nX);  % nU x nX
            fx = A(:,:,t);  % nX x nX
            fu = B(:,:,t);  % nX x nU
            
            Qx = cx + Vx * fx;  % 1 x nX
            Qu = cu + Vx * fu;  % 1 x nU
            Qxx = cxx + fx' * Vxx * fx;  % nX x nX
            Quu = cuu + fu' * Vxx * fu;  % nU x nU
            Qux = cux + fu' * Vxx * fx;  % nU x nX
            
            % Ensure Quu is positive definite
            [~,p] = chol(Quu);
            if p > 0
                Quu = Quu + eye(nU)*1e-3;
            end
            
            inv_Quu = inv(Quu);
            k(:,t) = -inv_Quu * Qu';  % nU x 1
            K(:,:,t) = -inv_Quu * Qux;  % nU x nX
            
            Vx = Qx + k(:,t)' * Qux;  % 1 x nX
            Vxx = Qxx + Qux' * K(:,:,t) + K(:,:,t)' * Qux + K(:,:,t)' * Quu * K(:,:,t);  % nX x nX
        end
        
        % Forward pass with line search
        alpha = 1.0;
        improved = false;
        for ls = 1:10
            x_new = zeros(nX, N);
            u_new = zeros(nU, N-1);
            x_new(:,1) = x0;
            for t = 1:N-1
                dx = x_new(:,t) - x_traj(:,t);
                du = alpha * k(:,t) + K(:,:,t) * dx;
                u_new(:,t) = u_traj(:,t) + du;
                u_new(:,t) = clippedu(u_new(:,t));
                x_next = x_new(:,t) + dyn(x_new(:,t), u_new(:,t)) * dt;
                q = x_next(7:10);
                q_norm = norm(q);
                if q_norm > 1e-6
                    q = q / q_norm;
                else
                    q = [1; 0; 0; 0];
                end
                if q(1) < 0
                    q = -q;
                end
                x_next(7:10) = q;
                x_new(:,t+1) = x_next;
            end
            new_cost = compute_trajectory_cost(x_new, u_new, Q, R, Qf, xd, phase, dt);
            if new_cost < previous_cost
                x_traj = x_new;
                u_traj = u_new;
                previous_cost = new_cost;
                improved = true;
                break;
            end
            alpha = alpha * 0.5;
        end
        if ~improved
            break;
        end
    end
    total_cost = previous_cost;
end

function cost = compute_trajectory_cost(xtraj, utraj, Q, R, Qf, xd, phase, dt)
    [nX, N] = size(xtraj);
    cost = 0;
    for t = 1:N-1
        err = xtraj(:,t) - xd;
        cost = cost + err' * Q * err + utraj(:,t)' * R * utraj(:,t);
    end
    err = xtraj(:,N) - xd;
    cost = cost + err' * Qf * err;
    if phase == 3
        if xtraj(3,N) < xd(3)
            cost = cost + 1000 * (xd(3) - xtraj(3,N))^2;
        end
    end
end

function u = clippedu(u)
    u(1) = max(u(1), 0);
    u(2) = max(u(2), 0);
    u(3) = min(max(u(3), -15), 15);
    u(4) = min(max(u(4), -15), 15);
end

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
DYNAMICS = @vtol3_quaternion;

nX = 13; % number of states (quaternion representation)
nU = 4;  % number of inputs

% Initial conditions (quaternion representation)
% [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
x0 = [0;0;0;0;0;0;0.7071;0;0;0.7071;0;0;0];  % quaternion starts as [1,0,0,0] (identity)
xd = [50;50;20;0;0;0;0.7071;0;0;0.7071;0;0;0];  % desired state with identity quaternion

% Define phase targets
xd_phase1 = [xd(1); xd(2); xd(3); 0;0;0; xd(7:10); 0;0;0];
xd_phase2 = xd;

% Initialization
num_samples = 1000;
N = 150;
utraj = zeros(nU, N-1);
uOpt = [];
xf = [];
dt = 0.02;
lambda = 10;
nu = 1000;
covu = diag([2.5,2.5,3e-2,3e-2]);

xtraj = zeros(nX, N);
R = lambda*inv(covu);

max_cost = 100;

x = x0;

phase = 1;
xd_current = xd_phase1;

% Vertical landing constraints (used in phase 2 primarily)
position_tolerance_horizontal = 20;     % meters for phase 1, will adjust for phase 2
position_tolerance_vertical = 10;       % meters
angle_tolerance = 0.78;                 % radians (~45 degrees)
velocity_tolerance = 15;                 % m/s for near-zero landing velocity
angular_velocity_tolerance = 5;       % rad/s

fprintf('=== MPPI VTOL Control with Quaternion Dynamics and Two-Phase Vertical Landing ===\n');
fprintf('Phase 1: Fly to hover at [%.1f, %.1f, %.1f]\n', xd_phase1(1), xd_phase1(2), xd_phase1(3));
fprintf('Phase 2: Vertical landing at [%.1f, %.1f, %.1f]\n', xd_phase2(1), xd_phase2(2), xd_phase2(3));
fprintf('Position tolerance (horizontal): %.1f m (phase 1), 20 m (phase 2)\n', position_tolerance_horizontal);
fprintf('Position tolerance (vertical): %.1f m\n', position_tolerance_vertical);
fprintf('Angle tolerance: %.2f rad (%.1f degrees)\n', angle_tolerance, rad2deg(angle_tolerance));
fprintf('Velocity tolerance: %.1f m/s\n', velocity_tolerance);
fprintf('Angular velocity tolerance: %.2f rad/s\n', angular_velocity_tolerance);
fprintf('\nStarting optimization...\n');

%% Run MPPI Optimization
computation_start = tic;

for iter = 1:1000
    xf = [xf,x]; % Append the simulated trajectory
    Straj = zeros(1,num_samples); % Initialize cost of rollouts
    
    % Start the rollouts and compute rollout costs
    for k = 1:num_samples
        du = covu*randn(nU, N-1);
        du = clippeddu(utraj, du);
        dU{k} = du;
        xtraj = [];
        xtraj(:,1) = x;
        for t = 1:N-1
            utraj(1, t) = max(utraj(1, t), 0);
            utraj(2, t) = max(utraj(2, t), 0);
            u = utraj(:,t);
            
            % Propagate dynamics
            x_next = xtraj(:,t) + DYNAMICS(xtraj(:,t), u+du(:,t))*dt;
            
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
            
            xtraj(:,t+1) = x_next;
            Straj(k) = Straj(k) + runningCost(xtraj(:,t), xd_current, R, u, du(:,t), nu, phase);
        end
        Straj(k) = Straj(k) + finalCost(xtraj(:,N), xd_current, phase);
    end

    minS = min(Straj); % Minimum rollout cost
    
    if mod(iter, 50) == 1
        fprintf('Iteration %d (Phase %d): Min cost = %.2f\n', iter, phase, minS);
    end

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
    x_next = x + DYNAMICS(x, utraj(:,1))*dt;
    
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
    uOpt = [uOpt, utraj(:,1)];
    
    % Shift the nominal inputs 
    for t = 2:N-1
        utraj(:,t-1) = utraj(:,t);
    end
    utraj(:,N-1) = [0;0;0;0];
    
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
    if phase == 1
        position_ok = pos_dist < position_tolerance_vertical; % Use vertical tol for all in phase 1, since hover
        altitude_ok = true; % No minimum altitude constraint in phase 1
        horizontal_tol = position_tolerance_horizontal; % For display
    else
        horizontal_tol = 20; % Larger horizontal tolerance in phase 2
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
            fprintf('\n=== PHASE 1 COMPLETED: REACHED HOVER POINT ===\n');
            fprintf('Switching to Phase 2 at iteration %d\n', iter);
            phase = 2;
            xd_current = xd_phase2;
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
final_pos_error = norm(xf(1:3,end) - xd_phase2(1:3));
final_altitude = xf(3,end);
altitude_satisfied = final_altitude >= xd_phase2(3);
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
fprintf('Final altitude: %.3f m (target: %.3f m, satisfied: %s)\n', final_altitude, xd_phase2(3), mat2str(altitude_satisfied));
fprintf('Final velocity magnitude: %.3f m/s\n', final_vel_magnitude);
fprintf('Final angle error: %.3f rad (%.1f degrees)\n', final_angle_error, rad2deg(final_angle_error));
fprintf('Final angular velocity: %.3f rad/s\n', final_angular_vel);
fprintf('Total control effort: %.3f\n', total_control_effort);
fprintf('Path length: %.3f m\n', path_length);
fprintf('Average speed: %.3f m/s\n', path_length / (size(xf, 2) * dt));

%% Visualization
figure('Position', [100, 100, 1400, 900]);

% Create subplots
subplot(3, 3, 1);
% 3D trajectory plot
plot3(xf(1,:), xf(2,:), xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot3(x0(1), x0(2), x0(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(xd_phase1(1), xd_phase1(2), xd_phase1(3), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm'); % Hover point
plot3(xd_phase2(1), xd_phase2(2), xd_phase2(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot3(xf(1,end), xf(2,end), xf(3,end), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Trajectory');
legend('Trajectory', 'Start', 'Hover', 'Target', 'Final', 'Location', 'best');

% X-Y trajectory plot with oriented VTOL rectangle
subplot(3, 3, 2);
plot(xf(1,:), xf(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd_phase2(1), xd_phase2(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

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
plot([0, time_steps(end)], [xd_phase2(3), xd_phase2(3)], 'r--', 'LineWidth', 1);
plot([0, time_steps(end)], [xd_phase1(3), xd_phase1(3)], 'm--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Altitude (m)');
title('Altitude vs Time');
legend('Actual', 'Landing Target', 'Hover Target', 'Location', 'best');

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
text(0.1, 0.55, sprintf('Final Altitude: %.2f m (>= %.2f)', final_altitude, xd_phase2(3)), 'FontSize', 10);
text(0.1, 0.5, sprintf('Final Angle Error: %.1f°', rad2deg(final_angle_error)), 'FontSize', 10);
text(0.1, 0.4, sprintf('Final Velocity: %.3f m/s', final_vel_magnitude), 'FontSize', 10);
text(0.1, 0.3, sprintf('Control Effort: %.3f', total_control_effort), 'FontSize', 10);
text(0.1, 0.2, sprintf('Path Length: %.3f m', path_length), 'FontSize', 10);

% Add overall title
sgtitle('MPPI VTOL Control with Quaternion Dynamics and Two-Phase Vertical Landing', 'FontSize', 16, 'FontWeight', 'bold');

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

function J = runningCost(x, xd_current, R, u, du, nu, phase)
    % Phase-specific cost function with quaternion state and angle penalties
    if phase == 1
        % Phase 1: Horizontal flight to hover point, balanced position weights
        Q_pos = diag([2.5, 2.5, 2.5]);
        Q_vel = diag([0.1, 0.1, 0.1]);      % Slight velocity penalty
        Q_quat = diag([0.5, 5, 5, 0.5]);    % Moderate penalty for roll/pitch
        Q_omega = diag([0.1, 0.1, 0.1]);    % Slight angular velocity penalty
    else
        % Phase 2: Vertical landing, relax horizontal, emphasize vertical and orientation
        Q_pos = diag([1, 1, 20]);           % Lower on x,y, high on z
        Q_vel = diag([1, 1, 1]);            % Higher velocity penalty for soft landing
        Q_quat = diag([0.5, 30, 30, 0.5]);  % High penalty for qx, qy (roll, pitch)
        Q_omega = diag([1, 1, 1]);          % Higher angular velocity penalty
    end
    
    Q = blkdiag(Q_pos, Q_vel, Q_quat, Q_omega);
    
    % State error computation
    pos_err = x(1:3) - xd_current(1:3);
    vel_err = x(4:6) - xd_current(4:6);
    
    % For vertical orientation, target quaternion should represent zero roll/pitch
    % Target: [1, 0, 0, qz] where qz can vary for yaw freedom
    target_quat = [1; 0; 0; x(10)]; % Keep current qz (yaw), zero qx, qy (roll, pitch)
    target_quat = target_quat / norm(target_quat); % Normalize
    
    quat_err = x(7:10) - target_quat;
    omega_err = x(11:13) - xd_current(11:13);
    
    x_err = [pos_err; vel_err; quat_err; omega_err];
    
    qx = x_err' * Q * x_err;
    J = qx + 1/2*u'*R*u + (1-1/nu)/2*du'*R*du + u'*R*du;
end

function J = finalCost(xT, xd_current, phase)
    % Phase-specific final cost with quaternion representation
    if phase == 1
        % Phase 1: Balanced weights, no altitude penalty
        Qf_pos = 20 * diag([5, 5, 5]);
        Qf_vel = 20 * diag([1, 1, 1]);
        Qf_quat = 20 * diag([0, 1, 1, 0]);
        Qf_omega = 20 * diag([1, 1, 1]);
        altitude_penalty = 0;
    else
        % Phase 2: Relax horizontal, high on vertical, orientation, velocities; altitude penalty
        Qf_pos = 20 * diag([1, 1, 10]);     % Lower on x,y
        Qf_vel = 20 * diag([5, 5, 5]);
        Qf_quat = 20 * diag([0, 10, 10, 0]);% High on roll/pitch
        Qf_omega = 20 * diag([5, 5, 5]);
        
        % Add penalty if altitude is below target (z < z_expected)
        altitude_penalty = 0;
        if xT(3) < xd_current(3)
            altitude_penalty = 1000 * (xd_current(3) - xT(3))^2; % Heavy penalty for being below target altitude
        end
    end
    
    Qf = blkdiag(Qf_pos, Qf_vel, Qf_quat, Qf_omega);
    
    % Target state for vertical landing
    xd_target = xd_current;
    % For vertical: qw=1, qx=0, qy=0, qz can be current value
    xd_target(7) = 1;    % qw = 1
    xd_target(8) = 0;    % qx = 0 (no roll)
    xd_target(9) = 0;    % qy = 0 (no pitch)
    xd_target(10) = xT(10); % qz = current qz (allow yaw freedom)
    
    % Normalize target quaternion
    q_target = xd_target(7:10);
    xd_target(7:10) = q_target / norm(q_target);
    
    J = (xT - xd_target)' * Qf * (xT - xd_target) + altitude_penalty;
end

function du = clippeddu(utraj, du)
    u = utraj + du;
    
    u(1,:) = max(u(1,:), 0);
    u(2,:) = max(u(2,:), 0);
    
    u(3,:) = min(max(u(3,:), -15), 15);
    u(4,:) = min(max(u(4,:), -15), 15);
    
    du = u - utraj;
end
DYNAMICS=@vtol2;

nX = 12;%number of states
nU = 4;%number of inputs

%initial conditions
x0= [0;0;0;0;0;0;0;0;0;0;0;0];
xd= [100;70;50;0;0;0;0;0;0;0;0;0]; % Changed target to match iLQR

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

% Vertical landing constraints
position_tolerance = 10;        % meters
angle_tolerance = 0.15;         % radians (~8.6 degrees)
velocity_tolerance = 5;         % m/s for near-zero landing velocity
angular_velocity_tolerance = 0.5; % rad/s

fprintf('=== MPPI VTOL Control with Vertical Landing ===\n');
fprintf('Target position: [%.1f, %.1f, %.1f]\n', xd(1), xd(2), xd(3));
fprintf('Position tolerance: %.1f m\n', position_tolerance);
fprintf('Angle tolerance: %.2f rad (%.1f degrees)\n', angle_tolerance, rad2deg(angle_tolerance));
fprintf('Velocity tolerance: %.1f m/s\n', velocity_tolerance);
fprintf('Angular velocity tolerance: %.2f rad/s\n', angular_velocity_tolerance);
fprintf('\nStarting optimization...\n');

%% Run MPPI Optimization
computation_start = tic;

for iter = 1:400
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
            xtraj(:,t+1) = xtraj(:,t) + DYNAMICS(xtraj(:,t), u+du(:,t))*dt;
            Straj(k) = Straj(k) + runningCost(xtraj(:,t), xd, R, u, du(:,t), nu);
        end
        Straj(k) = Straj(k) + finalCost(xtraj(:,N), xd);
    end

    minS = min(Straj); % Minimum rollout cost
    
    if mod(iter, 50) == 1
        fprintf('Iteration %d: Min cost = %.2f\n', iter, minS);
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
    x = x + DYNAMICS(x, utraj(:,1))*dt;
    uOpt = [uOpt, utraj(:,1)];
    
    % Shift the nominal inputs 
    for t = 2:N-1
        utraj(:,t-1) = utraj(:,t);
    end
    utraj(:,N-1) = [0;0;0;0];
    
    % Check convergence with both position and angle constraints
    current_pos = x(1:3);
    current_vel = x(4:6);
    current_angles = x(7:9);        % [phi, theta, psi] - roll, pitch, yaw
    current_angular_vel = x(10:12); % [wx, wy, wz]
    
    pos_dist = norm(current_pos - xd(1:3));
    vel_magnitude = norm(current_vel);
    angular_vel_magnitude = norm(current_angular_vel);
    
    % Check if angles are close to vertical (roll and pitch should be near zero)
    roll_error = abs(current_angles(1));   % phi should be ~0
    pitch_error = abs(current_angles(2));  % theta should be ~0
    max_angle_error = max(roll_error, pitch_error);
    
    % Display current status every 20 iterations
    if mod(iter, 20) == 0
        fprintf('Iter %d: Pos dist=%.2f, Vel=%.2f, Max angle err=%.3f rad (%.1f°), ω=%.2f\n', ...
                iter, pos_dist, vel_magnitude, max_angle_error, rad2deg(max_angle_error), angular_vel_magnitude);
    end
    
    % Check if all landing criteria are satisfied
    position_ok = pos_dist < position_tolerance;
    angle_ok = max_angle_error < angle_tolerance;
    velocity_ok = vel_magnitude < velocity_tolerance;
    angular_velocity_ok = angular_vel_magnitude < angular_velocity_tolerance;
    
    if position_ok && angle_ok && velocity_ok && angular_velocity_ok
        fprintf('\n=== SUCCESSFUL VERTICAL LANDING ===\n');
        fprintf('Target reached at iteration %d!\n', iter);
        fprintf('Final position distance: %.2f m\n', pos_dist);
        fprintf('Final velocity magnitude: %.2f m/s\n', vel_magnitude);
        fprintf('Final roll error: %.3f rad (%.1f degrees)\n', roll_error, rad2deg(roll_error));
        fprintf('Final pitch error: %.3f rad (%.1f degrees)\n', pitch_error, rad2deg(pitch_error));
        fprintf('Final angular velocity: %.2f rad/s\n', angular_vel_magnitude);
        break;
    end
    
    % Early termination if close to target but need better angle control
    if pos_dist < position_tolerance && ~angle_ok
        fprintf('Position reached but angle error too large: %.3f rad (%.1f°)\n', ...
                max_angle_error, rad2deg(max_angle_error));
    end
end

computation_time = toc(computation_start);

%% Calculate Performance Metrics
final_pos_error = norm(xf(1:3,end) - xd(1:3));
final_vel_magnitude = norm(xf(4:6,end));
final_angles = xf(7:9,end);
final_angle_error = max(abs(final_angles(1)), abs(final_angles(2))); % max of roll, pitch errors
final_angular_vel = norm(xf(10:12,end));

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
final_yaw = xf(9, end); % psi angle

% Create rectangle vertices in body frame
rect_body = [-vtol_width/2, -vtol_height/2;
              vtol_width/2, -vtol_height/2;
              vtol_width/2,  vtol_height/2;
             -vtol_width/2,  vtol_height/2;
             -vtol_width/2, -vtol_height/2];

% Rotate rectangle based on yaw angle
cos_yaw = cos(final_yaw);
sin_yaw = sin(final_yaw);
R_yaw = [cos_yaw, -sin_yaw; sin_yaw, cos_yaw];

rect_world = zeros(size(rect_body));
for i = 1:size(rect_body, 1)
    rotated_point = R_yaw * rect_body(i,:)';
    rect_world(i,:) = rotated_point' + [xf(1,end), xf(2,end)];
end

plot(rect_world(:,1), rect_world(:,2), 'k-', 'LineWidth', 2);
fill(rect_world(:,1), rect_world(:,2), [0.3, 0.3, 0.8], 'FaceAlpha', 0.7);

% Add orientation arrow showing forward direction
arrow_length = 2;
forward_dir = R_yaw * [arrow_length; 0];
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

% Euler Angles
subplot(3, 3, 5);
plot(time_steps, rad2deg(xf(7,:)), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Roll (φ)');
hold on;
plot(time_steps, rad2deg(xf(8,:)), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Pitch (θ)');
plot(time_steps, rad2deg(xf(9,:)), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Yaw (ψ)');
plot([0, time_steps(end)], [rad2deg(angle_tolerance), rad2deg(angle_tolerance)], 'r--', 'LineWidth', 1, 'DisplayName', 'Angle Limit');
plot([0, time_steps(end)], [-rad2deg(angle_tolerance), -rad2deg(angle_tolerance)], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Angle (degrees)');
title('Euler Angles (Vertical Landing)');
legend('Location', 'best');

% Control inputs
subplot(3, 3, 6);
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
subplot(3, 3, 7);
plot(time_steps, xf(10,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'ωx');
hold on;
plot(time_steps, xf(11,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'ωy');
plot(time_steps, xf(12,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'ωz');
plot([0, time_steps(end)], [angular_velocity_tolerance, angular_velocity_tolerance], 'r--', 'LineWidth', 1, 'DisplayName', 'ω Limit');
plot([0, time_steps(end)], [-angular_velocity_tolerance, -angular_velocity_tolerance], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
title('Angular Velocities');
legend('Location', 'best');

% Landing criteria satisfaction over time
subplot(3, 3, 8);
pos_errors = zeros(1, length(time_steps));
angle_errors = zeros(1, length(time_steps));
vel_magnitudes = zeros(1, length(time_steps));
for i = 1:length(time_steps)
    pos_errors(i) = norm(xf(1:3,i) - xd(1:3));
    angle_errors(i) = max(abs(xf(7,i)), abs(xf(8,i))); % max of roll, pitch
    vel_magnitudes(i) = norm(xf(4:6,i));
end

plot(time_steps, pos_errors, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Position Error');
hold on;
plot(time_steps, rad2deg(angle_errors), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Max Angle Error (deg)');
plot(time_steps, vel_magnitudes, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Velocity Magnitude');
plot([0, time_steps(end)], [position_tolerance, position_tolerance], 'b--', 'LineWidth', 1);
plot([0, time_steps(end)], [rad2deg(angle_tolerance), rad2deg(angle_tolerance)], 'r--', 'LineWidth', 1);
plot([0, time_steps(end)], [velocity_tolerance, velocity_tolerance], 'g--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Error Magnitude');
title('Landing Criteria Convergence');
legend('Location', 'best');

% Performance summary
subplot(3, 3, 9);
axis off;
text(0.1, 0.9, 'PERFORMANCE SUMMARY', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Computation Time: %.3f s', computation_time), 'FontSize', 10);
text(0.1, 0.7, sprintf('Iterations: %d', size(xf, 2)), 'FontSize', 10);
text(0.1, 0.6, sprintf('Final Position Error: %.3f m', final_pos_error), 'FontSize', 10);
text(0.1, 0.5, sprintf('Final Angle Error: %.1f°', rad2deg(final_angle_error)), 'FontSize', 10);
text(0.1, 0.4, sprintf('Final Velocity: %.3f m/s', final_vel_magnitude), 'FontSize', 10);
text(0.1, 0.3, sprintf('Control Effort: %.3f', total_control_effort), 'FontSize', 10);
text(0.1, 0.2, sprintf('Path Length: %.3f m', path_length), 'FontSize', 10);

% Add overall title
sgtitle('MPPI VTOL Control with Vertical Landing', 'FontSize', 16, 'FontWeight', 'bold');

%% Helper functions
function J = runningCost(x, xd, R, u, du, nu)
    % Enhanced cost function with angle penalties for vertical landing
    Q_pos = diag([2.5, 2.5, 20]);     % Position weights
    Q_vel = diag([0, 0, 0]);          % Velocity weights  
    Q_ang = diag([15, 15, 1]);        % Angle weights (high penalty for roll/pitch)
    Q_omega = diag([0, 0, 0]);        % Angular velocity weights
    
    Q = blkdiag(Q_pos, Q_vel, Q_ang, Q_omega);
    
    % Position and angle errors
    pos_err = x(1:3) - xd(1:3);
    vel_err = x(4:6) - xd(4:6);
    
    % For vertical landing, target angles should be [0, 0, current_yaw]
    target_angles = [0; 0; x(9)]; % Keep current yaw, but zero roll and pitch
    ang_err = x(7:9) - target_angles;
    omega_err = x(10:12) - xd(10:12);
    
    x_err = [pos_err; vel_err; ang_err; omega_err];
    
    qx = x_err' * Q * x_err;
    J = qx + 1/2*u'*R*u + (1-1/nu)/2*du'*R*du + u'*R*du;
end

function J = finalCost(xT, xd)
    % Enhanced final cost with strong penalties for non-vertical landing
    Qf_pos = 20 * diag([5, 5, 10]);    % Position weights
    Qf_vel = 20 * diag([5, 5, 5]);         % Velocity weights (penalty for landing with high velocity)
    Qf_ang = 20 * diag([5, 5, 0]);       % Angle weights (very high penalty for non-vertical)
    Qf_omega = 20 * diag([5, 5, 5]);       % Angular velocity weights
    
    Qf = blkdiag(Qf_pos, Qf_vel, Qf_ang, Qf_omega);
    
    % Target state for vertical landing
    xd_vertical = xd;
    xd_vertical(7:8) = 0;  % Target roll and pitch = 0 for vertical landing
    % Keep target yaw as in xd, or could be set to current yaw for flexibility
    
    J = (xT - xd_vertical)' * Qf * (xT - xd_vertical);
end

function du = clippeddu(utraj, du)
    u = utraj + du;
    
    u(1,:) = max(u(1,:), 0);
    u(2,:) = max(u(2,:), 0);
    
    u(3,:) = min(max(u(3,:), -15), 15);
    u(4,:) = min(max(u(4,:), -15), 15);
    
    du = u - utraj;
end
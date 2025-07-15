DYNAMICS=@vtol3_quaternion;

nX = 13;%number of states (now includes quaternion)
nU = 4;%number of inputs

%initial conditions (quaternion representation)
% [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
x0= [0;0;0;0;0;0;1;0;0;0;0;0;0];  % quaternion starts as [1,0,0,0] (identity)
xd= [100;10;50;0;0;0;1;0;0;0;0;0;0];  % desired state with identity quaternion

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

%% Run MPPI Optimization
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
            
            % Add error handling for rollout dynamics
            try
                x_next = xtraj(:,t) + DYNAMICS(xtraj(:,t), u+du(:,t))*dt;
                
                % Check for valid state
                if any(~isreal(x_next)) || any(isnan(x_next)) || any(isinf(x_next))
                    % If invalid, use previous state and penalize this rollout
                    xtraj(:,t+1) = xtraj(:,t);
                    Straj(k) = max_cost * 1000; % Heavy penalty
                    break; % Skip rest of this rollout
                else
                    xtraj(:,t+1) = x_next;
                end
                
                % Normalize quaternion to maintain unit quaternion with safety check
                q = xtraj(7:10,t+1);
                q_norm = norm(q);
                if q_norm > 1e-6
                    xtraj(7:10,t+1) = q / q_norm;
                else
                    xtraj(7:10,t+1) = [1; 0; 0; 0]; % Reset to identity
                end
                
                % Ensure positive scalar part
                if xtraj(7,t+1) < 0
                    xtraj(7:10,t+1) = -xtraj(7:10,t+1);
                end
                
            catch
                % If dynamics fail, heavily penalize this rollout
                Straj(k) = max_cost * 1000;
                break;
            end
            
            Straj(k) = Straj(k) + runningCost(xtraj(:,t), xd, R, u, du(:,t), nu);
        end
        % Only add final cost if rollout completed successfully
        if Straj(k) < max_cost * 100 % Check if rollout wasn't heavily penalized
            Straj(k) = Straj(k) + finalCost(xtraj(:,N), xd);
        end
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
    
    % Execute the utraj(0) with error handling
    try
        x_new = x + DYNAMICS(x, utraj(:,1))*dt;
        
        % Check for complex numbers or NaN/Inf
        if any(~isreal(x_new)) || any(isnan(x_new)) || any(isinf(x_new))
            fprintf('Warning: Invalid state detected at iteration %d, using previous state\n', iter);
            % Keep previous state and reduce control input
            utraj(:,1) = utraj(:,1) * 0.5;
        else
            x = x_new;
        end
        
        % Normalize quaternion after integration with safety check
        q = x(7:10);
        q_norm = norm(q);
        if q_norm > 1e-6
            x(7:10) = q / q_norm;
        else
            fprintf('Warning: Quaternion norm too small at iteration %d, resetting to identity\n', iter);
            x(7:10) = [1; 0; 0; 0]; % Reset to identity quaternion
        end
        
        % Ensure quaternion has positive scalar part (q0 >= 0) for consistency
        if x(7) < 0
            x(7:10) = -x(7:10);
        end
        
    catch ME
        fprintf('Error in dynamics at iteration %d: %s\n', iter, ME.message);
        fprintf('State: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', x);
        fprintf('Control: [%.3f, %.3f, %.3f, %.3f]\n', utraj(:,1));
        
        % Reduce control input and continue
        utraj(:,1) = utraj(:,1) * 0.1;
        fprintf('Reducing control input and continuing...\n');
    end
    
    uOpt = [uOpt, utraj(:,1)];
    
    % Shift the nominal inputs 
    for t = 2:N-1
        utraj(:,t-1) = utraj(:,t);
    end
    utraj(:,N-1) = [0;0;0;0];
    
    s = x(1:3) %Current distance to target
    dist = norm(x(1:3) - xd(1:3))
    if dist < 10
        fprintf('Target reached! Distance: %.2f m, Iteration: %d\n', dist, iter);
        break;
    end
end

%% Visualization for Quaternion-based VTOL
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

% X-Y trajectory plot with VTOL rectangle
subplot(3, 3, 2);
plot(xf(1,:), xf(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd(1), xd(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Draw VTOL as rectangle at final position with orientation
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
time_steps = 0:dt:(length(xf)-1)*dt;
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

% Quaternion components
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

% Euler angles (converted from quaternion)
subplot(3, 3, 6);
euler_angles = zeros(3, length(xf));
for i = 1:length(xf)
    q = xf(7:10, i)';
    [roll, pitch, yaw] = quat2angle(q, 'XYZ');
    euler_angles(:, i) = [roll; pitch; yaw] * 180/pi; % Convert to degrees
end
plot(time_steps, euler_angles(1,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Roll');
hold on;
plot(time_steps, euler_angles(2,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Pitch');
plot(time_steps, euler_angles(3,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Yaw');
grid on;
xlabel('Time (s)');
ylabel('Angle (degrees)');
title('Euler Angles');
legend('Location', 'best');

% Angular velocities
subplot(3, 3, 7);
plot(time_steps, xf(11,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'ωx');
hold on;
plot(time_steps, xf(12,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'ωy');
plot(time_steps, xf(13,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'ωz');
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
title('Angular Velocities');
legend('Location', 'best');

% Control inputs
subplot(3, 3, 8);
time_control = 0:dt:(length(uOpt)-1)*dt;
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

% Quaternion norm (should stay at 1)
subplot(3, 3, 9);
quat_norms = sqrt(sum(xf(7:10,:).^2, 1));
plot(time_steps, quat_norms, 'k-', 'LineWidth', 2);
hold on;
plot([0, time_steps(end)], [1, 1], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Quaternion Norm');
title('Quaternion Norm Verification');
legend('Actual', 'Target (1.0)', 'Location', 'best');
ylim([0.99, 1.01]);

% Add overall title
sgtitle('MPPI VTOL3 Quaternion Control Results', 'FontSize', 16, 'FontWeight', 'bold');

% Print final statistics
fprintf('\n=== QUATERNION VTOL SIMULATION RESULTS ===\n');
fprintf('Initial position: [%.2f, %.2f, %.2f]\n', x0(1), x0(2), x0(3));
fprintf('Target position: [%.2f, %.2f, %.2f]\n', xd(1), xd(2), xd(3));
fprintf('Final position: [%.2f, %.2f, %.2f]\n', xf(1,end), xf(2,end), xf(3,end));
fprintf('Final distance to target: %.2f m\n', norm(xf(1:3,end) - xd(1:3)));
fprintf('Total simulation time: %.2f s\n', time_steps(end));
fprintf('Number of iterations: %d\n', length(xf));

% Final quaternion and Euler angles
q_final = xf(7:10, end)';
[roll_final, pitch_final, yaw_final] = quat2angle(q_final, 'XYZ');
fprintf('Final orientation (degrees): Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ...
        roll_final*180/pi, pitch_final*180/pi, yaw_final*180/pi);
fprintf('Final quaternion norm: %.6f (should be 1.0)\n', norm(q_final));

%% Optional: 3D Animated trajectory with oriented VTOL
figure('Position', [200, 200, 900, 700]);
for i = 1:5:length(xf)
    clf;
    % Plot trajectory up to current point
    plot3(xf(1,1:i), xf(2,1:i), xf(3,1:i), 'b-', 'LineWidth', 2);
    hold on;
    
    % Plot target
    plot3(xd(1), xd(2), xd(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % Get current position and orientation
    current_pos = xf(1:3, i);
    current_quat = xf(7:10, i);
    
    % Convert quaternion to rotation matrix
    R_current = quat2rotm(current_quat');
    
    % Draw VTOL as oriented 3D rectangle
    % Define VTOL body vertices (centered at origin)
    vtol_vertices = [
        -vtol_width/2, -vtol_height/2, -0.2;  % bottom face
         vtol_width/2, -vtol_height/2, -0.2;
         vtol_width/2,  vtol_height/2, -0.2;
        -vtol_width/2,  vtol_height/2, -0.2;
        -vtol_width/2, -vtol_height/2,  0.2;  % top face
         vtol_width/2, -vtol_height/2,  0.2;
         vtol_width/2,  vtol_height/2,  0.2;
        -vtol_width/2,  vtol_height/2,  0.2
    ];
    
    % Rotate and translate vertices
    vtol_world = zeros(size(vtol_vertices));
    for j = 1:size(vtol_vertices, 1)
        rotated_vertex = R_current * vtol_vertices(j, :)';
        vtol_world(j, :) = (rotated_vertex + current_pos)';
    end
    
    % Define faces of the rectangular prism
    faces = [
        1, 2, 3, 4;  % bottom
        5, 6, 7, 8;  % top
        1, 2, 6, 5;  % front
        3, 4, 8, 7;  % back
        1, 4, 8, 5;  % left
        2, 3, 7, 6   % right
    ];
    
    % Draw the VTOL
    for f = 1:size(faces, 1)
        face_vertices = vtol_world(faces(f, :), :);
        fill3(face_vertices(:, 1), face_vertices(:, 2), face_vertices(:, 3), ...
              [0.3, 0.3, 0.8], 'FaceAlpha', 0.7, 'EdgeColor', 'k');
    end
    
    % Add orientation arrows (body frame axes)
    arrow_scale = 1.5;
    x_axis = R_current * [arrow_scale; 0; 0] + current_pos;
    y_axis = R_current * [0; arrow_scale; 0] + current_pos;
    z_axis = R_current * [0; 0; arrow_scale] + current_pos;
    
    % X-axis (red), Y-axis (green), Z-axis (blue)
    plot3([current_pos(1), x_axis(1)], [current_pos(2), x_axis(2)], ...
          [current_pos(3), x_axis(3)], 'r-', 'LineWidth', 3);
    plot3([current_pos(1), y_axis(1)], [current_pos(2), y_axis(2)], ...
          [current_pos(3), y_axis(3)], 'g-', 'LineWidth', 3);
    plot3([current_pos(1), z_axis(1)], [current_pos(2), z_axis(2)], ...
          [current_pos(3), z_axis(3)], 'b-', 'LineWidth', 3);
    
    grid on;
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    zlabel('Z Position (m)');
    title(sprintf('3D VTOL Animation - Step %d/%d', i, length(xf)));
    
    % Set consistent axis limits
    x_range = [min(min(xf(1,:)), xd(1))-10, max(max(xf(1,:)), xd(1))+10];
    y_range = [min(min(xf(2,:)), xd(2))-10, max(max(xf(2,:)), xd(2))+10];
    z_range = [min(min(xf(3,:)), xd(3))-5, max(max(xf(3,:)), xd(3))+10];
    xlim(x_range);
    ylim(y_range);
    zlim(z_range);
    
    view(45, 30); % Set viewing angle
    pause(0.1);
end

%% Helper functions
function J = runningCost(x, xd, R, u, du, nu)
    % Updated cost function for quaternion representation
    % Position, velocity, and angular velocity weights (12 elements total)
    Q = diag([2.5, 2.5, 20, 0, 0, 0, 1, 1, 15, 1, 1, 15]);
    
    % Handle quaternion error specially with safety checks
    % For quaternion, we need to compute the rotation error properly
    q_curr = x(7:10);
    q_des = xd(7:10);
    
    % Ensure quaternions are normalized
    q_curr = q_curr / max(norm(q_curr), 1e-6);
    q_des = q_des / max(norm(q_des), 1e-6);
    
    % Quaternion error: q_err = q_des^(-1) * q_curr
    q_des_inv = [q_des(1); -q_des(2:4)]; % conjugate of unit quaternion
    q_err = quatmultiply(q_des_inv', q_curr')';
    
    % Use the vector part of quaternion error (small angle approximation)
    quat_err = 2 * q_err(2:4); % 2 * vector part gives rotation error
    
    % Ensure real values
    quat_err = real(quat_err);
    
    % Construct error vector with quaternion error (12 elements)
    % [pos_err(3), vel_err(3), quat_err(3), omega_err(3)]
    xerr_quat = [x(1:6) - xd(1:6); quat_err; x(11:13) - xd(11:13)];
    
    % Quaternion-aware cost
    qx = xerr_quat'*Q*xerr_quat;
    J = qx + 1/2*u'*R*u + (1-1/nu)/2*du'*R*du + u'*R*du;
end

function J = finalCost(xT,xd)
    % Updated final cost function for quaternion representation
    Qf = 20*diag([2.5, 2.5, 20, 0, 0, 0, 1, 1, 15, 1, 1, 15]);
    
    % Handle quaternion error specially with safety checks
    % For quaternion, we need to compute the rotation error properly
    q_curr = xT(7:10);
    q_des = xd(7:10);
    
    % Ensure quaternions are normalized
    q_curr = q_curr / max(norm(q_curr), 1e-6);
    q_des = q_des / max(norm(q_des), 1e-6);
    
    % Quaternion error: q_err = q_des^(-1) * q_curr
    q_des_inv = [q_des(1); -q_des(2:4)]; % conjugate of unit quaternion
    q_err = quatmultiply(q_des_inv', q_curr')';
    
    % Use the vector part of quaternion error (small angle approximation)
    quat_err = 2 * q_err(2:4); % 2 * vector part gives rotation error
    
    % Ensure real values
    quat_err = real(quat_err);
    
    % Construct error vector with quaternion error (12 elements)
    % [pos_err(3), vel_err(3), quat_err(3), omega_err(3)]
    xerr_quat = [xT(1:6) - xd(1:6); quat_err; xT(11:13) - xd(11:13)];
    
    J = xerr_quat'*Qf*xerr_quat;
end

function du = clippeddu(utraj, du)
u = utraj + du;

u(1,:) = max(u(1,:), 0);
u(2,:) = max(u(2,:), 0);

u(3,:) = min(max(u(3,:), -15), 15);
u(4,:) = min(max(u(4,:), -15), 15);

du = u - utraj;

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
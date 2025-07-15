DYNAMICS=@vtol2;

nX = 12;%number of states
nU = 4;%number of inputs

%initial conditions
x0= [0;0;0;0;0;0;0;0;0;0;0;0];
xd= [100;100;50;0;0;0;0;0;0;0;0;0];

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
    utraj(:,N-1) = [0;0;0;0];
    
    s = x(1:3) %Current distance to target
    dist = norm(x(1:3) - xd(1:3))
    if dist < 10
        fprintf('Target reached! Distance: %.2f m, Iteration: %d\n', dist, iter);
        break;
    end
end

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Create subplots
subplot(2, 2, 1);
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
subplot(2, 2, 2);
plot(xf(1,:), xf(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(xd(1), xd(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Draw VTOL as rectangle at final position
vtol_width = 3;
vtol_height = 1.5;
vtol_x = xf(1,end) - vtol_width/2;
vtol_y = xf(2,end) - vtol_height/2;
rectangle('Position', [vtol_x, vtol_y, vtol_width, vtol_height], ...
          'FaceColor', [0.3, 0.3, 0.8], 'EdgeColor', 'k', 'LineWidth', 2);

% Add orientation arrow
arrow_length = 2;
arrow_x = [xf(1,end), xf(1,end) + arrow_length];
arrow_y = [xf(2,end), xf(2,end)];
plot(arrow_x, arrow_y, 'r-', 'LineWidth', 3);
plot(arrow_x(2), arrow_y(2), 'r>', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('X-Y Trajectory with VTOL');
legend('Trajectory', 'Start', 'Target', 'VTOL', 'Direction', 'Location', 'best');
axis equal;

% Altitude vs time
subplot(2, 2, 3);
time_steps = 0:dt:(length(xf)-1)*dt;
plot(time_steps, xf(3,:), 'b-', 'LineWidth', 2);
hold on;
plot([0, time_steps(end)], [xd(3), xd(3)], 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Altitude (m)');
title('Altitude vs Time');
legend('Actual', 'Target', 'Location', 'best');

% Control inputs
subplot(2, 2, 4);
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

% Add overall title
sgtitle('MPPI VTOL Control Results', 'FontSize', 16, 'FontWeight', 'bold');

% Print final statistics
fprintf('\n=== SIMULATION RESULTS ===\n');
fprintf('Initial position: [%.2f, %.2f, %.2f]\n', x0(1), x0(2), x0(3));
fprintf('Target position: [%.2f, %.2f, %.2f]\n', xd(1), xd(2), xd(3));
fprintf('Final position: [%.2f, %.2f, %.2f]\n', xf(1,end), xf(2,end), xf(3,end));
fprintf('Final distance to target: %.2f m\n', norm(xf(1:3,end) - xd(1:3)));
fprintf('Total simulation time: %.2f s\n', time_steps(end));
fprintf('Number of iterations: %d\n', length(xf));

%% Optional: Animated trajectory
figure('Position', [200, 200, 800, 600]);
for i = 1:5:length(xf)
    clf;
    % Plot trajectory up to current point
    plot3(xf(1,1:i), xf(2,1:i), xf(3,1:i), 'b-', 'LineWidth', 2);
    hold on;
    
    % Plot target
    plot3(xd(1), xd(2), xd(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % Draw VTOL rectangle at current position
    current_pos = xf(:,i);
    
    % For 3D visualization, we'll project the rectangle onto the X-Y plane
    % and show it at the current altitude
    vtol_corners_x = [current_pos(1)-vtol_width/2, current_pos(1)+vtol_width/2, ...
                     current_pos(1)+vtol_width/2, current_pos(1)-vtol_width/2, current_pos(1)-vtol_width/2];
    vtol_corners_y = [current_pos(2)-vtol_height/2, current_pos(2)-vtol_height/2, ...
                     current_pos(2)+vtol_height/2, current_pos(2)+vtol_height/2, current_pos(2)-vtol_height/2];
    vtol_corners_z = ones(1,5) * current_pos(3);
    
    plot3(vtol_corners_x, vtol_corners_y, vtol_corners_z, 'k-', 'LineWidth', 3);
    fill3(vtol_corners_x, vtol_corners_y, vtol_corners_z, [0.3, 0.3, 0.8], 'FaceAlpha', 0.7);
    
    % Add direction arrow
    arrow_end_x = current_pos(1) + arrow_length;
    arrow_end_y = current_pos(2);
    plot3([current_pos(1), arrow_end_x], [current_pos(2), arrow_end_y], ...
          [current_pos(3), current_pos(3)], 'r-', 'LineWidth', 2);
    
    grid on;
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    zlabel('Z Position (m)');
    title(sprintf('VTOL Animation - Step %d/%d', i, length(xf)));
    axis equal;
    
    % Set consistent axis limits
    x_range = [min(min(xf(1,:)), xd(1))-10, max(max(xf(1,:)), xd(1))+10];
    y_range = [min(min(xf(2,:)), xd(2))-10, max(max(xf(2,:)), xd(2))+10];
    z_range = [min(min(xf(3,:)), xd(3))-5, max(max(xf(3,:)), xd(3))+5];
    xlim(x_range);
    ylim(y_range);
    zlim(z_range);
    
    pause(0.1);
end

%% Helper functions
function J = runningCost(x, xd, R, u, du, nu)
    Q = diag([2.5, 2.5, 20, 0, 0, 0, 1, 1, 15, 0, 0, 0]);
    qx = (x-xd)'*Q*(x-xd);
    J = qx + 1/2*u'*R*u + (1-1/nu)/2*du'*R*du + u'*R*du;
end

function J = finalCost(xT,xd)
    Qf = 20*diag([2.5, 2.5, 20, 0, 0, 0, 1, 1, 15, 0, 0, 0]);
    J = (xT-xd)'*Qf*(xT-xd);
end

function du = clippeddu(utraj, du)
u = utraj + du;

u(1,:) = max(u(1,:), 0);
u(2,:) = max(u(2,:), 0);

u(3,:) = min(max(u(3,:), -15), 15);
u(4,:) = min(max(u(4,:), -15), 15);

du = u - utraj;

end
DYNAMICS = @quadrotor;

% Parameters
nX = 12; nU = 4; steps = 50;
dt = 0.05;
num_samples = 2000;
lambda = 100.0;
rho = 20.0;
var = diag([2.5, 5*10^(-3), 5*10^(-3), 5*10^(-3)]);

Q  = 0.01*diag([2 2 10 0 0 0 0 0 0 0 0 0]);
R  = 0.01 * eye(nU);
Qf = 10 * Q;

% Initial state and goal
x0 = zeros(nX,1);
xd = repmat([1;1;1; zeros(nX-3,1)], 1, steps+1);

% Initial control guess
u0 = zeros(nU, steps);

% Run MPPI
x = x0;
for i = 1:100
    x
    step_u = MPPI_quadrotor(x, u0, num_samples, steps, dt, Q, R, Qf, lambda, var, rho, DYNAMICS, xd);
    
    x = x + DYNAMICS(x, step_u(:,1))*dt;
    
    u0 = [step_u(:,2:end),zeros(nU,1)];

end


function step_u = MPPI_quadrotor(x0, u0, num_samples, steps, dt, Q, R, Qf, lambda, var, rho, DYNAMICS, xd)
    % MPPI controller for a quadrotor
    % x0: initial state [12x1]
    % u0: initial control guess [4xsteps]
    % xd: desired state trajectory [12x(steps+1)]

    nX = length(x0);
    nU = size(u0, 1);

    costs = zeros(1, num_samples);
    du_all = zeros(nU, steps, num_samples);
    
    for k = 1:num_samples
        % Sample control noise
        du = sqrt(var) * randn(nU, steps) / sqrt(rho * dt);
        u_rollout = u0 + du;
        
        x = x0;
        total_cost = 0;
        
        for t = 1:steps
            x = DYNAMICS(x, u_rollout(:,t));
            state_error = x - xd(:,t);
            total_cost = total_cost + state_error' * Q * state_error + u_rollout(:,t)' * R * u_rollout(:,t);
        end
        
        % Final cost
        final_error = x - xd(:,end);
        total_cost = total_cost + final_error' * Qf * final_error;
        
        costs(k) = total_cost;
        du_all(:,:,k) = du;
    end

    % Compute weights
    beta = min(costs);  % numerical stability
    weights = exp(-1/lambda * (costs - beta));
    weights = weights / sum(weights + 1e-10);

    % Weighted sum of perturbations
    delta_u = zeros(nU, steps);
    for k = 1:num_samples
        delta_u = delta_u + weights(k) * du_all(:,:,k);
    end
    
    % Update control
    step_u = u0 + delta_u;
end


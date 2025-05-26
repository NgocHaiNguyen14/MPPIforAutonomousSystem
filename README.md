# MPPI_forAutonomousSystem

Algorithm

init LQR params: Q, R, Qf
init MPPI params: lambda, N, scale
init traj params: x0, xd, u_opt = [], t, steps
dt = t / steps

solve u_baseline by iLQR

for step in 0 to steps-1:

    generate N perturbations delta_u ~ N(0, scale)
    traj_cost = []

    for i in 0 to N-1:
        utraj_i = u_baseline + delta_u[i]
        
        if len(u_opt) > 0:
            utraj_i[step] = u_opt[-1]
        
        xtraj_i = forward_pass(x0, utraj_i, dt)
        cost_i = trajectory_cost(xtraj_i, utraj_i, xd)
        traj_cost.append(cost_i)
    
    weights = exp(-1/lambda * traj_cost)
    step_u = sum_i weights[i] * (u_baseline[step] + delta_u[i, step]) / sum(weights)

    u_opt.append(step_u)

endfor



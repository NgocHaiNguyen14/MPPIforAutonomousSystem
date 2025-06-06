# MPPI_forAutonomousSystem

Algorithm MPPI

# --- Initialization ---
Initialize cost parameters: Q, R, Qf
Initialize MPPI parameters: λ (lambda), K (num_samples), N (time steps), CovU (noise std)
Initialize trajectory parameters:
    x0 (initial state), xd (desired state), 
    u_opt = [] (optimal control sequence), 
    T (total time), steps (planning horizon)
Set dt = T / steps

Initialize utraj = [u0, ..., uN-1]

# --- MPPI Optimization Loop ---
while task not done:

	x = getCurrentState()
	Initialize Straj storing cost of rollouts
	for k = 1 to K
		du = getRandomRollouts du E R^(nUxN-1)
		store du to DU cell
		Initalize xtraj E R^(nX x N)
		for t = 1:N-1
			xtraj(t+1) = xtraj(t) + f(xtraj(t), utraj(t) + du(t))*dt
			Straj(k) += runningCost(xtraj(t), xd, utraj(t), du(t))
		endfor
		
		Straj(k) += finalCost(xtraj(N),xd)
	endfor
	
	minS = min(Straj)
	
	for t = 1:N-1
		Initalize sum of weights(ss) and sum of weight*inputs(su) = 0
		for k = 1:K
			ss += exp(-1/lambda*Straj(k) - minS)
			su += exp(-1/lambda*Straj(k) - minS)*DU{k}(:,t)
		endfor
		
		update nominal input utraj(t) += su/ss
	endfor
	
	Execute utraj(1)
	
	Shift the nominal input
	for t = 2:N-1
		utraj(t-1) = utraj(t)
	endfor
	
	initialize utraj(N-1)
	
endwhile
	




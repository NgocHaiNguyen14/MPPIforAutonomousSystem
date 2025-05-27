%==========================================================================
%   iterativeLQR function
%==========================================================================
function [xtraj, utraj, ktraj, Ktraj] = iLQR_quadrotor(x0, xtraj, utraj, ktraj, Ktraj,N,dt, Q, R, Qf,xd, DYNAMICS, GRADIENTS)
    J=1e6;
    Jlast = J;
    for i=1:1000
        fprintf("Iteration: %d\n", i)
        [xtraj, utraj, J]=forward_pass(x0, xtraj, utraj, ktraj, Ktraj, N, dt,Q,R,Qf,xd,J, DYNAMICS);
        [Ktraj, ktraj]=backward_pass(xtraj, utraj, ktraj, Ktraj, Q, R, Qf, xd,N,dt, GRADIENTS);
        if abs(J-Jlast) < 0.000001
            break;
        end
        Jlast = J;

    end
end

%==========================================================================
%   get Q-terms (fill in)
%==========================================================================
function [Qx, Qu, Qxx, Qux, Quu]= Q_terms (gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx)
Qx = gx+fx'*Vx;
Qu = gu+fu'*Vx;
Qxx = gxx + fx'*Vxx*fx;
Qux = 2*gux + fu'*Vxx*fx;
Quu = guu + fu'*Vxx*fu;
end


%==========================================================================
%   get gains (fill in)
%==========================================================================
function [K, v]= gains(Qx,Qu,Qxx,Qux,Quu)
    lambda = 0.1;  % parameter to avoid singularity
    Quu_reg = Quu + lambda * eye(size(Quu));
    v = -Quu_reg \ Qu;
    K = -Quu_reg \ Qux;
end

%==========================================================================
%   get V terms (fill in)
%==========================================================================
function [Vx, Vxx] = Vterms (Qx,Qu,Qxx,Qux,Quu,K,v)
    Vx = (Qx' + Qu'  *K + v'*Qux + v'*Quu*K)';
    Vxx = Qxx + K'*Qux+ Qux'*K+K'*Quu*K;
end

%==========================================================================
%   compute backward passQ_terms
%==========================================================================
function [Ktraj, ktraj] = backward_pass(xtraj, utraj, ktraj, Ktraj, Q, R, Qf, xd,N,dt, GRADIENTS)
    global t
    [gxN, guN, gxxN, guxN, guuN] = final_cost_gradients(xtraj(:,N), utraj(:,N-1), xd, Qf);
    Vxx = gxxN;%fill in Vxx
    Vx = gxN;%fill in Vx
    for i=N-1:-1:1
        [gx, gu, gxx, gux, guu] = cost_gradients(xtraj(:,i),utraj(:,i),xd,Q,R);
        [fx, fu] = GRADIENTS(xtraj(:,i),utraj(:,i));
        fu = fu*dt;
        fx = (fx*dt+eye(length(fx)));
        [Qx, Qu, Qxx, Qux, Quu]= Q_terms (gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx);
        [Ktraj(:,:,i), ktraj(:,i)]= gains(Qx,Qu,Qxx,Qux,Quu);
        [Vx, Vxx] = Vterms (Qx,Qu,Qxx,Qux,Quu,Ktraj(:,:,i), ktraj(:,i));

    end
end

%==========================================================================
%   cost function
%==========================================================================
function J = cost(x,u,Q,R)

J = 0.5*x'*Q*x + 0.5*u'*R*u;

end

%==========================================================================
%   final cost function
%==========================================================================
function Jf = final_cost(x,u,xd,Qf)
Jf = 0.5*(x-xd)'*Qf*(x-xd);
end

%==========================================================================
%   final cost gradients (fill in)
%==========================================================================
function [gx, gu, gxx, gux, guu] = final_cost_gradients(x,u,xd,Qf)
    gx = Qf * (x - xd);
    gu = zeros(size(x,1), 1);
    gxx = Qf;
    gux = zeros(size(x,1), size(x,1));
    guu = zeros(size(x,1), size(x,1));
end

%==========================================================================
%   cost gradients (fill in)
%==========================================================================
function [gx, gu, gxx, gux, guu] = cost_gradients(x,u,xd,Q,R)
    gx = Q * x;
    gu = R * u;
    gxx = Q;
    gux = zeros(size(gu,1), size(gx,1));
    guu = R;
end

%==========================================================================
%   forward pass
%==========================================================================
function [xtraj, utraj, J] = forward_pass(x0, xtraj0, utraj0, ktraj, Ktraj, N, dt,Q,R,Qf,xd,J0, DYNAMICS)
    J = 1e7;
    alpha = 10;
    while(J0<J)
        t = 0;
        x = x0;
        J = 0;
        for i=1:N-1
            xtraj(:,i) = x;
            utraj(:,i) = utraj0(:,i) + alpha*ktraj(:,i)+Ktraj(:,:,i)*(x-xtraj0(:,i));% fill in input for forward pass
            J = J+cost(x,utraj(:,i),Q,R);%compute cost of current iteration
            xdot = DYNAMICS(x,utraj(:,i));%compute cartpole derivatives
            t = t + dt;%increment time step
            x=x+xdot*dt;%integrate state
        end
        xtraj(:,N) = x;
        J = J+final_cost(x,utraj(:,N-1),xd,Qf);%compute total cost    
        alpha = alpha/2;
    end
end


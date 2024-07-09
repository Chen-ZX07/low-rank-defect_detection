function [A_hat, E_hat, F_hat] = inexact_alm_ppnlreg_lam(D,lambda,bita,perior,tol, maxIter)
%%    minimize (inexactly, update A and E and F only once)

% addpath PROPACK;

[m, n] = size(D);

if nargin < 2
    lambda = 1 / sqrt(m);
end
if nargin < 3
    bita = 1e-7 / sqrt(m);
end
if nargin<4
    perior=ones(m,n);
end
if nargin < 5
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 6
    maxIter = 2000;
elseif maxIter == -1
    maxIter = 2000;
end

% initialize
p = 0.9;
Y1 = D;

norm_two = lansvd(Y1, 1, 'L');
norm_inf = norm( Y1(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y1 = Y1 / dual_norm;

J=zeros(m,n);
A_hat = zeros( m, n);
E_hat = zeros( m, n);
%% 改进 加入小噪声部分 F
F_hat = zeros( m, n);
Y2 = A_hat.*F_hat;
[~,S,~] = svd(A_hat,'econ');
diagS = diag(S);
Skk = diagS;
akk = A_hat;
ekk = E_hat;
fkk = F_hat;
%% 
mu = 3/norm_two ;% this one can be tuned
mu1 = 3/norm_two ;

mu_bar = mu * 1e7;
theta = 1.8;
% rho = 20;       % this one can be tuned
% rho1 = 100;      % this one can be tuned
d_norm = norm(D, 'fro');

Ak_hat = A_hat;
Ek_hat = zeros(m,n);
Fk_hat = F_hat;

tau = 0;
iter = 0;
total_svd = 0;
converged = false;

while ~converged       
    iter = iter + 1; 
    
        theta1 = sum(sum(lambda*(E_hat-Ek_hat)))/sum(sum((E_hat-Ek_hat)));
    if theta1 <= 0.5
        lambda = (1-tau)*lambda;
    elseif theta1 > 2
        lambda = (1+tau)*lambda;
    else
        lambda = lambda;
    end
    
    Ek_hat = E_hat;
    temp_T1 = D - A_hat -F_hat+ (1/mu)*Y1;
    Temp_T2 = Y2.*F_hat-mu1*tau*F_hat;
    E_hat = max((temp_T1-Temp_T2 -lambda*exp(-perior))./(mu+mu1*F_hat.*F_hat), 0);
    E_hat = E_hat+min((temp_T1 -Temp_T2 + lambda/mu*exp(-perior))./(mu+mu1*F_hat.*F_hat), 0);

    Fk_hat = F_hat;
    F_hat=(mu*(D-A_hat-E_hat)+Y1-Y2.*E_hat+mu1*tau*E_hat)./(bita+mu+mu1*E_hat.*E_hat);

    
    %% Update of Low-Rank matrix
    dey=D - E_hat- F_hat + (1/mu)*Y1;
    temp = dey*J';
    [U , ~, V] = svd(temp,'econ');
    Q = U*V';
    temp = Q'*dey;
    [U, S, V] = svd(temp,'econ');
    diagS = diag( S );

    diagS = sign( diagS ) .* max( abs( diagS ) - 1/mu,0);
%     diagS = max( abs( diagS ) - (p*akk.^(p-1))/mu,0);
    
    J = U * diag( diagS ) * V'; 
    Ak_hat = A_hat;
    A_hat=Q*J; 
    
    %%
    total_svd = total_svd + 1;
    
    Z1 = D - A_hat - E_hat-F_hat;   Zk1 = D-akk-ekk-fkk;
    Z2 = E_hat.*F_hat-tau;          Zk2 = ekk.*fkk-tau;
    Y1 = Y1 + mu*Z1;
    Y2 = Y2 + mu*Z2;
    
    g1 = sum(diagS)+lambda*norm(E_hat)+bita/2*norm(F_hat,'fro')+mu/2*norm(Z1,'fro');
    g2 = sum(Skk)+lambda*norm(ekk)+bita/2*norm(fkk,'fro')+mu/2*norm(Zk1,'fro');
    rho = theta*g1/g2;
    
    g3 = lambda*norm(E_hat)+bita/2*norm(F_hat,'fro')+mu1/2*norm(Z2,'fro');
    g4 = lambda*norm(ekk)+bita/2*norm(fkk,'fro')+mu1/2*norm(Zk2,'fro');
    rho1 = theta*g3/g4;
    
    mu = min(mu*rho, mu_bar);
    mu1 = min(mu1*rho1, mu_bar);
    
    Skk = diagS;
    akk = A_hat;
    ekk = E_hat;
    fkk = F_hat;
% stop Criterion    
    stopCriterion = norm(Z1, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
    
    if mod( total_svd, 10) == 0
        disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion ' num2str(stopCriterion)]);
    end    
    
    if ~converged && iter >= maxIter
        %disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end


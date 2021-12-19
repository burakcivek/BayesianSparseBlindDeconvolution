function [theta_chain,diagnostics] = sparse_BGS_solver(data,params)
    % Extracting data
    y = data.y;
    N = length(y);
    
    % General parameters
    num_of_MCMC_iteration = params.num_of_MCMC_iteration;
    
    % Pulse subsapce parameters
    H_subspace = params.H_subspace;
    [T,gamma_dim] = size(H_subspace);
    K = N-T;
    lambda_g = params.lambda_g;
    
    % Noise level parameters
    alpha_v = params.alpha_v;
    beta_v = params.beta_v;
    
    % Parameters of the prior distribution of latent variable's scale
    p0 = params.p0;
    lambda_x = params.lambda_x;
    
    % Parameter initializations
    s_init = binornd(1,1-p0,[K,1]);
    lambda_v_init = 1e-4;
    x_init = normrnd(0,sqrt(1/lambda_x),[K,1]).*s_init;
    g_init = normrnd(0,1./sqrt(lambda_g),gamma_dim,1);
    
    % Placeholder for Generated Samples
    s_chain = zeros(K,num_of_MCMC_iteration);
    lambda_v_chain = zeros(1,num_of_MCMC_iteration);
    x_chain = zeros(K,num_of_MCMC_iteration);
    gamma_chain = zeros(gamma_dim,num_of_MCMC_iteration);
    
    % Placeholders for General Diagnostics
    log_posterior = zeros(num_of_MCMC_iteration,1);
    tic
    for i = 1:num_of_MCMC_iteration
        if i == 1
            s = s_init;
            x = x_init;
            g = g_init;
            lambda_v = lambda_v_init;
        else
            s = s_chain(:,i-1);
            x = x_chain(:,i-1);
            g = gamma_chain(:,i-1);
            lambda_v = lambda_v_chain(:,i-1);
        end
        h = H_subspace*g;
        H = toeplitz([h;zeros(N-T,1)],[h(1),zeros(1,K-1)]);
        
        % Step 1. Sample s and x tuples
        for k = 1:K
            [s(k),x(k)] = sample_s_and_x(k,y,H,x,lambda_v,lambda_x,p0);
        end
        
        % Step 2. Sample pulse coefficients and propose time-shift
        if rand > 0.5
            shift = 1;
        else
            shift = -1;
        end
        x_p = circshift(x,shift,1);
        s_p = circshift(s,shift,1);
        
        X_p = toeplitz([x_p;zeros(T,1)],[x_p(1),zeros(1,T-1)]);
        A_p = X_p*H_subspace;
        U_p = chol(A_p'*A_p*lambda_v + lambda_g*eye(gamma_dim));
       
        X = toeplitz([x;zeros(T,1)],[x(1),zeros(1,T-1)]);
        A = X*H_subspace;
        U = chol(A'*A*lambda_v + lambda_g*eye(gamma_dim));
        
        B_p = U_p'\(A_p'*y*lambda_v);
        B = U'\(A'*y*lambda_v);
        
        alpha = B_p'*B_p - B'*B + 2*sum(log(diag(U)) - log(diag(U_p)));
        if 2*log(rand) < alpha
            x = x_p;
            s = s_p;
            X = X_p;
            g = U_p\(B_p + randn(gamma_dim,1));
        else
            g = U\(B + randn(gamma_dim,1));
        end
        
        % Step 4. Sample noise variance
        y_rec = X*H_subspace*g;
        lambda_v = sample_lambda_v(y,y_rec,alpha_v,beta_v);
        
        % Update chains
        s_chain(:,i) = s;
        x_chain(:,i) = x;
        gamma_chain(:,i) = g;
        lambda_v_chain(:,i) = lambda_v;

        % Calculate log-posterior
        log_likelihood = -0.5*(y-y_rec)'*(y-y_rec)*lambda_v + 0.5*length(y)*log(lambda_v);
%         log_prior_g = -0.5*(g'*g)*lambda_g;
%         log_prior_x_s = -0.5*(x'*x)*lambda_x + log(lambda_x)*sum(s);
%         log_prior_s = sum(s)*log(1-p0) + (K-sum(s))*log(p0);
%         log_prior_lambda_v = (alpha_v-1)*sum(log(lambda_v)) - sum(lambda_v)*beta_v;
%         log_posterior(i) = log_likelihood + log_prior_x_s + log_prior_s ...
%                            + log_prior_lambda_v + log_prior_g;
        log_posterior(i) = log_likelihood;
                       
%         if mod(i,100) == 0
%             fprintf('Iteration %d is completed in %.02f seconds\n',i,toc);
%         end
    end
    fprintf('Completed in %.02f seconds\n',toc);
    % Collect all samples
    theta_chain.x_chain = x_chain;
    theta_chain.gamma_chain = gamma_chain;
    theta_chain.lambda_v_chain = lambda_v_chain;
    theta_chain.s_chain = s_chain;
    
    % Collect diagnostics
    diagnostics.log_posterior = log_posterior;
end

function [lambda_v] = sample_lambda_v(y,y_rec,alpha_v,beta_v)    
    beta_v = 0.5*(y-y_rec)'*(y-y_rec) + beta_v;
    alpha_v = alpha_v + 0.5*length(y);
    lambda_v = gamrnd(alpha_v,1./beta_v);
end

function [s,x] = sample_s_and_x(idx,y,H,x,lambda_v,lambda_x,p0)
    x(idx) = 0;
    H_idx = H(:,idx);
    y_tilde = y - H*x;
    
    log_prob_s0 = log(p0);
    
    sigma_s = 1/(H_idx'*H_idx*lambda_v + lambda_x);
    mu_s = sigma_s*H_idx'*y_tilde*lambda_v;
    log_prob_s1 = 0.5*(log(sigma_s) + log(lambda_x)) + 0.5*mu_s^2/sigma_s + log(1-p0);
    
    J = exp(log_prob_s0 - log_prob_s1);
    alpha = 1/(1+J);
    if rand < alpha
        s = 1;
        x = normrnd(mu_s,sqrt(sigma_s));
    else
        s = 0;
        x = 0;
    end
end

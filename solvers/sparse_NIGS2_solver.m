function [theta_chain,diagnostics] = sparse_NIGS2_solver(data,params)
    % Extracting data
    y = data.y;
    N = length(y);
    
    % General parameters
    num_of_MCMC_iteration = params.num_of_MCMC_iteration;
    window_length = params.NIGS2_window_size;
    
    % Pulse subsapce parameters
    H_subspace = params.H_subspace;
    [T,gamma_dim] = size(H_subspace);
    K = N-T;
    lambda_g = params.lambda_g;
    
    % Parameter initializations
    beta_x_init = 1e-6;
    alpha_x_init = 0.1;
    lambda_v_init = 1e-4;
    lambda_x_init = gamrnd(alpha_x_init*ones(K,1),ones(K,1)./beta_x_init);
    x_init = mvnrnd(zeros(1,K),diag(1./lambda_x_init))';
    g_init = normrnd(0,1./sqrt(lambda_g),gamma_dim,1);
    
    % Placeholder for Generated Samples
    beta_x_chain = zeros(1,num_of_MCMC_iteration);
    alpha_x_chain = zeros(1,num_of_MCMC_iteration);
    lambda_v_chain = zeros(1,num_of_MCMC_iteration);
    lambda_x_chain = zeros(K,num_of_MCMC_iteration);
    x_chain = zeros(K,num_of_MCMC_iteration);
    gamma_chain = zeros(gamma_dim,num_of_MCMC_iteration);
    
    % Placeholders for General Diagnostics
    log_posterior = zeros(num_of_MCMC_iteration,1);
    log_likelihood = zeros(num_of_MCMC_iteration,1);
    log_priors = zeros(num_of_MCMC_iteration,1);
    tic
    for i = 1:num_of_MCMC_iteration
        if i == 1
            lambda_x = lambda_x_init;
            x = x_init;
            g = g_init;
            lambda_v = lambda_v_init;
            beta_x = beta_x_init;
            alpha_x = alpha_x_init;
        else
            lambda_x = lambda_x_chain(:,i-1);
            x = x_chain(:,i-1);
            g = gamma_chain(:,i-1);
            lambda_v = lambda_v_chain(:,i-1);
            beta_x = beta_x_chain(:,i-1);
            alpha_x = alpha_x_chain(:,i-1);
        end
        h = H_subspace*g;
        H = toeplitz([h;zeros(N-T,1)],[h(1),zeros(1,K-1)]);
        
        % Step 1. Sample shape parameter of the latent variables
        alpha_x = sample_alpha_x(lambda_x,beta_x,alpha_x);
        
        % Step 2. Sample scale parameter of the latent variables
        beta_x = sample_beta_x(lambda_x,alpha_x);
        
        % Step 3. Sample latent variable
        lambda_x = sample_lambda_x(x,alpha_x,beta_x);
        
        % Get current window
        idx_start = mod(i-1,K-window_length+1)+1;
        window_0 = [1:idx_start-1,idx_start+window_length:K];
        window_1 = idx_start:idx_start+window_length-1;
        
        % Step 4. Correction for sparse sequence blocks
        H0 = H(:,window_0);
        H1 = H(:,window_1);
        lambda_x_0 = lambda_x(window_0);
        lambda_x_1 = lambda_x(window_1);
        x(window_0) = sample_x_0(y,H0,H1,lambda_x_0,lambda_x_1,lambda_v);
        
        % Step 5. Correction for latent variable blocks
        y_tilde = y - H0*x(window_0);
        for k = 1:window_length
            func = @(parameter)get_log_posterior_lambda_x_1_uv(parameter,k,lambda_x_1,y_tilde,H1,lambda_v,alpha_x,beta_x);
            lambda_x_1(k) = self_slice_sampler_for_uv_case(lambda_x_1(k),func,1./beta_x);
        end
        lambda_x(window_1) = lambda_x_1;
        
        % Step 6. Sample sparse sequence
        U = chol(H1'*H1*lambda_v + diag(lambda_x_1));
        x(window_1) = U\(U'\(H1'*y_tilde*lambda_v) + randn(window_length,1));
        
        % Step 7. First step of sampling noise variance jointly with pulse coefficients
        A = toeplitz([x;zeros(T,1)],[x(1),zeros(1,T-1)])*H_subspace;
        lambda_v = sample_lambda_v_v2(lambda_v,y,A,lambda_g);
        
        % Step 8. Sample pulse coefficients and propose time-shift
        if rand > 0.5
            shift = 1;
        else
            shift = -1;
        end
        x_p = circshift(x,shift,1);
        lambda_x_p = circshift(lambda_x,shift,1);
        
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
            lambda_x = lambda_x_p;
            X = X_p;
            g = U_p\(B_p + randn(gamma_dim,1));
        else
            g = U\(B + randn(gamma_dim,1));
        end
          
        % Update chains
        x_chain(:,i) = x;
        gamma_chain(:,i) = g;
        lambda_x_chain(:,i) = lambda_x;
        lambda_v_chain(:,i) = lambda_v;
        beta_x_chain(:,i) = beta_x;
        alpha_x_chain(:,i) = alpha_x;
        
        % Calculate log-posterior
        y_rec = X*H_subspace*g;  
        log_likelihood(i) = -0.5*(y-y_rec)'*(y-y_rec)*lambda_v + 0.5*length(y)*log(lambda_v);
        log_prior_g = -0.5*(g'*g)*lambda_g + 0.5*length(g)*log(lambda_g);
        log_prior_x_lambda_x = -0.5*x'*diag(lambda_x)*x + 0.5*sum(log(lambda_x));
        log_prior_lambda_x = (alpha_x-1)*sum(log(lambda_x)) - sum(lambda_x)*beta_x + length(lambda_x)*(alpha_x*log(beta_x) - log(gamma(alpha_x)));
        log_prior_lambda_v = log(lambda_v);
        log_prior_beta_x = -log(beta_x);
        log_prior_alpha_x = -log(alpha_x);
        log_priors(i) = log_prior_x_lambda_x + log_prior_lambda_x + log_prior_alpha_x ...
                           + log_prior_lambda_v + log_prior_g + log_prior_beta_x;
        log_posterior(i) = log_likelihood(i) + log_priors(i);
        if mod(i,100) == 0
            fprintf('Iteration %d is completed in %.02f seconds\n',i,toc);
        end
    end
    fprintf('Completed in %.02f seconds\n',toc);
    
    % Collect all samples
    theta_chain.x_chain = x_chain;
    theta_chain.gamma_chain = gamma_chain;
    theta_chain.lambda_v_chain = lambda_v_chain;
    theta_chain.lambda_x_chain = lambda_x_chain;
    theta_chain.beta_x_chain = beta_x_chain;
    theta_chain.alpha_x_chain = alpha_x_chain;
    
    % Collect diagnostics
    diagnostics.log_posterior = log_posterior;
    diagnostics.log_likelihood = log_likelihood;
    diagnostics.log_priors = log_priors;
end

function [x_0] = sample_x_0(y,H0,H1,lambda_x_0,lambda_x_1,lambda_v)
    U = chol(H1'*H1*lambda_v + diag(lambda_x_1));
    B = U'\H1';
    A = eye(length(y)) - (B'*B)*lambda_v;
    U = chol(H0'*A*H0*lambda_v + diag(lambda_x_0));
    x_0 = U\(U'\(H0'*A'*y*lambda_v) + randn(length(lambda_x_0),1));
end

function [lambda_x] = sample_lambda_x(x,alpha_x,beta_x)
    beta_x = 0.5*x.^2 + beta_x;
    alpha_x = alpha_x + 0.5;
    lambda_x = gamrnd(alpha_x,1./beta_x);
end

function [log_posterior_lambda_x_1] = get_log_posterior_lambda_x_1_uv(param,idx,lambda_x_1,y_tilde,H1,lambda_v,alpha_x,beta_x)
    if param <= 0
        log_posterior_lambda_x_1 = -Inf;
    else
        lambda_x_1(idx) = param;
        U = chol(H1'*H1*lambda_v + diag(lambda_x_1));
        B = U'\(H1'*y_tilde*lambda_v);
        log_prior_lambda_x_1 = (alpha_x-1)*sum(log(lambda_x_1)) - sum(lambda_x_1)*beta_x;
        log_posterior_lambda_x_1 = -sum(log(diag(U))) + 0.5*sum(log(lambda_x_1)) + 0.5*(B'*B) + log_prior_lambda_x_1;       
    end
end

function [beta_x] = sample_beta_x(lambda_x,alpha_x)    
    beta = sum(lambda_x);
    alpha = alpha_x*length(lambda_x);
    beta_x = gamrnd(alpha,1/beta);
end

function [alpha_x] = sample_alpha_x(lambda_x,beta_x,alpha_x)
    func = @(x)get_log_posterior_alpha_x(x,lambda_x,beta_x);
    alpha_x = slicesample(alpha_x,1,'logpdf',func,'thin',1,'burnin',0,'width',1);
end

function [log_posterior] = get_log_posterior_alpha_x(alpha_x,lambda_x,beta_x)
    if alpha_x <= 0
        log_posterior = -Inf;
    else
        log_posterior = (length(lambda_x)*log(beta_x) + sum(log(lambda_x)))*alpha_x...
                        - length(lambda_x)*log(gamma(alpha_x)) - log(alpha_x);
    end
end

function [lambda_v] = sample_lambda_v_v2(lambda_v,y,A,lambda_g)    
    func = @(parameter)get_log_posterior_lambda_v(parameter,y,A,lambda_g);
    lambda_v = slicesample(lambda_v,1,'logpdf',func,'thin',1,'burnin',0,'width',1e4);
end

function [log_posterior] = get_log_posterior_lambda_v(lambda_v,y,A,lambda_g)
    if lambda_v <= 0
        log_posterior = -Inf;
    else
        U = chol(A'*A*lambda_v + eye(size(A,2))*lambda_g);
        B = U'\(A'*y*lambda_v);
        log_posterior = (0.5*length(y)-1)*log(lambda_v) - 0.5*(y'*y)*lambda_v + 0.5*(B'*B) ...
                        - sum(log(diag(U)));
    end
end
function [theta_chain,diagnostics] = sparse_BGS_Ktuple_solver(data,params)
    % Extracting data
    y = data.y;
    N = length(y);
    
    % General parameters
    num_of_MCMC_iteration = params.num_of_MCMC_iteration;
    K_tuple = params.K_tuple;
    
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
    
    candidates = dec2bin(0:2^K_tuple-1,K_tuple)-'0';
    R = cell(size(candidates,1)-1,1);
    a = zeros(size(candidates,1)-1,1);
    d = zeros(size(candidates,1)-1,1);
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
        for m = 1:size(candidates,1)-1
            H1 = H(:,find(candidates(m+1,:)));
            R{m} = chol(H1'*H1*lambda_v + lambda_x*eye(size(H1,2)));
            d(m) = -sum(log(diag(R{m})));
            a(m) = sum(candidates(m+1,:));
        end
        for k = 1:K-K_tuple+1
            window = k:k+K_tuple-1;
            x(window) = 0;
            y1 = y - H*x;
            H1 = H(:,window);
            [s(window),x(window)] = sample_s_and_x(candidates,y1,R,d,a,H1,lambda_v,lambda_x,p0);
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

function [s,x] = sample_s_and_x(candidates,y,U,d,a,H,lambda_v,lambda_x,p0)
    [num_of_cand,K_tuple] = size(candidates);
    log_probs = zeros(num_of_cand,1);
    log_probs(1) = log(p0)*K_tuple;
    B = cell(num_of_cand-1,1);
    for m = 1:num_of_cand-1
        H1 = H(:,candidates(m+1,:)==1);
        B{m} = U{m}'\(H1'*y*lambda_v);
        log_p_s = log(p0)*(K_tuple - a(m)) + log(1-p0)*a(m);
        log_probs(m+1) = 0.5*a(m)*log(lambda_x) + 0.5*(B{m}'*B{m}) + d(m) + log_p_s;
    end
    lin_probs = convert_to_lin_probs(log_probs);
    selection = discretesample(lin_probs,1);
    s = candidates(selection,:);
    
    x = zeros(K_tuple,1);
    if sum(s) ~= 0
        x(s==1) = U{selection-1}\(B{selection-1} + randn(a(selection-1),1));
    end
end

function [lin_probs] = convert_to_lin_probs(log_probs)
    lin_probs = 1./sum(exp(repmat(log_probs,1,length(log_probs))-log_probs'),1)';
    lin_probs(isnan(lin_probs)) = 0;
end

function x = discretesample(p, n)
    % Samples from a discrete distribution
    %
    %   x = discretesample(p, n)
    %       independently draws n samples (with replacement) from the 
    %       distribution specified by p, where p is a probability array 
    %       whose elements sum to 1.
    %
    %       Suppose the sample space comprises K distinct objects, then
    %       p should be an array with K elements. In the output, x(i) = k
    %       means that the k-th object is drawn at the i-th trial.
    %       
    %   Remarks
    %   -------
    %       - This function is mainly for efficient sampling in non-uniform 
    %         distribution, which can be either parametric or non-parametric.         
    %
    %       - The function is implemented based on histc, which has been 
    %         highly optimized by mathworks. The basic idea is to divide
    %         the range [0, 1] into K bins, with the length of each bin 
    %         proportional to the probability mass. And then, n values are
    %         drawn from a uniform distribution in [0, 1], and the bins that
    %         these values fall into are picked as results.
    %
    %       - This function can also be employed for continuous distribution
    %         in 1D/2D dimensional space, where the distribution can be
    %         effectively discretized.
    %
    %       - This function can also be useful for sampling from distributions
    %         which can be considered as weighted sum of "modes". 
    %         In this type of applications, you can first randomly choose 
    %         a mode, and then sample from that mode. The process of choosing
    %         a mode according to the weights can be accomplished with this
    %         function.
    %
    %   Examples
    %   --------
    %       % sample from a uniform distribution for K objects.
    %       p = ones(1, K) / K;
    %       x = discretesample(p, n);
    %
    %       % sample from a non-uniform distribution given by user
    %       x = discretesample([0.6 0.3 0.1], n);
    %
    %       % sample from a parametric discrete distribution with
    %       % probability mass function given by f.
    %       p = f(1:K);
    %       x = discretesample(p, n);
    %
    %   Created by Dahua Lin, On Oct 27, 2008
    %
    %% parse and verify input arguments
    assert(isfloat(p), 'discretesample:invalidarg', ...
        'p should be an array with floating-point value type.');
    assert(isnumeric(n) && isscalar(n) && n >= 0 && n == fix(n), ...
        'discretesample:invalidarg', ...
        'n should be a nonnegative integer scalar.');
    %% main
    % process p if necessary
    K = numel(p);
    if ~isequal(size(p), [1, K])
        p = reshape(p, [1, K]);
    end
    % construct the bins
    edges = [0, cumsum(p)];
    s = edges(end);
    if abs(s - 1) > eps
        edges = edges * (1 / s);
    end
    % draw bins
    rv = rand(1, n);
    c = histc(rv, edges);
    ce = c(end);
    c = c(1:end-1);
    c(end) = c(end) + ce;
    % extract samples
    xv = find(c);
    if numel(xv) == n  % each value is sampled at most once
        x = xv;
    else                % some values are sampled more than once
        xc = c(xv);
        d = zeros(1, n);
        dv = [xv(1), diff(xv)];
        dp = [1, 1 + cumsum(xc(1:end-1))];
        d(dp) = dv;
        x = cumsum(d);
    end
    % randomly permute the sample's order
    x = x(randperm(n));
end


function [h_chain,h_corrected,x_corrected,lambda_x_corrected,log_posterior] = solver_main(params,y,fhi)
    [N,L] = size(y);
    
    num_of_MCMC_iteration = params.num_of_MCMC_iteration;
    solver = params.solver;
    alpha_x = params.alpha_x;
    alpha_v = params.alpha_v;
    beta_x = params.beta_x;
    beta_v = params.beta_v;
    subspace_type = params.subspace;
    time_limit_type = params.time_limit;
    burn_in_ratio = params.burn_in_ratio;
    T = params.pulse_length;
    
    [H_subspace,cov_p] = pulseSubspace(subspace_type,time_limit_type,T,N,fhi);
    
    if solver == "NIGS1"
        [theta_chain,diagnostics] = sparse_NIG_solver(num_of_MCMC_iteration,y,H_subspace,alpha_x,beta_x,alpha_v,beta_v,cov_p);
    elseif solver == "NIGS2"
        K = params.NIGS2_tuple_size;
        [theta_chain,diagnostics] = sparse_NIG_solver_2(num_of_MCMC_iteration,y,H_subspace,alpha_x,beta_x,alpha_v,beta_v,cov_p,K);
    elseif solver == "BGS"
        p0 = 1 - params.BGS_success_probability;
        lambda_x = params.BGS_lambda_x;
        [theta_chain,diagnostics] = sparse_BG_solver(num_of_MCMC_iteration,y,H_subspace,p0,lambda_x,alpha_v,beta_v,cov_p);
    elseif solver == "NIGS1_complex_valued"
        F = exp(-1i*2*pi*(sub_band-1)'*(0:N-1)/N);
        [theta_chain,diagnostics] = sparse_NIG_solver_complex_valued(num_of_MCMC_iteration,y,H_subspace,F,alpha_x,beta_x,alpha_v,beta_v,cov_p);
    else
        fprintf('No such solver!\n')
    end
    
    x_chain = theta_chain.x_chain;
    h_chain = H_subspace*theta_chain.p_chain;
    lambda_x_chain = theta_chain.lambda_x_chain;
    log_posterior = diagnostics.log_posterior;
    
%     x_chain = theta_chain.x_chain;
%     p_chain = theta_chain.p_chain;
%     lambda_v_chain = theta_chain.lambda_v_chain;
%     log_posterior = diagnostics.log_posterior;
%     
    %p_samples = p_chain(:,num_of_MCMC_iteration-num_of_realizations+1:end);
    %x_samples = x_chain(:,:,num_of_MCMC_iteration-num_of_realizations+1:end);
    %lambda_v_samples = lambda_v_chain(:,num_of_MCMC_iteration-num_of_realizations+1:end);
    %log_posterior_samples = log_posterior(num_of_MCMC_iteration-num_of_realizations+1:end);

    num_of_realizations = round(num_of_MCMC_iteration*(1-burn_in_ratio));
    h_samples = h_chain(:,num_of_MCMC_iteration-num_of_realizations+1:end);
    x_samples = x_chain(:,:,num_of_MCMC_iteration-num_of_realizations+1:end);
    lambda_x_samples = lambda_x_chain(:,:,num_of_MCMC_iteration-num_of_realizations+1:end);
    log_posterior_samples = log_posterior(num_of_MCMC_iteration-num_of_realizations+1:end);
    
    [h_ref] = get_h_ref(h_samples,log_posterior_samples);
    [h_corrected,x_corrected,lambda_x_corrected] = correct_shift_and_scale(h_ref,h_samples,x_samples,lambda_x_samples);
    
%     h_chain = H_subspace*p_chain;
%     
%     [~,max_idx] = max(log_posterior);
%     h_ref = h_chain(:,max_idx);
%     
%     delay = -4:4;
%     if params.time_limit == "soft"
%         h_corrected = zeros(N,num_of_MCMC_iteration);
%     else
%         h_corrected = zeros(T,num_of_MCMC_iteration);
%     end
%     x_corrected = zeros(N,L,num_of_MCMC_iteration);
%     for i = 1:num_of_MCMC_iteration
%         h = h_chain(:,i);
%         x = x_chain(:,:,i);
%         error = [];
%         alpha = zeros(length(delay),1);
%         for j = 1:length(delay)
%             h_shifted = circshift(h,delay(j));
%             alpha(j) = (h_shifted'*h_ref)/(h_shifted'*h_shifted);
%             h_shifted_scaled = alpha(j)*h_shifted;
%             error = [error,sum((h_shifted_scaled - h_ref).^2)];
%         end
%         [~,min_idx] = min(error);
%         h_corrected(:,i) = circshift(h,delay(min_idx))*alpha(min_idx);
%         x_corrected(:,:,i) = circshift(x,-delay(min_idx),1)/alpha(min_idx);
%     end
%     
%     num_of_realizations = round(num_of_MCMC_iteration*(1-burn_in_ratio));
%     x_rec = mean(x_corrected(:,:,num_of_MCMC_iteration-num_of_realizations+1:end),3);
%     h_rec = mean(h_corrected(:,num_of_MCMC_iteration-num_of_realizations+1:end),2);
%     lambda_v_rec = mean(lambda_v_chain(:,num_of_MCMC_iteration-num_of_realizations+1:end));
end

function [h_ref] = get_h_ref(h_samples,log_posterior_samples)
    [T,num_of_realizations] = size(h_samples);
    delay_array = -round(T/2):round(T/2);
    h_est = h_samples(:,end);
    
    h_delay = zeros(1,num_of_realizations);
    for i = 1:num_of_realizations
        h = h_samples(:,i);
        error = zeros(length(delay_array),1);
        alpha = zeros(length(delay_array),1);
        for j = 1:length(delay_array)
            h_shifted = circshift(h,delay_array(j));
            alpha(j) = (h_shifted'*h_est)/(h_shifted'*h_shifted);
            h_shifted_scaled = alpha(j)*h_shifted;
            error(j) = sum(abs(h_shifted_scaled - h_est).^2);
        end
        [~,min_idx] = min(error);
        h_delay(i) = delay_array(min_idx);
    end
    [counts,bin_edges] = histcounts(h_delay);
    [~,max_idx] = max(counts);
    selected_delay = round(mean(bin_edges([max_idx,max_idx+1])));
    same_delay_idx = find(h_delay==selected_delay);
    [~,max_idx] = max(log_posterior_samples(same_delay_idx));
    h_ref = h_samples(:,same_delay_idx(max_idx));
end

function [h_corrected,x_corrected,lambda_x_corrected] = correct_shift_and_scale(h_ref,h_samples,x_samples,lambda_x_samples)
    num_of_iter = 20;
    [T,num_of_realizations] = size(h_samples);
    delay_array = -round(T/2):round(T/2);
    h_est = h_ref;
    
    h_est_array = zeros(size(h_samples,1),num_of_iter);
    x_est_array = zeros(size(x_samples,1),size(x_samples,2),num_of_iter);
    lambda_x_est_array = zeros(size(lambda_x_samples,1),size(lambda_x_samples,2),num_of_iter);
    for iter = 1:num_of_iter
        h_corrected = zeros(size(h_samples,1),num_of_realizations);
        x_corrected = zeros(size(x_samples,1),size(x_samples,2),num_of_realizations);
        lambda_x_corrected = zeros(size(lambda_x_samples,1),size(lambda_x_samples,2),num_of_realizations);
        for i = 1:num_of_realizations
            h = h_samples(:,i);
            x = x_samples(:,:,i);
            lambda_x = lambda_x_samples(:,:,i);
            error = zeros(length(delay_array),1);
            alpha = zeros(length(delay_array),1);
            for j = 1:length(delay_array)
                h_shifted = circshift(h,delay_array(j));
                alpha(j) = (h_shifted'*h_est)/(h_shifted'*h_shifted);
                h_shifted_scaled = alpha(j)*h_shifted;
                error(j) = sum(abs(h_shifted_scaled - h_est).^2);
            end
            [~,min_idx] = min(error);
            h_corrected(:,i) = circshift(h,delay_array(min_idx))*alpha(min_idx);
            x_corrected(:,:,i) = circshift(x,-delay_array(min_idx),1)/alpha(min_idx);
            lambda_x_corrected(:,:,i) = circshift(lambda_x,-delay_array(min_idx),1)*(alpha(min_idx)^2);
        end
        h_est = mean(h_corrected,2);
        x_est = mean(x_corrected,3);
        lambda_x_est = mean(lambda_x_corrected,3);
        
        h_est_array(:,iter) = h_est;
        x_est_array(:,:,iter) = x_est;
        lambda_x_est_array(:,:,iter) = lambda_x_est;
    end
end
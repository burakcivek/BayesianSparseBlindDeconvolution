function [samples,diagnostics] = run_MCMC(data,params)
    if params.solver == "NIGS1"
        [theta_chain,diagnostics] = sparse_NIGS1_solver(data,params);
    elseif params.solver == "NIGS2"
        [theta_chain,diagnostics] = sparse_NIGS2_solver(data,params);
    elseif params.solver == "BGS"
        [theta_chain,diagnostics] = sparse_BGS_solver(data,params);
    elseif params.solver == "BGS_Tuple"
        [theta_chain,diagnostics] = sparse_BGS_Ktuple_solver(data,params);
    else
        fprintf('No such solver!\n');
    end
    
    H_subspace = params.H_subspace;
    % Extract original samples
    x_chain = theta_chain.x_chain;
    h_chain = H_subspace*theta_chain.gamma_chain;
    lambda_v_chain = theta_chain.lambda_v_chain;
    if params.solver == "NIGS1" || params.solver == "NIGS2"
        lambda_x_chain = theta_chain.lambda_x_chain;
        beta_x_chain = theta_chain.beta_x_chain;
        alpha_x_chain = theta_chain.alpha_x_chain;
    else
        s_chain = theta_chain.s_chain;
    end
    log_posterior = diagnostics.log_posterior;

    % Calculate the reference pulse shape
    [h_ref] = get_h_ref(h_chain,log_posterior);
    
    % Correct shift-scale ambiguities based on reference pulse shape
    if params.solver == "NIGS1" || params.solver == "NIGS2"
        [h_corrected,x_corrected,lambda_x_corrected] = correct_shift_and_scale_NIGS(h_ref,h_chain,x_chain,lambda_x_chain);
    else
        [h_corrected,x_corrected,s_corrected] = correct_shift_and_scale_BGS(h_ref,h_chain,x_chain,s_chain);
    end
    
    % Collect corrected samples
    samples.h = h_corrected;
    samples.x = x_corrected;
    samples.lambda_v = lambda_v_chain;
    if params.solver == "NIGS1" || params.solver == "NIGS2"
        samples.lambda_x = lambda_x_corrected;
        samples.beta_x = beta_x_chain;
        samples.alpha_x = alpha_x_chain;
    else
        samples.s = s_corrected;
    end
end

function [h_ref] = get_h_ref(h_samples,log_posterior_samples)
    % This function calculates the reference pulse shape that achieves the 
    % highest posterior value. This will be used to correct different time
    % shift and scale configurations.
    
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

function [h_corrected,x_corrected,lambda_x_corrected] = correct_shift_and_scale_NIGS(h_ref,h_samples,x_samples,lambda_x_samples)
    num_of_iter = 20;
    [T,num_of_realizations] = size(h_samples);
    delay_array = -round(T/2):round(T/2);
    h_est = h_ref;
    
    h_est_array = zeros(size(h_samples,1),num_of_iter);
    x_est_array = zeros(size(x_samples,1),num_of_iter);
    lambda_x_est_array = zeros(size(lambda_x_samples,1),num_of_iter);
    for iter = 1:num_of_iter
        h_corrected = zeros(size(h_samples,1),num_of_realizations);
        x_corrected = zeros(size(x_samples,1),num_of_realizations);
        lambda_x_corrected = zeros(size(lambda_x_samples,1),num_of_realizations);
        for i = 1:num_of_realizations
            h = h_samples(:,i);
            x = x_samples(:,i);
            lambda_x = lambda_x_samples(:,i);
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
            x_corrected(:,i) = circshift(x,-delay_array(min_idx))/alpha(min_idx);
            lambda_x_corrected(:,i) = circshift(lambda_x,-delay_array(min_idx))*(alpha(min_idx)^2);
        end
        h_est = mean(h_corrected,2);
        x_est = mean(x_corrected,2);
        lambda_x_est = mean(lambda_x_corrected,2);
        
        h_est_array(:,iter) = h_est;
        x_est_array(:,iter) = x_est;
        lambda_x_est_array(:,iter) = lambda_x_est;
    end
end

function [h_corrected,x_corrected,s_corrected] = correct_shift_and_scale_BGS(h_ref,h_samples,x_samples,s_samples)
    num_of_iter = 20;
    [T,num_of_realizations] = size(h_samples);
    delay_array = -round(T/2):round(T/2);
    h_est = h_ref;
    
    h_est_array = zeros(size(h_samples,1),num_of_iter);
    x_est_array = zeros(size(x_samples,1),num_of_iter);
    s_est_array = zeros(size(s_samples,1),num_of_iter);
    for iter = 1:num_of_iter
        h_corrected = zeros(size(h_samples,1),num_of_realizations);
        x_corrected = zeros(size(x_samples,1),num_of_realizations);
        s_corrected = zeros(size(s_samples,1),num_of_realizations);
        for i = 1:num_of_realizations
            h = h_samples(:,i);
            x = x_samples(:,i);
            s = s_samples(:,i);
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
            x_corrected(:,i) = circshift(x,-delay_array(min_idx))/alpha(min_idx);
            s_corrected(:,i) = circshift(s,-delay_array(min_idx));
        end
        h_est = mean(h_corrected,2);
        x_est = mean(x_corrected,2);
        s_est = mean(s_corrected,2);
        
        h_est_array(:,iter) = h_est;
        x_est_array(:,iter) = x_est;
        s_est_array(:,iter) = s_est;
    end
end
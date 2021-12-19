function [params] = get_params(data)
    %------------------- Pulse Shape Model Parameters ---------------------
    % Set pulse_length to the length of true pulse shape, if known,
    % otherwise, set it to an arbitrary constant
    params.lambda_g = 10;
    if isfield(data,'h_true')
        params.pulse_length = length(data.h_true);
    else
        params.pulse_length = 23;
    end
    % Set upper limit for pulse bandwidth to an arbitrary constant, if it
    % is not provided within the data
    if isfield(data,'f_max')
        params.f_max = data.f_max;
    else
        params.f_max = 16e9;
    end
    % Construct subspace matrix using either dps sequences, or set it as
    % identity, if it is not provided within the data
    params.subspace_type = "identity"; % dpss | identity
    if isfield(data,'H_subspace')
        params.H_subspace = data.H_subspace;
        params.coeff_order = size(params.H_subspace,2); 
    else
        params.H_subspace = get_pulse_subspace(params);
        params.coeff_order = size(params.H_subspace,2); 
    end
    
    % General sampler parameters
    params.num_of_MCMC_iteration = 10000;
    params.burn_in_ratio = 0.75;
    params.solver = "NIGS2";
    
    % NIGS2 parameters
    params.NIGS2_window_size = 10;
    
    % BGS parameters
    params.p0 = 0.9;
    params.lambda_x = 1;
    
    % BGS_Tuple parameters
    params.K_tuple = 3;
end

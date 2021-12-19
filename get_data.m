function [data] = get_data(file_name)
    load(file_name);
    
    %--------------- Required Fields ----------------------
    % y      : measurement in time domain

    %--------------- Optional Fields ----------------------
    % x_true     : true sparse sequence in time domain  
    % h_true     : true pulse sequence in time domain
    % f_max      : upper limit for bandwidth of the pulse
    % H_subspace : subspace matric for pulse shape
    
    data.y = y_25dB;
    data.x_true = x;
    data.h_true = h;
    %data.f_max = fhi;
    %data.H_subspace = H_subspace;
end


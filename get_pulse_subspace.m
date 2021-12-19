function [H_subspace] = get_pulse_subspace(params)
    T = params.pulse_length;
    if params.subspace_type == "dpss"
        fs = params.fs;
        f_max = params.f_max;
        [dps_seq,~] = dpss(T,T*f_max/fs,floor(2*T*f_max/fs - 1));
        B = [eye(T-1);-1*ones(1,T-1)];
        M = null([B,dps_seq]);
        H_subspace = B*M(1:T-1,:);
    elseif params.subspace_type == "identity"
        H_subspace = eye(T);
    else
        fprintf('Incorrect key!');
    end
end


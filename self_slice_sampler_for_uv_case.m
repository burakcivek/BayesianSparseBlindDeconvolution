function [xp] = self_slice_sampler_for_uv_case(initial,logpdf,width)
    maxiter = 200;
    e = exprnd(1,1,1); % needed for the vertical position of the slice.

    RW = rand(1,1); % factors of randomizing the width
    RD = rand(1,1); % uniformly draw the point within the slice
    x0 = initial; % current value

    inside = @(x,th) (logpdf(x) > th);

    z = logpdf(x0) - e;

    r = width.*RW; % random width/stepsize
    xl = x0 - r; 
    xr = xl + width; 
    
    iter = 0;
    % step out to the left.
    while inside(xl,z) && iter<maxiter
        xl = xl - width;
        iter = iter +1;
    end
    % step out to the right
    iter = 0;  
    while (inside(xr,z)) && iter<maxiter
        xr = xr + width;
        iter = iter+1;        
    end

    xp = RD.*(xr-xl) + xl;

    iter = 0;  
    while(~inside(xp,z))&& iter<maxiter 
        rshrink = (xp>x0);
        xr(rshrink) = xp(rshrink);
        lshrink = ~rshrink;
        xl(lshrink) = xp(lshrink);
        xp = rand(1,1).*(xr-xl) + xl; % draw again
        iter = iter+1;
    end
end


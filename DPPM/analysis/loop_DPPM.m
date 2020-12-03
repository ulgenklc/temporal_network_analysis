function out = loop_DPPM(TA)
    for i= 1:length(TA)/9
        [vertices, communities] = DPPM(TA((i-1)*9+1),TA((i-1)*9+2),TA((i-1)*9+3),TA((i-1)*9+4),TA((i-1)*9+5),TA((i-1)*9+6),TA((i-1)*9+7),TA((i-1)*9+8),TA((i-1)*9+9));
        save(sprintf('comms_%d.mat',i), 'communities')
        save(sprintf('vertices_%d.mat',i), 'vertices')
    end
end


    
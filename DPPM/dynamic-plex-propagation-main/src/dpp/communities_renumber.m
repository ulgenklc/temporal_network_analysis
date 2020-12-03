function communities = communities_renumber(communities)
%COMP_CLEAN Renumber components sequentially.

vs = sort(unique(communities));
for k = 1:length(vs)
    communities(communities == vs(k)) = k;
end

end


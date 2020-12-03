function [comms,vertices] = DPPM(t0,t1,t2,t3,t4,t5,t6,t7,t8, index)
    matrix = cat(3,t0,t1,t2,t3,t4,t5,t6,t7,t8);
    [comms_4_1,vertices_4_1] = dpp(matrix,4,1);
    [comms_4_2,vertices_4_2] = dpp(matrix,4,2);
    [comms_4_3,vertices_4_3] = dpp(matrix,4,3);
    [comms_5_1,vertices_5_1] = dpp(matrix,5,1);
    [comms_5_2,vertices_5_2] = dpp(matrix,5,2);
    [comms_5_3,vertices_5_3] = dpp(matrix,5,3);
    [comms_5_4,vertices_5_4] = dpp(matrix,5,4);
    comms = {comms_4_1,comms_4_2,comms_4_3,comms_5_1,comms_5_2,comms_5_3,comms_5_4};
    vertices = {vertices_4_1,vertices_4_2,vertices_4_3,vertices_5_1,vertices_5_2,vertices_5_3,vertices_5_4};
end
    
        
    
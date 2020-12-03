function themat = generate_circmat(xy)
%GENERATE_CIRCMAT 
% Copyright Wes Viles 2013

[fir, ~, thir] = inbet(xy);
four = dmat(xy);
themat = bending(fir, four, thir, xy);

function [y,z,r] = inbet(xy)
%INBET
%       xy
%       An n x 2 matrix, where each row are the two-dimensional coordinates
%       of a vertex in space. Used for plotting.

    n = size(xy, 1); % number of vertices
    y = zeros(n);
    z = zeros(n);
    for i = 1:(n-1)
        for j = (i+1):n
            vals = [1:n];
            vals([i,j]) = [];
            for k = vals
                [yt,zt] = onaline(xy(k,1),xy(k,2),xy(i,1),xy(i,2),xy(j,1),xy(j,2));
                y(i,j) = y(i,j) + yt;
                z(i,j) = z(i,j) + zt;
            end
        end
    end
    y = y + y';
    z = z + z';
    r = zeros(n);
    for i = 1:(n-1)
        for j = (i+1):n
            if z(i,j) > 0
                r(i,j) = y(i,j)/z(i,j);
            end
        end
    end
    r = r + r';
end

function y = dmat(xy,h)
    if nargin < 2
        h = 1;
    end
    n = size(xy,1);
    y = 1./eye(n)-eye(n);
    for i = 1:(n-1)
        for j = (i+1):n
            vals = [1:n];
            vals([i,j]) = [];
            minx = min(xy(i,1),xy(j,1));
            maxx = max(xy(i,1),xy(j,1));
            miny = min(xy(i,2),xy(j,2));
            maxy = max(xy(i,2),xy(j,2));
            for k = vals
                if (xy(k,1) >= minx && xy(k,1) <= maxx) || (xy(k,2) >= miny && xy(k,2) <= maxy)
                    temp = point2line(xy(k,1),xy(k,2),xy(i,1),xy(i,2),xy(j,1),xy(j,2));
                    if temp > 10^(-h)
                        y(i,j) = min(y(i,j),temp);
                        y(j,i) = y(i,j);
                    end
                end
            end
        end
    end
end

function y = bending(bet,dist,rat,xy)
    n = size(xy,1);
    y = cell(n);
    for i = 1:(n-1)
        for j = (i+1):n
            if bet(i,j) > 0
                [xp,yp,r,ang,arcang] = comp(bet(i,j),dist(i,j),rat(i,j),xy(i,1),xy(i,2),xy(j,1),xy(j,2));
                y{i,j} = [xp,yp,r,ang,arcang];
            end
        end
    end
end

function y = point2line(xt,yt,x0,y0,x1,y1)
    base = sqrt((x1-x0)^2+(y1-y0)^2);
    side1 = sqrt((xt-x0)^2+(yt-y0)^2);
    side2 = sqrt((xt-x1)^2+(yt-y1)^2);
    s = (base+side1+side2)/2;
    y = 2*real(sqrt(s*(s-base)*(s-side1)*(s-side2)))/base;
end

function [x,y,r,ang,arcang] = comp(bet,dist,rat,x1,y1,x3,y3,param)
    if nargin < 8
        param = .8;
    end
    dist = ((-1)^rem(bet,2))*dist*param;
    if y1 == y3
        x2 = (x1+x3)/2;
        y2 = y1+dist*rat;
        ma = (y2-y1)/(x2-x1);
        mb = (y3-y2)/(x3-x2);
        x = (ma*mb*(y1-y3)+mb*(x1+x2)-ma*(x2+x3))/(2*(mb-ma));
        y = -(1/ma)*(x-(x1+x2)/2)+(y1+y2)/2;
        r = real(sqrt((x1-x)^2+(y1-y)^2));
        if y1 > y
            if x1 > x3
                ang = atan(abs(y1-y)/abs(x1-x));
                arcang = pi-2*atan(abs(y1-y)/abs(x1-x));
            else
                ang = atan(abs(y3-y)/abs(x3-x));
                arcang = pi-2*atan(abs(y3-y)/abs(x3-x));
            end
        else
            if x1 > x3
                ang = pi+atan(abs(y3-y)/abs(x3-x));
                arcang = pi-2*atan(abs(y3-y)/abs(x3-x));
            else
                ang = pi+atan(abs(y1-y)/abs(x1-x));
                arcang = pi-2*atan(abs(y1-y)/abs(x1-x));
            end
        end
    elseif x1 == x3
        x2 = x1+dist*rat;
        y2 = (y1+y3)/2;
        ma = (y2-y1)/(x2-x1);
        mb = (y3-y2)/(x3-x2);
        x = (ma*mb*(y1-y3)+mb*(x1+x2)-ma*(x2+x3))/(2*(mb-ma));
        y = -(1/ma)*(x-(x1+x2)/2)+(y1+y2)/2;
        r = real(sqrt((x1-x)^2+(y1-y)^2));
        if x1 > x
            if y1 > y3
                ang = -atan(abs(y3-y)/abs(x3-x));
                arcang = 2*atan(abs(y1-y)/abs(x1-x));
            else
                ang = -atan(abs(y1-y)/abs(x1-x));
                arcang = 2*atan(abs(y3-y)/abs(x3-x));
            end
        else
            if y1 > y3
                ang = pi-atan(abs(y1-y)/abs(x1-x));
                arcang = 2*atan(abs(y1-y)/abs(x1-x));
            else
                ang = pi-atan(abs(y3-y)/abs(x3-x));
                arcang = 2*atan(abs(y3-y)/abs(x3-x));
            end
        end
    else
        m = (y1-y3)/(x1-x3);
        x2 = (x1+x3)/2+((-1)^rem(bet,2))*sqrt((rat*dist^2)/(1/m^2+1));
        y2 = (y1+y3)/2-(1/m)*((-1)^rem(bet,2))*sqrt((rat*dist^2)/(1/m^2+1));
        ma = (y2-y1)/(x2-x1);
        mb = (y3-y2)/(x3-x2);
        x = (ma*mb*(y1-y3)+mb*(x1+x2)-ma*(x2+x3))/(2*(mb-ma));
        y = -(1/ma)*(x-(x1+x2)/2)+(y1+y2)/2;
        r = real(sqrt((x1-x)^2+(y1-y)^2));
        xlow = min(x1,x3);
        xhigh = max(x1,x3);
        ylow = min(y1,y3);
        yhigh = max(y1,y3);
        ang = 0;
        arcang = 0;
        if x <= xlow && y <= ylow
            if x1 > x3
                ang = atan((y1-y)/(x1-x));
                arcang = atan((y3-y)/(x3-x))-ang;
            else
                ang = atan((y3-y)/(x3-x));
                arcang = atan((y1-y)/(x1-x))-ang;
            end
        elseif x <= xlow && y >= ylow && y <= yhigh
            if y3 > y1
                ang = atan((y1-y)/(x1-x));
                arcang = atan((y3-y)/(x3-x))-ang;
            else
                ang = atan((y3-y)/(x3-x));
                arcang = atan((y1-y)/(x1-x))-ang;
            end
        elseif x <= xlow && y >= yhigh
            if x1 > x3
                ang = atan((y3-y)/(x3-x));
                arcang = abs(ang)-abs(atan((y1-y)/(x1-x)));
            else
                ang = atan((y1-y)/(x1-x));
                arcang = abs(ang)-abs(atan((y3-y)/(x3-x)));
            end
        elseif x >= xlow && x <= xhigh && y <= ylow
            if x1 > x3
                ang = atan((y1-y)/(x1-x));
                arcang = pi-atan(abs((y3-y)/(x3-x)))-ang;
            else
                ang = atan((y3-y)/(x3-x));
                arcang = pi-atan(abs((y1-y)/(x1-x)))-ang;
            end
        elseif x >= xlow && x <= xhigh && y >= ylow && y <= yhigh
            if m > 0
                if x1 > x3
                    ang = atan((y1-y)/(x1-x));
                    arcang = pi+atan(abs((y3-y)/(x3-x)))-ang;
                else
                    ang = atan((y3-y)/(x3-x));
                    arcang = pi+atan(abs((y1-y)/(x1-x)))-ang;
                end
            else
                if x1 > x3
                    ang = atan((y-y1)/(x-x1));
                    arcang = pi-atan(abs((y3-y)/(x3-x)))-ang;
                else
                    ang = atan((y-y3)/(x-x3));
                    arcang = pi-atan(abs((y1-y)/(x1-x)))-ang;
                end
            end
        elseif x >= xlow && x <= xhigh && y >= yhigh
            if x1 > x3
                ang = pi+atan(abs((y3-y)/(x3-x)));
                arcang = pi-atan(abs((y3-y)/(x3-x)))-atan(abs((y1-y)/(x1-x)));
            else
                ang = pi+atan(abs((y1-y)/(x1-x)));
                arcang = pi-atan(abs((y3-y)/(x3-x)))-atan(abs((y1-y)/(x1-x)));
            end
        elseif x >= xhigh && y <= ylow
            if x1 > x3
                ang = pi-atan(abs((y1-y)/(x1-x)));
                arcang = atan(abs((y1-y)/(x1-x)))-atan(abs((y3-y)/(x3-x)));
            else
                ang = pi-atan(abs((y3-y)/(x3-x)));
                arcang = atan(abs((y3-y)/(x3-x)))-atan(abs((y1-y)/(x1-x)));
            end
        elseif x >= xhigh && y >= ylow && y <= yhigh
            if y1 > y3
                ang = pi-atan(abs((y1-y)/(x1-x)));
                arcang = atan(abs((y1-y)/(x1-x)))+atan(abs((y3-y)/(x3-x)));
            else
                ang = pi-atan(abs((y3-y)/(x3-x)));
                arcang = atan(abs((y3-y)/(x3-x)))+atan(abs((y1-y)/(x1-x)));
            end
        elseif x >= xhigh && y >= yhigh
            if x1 > x3
                ang = pi+atan(abs((y3-y)/(x3-x)));
                arcang = atan(abs((y1-y)/(x1-x)))-atan(abs((y3-y)/(x3-x)));
            else
                ang = pi+atan(abs((y1-y)/(x1-x)));
                arcang = atan(abs((y3-y)/(x3-x)))-atan(abs((y1-y)/(x1-x)));
            end
        end
    end
end

function [y,z] = onaline(xt,yt,x0,y0,x1,y1,k)
    if nargin < 7
        k = 1;
    end
    m = (y1-y0)/(x1-x0);
    minx = min(x0,x1);
    miny = min(y0,y1);
    maxx = max(x0,x1);
    maxy = max(y0,y1);
    if abs(x1-x0) < 10^(-k) && abs(xt-x1) < 10^(-k)
        z = 1;
        if xt >= minx && xt <= maxx && yt >= miny && yt <= maxy
            y = 1;
        else
            y = 0;
        end
    elseif abs((yt-y0)-m*(xt-x0)) < 10^(-k)
        z = 1;
        if xt >= minx && xt <= maxx && yt >= miny && yt <= maxy
            y = 1;
        else
            y = 0;
        end
    else
        z = 0;
        y = 0;
    end
end

end


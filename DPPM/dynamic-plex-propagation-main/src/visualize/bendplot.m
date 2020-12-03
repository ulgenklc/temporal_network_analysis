function [] = bendplot(a, comm, circmat, xy, labels)
% Copyright Wes Viles 2013

    p = size(a,1);
    plot(xy(1,1),xy(1,2),'.','Color',[0,0,0],'MarkerSize',15);
    hold on
    for h = 1:p
        plot(xy(h,1),xy(h,2),'.','Color',[0,0,0],'MarkerSize',15);
        if exist('labels', 'var') && ~isempty(labels)
            text(xy(h,1)-0.2,xy(h,2)+0.15,labels{h});
        else
            text(xy(h,1)-0.2,xy(h,2)+0.15,num2str(h));
        end
    end
    hold off
    ylim([min(xy(:,2))-1,max(xy(:,2))+1]); 
    xlim([min(xy(:,1))-1,max(xy(:,1))+1]);
    cmap = colormap(jet);
    colormapping = spacer();
    hold on
    for i = 1:(p-1)
        for j = (i+1):p
            if a(i,j) == 1
                if isempty(circmat{i,j})
                    plot([xy(i,1),xy(j,1)],[xy(i,2),xy(j,2)],'Color',[0,0,0]);
                else
                    t = 0:.0001:circmat{i,j}(5);
                    xp = circmat{i,j}(3)*cos(t+circmat{i,j}(4))+circmat{i,j}(1);
                    yp = circmat{i,j}(3)*sin(t+circmat{i,j}(4))+circmat{i,j}(2);
                    plot(xp,yp,'Color',[0,0,0]);
                end
                if length(comm{i,j}) == 1
                    if isempty(circmat{i,j})
                        c = 1 + mod(comm{i, j} - 1, length(colormapping));
                        plot([xy(i,1),xy(j,1)],[xy(i,2),xy(j,2)],'Color',cmap(colormapping(c),:),'LineWidth',2);
                    else
                        t = 0:.0001:circmat{i,j}(5);
                        xp = circmat{i,j}(3)*cos(t+circmat{i,j}(4))+circmat{i,j}(1);
                        yp = circmat{i,j}(3)*sin(t+circmat{i,j}(4))+circmat{i,j}(2);
                        c = 1 + mod(comm{i, j} - 1, length(colormapping));
                        plot(xp,yp,'Color',cmap(colormapping(c),:),'LineWidth',2);
                    end
                elseif length(comm{i,j}) > 1
                    if isempty(circmat{i,j})
                        plot([xy(i,1),xy(j,1)],[xy(i,2),xy(j,2)],'Color',[0,0,0],'LineWidth',4);
                    else
                        t = 0:.0001:circmat{i,j}(5);
                        xp = circmat{i,j}(3)*cos(t+circmat{i,j}(4))+circmat{i,j}(1);
                        yp = circmat{i,j}(3)*sin(t+circmat{i,j}(4))+circmat{i,j}(2);
                        plot(xp,yp,'Color',[0,0,0],'LineWidth',4);
                    end
                end
            end
        end
    end
    hold off
    
    function y = spacer()
        y = zeros(21,1);
        y(1) = 1;
        y(2) = 64;
        y(3) = 32;
        y(4) = 16;
        y(5) = 48;
        y(6) = 8;
        y(7) = 40;
        y(8) = 24;
        y(9) = 56;
        y(10) = 4;
        y(11) = 36;
        y(12) = 20;
        y(13) = 52;
        y(14) = 12;
        y(15) = 44;
        y(16) = 28;
        y(17) = 60;
        y(18) = 2;
        y(19) = 34;
        y(20) = 18;
        y(21) = 50;
    end
end
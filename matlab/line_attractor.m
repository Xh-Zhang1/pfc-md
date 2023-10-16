close all;
% light to dark blue colormap
n=256; % number of colors
cmap = [linspace(.9,0,n)', linspace(.9447,.447,n)', linspace(.9741,.741,n)']; % light to dark blue colormap
cmap_inverted = flip(cmap);


load('arrdata.mat')
V=arr([20:50],:);
A=arr([70:100],:);
figure;
hold on;
plot3(0,0,0,'r.','markersize',30);
%plot3(V(:,1),V(:,2),V(:,3),'x','markersize',15)
hold on
xyz = double(V');
fnplt(cscvn(xyz(:,[1:end])),'r',2)
hold off

C = linspace(1,10,length(V));
hold on;
scatter3(V(:,1),V(:,2),V(:,3),100,C,'x')
colorbar


%hold on;plot3(A(:,1),A(:,2),A(:,3),'x','markersize',15)
hold on
xyz = double(A');
fnplt(cscvn(xyz(:,[1:end])),'b',2)
hold off


hold on;
%scatter3(A(:,1),A(:,2),A(:,3),100,C,'filled')
scatter3(A(:,1),A(:,2),A(:,3),100,C,'x')
colorbar


view(42,39);
set(gca,'fontsize',20);
set(gca,'linewidth',1);
box on

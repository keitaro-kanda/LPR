clear all;
close all;

[mesh2dFilename,mesh2dFilePath]=uigetfile({'*.2B;*.2A;*.2C'}); 
fid=fopen(mesh2dFilename);

sum2B = [];  % 保存2B通道数据

[ntrace,DAT] = readlpr02E(mesh2dFilePath,mesh2dFilename); 
sum2B = [sum2B DAT.data];

[nsample ntrace] = size(sum2B);

dt = 0.3125;  % 第二通道的时间步长(Time step of channel 2)
% dt = 2.5; % 第一通道的时间步长(Time step of channel 1)
t = 1:nsample;
t = t*dt;
x = 1:ntrace;

sum2B_gain=sum2B;

for i=1:ntrace
    sum2B_gain(:,i)=sum2B(:,i).*((1:nsample)');
end

figure
set(gca,'fontsize',64);
imagesc(x, t ,sum2B);
caxis([-300 500]);
colormap(gray);
ylabel('Time （ns）');
xlabel('Trace');
title('CE4 CH2B Data');


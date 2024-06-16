clear all; % 初期化
close all;

[mesh2dFilename,mesh2dFilePath]=uigetfile({'*.2B;*.2A;*.2C'}); % ファイル選択
fid=fopen(mesh2dFilename); % ファイルオープン

sum2B = [];  % ����2Bͨ������ 2B channelのデータを保存するための空配列？

[ntrace,DAT] = readlpr02E(mesh2dFilePath,mesh2dFilename); % データ読み込み，関数はreadlpr02E.mファイルで定義されている
sum2B = [sum2B DAT.data]; % データをsum2Bに追加

[nsample ntrace] = size(sum2B); % データサイズ取得

dt = 0.3125;  % �ڶ�ͨ����ʱ�䲽��(Time step of channel 2)
% dt = 2.5; % ��һͨ����ʱ�䲽��(Time step of channel 1)
t = 1:nsample; % タイムステップ数
t = t*dt; % 時間軸
x = 1:ntrace; % トレース番号

sum2B_gain=sum2B; % 各トレースにゲインを適用

for i=1:ntrace
    sum2B_gain(:,i)=sum2B(:,i).*((1:nsample)');
end

figure % プロット
set(gca,'fontsize',64);
imagesc(x, t ,sum2B);
caxis([-300 500]);
colormap(gray);
ylabel('Time ��ns��');
xlabel('Trace');
title('CE4 CH2B Data');


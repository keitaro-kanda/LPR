function [ntr,DAT] = readlpr02E(pname,fname)
%
% READLPR02 : Imports LPR 2A/2B/2C Level data.
%
%   Usage : [ntrace,DATA] = readlpr02(pname,fname);
%
% RETURNS : The trace number ntrace and a structure DATA containing the 
%           data and all other necessary parameters
%
%
%  Author : Zhou Bin
%           Institute of Electronics, Chinese Academy of Science

%% Explanation of Key Parameters:
% 
% tag: Data trace header identifier
% tstamp_sec: Time code in seconds since the base time 2010-01-01T00:00:00.000
% tstamp_msec: Time code in milliseconds
% tstr: Converted timestamp to string format
% velocity: Velocity in floating point
% position_x, position_y, position_z: X, Y, Z coordinates in floating point
% pose_u, pose_v, pose_w: Pitch, roll, yaw components in floating point
% position_xref, position_yref, position_zref: Reference positions (X, Y, Z) in floating point
% pose_uref, pose_vref, pose_wref: Reference pitch, roll, yaw components in floating point
% data_block_num: Data block count
% rs_*: Various remote sensing telemetry parameters (e.g., voltage, current, temperature)
% para_*: Various radar parameters (e.g., work mode, gain mode, time window, PRF)
% FPGA_status: FPGA status byte
% channel: Antenna channel identifier
% nbyte_trace: Effective data length of the trace in bytes
% tracenum1, tracenum2: First and second channel trace counts
% nsamp: Number of samples in the data
% data: Scientific data of the trace
% qs: Data quality status of the trace



%% ファイルオープンとエラーチェック
fid = fopen([pname,fname],'r');
if fid == -1
    disp([fname,' does not exist!'])
    ntr = 0; DAT = {};
    return;
end

%% チャンネルとデータレベルのチェック
if ~isempty(strfind(fname,'LPR-1'))
    ch = 1;
elseif ~isempty(strfind(fname,'LPR-2'))
    ch = 2;
else
    ch = 0;
    disp([fname,' is not a correct LPR data file name'])
    ntr = 0; DAT = {};
    return;
end
if strcmp(fname(end-1:end),'2A') || strcmp(fname(end-1:end),'2a')
    dlevel = 210;
elseif strcmp(fname(end-1:end),'2B') || strcmp(fname(end-1:end),'2b')
    dlevel = 211;
elseif strcmp(fname(end-1:end),'2C') || strcmp(fname(end-1:end),'2c')
    dlevel = 212;
else
    dlevel = -1;
    disp([fname,' is not a correct 2 level LPR data file name!'])
    ntr = 0; DAT = {};
    return;
end

%% データ読み込み
%% 特定のバイトパターンに基づいてトレースの開始位置を特性する．
dchar = fread(fid,'uint8');
if ch == 1
    a1 = find(dchar == hex2dec('14'));
    a1 = a1(find(a1 < (length(dchar) - 4)));
    a2 = a1(find(dchar(a1 + 1) == hex2dec('6F')));
    a3 = a2(find(dchar(a2 + 2) == hex2dec('11')));
    trace_bpos = a3(find(dchar(a3 + 3) == hex2dec('11')));
else
    a1 = find(dchar == hex2dec('14'));
    a1 = a1(find(a1 < (length(dchar) - 4)));
    a2 = a1(find(dchar(a1 + 1) == hex2dec('6F')));
    a3 = a2(find(dchar(a2 + 2) == hex2dec('22')));
    trace_bpos = a3(find(dchar(a3 + 3) == hex2dec('22')));
end

%% もしトレースの開始位置が特例された場合，その位置までのデータをテキストファイルに書き出す．
if trace_bpos > 1
    foid = fopen([pname,fname,'.txt'],'w');
    fprintf(foid,'%s',dchar(1:trace_bpos-1));
    fclose(foid);
end
clear dchar;

%% トレースデータの読み込み
fseek(fid, 0, 'bof');
n = length(trace_bpos);
fprintf(1, '%s\n', ['Reading data from ''' fname ' ''...']);
nc = 0;
for k = 1:n
    if k == floor(k / 10) * 10
        fprintf(1, repmat('\b', 1, nc));
        nc = fprintf(1, '%s', [num2str(k) '/' num2str(n)]);
    end
    fseek(fid, trace_bpos(k) - 1, 'bof');
    tag = dec2hex(fread(fid, 1, 'uint32', 'b')); % Data trace header identifier
    DAT(k).tag = tag;
    DAT(k).tstamp_sec = fread(fid, 1, 'uint32'); % Time code in seconds since 2010-01-01T00:00:00.000
    DAT(k).tstamp_msec = fread(fid, 1, 'uint16'); % Time code in milliseconds
    DAT(k).tstr = datestr((DAT(k).tstamp_sec + DAT(k).tstamp_msec / 1000) / 3600 / 24 + datenum(2010,1,1,0,0,0), 'yyyy-mm-dd HH:MM:SS.FFF');
    DAT(k).velocity = fread(fid, 1, 'single'); % Velocity in floating point
    DAT(k).position_x = fread(fid, 1, 'single'); % X-coordinate position in floating point
    DAT(k).position_y = fread(fid, 1, 'single'); % Y-coordinate position in floating point
    DAT(k).position_z = fread(fid, 1, 'single'); % Z-coordinate position in floating point
    DAT(k).pose_u = fread(fid, 1, 'single'); % Pitch component in floating point
    DAT(k).pose_v = fread(fid, 1, 'single'); % Roll component in floating point
    DAT(k).pose_w = fread(fid, 1, 'single'); % Yaw component in floating point
    
    if dlevel == 211 || dlevel == 212
        DAT(k).position_xref = fread(fid, 1, 'single'); % Reference X-coordinate position in floating point
        DAT(k).position_yref = fread(fid, 1, 'single'); % Reference Y-coordinate position in floating point
        DAT(k).position_zref = fread(fid, 1, 'single'); % Reference Z-coordinate position in floating point
        DAT(k).pose_uref = fread(fid, 1, 'single'); % Reference pitch component in floating point
        DAT(k).pose_vref = fread(fid, 1, 'single'); % Reference roll component in floating point
        DAT(k).pose_wref = fread(fid, 1, 'single'); % Reference yaw component in floating point
    end
    
    DAT(k).data_block_num = fread(fid, 1, 'uint16'); % Data block count
    DAT(k).rs_5V = fread(fid, 1, 'uint8'); % Remote sensing +5V telemetry
    DAT(k).rs_3V3 = fread(fid, 1, 'uint8'); % Remote sensing +3.3V telemetry
    DAT(k).rs_12V = fread(fid, 1, 'uint8'); % Remote sensing +-12V telemetry
    DAT(k).rs_I = fread(fid, 1, 'uint8'); % Remote sensing total current telemetry
    DAT(k).rs_payload = fread(fid, 1, 'uint8'); % Payload multiplexing identifier, 0x01 for main control box, 0x09 for main control box + lunar radar
    DAT(k).rs_HV1 = fread(fid, 1, 'uint8'); % High voltage 1 telemetry
    DAT(k).rs_HV2 = fread(fid, 1, 'uint8'); % High voltage 2 telemetry
    DAT(k).rs_PRF1 = fread(fid, 1, 'uint8'); % PRF1 telemetry
    DAT(k).rs_PRF2 = fread(fid, 1, 'uint8'); % PRF2 telemetry
    
    DAT(k).para_workmode = dec2hex(fread(fid, 1, 'uint8')); % Radar work mode
    DAT(k).para_gainmode1 = dec2hex(fread(fid, 1, 'uint8')); % Radar first channel gain mode
    DAT(k).para_gain1 = fread(fid, 1, 'uint8'); % Radar first channel attenuation value
    DAT(k).para_gainmode2 = dec2hex(fread(fid, 1, 'uint8')); % Radar second channel gain mode
    DAT(k).para_gain2A = fread(fid, 1, 'uint8'); % Radar 2A channel attenuation value
    DAT(k).para_gain2B = fread(fid, 1, 'uint8'); % Radar 2B channel attenuation value
    DAT(k).para_timewindow1 = dec2hex(fread(fid, 1, 'uint8')); % Radar first channel time window
    DAT(k).para_timedelay1 = fread(fid, 1, 'uint8'); % Radar first channel time delay value
    DAT(k).para_timewindow2 = dec2hex(fread(fid, 1, 'uint8')); % Radar second channel time window
    DAT(k).para_timedelay2 = fread(fid, 1, 'uint8'); % Radar second channel time delay value
    DAT(k).para_prf1 = dec2hex(fread(fid, 1, 'uint8')); % Radar first channel PRF
    DAT(k).para_nsum1 = fread(fid, 1, 'uint8'); % Radar first channel accumulation times, 0 means no accumulation
    DAT(k).para_prf2 = dec2hex(fread(fid, 1, 'uint8')); % Radar second channel PRF
    DAT(k).para_nsum2 = fread(fid, 1, 'uint8'); % Radar second channel accumulation times, 0 means no accumulation
    DAT(k).para_tracenum1 = fread(fid, 1, 'uint16'); % Radar first channel trace count
    DAT(k).para_tracenum2 = fread(fid, 1, 'uint16'); % Radar second channel trace count
    DAT(k).para_prf_status = dec2hex(fread(fid, 1, 'uint8')); % Radar PRF selection status
    DAT(k).para_rxpower_en_status = dec2hex(fread(fid, 1, 'uint8')); % Radar receiver power enable status
    DAT(k).FPGA_status = dec2hex(fread(fid, 1, 'uint8')); % FPGA status byte
    DAT(k).para_reserved = dec2hex(fread(fid, 1, 'uint32', 'b')); % 4 reserved bytes
    
    DAT(k).rs_Vrx = fread(fid, 1, 'uint8'); % Radar receiver voltage telemetry
    DAT(k).rs_Vctrl = fread(fid, 1, 'uint8'); % Radar controller voltage telemetry
    DAT(k).rs_Itx = fread(fid, 1, 'uint8'); % Radar transmitter current telemetry
    DAT(k).rs_Tps = fread(fid, 1, 'uint8'); % Main control box power unit temperature telemetry
    DAT(k).rs_5Vref = fread(fid, 1, 'uint8'); % Main control box +5V reference voltage telemetry
    DAT(k).rs_Ttx1 = fread(fid, 1, 'uint8'); % Radar transmitter 1 temperature telemetry
    DAT(k).rs_Ttx2 = fread(fid, 1, 'uint8'); % Radar transmitter 2 temperature telemetry
    DAT(k).rs_Trx1 = fread(fid, 1, 'uint8'); % Radar receiver 1 temperature telemetry
    DAT(k).rs_Trx2 = fread(fid, 1, 'uint8'); % Radar receiver 2 temperature telemetry
    
    DAT(k).nbyte_trace = fread(fid, 1, 'uint16'); % Effective data length of the trace (in bytes)
    DAT(k).tracenum1 = fread(fid, 1, 'uint16'); % First channel trace count
    DAT(k).tracenum2 = fread(fid, 1, 'uint16'); % Second channel trace count
    
    channel = dec2hex(fread(fid, 1, 'uint8')); % Antenna channel identifier
    DAT(k).channel = channel;
    if strcmp(channel, '11')
        DAT(k).nsamp = DAT(k).nbyte_trace / 4;
        DAT(k).data = fread(fid, 8192, 'single'); % Scientific data of the trace
    elseif strcmp(channel, '2A')
        DAT(k).nsamp = DAT(k).nbyte_trace / 8;
        DAT(k).data = fread(fid, 2048, 'single'); % Scientific data of the trace
    elseif strcmp(channel, '2B')
        DAT(k).nsamp = DAT(k).nbyte_trace / 8;
        DAT(k).data = fread(fid, 2048, 'single'); % Scientific data of the trace
    end
    
    DAT(k).qs = fread(fid, 1, 'uint8'); % Data quality status of the trace
end

fprintf(1, repmat('\b', 1, nc));
nc = fprintf(1, '%s', [num2str(k) '/' num2str(n)]);
fprintf('\n');
ntr = k;

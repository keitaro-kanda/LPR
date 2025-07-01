#!/usr/bin/env python3
"""
岩石ラベルファイルの地形補正変換ツール
JSONファイル内の時刻情報を地形補正前後で相互変換する
"""

import json
import os
import numpy as np
from tqdm import tqdm

def load_position_profile(position_profile_path):
    """
    位置プロファイルデータを読み込む
    
    Parameters:
    -----------
    position_profile_path : str
        位置プロファイルファイルのパス
    
    Returns:
    --------
    z_profile : np.ndarray
        地形高度プロファイル [m]
    """
    if not os.path.exists(position_profile_path):
        raise FileNotFoundError(f"位置プロファイルファイルが見つかりません: {position_profile_path}")
    
    try:
        position_data = np.loadtxt(position_profile_path, delimiter=' ', skiprows=1)
        z_profile = position_data[:, 2]  # 3列目が高度データ
        print(f"位置プロファイル読み込み完了: {len(z_profile)} 点")
        print(f"高度範囲: {z_profile.min():.3f} - {z_profile.max():.3f} m")
        return z_profile
    except Exception as e:
        raise ValueError(f"位置プロファイルファイルの読み込みエラー: {e}")

def calculate_terrain_correction_params(z_profile):
    """
    地形補正パラメータを計算
    
    Parameters:
    -----------
    z_profile : np.ndarray
        地形高度プロファイル [m]
    
    Returns:
    --------
    dict: 地形補正パラメータ
    """
    # run_data_processing.pyと同じパラメータ
    sample_interval = 0.312500e-9  # [s]
    c = 299792458  # [m/s]
    
    z_max = np.max(z_profile)
    z_min = np.min(z_profile)
    
    t_expand_min = int(np.abs(z_max) / c / sample_interval)
    t_expand_max = int(np.abs(z_min) / c / sample_interval)
    
    params = {
        'sample_interval': sample_interval,
        'c': c,
        'z_max': z_max,
        'z_min': z_min,
        't_expand_min': t_expand_min,
        't_expand_max': t_expand_max,
        'time_min': -t_expand_min * sample_interval,
        'time_max_offset': t_expand_max * sample_interval
    }
    
    print(f"地形補正パラメータ:")
    print(f"  z_max: {z_max:.3f} m, z_min: {z_min:.3f} m")
    print(f"  t_expand_min: {t_expand_min}, t_expand_max: {t_expand_max}")
    print(f"  time_min: {params['time_min']*1e9:.2f} ns")
    
    return params

def convert_time_terrain_to_original(y_terrain, x_position, z_profile, params):
    """
    地形補正後の時刻を地形補正前の時刻に変換
    
    Parameters:
    -----------
    y_terrain : float
        地形補正後の時刻 [ns]
    x_position : float
        水平位置 [m]
    z_profile : np.ndarray
        地形高度プロファイル [m]
    params : dict
        地形補正パラメータ
    
    Returns:
    --------
    float: 地形補正前の時刻 [ns]
    """
    # run_data_processing.pyのロジックを逆算
    trace_interval = 3.6e-2  # [m]
    trace_index = int(x_position / trace_interval)
    
    if trace_index >= len(z_profile):
        trace_index = len(z_profile) - 1
    elif trace_index < 0:
        trace_index = 0
    
    # 地形補正の逆変換
    equivalent_time = z_profile[trace_index] / params['c']  # [s]
    start_index = int(equivalent_time / params['sample_interval'])
    start_row = params['t_expand_min'] - start_index
    
    # 地形補正後の時刻をインデックスに変換
    y_terrain_s = y_terrain * 1e-9  # ns -> s
    terrain_index = int((y_terrain_s - params['time_min']) / params['sample_interval'])
    
    # 元のデータでのインデックス
    original_index = terrain_index - start_row
    
    # 元の時刻に変換
    y_original_s = original_index * params['sample_interval']
    y_original_ns = y_original_s * 1e9
    
    return y_original_ns

def convert_time_original_to_terrain(y_original, x_position, z_profile, params):
    """
    地形補正前の時刻を地形補正後の時刻に変換
    
    Parameters:
    -----------
    y_original : float
        地形補正前の時刻 [ns]
    x_position : float
        水平位置 [m]
    z_profile : np.ndarray
        地形高度プロファイル [m]
    params : dict
        地形補正パラメータ
    
    Returns:
    --------
    float: 地形補正後の時刻 [ns]
    """
    # run_data_processing.pyのロジックを適用
    trace_interval = 3.6e-2  # [m]
    trace_index = int(x_position / trace_interval)
    
    if trace_index >= len(z_profile):
        trace_index = len(z_profile) - 1
    elif trace_index < 0:
        trace_index = 0
    
    # 地形補正の適用
    equivalent_time = z_profile[trace_index] / params['c']  # [s]
    start_index = int(equivalent_time / params['sample_interval'])
    start_row = params['t_expand_min'] - start_index
    
    # 元の時刻をインデックスに変換
    y_original_s = y_original * 1e-9  # ns -> s
    original_index = int(y_original_s / params['sample_interval'])
    
    # 地形補正後のインデックス
    terrain_index = original_index + start_row
    
    # 地形補正後の時刻に変換
    y_terrain_s = params['time_min'] + terrain_index * params['sample_interval']
    y_terrain_ns = y_terrain_s * 1e9
    
    return y_terrain_ns

def convert_labels_file(input_path, output_path, z_profile, params, conversion_mode):
    """
    ラベルファイルの時刻情報を変換
    
    Parameters:
    -----------
    input_path : str
        入力JSONファイルのパス
    output_path : str
        出力JSONファイルのパス
    z_profile : np.ndarray
        地形高度プロファイル
    params : dict
        地形補正パラメータ
    conversion_mode : str
        変換モード ('terrain_to_original' または 'original_to_terrain')
    """
    print(f"\nファイル変換中: {os.path.basename(input_path)}")
    print(f"変換モード: {conversion_mode}")
    
    # JSONファイル読み込み
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'results' not in data:
        raise ValueError("JSONファイルに'results'キーが見つかりません")
    
    results = data['results']
    converted_count = 0
    
    # 各ラベルの時刻を変換
    for key, label_data in tqdm(results.items(), desc="ラベル変換中"):
        try:
            x_pos = label_data['x']
            y_time = label_data['y']
            
            if conversion_mode == 'terrain_to_original':
                # 地形補正後 → 地形補正前
                y_converted = convert_time_terrain_to_original(y_time, x_pos, z_profile, params)
            elif conversion_mode == 'original_to_terrain':
                # 地形補正前 → 地形補正後
                y_converted = convert_time_original_to_terrain(y_time, x_pos, z_profile, params)
            else:
                raise ValueError(f"無効な変換モード: {conversion_mode}")
            
            # 変換後の値を設定
            label_data['y'] = y_converted
            converted_count += 1
            
        except Exception as e:
            print(f"警告: ラベル {key} の変換に失敗: {e}")
            continue
    
    # 変換結果を保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"変換完了: {converted_count} 個のラベルを変換")
    print(f"出力ファイル: {output_path}")

def main():
    """
    メイン処理
    """
    print("岩石ラベルファイル地形補正変換ツール")
    print("=" * 60)
    
    # 変換モード選択
    print("\n変換モードを選択してください:")
    print("1: 地形補正後 → 地形補正前")
    print("2: 地形補正前 → 地形補正後")
    mode_choice = input("選択 (1-2): ").strip()
    
    mode_map = {
        '1': 'terrain_to_original',
        '2': 'original_to_terrain'
    }
    
    if mode_choice not in mode_map:
        print("エラー: 無効な選択です。1-2の数字を入力してください。")
        return
    
    conversion_mode = mode_map[mode_choice]
    print(f"選択された変換モード: {conversion_mode}")
    
    # 位置プロファイルファイルのパス
    default_position_path = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/position_plot/total_position.txt"
    print(f"\n位置プロファイルファイルのパス:")
    print(f"デフォルト: {default_position_path}")
    position_path = input("パスを入力 (Enterでデフォルト使用): ").strip()
    
    if not position_path:
        position_path = default_position_path
    
    # 位置プロファイル読み込み
    try:
        z_profile = load_position_profile(position_path)
        params = calculate_terrain_correction_params(z_profile)
    except Exception as e:
        print(f"エラー: {e}")
        return
    
    # 入力ファイル/ディレクトリ選択
    print("\n入力を選択してください:")
    print("1: 単一JSONファイル")
    print("2: ディレクトリ内の全JSONファイル")
    input_choice = input("選択 (1-2): ").strip()
    
    if input_choice == '1':
        # 単一ファイル処理
        input_path = input("入力JSONファイルのパス: ").strip()
        if not os.path.exists(input_path):
            print("エラー: ファイルが見つかりません")
            return
        
        # 出力ファイル名生成
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        if conversion_mode == 'terrain_to_original':
            suffix = '_original'
        else:
            suffix = '_terrain_corrected'
        
        output_path = os.path.join(base_dir, f"{base_name}{suffix}.json")
        
        try:
            convert_labels_file(input_path, output_path, z_profile, params, conversion_mode)
        except Exception as e:
            print(f"エラー: {e}")
            return
    
    elif input_choice == '2':
        # ディレクトリ処理
        input_dir = input("入力ディレクトリのパス: ").strip()
        if not os.path.isdir(input_dir):
            print("エラー: ディレクトリが見つかりません")
            return
        
        # JSONファイルを検索
        json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
        if not json_files:
            print("エラー: JSONファイルが見つかりません")
            return
        
        print(f"\n見つかったJSONファイル: {len(json_files)} 個")
        
        # 出力ディレクトリ
        if conversion_mode == 'terrain_to_original':
            suffix = '_original'
        else:
            suffix = '_terrain_corrected'
        
        output_dir = os.path.join(input_dir, f"converted{suffix}")
        
        # 各ファイルを処理
        success_count = 0
        for json_file in sorted(json_files):
            input_path = os.path.join(input_dir, json_file)
            output_path = os.path.join(output_dir, json_file)
            
            try:
                convert_labels_file(input_path, output_path, z_profile, params, conversion_mode)
                success_count += 1
            except Exception as e:
                print(f"エラー: ファイル {json_file} の処理に失敗: {e}")
                continue
        
        print(f"\n処理完了: {success_count}/{len(json_files)} ファイルが正常に変換されました")
        print(f"出力ディレクトリ: {output_dir}")
    
    else:
        print("エラー: 無効な選択です")
        return
    
    print("\n変換処理が完了しました！")

if __name__ == "__main__":
    main()
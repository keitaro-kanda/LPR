import numpy as np
import matplotlib.pyplot as plt
import os


# RSFDと年齢データ
# 形式：（中央年代 (Ma), 年代誤差 (Ma), べき指数, in-situ or remote, reference）
data_exponent = [
    (1, 0, 5.0, "remote", "Aussel et al., 2025, (R)"), # 年代は<1 Maとの記述
    (1, 0, 5.0, "remote", "Aussel et al., 2025, (R)"), # 年代は<1 Maとの記述
    (1, 0, 4.6, "remote", "Aussel et al., 2025, (R)"), # 年代は<1 Maとの記述
    (1, 0, 4.8, "remote", "Aussel et al., 2025, (R)"), # 年代は<1 Maとの記述
    (1, 0, 4.5, "remote", "Aussel et al., 2025, (R)"), # 年代は<1 Maとの記述
    (1, 0, 4.9, "remote", "Aussel et al., 2025, (R)"), # 年代は<1 Maとの記述
    (2750, 250, 1.779, "in-situ", "Cintala & Macbride, 1995, (I)"),
    (2500, 250, 2.505, "remote", "Cintala & Macbride, 1995, (R)"),
    (3150, 50, 2.559, "in-situ", "Cintala & Macbride, 1995, (I)"),
    (3850, 50, 5.653, "remote", "Cintala & Macbride, 1995, (R)"),
    (3150, 50, 2.286, "in-situ", "Cintala & Macbride, 1995, (I)"), # 年代はIIIと同じくらい、との記述
    (3150, 50, 6.021, "remote", "Cintala & Macbride, 1995, (R)"), # 年代はIIIと同じくらい、との記述
    (500, 500, 1.802, "in-situ", "Cintala & Macbride, 1995, (I)"), # 年代は100 Myrオーダーとの記述
    (500, 500, 4.031, "remote", "Cintala & Macbride, 1995, (R)"), # 年代は100 Myrオーダーとの記述
    (3, 0, 2.76, "remote", "Krishna & Kumar, 2016, (R)"),
    (53.5, 26.5, 2.629, "remote", "Li et al., 2017, (R)"),
    (53.5, 26.5, 1.86, "in-situ", "Li et al., 2017, (I)"),
    (2750, 250, 2.73, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照
    (2750, 250, 2.23, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照
    (3150, 50, 3.51, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照
    (3150, 50, 2.58, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照
    (3150, 50, 3.68, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照
    (3150, 50, 2.77, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照
    (500, 500, 2.44, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照
    (500, 500, 2.53, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照
    (53.5, 26.5, 5.46, "remote", "Li et al., 2018, (R)"),
    (2750, 250, 2.11, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照
    (3150, 50, 2.56, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照
    (3150, 50, 2.51, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照
    (500, 500, 1.82, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照
    (10, 0, 4.03, "remote", "Pajola et al., 2019, (R)"),
    (2, 0, 5.3, "remote", "Watkins et al., 2019, (R)"),
    (26, 0, 5.6, "remote", "Watkins et al., 2019, (R)"),
    (50, 0, 3.8, "remote", "Watkins et al., 2019, (R)"),
    (80, 0, 4.4, "remote", "Watkins et al., 2019, (R)"),
    (105, 0, 6.8, "remote", "Watkins et al., 2019, (R)"),
    (200, 0, 4.7, "remote", "Watkins et al., 2019, (R)"),
]

# /km^2に統一
data_const = [
    (2750, 250, 79.7, "in-situ", "Cintala & Macbride, 1995, (I)"), # /100 m^2を/km^2に変換
    (2500, 250, 5.0, "remote", "Cintala & Macbride, 1995, (R)"), # /100 m^2を/km^2に変換
    (3150, 50, 4.9, "in-situ", "Cintala & Macbride, 1995, (I)"), # /100 m^2を/km^2に変換
    (3850, 50, 115.4, "remote", "Cintala & Macbride, 1995, (R)"), # /100 m^2を/km^2に変換
    (3150, 50, 15.4, "in-situ", "Cintala & Macbride, 1995, (I)"), # 年代はIIIと同じくらい、との記述, 100 m^2を/km^2に変換
    (3150, 50, 19.2, "remote", "Cintala & Macbride, 1995, (R)"), # 年代はIIIと同じくらい、との記述, 100 m^2を/km^2に変換
    (500, 500, 266.6, "in-situ", "Cintala & Macbride, 1995, (I)"), # 年代は100 Myrオーダーとの記述, 100 m^2を/km^2に変換
    (500, 500, 1826.1, "remote", "Cintala & Macbride, 1995, (R)"), # 年代は100 Myrオーダーとの記述, 100 m^2を/km^2に変換
    (53.5, 26.5, 0.00528e6, "remote", "Li et al., 2017, (R)"), # /m^2を/km^2に変換
    (53.5, 26.5, 0.00275e6, "in-situ", "Li et al., 2017, (I)"), # /m^2を/km^2に変換
    (2750, 250, 0.0015e6, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (2750, 250, 0.00114e6, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (3150, 50, 0.962e6, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (3150, 50, 0.00060e6, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (3150, 50, 0.00027e6, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (3150, 50, 0.00016e6, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (500, 500, 0.00324e6, "remote", "Li et al., 2017, (R)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (500, 500, 0.00311e6, "in-situ", "Li et al., 2017, (I)"), # Cintala & Macbride, 1995の年代を参照, /m^2を/km^2に変換
    (53.5, 26.5, 5.09, "remote", "Li et al., 2018, (R)"),
    (2750, 250, 5e8, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照, /100 m^2を/km^2に変換
    (3150, 50, 3.3e8, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照, /100 m^2を/km^2に変換
    (3150, 50, 1.25e8, "in-situ", "Shoemaker & Morris, 1969, (I)"), # Cintala & Macbride, 1995の年代を参照, /100 m^2を/km^2に変換
    (500, 500, 7.9e7, "in-situ", "Shoemaker & Morris, 1969, (I)") # Cintala & Macbride, 1995の年代を参照, /100 m^2を/km^2に変換
]

# Calculate correlation coefficient
# Calculate in linear scale
ages = np.array([d[0] for d in data_exponent])
rsfd_values = np.array([d[2] for d in data_exponent])
correlation_matrix = np.corrcoef(ages, rsfd_values)
correlation_coefficient = correlation_matrix[0, 1]
print(f"Correlation coefficient (linear scale): {correlation_coefficient}")
# Calculate in log scale
log_ages = np.log10(ages + 1e-6)  # avoid
correlation_matrix_log = np.corrcoef(log_ages, rsfd_values)
correlation_coefficient_log = correlation_matrix_log[0, 1]
print(f"Correlation coefficient (log scale): {correlation_coefficient_log}")
# Calculate correlation only for remote data in log scale
remote_ages = np.array([d[0] for d in data_exponent if d[3] == "remote"])
remote_rsfd_values = np.array([d[2] for d in data_exponent if d[3] == "remote"])
log_remote_ages = np.log10(remote_ages + 1e-6)
correlation_matrix_remote_log = np.corrcoef(log_remote_ages, remote_rsfd_values)
correlation_coefficient_remote_log = correlation_matrix_remote_log[0, 1]
print(f"Correlation coefficient for remote data only (log scale): {correlation_coefficient_remote_log}")
# Calculate correlation only for in-situ data_exponent in log scale
in_situ_ages = np.array([d[0] for d in data_exponent if d[3] == "in-situ"])
in_situ_rsfd_values = np.array([d[2] for d in data_exponent if d[3] == "in-situ"])
log_in_situ_ages = np.log10(in_situ_ages + 1e-6)
correlation_matrix_in_situ_log = np.corrcoef(log_in_situ_ages, in_situ_rsfd_values)
correlation_coefficient_in_situ_log = correlation_matrix_in_situ_log[0, 1]
print(f"Correlation coefficient for in-situ data only (log scale): {correlation_coefficient_in_situ_log}")

# Calculate correlation coefficient for data_const
print("\n=== Correlation coefficients for constant term (/km^2) ===")
# Calculate in linear scale
ages_const = np.array([d[0] for d in data_const])
const_values = np.array([d[2] for d in data_const])
correlation_matrix_const = np.corrcoef(ages_const, const_values)
correlation_coefficient_const = correlation_matrix_const[0, 1]
print(f"Correlation coefficient (linear scale): {correlation_coefficient_const}")
# Calculate in log-log scale
log_ages_const = np.log10(ages_const + 1e-6)
log_const_values = np.log10(const_values + 1e-6)
correlation_matrix_const_log = np.corrcoef(log_ages_const, log_const_values)
correlation_coefficient_const_log = correlation_matrix_const_log[0, 1]
print(f"Correlation coefficient (log-log scale): {correlation_coefficient_const_log}")
# Calculate correlation only for remote data in log-log scale
remote_ages_const = np.array([d[0] for d in data_const if d[3] == "remote"])
remote_const_values = np.array([d[2] for d in data_const if d[3] == "remote"])
log_remote_ages_const = np.log10(remote_ages_const + 1e-6)
log_remote_const_values = np.log10(remote_const_values + 1e-6)
correlation_matrix_remote_const_log = np.corrcoef(log_remote_ages_const, log_remote_const_values)
correlation_coefficient_remote_const_log = correlation_matrix_remote_const_log[0, 1]
print(f"Correlation coefficient for remote data only (log-log scale): {correlation_coefficient_remote_const_log}")
# Calculate correlation only for in-situ data in log-log scale
in_situ_ages_const = np.array([d[0] for d in data_const if d[3] == "in-situ"])
in_situ_const_values = np.array([d[2] for d in data_const if d[3] == "in-situ"])
log_in_situ_ages_const = np.log10(in_situ_ages_const + 1e-6)
log_in_situ_const_values = np.log10(in_situ_const_values + 1e-6)
correlation_matrix_in_situ_const_log = np.corrcoef(log_in_situ_ages_const, log_in_situ_const_values)
correlation_coefficient_in_situ_const_log = correlation_matrix_in_situ_const_log[0, 1]
print(f"Correlation coefficient for in-situ data only (log-log scale): {correlation_coefficient_in_situ_const_log}")

# Save correlation coefficients to a text file
output_dir = "/Volumes/SSD_Kanda_SAMSUNG/RSFD_vs_age"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, "correlation_coefficients.txt"), "w") as f:
    f.write("=== Power law exponent ===\n")
    f.write(f"Correlation coefficient (linear scale): {correlation_coefficient}\n")
    f.write(f"Correlation coefficient (log scale): {correlation_coefficient_log}\n")
    f.write(f"Correlation coefficient for remote data only (log scale): {correlation_coefficient_remote_log}\n")
    f.write(f"Correlation coefficient for in-situ data only (log scale): {correlation_coefficient_in_situ_log}\n")
    f.write("\n=== Constant term (/km^2) ===\n")
    f.write(f"Correlation coefficient (linear scale): {correlation_coefficient_const}\n")
    f.write(f"Correlation coefficient (log-log scale): {correlation_coefficient_const_log}\n")
    f.write(f"Correlation coefficient for remote data only (log-log scale): {correlation_coefficient_remote_const_log}\n")
    f.write(f"Correlation coefficient for in-situ data only (log-log scale): {correlation_coefficient_in_situ_const_log}\n")
print(f"Correlation coefficients saved to {output_dir}/correlation_coefficients.txt")



# Add color parameter based on reference
reference_colors = {
    "Aussel et al., 2025, (R)": "red",
    "Cintala & Macbride, 1995, (I)": "green",
    "Cintala & Macbride, 1995, (R)": "green",
    "Krishna & Kumar, 2016, (R)": "blue",
    "Li et al., 2017, (R)": "orange",
    "Li et al., 2017, (I)": "orange",
    "Li et al., 2018, (R)": "cyan",
    "Watkins et al., 2019, (R)": "purple",
    "Pajola et al., 2019, (R)": "black",
    "Shoemaker & Morris, 1969, (I)": "brown",
}


def plot_rsfd_vs_age(data, data_type, filter_type, show_error_bars, output_filename, ylabel, output_dir):
    """
    汎用RSFD vs Age プロット関数

    Parameters:
    -----------
    data : list of tuples
        データセット (age, age_error, value, in-situ/remote, reference)
    data_type : str
        "exponent" または "const"
    filter_type : str
        "all", "in-situ", "remote"
    show_error_bars : bool
        エラーバー表示の有無
    output_filename : str
        出力ファイル名（拡張子なし）
    ylabel : str
        Y軸ラベル
    output_dir : str
        出力ディレクトリパス
    """
    # データ抽出
    ages = np.array([d[0] for d in data])
    age_errors = np.array([d[1] for d in data])
    values = np.array([d[2] for d in data])
    in_situ_flags = np.array([d[3] == "in-situ" for d in data])
    markers = np.where(in_situ_flags, 'o', 'x')

    # フィルタリング
    if filter_type == "in-situ":
        mask = in_situ_flags
    elif filter_type == "remote":
        mask = ~in_situ_flags
    else:  # "all"
        mask = np.ones(len(data), dtype=bool)

    # プロット作成
    plt.figure(figsize=(12, 8))

    for i, d in enumerate(data):
        if not mask[i]:
            continue

        if show_error_bars:
            plt.errorbar(ages[i], values[i], xerr=age_errors[i], fmt=markers[i],
                        label=d[4], capsize=5, color=reference_colors[d[4]], markersize=7)
        else:
            plt.scatter(ages[i], values[i], marker=markers[i], label=d[4],
                       color=reference_colors[d[4]], s=75)

    plt.xlabel("Age (Ma)", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlim(0.1, 5000)

    # Y軸範囲の設定
    if data_type == "exponent":
        plt.ylim(0, 7)
    else:  # "const"
        # 対数スケールを使用
        plt.yscale('log')

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xscale('log')

    # 凡例作成
    handles = []
    labels = []
    for ref, color in reference_colors.items():
        # フィルタ条件に合致するデータがあるか確認
        has_data = False
        marker_type = None
        for d in data:
            if d[4] == ref:
                if filter_type == "all":
                    has_data = True
                    marker_type = 'o' if d[3] == "in-situ" else 'x'
                    break
                elif filter_type == "in-situ" and d[3] == "in-situ":
                    has_data = True
                    marker_type = 'o'
                    break
                elif filter_type == "remote" and d[3] == "remote":
                    has_data = True
                    marker_type = 'x'
                    break

        if has_data and ref not in labels:
            handles.append(plt.Line2D([], [], color=color, marker=marker_type,
                                     linestyle='None', markersize=8))
            labels.append(ref)

    # 凡例を図の上部外側に配置
    plt.legend(handles, labels, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid()
    plt.tight_layout()

    # 保存
    plt.savefig(os.path.join(output_dir, f"{output_filename}.png"), dpi=120)
    plt.savefig(os.path.join(output_dir, f"{output_filename}.pdf"), dpi=600)
    print(f"プロットを保存しました: {output_dir}/{output_filename}.png")
    plt.show()


# 出力ディレクトリの設定
output_dir = "/Volumes/SSD_Kanda_SAMSUNG/RSFD_vs_age"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# data_exponentの4種類のプロット（汎用関数を使用）
print("\n=== Power law exponent プロットの作成 ===")

# 1. With error bars (all data)
plot_rsfd_vs_age(
    data=data_exponent,
    data_type="exponent",
    filter_type="all",
    show_error_bars=True,
    output_filename="exponent_error_bars",
    ylabel="Power law exponent",
    output_dir=output_dir
)

# 2. Without error bars (all data)
plot_rsfd_vs_age(
    data=data_exponent,
    data_type="exponent",
    filter_type="all",
    show_error_bars=False,
    output_filename="exponent_no_error_bars",
    ylabel="Power law exponent",
    output_dir=output_dir
)

# 3. In-situ only
plot_rsfd_vs_age(
    data=data_exponent,
    data_type="exponent",
    filter_type="in-situ",
    show_error_bars=True,
    output_filename="exponent_in_situ_only",
    ylabel="Power law exponent",
    output_dir=output_dir
)

# 4. Remote only
plot_rsfd_vs_age(
    data=data_exponent,
    data_type="exponent",
    filter_type="remote",
    show_error_bars=True,
    output_filename="exponent_remote_only",
    ylabel="Power law exponent",
    output_dir=output_dir
)

# data_constの4種類のプロット（汎用関数を使用）
print("\n=== Constant term (/km^2) プロットの作成 ===")

# 1. With error bars (all data)
plot_rsfd_vs_age(
    data=data_const,
    data_type="const",
    filter_type="all",
    show_error_bars=True,
    output_filename="const_error_bars",
    ylabel="Constant term (/km$^2$)",
    output_dir=output_dir
)

# 2. Without error bars (all data)
plot_rsfd_vs_age(
    data=data_const,
    data_type="const",
    filter_type="all",
    show_error_bars=False,
    output_filename="const_no_error_bars",
    ylabel="Constant term (/km$^2$)",
    output_dir=output_dir
)

# 3. In-situ only
plot_rsfd_vs_age(
    data=data_const,
    data_type="const",
    filter_type="in-situ",
    show_error_bars=True,
    output_filename="const_in_situ_only",
    ylabel="Constant term (/km$^2$)",
    output_dir=output_dir
)

# 4. Remote only
plot_rsfd_vs_age(
    data=data_const,
    data_type="const",
    filter_type="remote",
    show_error_bars=True,
    output_filename="const_remote_only",
    ylabel="Constant term (/km$^2$)",
    output_dir=output_dir
)

print("\n=== すべてのプロット作成が完了しました ===")
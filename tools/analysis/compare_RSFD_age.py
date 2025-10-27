import numpy as np
import matplotlib.pyplot as plt
import os


# RSFDと年齢データ
# 形式：（中央年代 (Ma), 年代誤差 (Ma), べき指数, in-situ or remote, reference）
data = [
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
    (2, 0, 5.3, "remote", "Watkins et al., 2019, (R)"),
    (26, 0, 5.6, "remote", "Watkins et al., 2019, (R)"),
    (50, 0, 3.8, "remote", "Watkins et al., 2019, (R)"),
    (80, 0, 4.4, "remote", "Watkins et al., 2019, (R)"),
    (105, 0, 6.8, "remote", "Watkins et al., 2019, (R)"),
    (200, 0, 4.7, "remote", "Watkins et al., 2019, (R)"),
    (10, 0, 4.03, "remote", "Pajola et al., 2019, (R)")
]


# Calculate correlation coefficient
# Calculate in linear scale
ages = np.array([d[0] for d in data])
rsfd_values = np.array([d[2] for d in data])
correlation_matrix = np.corrcoef(ages, rsfd_values)
correlation_coefficient = correlation_matrix[0, 1]
print(f"Correlation coefficient (linear scale): {correlation_coefficient}")
# Calculate in log scale
log_ages = np.log10(ages + 1e-6)  # avoid
correlation_matrix_log = np.corrcoef(log_ages, rsfd_values)
correlation_coefficient_log = correlation_matrix_log[0, 1]
print(f"Correlation coefficient (log scale): {correlation_coefficient_log}")
# Calculate correlation only for remote data in log scale
remote_ages = np.array([d[0] for d in data if d[3] == "remote"])
remote_rsfd_values = np.array([d[2] for d in data if d[3] == "remote"])
log_remote_ages = np.log10(remote_ages + 1e-6)
correlation_matrix_remote_log = np.corrcoef(log_remote_ages, remote_rsfd_values)
correlation_coefficient_remote_log = correlation_matrix_remote_log[0, 1]
print(f"Correlation coefficient for remote data only (log scale): {correlation_coefficient_remote_log}")
# Calculate correlation only for in-situ data in log scale
in_situ_ages = np.array([d[0] for d in data if d[3] == "in-situ"])
in_situ_rsfd_values = np.array([d[2] for d in data if d[3] == "in-situ"])
log_in_situ_ages = np.log10(in_situ_ages + 1e-6)
correlation_matrix_in_situ_log = np.corrcoef(log_in_situ_ages, in_situ_rsfd_values)
correlation_coefficient_in_situ_log = correlation_matrix_in_situ_log[0, 1]
print(f"Correlation coefficient for in-situ data only (log scale): {correlation_coefficient_in_situ_log}")
# Save correlation coefficients to a text file
output_dir = "/Volumes/SSD_Kanda_SAMSUNG/RSFD_vs_age"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, "correlation_coefficients.txt"), "w") as f:
    f.write(f"Correlation coefficient (linear scale): {correlation_coefficient}\n")
    f.write(f"Correlation coefficient (log scale): {correlation_coefficient_log}\n")
    f.write(f"Correlation coefficient for remote data only (log scale): {correlation_coefficient_remote_log}\n")
    f.write(f"Correlation coefficient for in-situ data only (log scale): {correlation_coefficient_in_situ_log}\n")
print(f"Correlation coefficients saved to {output_dir}/correlation_coefficients.txt")



# Add color parameter based on reference
reference_colors = {
    "Aussel et al., 2025, (R)": "blue",
    "Cintala & Macbride, 1995, (I)": "orange",
    "Cintala & Macbride, 1995, (R)": "orange",
    "Krishna & Kumar, 2016, (R)": "green",
    "Li et al., 2017, (R)": "purple",
    "Li et al., 2017, (I)": "purple",
    "Watkins et al., 2019, (R)": "brown",
    "Pajola et al., 2019, (R)": "red"
}

# prepare data for plotting
ages = np.array([d[0] for d in data])
age_errors = np.array([d[1] for d in data])
rsfd_values = np.array([d[2] for d in data])
in_situ_flags = np.array([d[3] == "in-situ" for d in data])
markers = np.where(in_situ_flags, 'o', 'x')


# Plot
# With error bars
plt.figure(figsize=(12, 8))
for i, d in enumerate(data):
    plt.errorbar(ages[i], rsfd_values[i], xerr=age_errors[i], fmt=markers[i], label=d[4], capsize=5,
                 color=reference_colors[d[4]], markersize=8)

plt.xlabel("Age (Ma)", fontsize=18)
plt.ylabel("Power law exponent", fontsize=18)
plt.xlim(0.1, 5000)
plt.ylim(0, 7)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')

# Add legend at only one entry per reference with custom markers
handles = []
labels = []
for ref, color in reference_colors.items():
    if ref not in labels:
        marker = 'o' if any(d[4] == ref and d[3] == "in-situ" for d in data) else 'x'
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8))
        labels.append(ref)
# Add legend outside the plot
plt.legend(handles, labels, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

plt.grid()
plt.tight_layout()

# save
output_dir = "/Volumes/SSD_Kanda_SAMSUNG/RSFD_vs_age"
plt.savefig(os.path.join(output_dir, "error_bars.png"), dpi=120)
plt.savefig(os.path.join(output_dir, "error_bars.pdf"), dpi=600)
print(f"プロットを保存しました: {output_dir}/error_bars.png")
plt.show()


# Without error bars
plt.figure(figsize=(12, 8))
for i, d in enumerate(data):
    plt.scatter(ages[i], rsfd_values[i], marker=markers[i], label=d[4],
                color=reference_colors[d[4]], s=100)
plt.xlabel("Age (Ma)", fontsize=18)
plt.ylabel("Power law exponent", fontsize=18)
plt.xlim(0.1, 5000)
plt.ylim(0, 7)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')

# Add legend at only one entry per reference with custom markers
handles = []
labels = []
for ref, color in reference_colors.items():
    if ref not in labels:
        marker = 'o' if any(d[4] == ref and d[3] == "in-situ" for d in data) else 'x'
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8))
        labels.append(ref)
# Add legend outside the plot
plt.legend(handles, labels, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

plt.grid()
plt.tight_layout()
# save
plt.savefig(os.path.join(output_dir, "no_error_bars.png"), dpi=120)
plt.savefig(os.path.join(output_dir, "no_error_bars.pdf"), dpi=600)
print(f"プロットを保存しました: {output_dir}/no_error_bars.png")
plt.show()


# Plot only in-situ data
plt.figure(figsize=(12, 8))
for i, d in enumerate(data):
    if d[3] == "in-situ":
        plt.errorbar(ages[i], rsfd_values[i], xerr=age_errors[i], fmt=markers[i], label=d[4], capsize=5,
                     color=reference_colors[d[4]], markersize=8)
plt.xlabel("Age (Ma)", fontsize=18)
plt.ylabel("Power law exponent", fontsize=18)
plt.xlim(0.1, 5000)
plt.ylim(0, 7)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')
# Add legend at only one entry per reference with custom markers
handles = []
labels = []
for ref, color in reference_colors.items():
    if ref not in labels and any(d[4] == ref and d[3] == "in-situ" for d in data):
        marker = 'o'  # only in-situ data
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8))
        labels.append(ref)
# Add legend outside the plot
plt.legend(handles, labels, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.tight_layout()
# save
plt.savefig(os.path.join(output_dir, "in_situ_only.png"), dpi=120)
plt.savefig(os.path.join(output_dir, "in_situ_only.pdf"), dpi=600)
print(f"プロットを保存しました: {output_dir}/in_situ_only.png")
plt.show()


# Plot only remote data
plt.figure(figsize=(12, 8))
for i, d in enumerate(data):
    if d[3] == "remote":
        plt.errorbar(ages[i], rsfd_values[i], xerr=age_errors[i], fmt=markers[i], label=d[4], capsize=5,
                     color=reference_colors[d[4]], markersize=8)
plt.xlabel("Age (Ma)", fontsize=18)
plt.ylabel("Power law exponent", fontsize=18)
plt.xlim(0.1, 5000)
plt.ylim(0, 7)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')
# Add legend at only one entry per reference with custom markers
handles = []
labels = []
for ref, color in reference_colors.items():
    if ref not in labels and any(d[4] == ref and d[3] == "remote" for d in data):
        marker = 'x'  # only remote data
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8))
        labels.append(ref)
# Add legend outside the plot
plt.legend(handles, labels, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.tight_layout()
# save
plt.savefig(os.path.join(output_dir, "remote_only.png"), dpi=120)
plt.savefig(os.path.join(output_dir, "remote_only.pdf"), dpi=600)
print(f"プロットを保存しました: {output_dir}/remote_only.png")
plt.show()
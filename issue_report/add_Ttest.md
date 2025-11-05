【作業指示書】 RSFD_generator.py の改修TO: Claude Code (開発担当)FROM: プロジェクトマネージャーTICKET-ID: PY-RSFD-002件名: statsmodels を用いた回帰係数のt検定機能の実装と結果出力の強化1. 概要 (Overview)現行の RSFD_generator.py は、numpy.polyfit を使用して「べき則」および「指数関数」のフィッティングを実行しています。本タスクの目的は、このフィッティングロジックを statsmodels ライブラリに置き換え、最小二乗法（OLS）による**回帰係数（傾き $r$）の統計的優位性（t値, p値, 標準誤差）**を算出し、プロット凡例および新規サマリーファイルに出力するようスクリプトを改修することです。2. 受入基準 (Acceptance Criteria)[ ] スクリプトが import statsmodels.api as sm を正しくインポートする。[ ] calc_fitting 関数が sm.OLS を使用してフィッティングを行い、傾き $r$ とその統計量（t値, p値, 標準誤差）を正しく返す。[ ] 出力されるすべてのフィッティングプロット（.png, .pdf）の凡例に、t値とp値が書式指定に従って表示される。[ ] output_dir 直下に RSFD_fitting_summary.txt が新規作成される。[ ] RSFD_fitting_summary.txt には、3つのデータセット（従来, Group2推定, Group2-3のみ）のそれぞれについて、べき則と指数関数の計6パターンのフィッティング結果（$k, r, R^2, r\_StdErr, r\_t\_value, r\_p\_value, N\_points, DOF$）がタブ区切りで記録される。3. タスク詳細 (Task Details)Task 1: 依存関係の追加対象ファイル: RSFD_generator.py （スクリプト冒頭）指示: import ブロックに以下の行を追加してください。Pythonimport statsmodels.api as sm
Task 2: calc_fitting 関数のロジック置換対象関数: calc_fitting (# 8.)現状 (As-Is): np.polyfit を使用し、$R^2$ を手動で計算している。変更後 (To-Be): sm.OLS を使用し、詳細な統計レポートを取得する。実装ステップ:べき則 (Power-law): $\log N = r \log D + \log k$独立変数 $X = \log D$, 従属変数 $Y = \log N$ とします。np.polyfit の行を以下に置き換えてください。Python# 定数項 (切片) のために X に '1' の列を追加
X_pow = sm.add_constant(log_D)
# OLSモデルの実行
model_pow = sm.OLS(log_N, X_pow)
results_pow = model_pow.fit()
パラメータを results_pow から抽出します。Pythonlog_k_pow, r_pow = results_pow.params
k_pow = np.exp(log_k_pow)
R2_pow = results_pow.rsquared
# 傾き(r) の統計量を取得
r_pow_se = results_pow.bse[1]
r_pow_t = results_pow.tvalues[1]
r_pow_p = results_pow.pvalues[1]
dof_pow = results_pow.df_resid
n_pow = int(results_pow.nobs)
指数関数 (Exponential): $\log N = rD + \log k$独立変数 $X = sizes$ (非対数), 従属変数 $Y = \log N$ とします。同様に np.polyfit の行を置き換えてください。Python# 定数項 (切片) のために X に '1' の列を追加
X_exp = sm.add_constant(sizes)
# OLSモデルの実行
model_exp = sm.OLS(log_N, X_exp)
results_exp = model_exp.fit()
パラメータを results_exp から抽出します。Pythonlog_k_exp, r_exp = results_exp.params
k_exp = np.exp(log_k_exp)
R2_exp = results_exp.rsquared
# 傾き(r) の統計量を取得
r_exp_se = results_exp.bse[1]
r_exp_t = results_exp.tvalues[1]
r_exp_p = results_exp.pvalues[1]
dof_exp = results_exp.df_resid
n_exp = int(results_exp.nobs)
戻り値の変更:calc_fitting 関数の return 文を、上記で取得したすべての統計量を含むように変更してください。戻り値のタプル構造（例）:Python# べき則の結果, 指数関数の結果, D_fit
return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
       (k_exp, np.abs(r_exp), R2_exp, N_exp_fit, r_exp_t, r_exp_p, r_exp_se, n_exp, dof_exp), \
       D_fit
Task 3: calc_fitting 呼び出し箇所の修正対象: # 8. の直後、3箇所の calc_fitting(...) 呼び出し指示: Task 2 で変更した戻り値（タプル）をすべて受け取れるよう、アンパックする変数を追加してください。（例） (k_pow_trad, r_pow_trad, R2_pow_trad, N_pow_fit_trad, t_pow_trad, p_pow_trad, ...), ... = calc_fitting(...)Task 4: プロット凡例の更新対象: # 9., # 10., # 11. の fit_lines... 辞書定義指示: label 文字列を、t値とp値を含むように更新してください。p値の書式指定:Python# p値のフォーマットを補助する
def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p={p:.3f}"

# (この関数は create_rsfd_plot の近くか、凡例作成の直前で定義してください)
凡例ラベルのテンプレート:（例）# 9.1 べき則フィット（従来手法） の label 文字列を以下のように変更します。Pythonp_str_pow_trad = format_p_value(p_pow_trad) # p値の書式設定
fit_lines_pow_trad = [{
    'x': D_fit_trad, 'y': N_pow_fit_trad,
    'label': f'Power-law: r={r_pow_trad:.3f}, R²={R2_pow_trad:.4f}, t={t_pow_trad:.2f}, {p_str_pow_trad}',
    'color': 'red', 'linestyle': '--'
}]
この変更を、指数関数（Exponential）および他のデータセット（_est_grp2, _grp2_3）のすべての label 文字列に対しても同様に適用してください。Task 5: 統計サマリーファイルの新規作成対象: スクリプト末尾、print('すべて完了しました！') の直前指示: 3つのデータセット × 2つのモデル（計6パターン）のフィッティング結果を、RSFD_fitting_summary.txt にタブ区切り（TSV）で出力するコードを追加してください。出力ファイルパス:Pythonsummary_file_path = os.path.join(output_dir, 'RSFD_fitting_summary.txt')
出力内容（ヘッダーとデータ行）:Pythonwith open(summary_file_path, 'w') as f:
    # ヘッダー (タブ区切り)
    f.write('DataSet\tModel\tk\tr\tR_squared\tr_StdErr\tr_t_value\tr_p_value\tN_points\tDOF\n')

    # 1. 従来手法 (Traditional)
    f.write(f'Traditional\tPower\t{k_pow_trad:.4e}\t{r_pow_trad:.4f}\t{R2_pow_trad:.4f}\t'
            f'{se_pow_trad:.4f}\t{t_pow_trad:.3f}\t{p_pow_trad:.4e}\t{n_pow_trad}\t{dof_pow_trad}\n')
    f.write(f'Traditional\tExponential\t{k_exp_trad:.4e}\t{r_exp_trad:.4f}\t{R2_exp_trad:.4f}\t'
            f'{se_exp_trad:.4f}\t{t_exp_trad:.3f}\t{p_exp_trad:.4e}\t{n_exp_trad}\t{dof_exp_trad}\n')

    # 2. Group2 推定 (Estimate_Group2)
    f.write(f'Estimate_Group2\tPower\t{k_pow_est_grp2:.4e}\t{r_pow_est_grp2:.4f}\t{R2_pow_est_grp2:.4f}\t'
            f'{se_pow_est_grp2:.4f}\t{t_pow_est_grp2:.3f}\t{p_pow_est_grp2:.4e}\t{n_pow_est_grp2}\t{dof_pow_est_grp2}\n')
    f.write(f'Estimate_Group2\tExponential\t{k_exp_est_grp2:.4e}\t{r_exp_est_grp2:.4f}\t{R2_exp_est_grp2:.4f}\t'
            f'{se_exp_est_grp2:.4f}\t{t_exp_est_grp2:.3f}\t{p_exp_est_grp2:.4e}\t{n_exp_est_grp2}\t{dof_exp_est_grp2}\n')

    # 3. Group2-3 のみ (Group2-3_Only)
    f.write(f'Group2-3_Only\tPower\t{k_pow_grp2_3:.4e}\t{r_pow_grp2_3:.4f}\t{R2_pow_grp2_3:.4f}\t'
            f'{se_pow_grp2_3:.4f}\t{t_pow_grp2_3:.3f}\t{p_pow_grp2_3:.4e}\t{n_pow_grp2_3}\t{dof_pow_grp2_3}\n')
    f.write(f'Group2-3_Only\tExponential\t{k_exp_grp2_3:.4e}\t{r_exp_grp2_3:.4f}\t{R2_exp_grp2_3:.4f}\t'
            f'{se_exp_grp2_3:.4f}\t{t_exp_grp2_3:.3f}\t{p_exp_grp2_3:.4e}\t{n_exp_grp2_3}\t{dof_exp_grp2_3}\n')

print(f'フィッティングサマリー保存: {summary_file_path}')
（注: 上記コード内の変数は、Task 3 で定義した変数名に正確に合わせてください）作業指示は以上です。
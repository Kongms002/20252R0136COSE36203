"""
Optuna 최적화 결과 시각화

생성되는 그래프:
1. optuna_comparison.png - 최적화 전후 성능 비교
2. model_performance.png - 모델별 성능 비교
3. ensemble_comparison.png - 앙상블 방법 비교
4. improvement_analysis.png - 개선율 분석
5. optimization_summary.png - 최적화 요약
6. cv_fold_results.png - CV Fold별 결과
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
from pathlib import Path

# 경로 설정
OUTPUT_DIR = Path(__file__).parent

# 스타일 먼저 적용 (폰트 설정 전에!)
plt.style.use('seaborn-v0_8-whitegrid')

# 한글 폰트 설정 (스타일 적용 후에!)
font_path = Path.home() / '.fonts' / 'NanumGothicCoding.ttf'
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'NanumGothicCoding'
    print(f"Using Korean font: NanumGothicCoding")
else:
    print("Korean font not found, using default")

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 색상 팔레트
COLORS = {
    'default': '#95a5a6',
    'lightgbm': '#3498db',
    'xgboost': '#e74c3c',
    'catboost': '#2ecc71',
    'ensemble': '#9b59b6',
    'best': '#f39c12'
}


def create_optuna_comparison():
    """최적화 전후 성능 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE 비교
    ax1 = axes[0]
    models = ['LightGBM\nDefault', 'LightGBM\nOptuna', 'XGBoost\nOptuna', 'CatBoost\nOptuna', 'Median\nEnsemble']
    mae_values = [241.96, 199.53, 172.69, 190.94, 168.81]
    colors = [COLORS['default'], COLORS['lightgbm'], COLORS['xgboost'], COLORS['catboost'], COLORS['best']]
    
    bars = ax1.bar(models, mae_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('MAE (persons)', fontsize=12, fontweight='bold')
    ax1.set_title('MAE Comparison: Before vs After Optuna', fontsize=14, fontweight='bold')
    ax1.axhline(y=241.96, color='red', linestyle='--', alpha=0.5, label='Baseline')
    
    # 값 표시
    for bar, val in zip(bars, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.legend(loc='upper right')
    
    # R² 비교
    ax2 = axes[1]
    r2_values = [0.9187, 0.9395, 0.9291, 0.9451, 0.9494]
    
    bars = ax2.bar(models, r2_values, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² Comparison: Before vs After Optuna', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.9, 0.96)
    ax2.axhline(y=0.9187, color='red', linestyle='--', alpha=0.5, label='Baseline')
    
    for bar, val in zip(bars, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optuna_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: optuna_comparison.png")


def create_model_performance():
    """모델별 성능 비교 상세"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['LightGBM\nDefault', 'LightGBM\nOptuna', 'XGBoost\nOptuna', 'CatBoost\nOptuna']
    mae_values = [241.96, 199.53, 172.69, 190.94]
    mae_std = [52.9, 37.9, 41.8, 35.4]
    colors = [COLORS['default'], COLORS['lightgbm'], COLORS['xgboost'], COLORS['catboost']]
    
    x = np.arange(len(models))
    bars = ax.bar(x, mae_values, yerr=mae_std, color=colors, 
                  edgecolor='black', linewidth=1.5, capsize=8, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (persons)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    
    # 개선율 표시
    baseline = 241.96
    for i, (bar, val) in enumerate(zip(bars, mae_values)):
        if i > 0:
            improvement = (baseline - val) / baseline * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mae_std[i] + 5,
                   f'{val:.1f}\n(-{improvement:.1f}%)', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='green')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mae_std[i] + 5,
                   f'{val:.1f}\n(baseline)', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='gray')
    
    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: model_performance.png")


def create_ensemble_comparison():
    """앙상블 방법 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 개별 모델 vs 앙상블
    ax1 = axes[0]
    categories = ['XGBoost\nOptuna', 'CatBoost\nOptuna', 'LightGBM\nOptuna', 
                  'Simple\nAverage', 'Weighted\nAverage', 'Median\nEnsemble']
    mae_values = [172.69, 190.94, 199.53, 171.26, 171.29, 168.81]
    colors = [COLORS['xgboost'], COLORS['catboost'], COLORS['lightgbm'],
              COLORS['ensemble'], COLORS['ensemble'], COLORS['best']]
    
    bars = ax1.barh(categories, mae_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('MAE (persons)', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Models vs Ensemble Methods', fontsize=14, fontweight='bold')
    ax1.axvline(x=168.81, color='gold', linestyle='--', linewidth=2, label='Best (Median)')
    
    for bar, val in zip(bars, mae_values):
        ax1.text(val + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=11, fontweight='bold')
    
    ax1.legend(loc='lower right')
    ax1.set_xlim(150, 220)
    
    # 앙상블 R² 비교
    ax2 = axes[1]
    ens_methods = ['Simple Average', 'Weighted Average', 'Median Ensemble']
    r2_values = [0.9477, 0.9479, 0.9494]
    colors = [COLORS['ensemble'], COLORS['ensemble'], COLORS['best']]
    
    bars = ax2.bar(ens_methods, r2_values, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Ensemble Method R² Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.94, 0.955)
    
    for bar, val in zip(bars, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ensemble_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: ensemble_comparison.png")


def create_improvement_analysis():
    """개선율 분석"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 단계별 개선
    stages = ['V6 Baseline\n(LightGBM Default)', 'LightGBM\n+ Optuna', 
              'XGBoost\n+ Optuna', 'Median\nEnsemble']
    mae_values = [241.96, 199.53, 172.69, 168.81]
    
    # 개선율 계산
    baseline = mae_values[0]
    improvements = [(baseline - v) / baseline * 100 for v in mae_values]
    
    # 선 그래프
    x = range(len(stages))
    ax.plot(x, mae_values, 'o-', color=COLORS['best'], linewidth=3, markersize=15, 
            markeredgecolor='black', markeredgewidth=2)
    
    # 영역 채우기
    ax.fill_between(x, mae_values, baseline, alpha=0.3, color='green')
    
    # 값 및 개선율 표시
    for i, (stage, mae, imp) in enumerate(zip(stages, mae_values, improvements)):
        ax.annotate(f'MAE: {mae:.1f}\n({imp:.1f}% ↓)', 
                   xy=(i, mae), xytext=(0, -40 if i % 2 == 0 else 40),
                   textcoords='offset points', ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel('MAE (persons)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement Journey with Optuna', fontsize=14, fontweight='bold')
    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Baseline')
    ax.set_ylim(140, 280)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'improvement_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: improvement_analysis.png")


def create_optimization_summary():
    """최적화 요약 인포그래픽"""
    fig = plt.figure(figsize=(14, 10))
    
    # 제목
    fig.suptitle('Optuna Hyperparameter Optimization Summary', fontsize=18, fontweight='bold', y=0.98)
    
    # 서브플롯 설정
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 최적화 전후 비교 (좌상)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Before\n(Default)', 'After\n(Best)']
    values = [241.96, 168.81]
    colors_bar = [COLORS['default'], COLORS['best']]
    bars = ax1.bar(labels, values, color=colors_bar, edgecolor='black', linewidth=2)
    ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax1.set_title('MAE Improvement', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
    
    # 2. 개선율 원그래프 (중상)
    ax2 = fig.add_subplot(gs[0, 1])
    improvement = 30.2
    remaining = 100 - improvement
    ax2.pie([improvement, remaining], labels=[f'Improved\n{improvement:.1f}%', ''],
           colors=['#2ecc71', '#ecf0f1'], startangle=90, 
           wedgeprops={'edgecolor': 'black', 'linewidth': 2},
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Performance Improvement', fontsize=12, fontweight='bold')
    
    # 3. 최적화 시간 (우상)
    ax3 = fig.add_subplot(gs[0, 2])
    models_time = ['LightGBM', 'XGBoost', 'CatBoost']
    times = [6.9, 35.3, 45.8]  # 분
    colors_time = [COLORS['lightgbm'], COLORS['xgboost'], COLORS['catboost']]
    bars = ax3.barh(models_time, times, color=colors_time, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax3.set_title('Optimization Time', fontsize=12, fontweight='bold')
    for bar, t in zip(bars, times):
        ax3.text(t + 1, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}m', va='center', fontsize=11, fontweight='bold')
    
    # 4. 모델별 MAE (좌하)
    ax4 = fig.add_subplot(gs[1, 0])
    models = ['LGB_Def', 'LGB_Opt', 'XGB_Opt', 'Cat_Opt', 'Median']
    maes = [241.96, 199.53, 172.69, 190.94, 168.81]
    colors_model = [COLORS['default'], COLORS['lightgbm'], COLORS['xgboost'], 
                   COLORS['catboost'], COLORS['best']]
    bars = ax4.bar(models, maes, color=colors_model, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax4.set_title('Model MAE Comparison', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. R² 비교 (중하)
    ax5 = fig.add_subplot(gs[1, 1])
    r2s = [0.9187, 0.9395, 0.9291, 0.9451, 0.9494]
    bars = ax5.bar(models, r2s, color=colors_model, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax5.set_title('Model R² Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylim(0.9, 0.96)
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. 최종 결과 텍스트 (우하)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = """
    ===============================
       OPTIMIZATION RESULTS
    ===============================
    
    [*] Best Model: Median Ensemble
    
    [-] MAE: 168.81 (30.2% down)
    
    [+] R2: 0.9494
    
    [T] Total Time: 88.1 minutes
    
    [#] Trials per Model: 50
    
    ===============================
    """
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                     edgecolor='orange', linewidth=2))
    
    plt.savefig(OUTPUT_DIR / 'optimization_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: optimization_summary.png")


def create_cv_fold_results():
    """CV Fold별 결과"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    x = np.arange(len(folds))
    width = 0.2
    
    # 각 모델의 Fold별 결과
    lgb_optuna_mae = [226.89, 181.97, 160.14, 168.85, 259.79]
    xgb_optuna_mae = [197.29, 138.69, 135.43, 148.57, 243.45]
    cat_optuna_mae = [200.88, 174.48, 156.16, 167.80, 255.40]
    lgb_default_mae = [255.35, 212.73, 186.68, 216.80, 338.22]
    
    lgb_optuna_r2 = [0.9142, 0.9506, 0.9643, 0.9406, 0.9278]
    xgb_optuna_r2 = [0.9105, 0.9513, 0.9661, 0.9518, 0.8655]
    cat_optuna_r2 = [0.9361, 0.9534, 0.9632, 0.9449, 0.9281]
    lgb_default_r2 = [0.9056, 0.9375, 0.9532, 0.9229, 0.8745]
    
    # MAE by Fold
    ax1 = axes[0, 0]
    ax1.bar(x - 1.5*width, lgb_default_mae, width, label='LGB Default', color=COLORS['default'])
    ax1.bar(x - 0.5*width, lgb_optuna_mae, width, label='LGB Optuna', color=COLORS['lightgbm'])
    ax1.bar(x + 0.5*width, xgb_optuna_mae, width, label='XGB Optuna', color=COLORS['xgboost'])
    ax1.bar(x + 1.5*width, cat_optuna_mae, width, label='Cat Optuna', color=COLORS['catboost'])
    ax1.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax1.set_title('MAE by Fold', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.legend(loc='upper right', fontsize=9)
    
    # R² by Fold
    ax2 = axes[0, 1]
    ax2.bar(x - 1.5*width, lgb_default_r2, width, label='LGB Default', color=COLORS['default'])
    ax2.bar(x - 0.5*width, lgb_optuna_r2, width, label='LGB Optuna', color=COLORS['lightgbm'])
    ax2.bar(x + 0.5*width, xgb_optuna_r2, width, label='XGB Optuna', color=COLORS['xgboost'])
    ax2.bar(x + 1.5*width, cat_optuna_r2, width, label='Cat Optuna', color=COLORS['catboost'])
    ax2.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax2.set_title('R² by Fold', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folds)
    ax2.set_ylim(0.85, 1.0)
    ax2.legend(loc='lower right', fontsize=9)
    
    # 최고 성능 모델 (XGBoost) 상세
    ax3 = axes[1, 0]
    ax3.plot(folds, xgb_optuna_mae, 'o-', color=COLORS['xgboost'], linewidth=2, 
             markersize=10, label='XGBoost Optuna MAE')
    ax3.fill_between(folds, xgb_optuna_mae, alpha=0.3, color=COLORS['xgboost'])
    ax3.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax3.set_title('XGBoost Optuna (Best Single Model)', fontsize=12, fontweight='bold')
    mean_mae = np.mean(xgb_optuna_mae)
    ax3.axhline(y=mean_mae, color='red', linestyle='--', label=f'Mean: {mean_mae:.1f}')
    ax3.legend()
    
    # 모델 간 개선율
    ax4 = axes[1, 1]
    models = ['LGB Optuna', 'XGB Optuna', 'Cat Optuna']
    baseline_mean = np.mean(lgb_default_mae)
    improvements = [
        (baseline_mean - np.mean(lgb_optuna_mae)) / baseline_mean * 100,
        (baseline_mean - np.mean(xgb_optuna_mae)) / baseline_mean * 100,
        (baseline_mean - np.mean(cat_optuna_mae)) / baseline_mean * 100
    ]
    colors_imp = [COLORS['lightgbm'], COLORS['xgboost'], COLORS['catboost']]
    bars = ax4.bar(models, improvements, color=colors_imp, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('MAE Improvement vs Baseline', fontsize=12, fontweight='bold')
    for bar, imp in zip(bars, improvements):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{imp:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cv_fold_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: cv_fold_results.png")


def create_hyperparameter_analysis():
    """최적 하이퍼파라미터 분석"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # LightGBM 최적 파라미터
    ax1 = axes[0]
    lgb_params = {
        'n_estimators': 493,
        'learning_rate': 0.227,
        'max_depth': 9,
        'num_leaves': 73,
        'min_child_samples': 56,
        'subsample': 0.933,
        'colsample_bytree': 0.892
    }
    params = list(lgb_params.keys())
    values = list(lgb_params.values())
    y_pos = np.arange(len(params))
    
    # 정규화 (시각화용)
    normalized = [v / max(values) for v in values]
    ax1.barh(y_pos, normalized, color=COLORS['lightgbm'], edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params, fontsize=10)
    ax1.set_title('LightGBM Optimal Parameters', fontsize=12, fontweight='bold')
    for i, (v, n) in enumerate(zip(values, normalized)):
        ax1.text(n + 0.02, i, f'{v:.3g}', va='center', fontsize=10)
    
    # XGBoost 최적 파라미터
    ax2 = axes[1]
    xgb_params = {
        'n_estimators': 594,
        'learning_rate': 0.015,
        'max_depth': 12,
        'min_child_weight': 4,
        'subsample': 0.972,
        'colsample_bytree': 0.999
    }
    params = list(xgb_params.keys())
    values = list(xgb_params.values())
    y_pos = np.arange(len(params))
    
    normalized = [v / max(values) for v in values]
    ax2.barh(y_pos, normalized, color=COLORS['xgboost'], edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(params, fontsize=10)
    ax2.set_title('XGBoost Optimal Parameters', fontsize=12, fontweight='bold')
    for i, (v, n) in enumerate(zip(values, normalized)):
        ax2.text(n + 0.02, i, f'{v:.3g}', va='center', fontsize=10)
    
    # CatBoost 최적 파라미터
    ax3 = axes[2]
    cat_params = {
        'iterations': 776,
        'learning_rate': 0.136,
        'depth': 10,
        'bagging_temp': 0.667,
        'random_strength': 0.0
    }
    params = list(cat_params.keys())
    values = list(cat_params.values())
    y_pos = np.arange(len(params))
    
    normalized = [v / max(values) if max(values) > 0 else 0 for v in values]
    ax3.barh(y_pos, normalized, color=COLORS['catboost'], edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(params, fontsize=10)
    ax3.set_title('CatBoost Optimal Parameters', fontsize=12, fontweight='bold')
    for i, (v, n) in enumerate(zip(values, normalized)):
        ax3.text(max(n, 0.05) + 0.02, i, f'{v:.3g}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hyperparameter_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: hyperparameter_analysis.png")


def main():
    """메인 실행"""
    print("=" * 50)
    print(" Creating Optuna Visualization")
    print("=" * 50)
    
    create_optuna_comparison()
    create_model_performance()
    create_ensemble_comparison()
    create_improvement_analysis()
    create_optimization_summary()
    create_cv_fold_results()
    create_hyperparameter_analysis()
    
    print("\n" + "=" * 50)
    print(" All visualizations created successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()


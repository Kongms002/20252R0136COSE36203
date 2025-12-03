"""
Optuna 최적화 모델 Feature Importance 분석 및 시각화
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
from pathlib import Path
import pandas as pd

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
plt.rcParams['font.size'] = 11

# 색상
COLORS = {
    'time': '#3498db',
    'location': '#e74c3c', 
    'weather': '#2ecc71',
    'event': '#9b59b6',
    'interaction': '#f39c12'
}


def load_feature_importances():
    """Feature importance 로드"""
    with open(OUTPUT_DIR / 'feature_importances.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_feature(feature_name):
    """Feature를 카테고리로 분류"""
    feature_lower = feature_name.lower()
    
    # 위치 관련
    if any(k in feature_lower for k in ['순서', '호선', '환승', 'station', 'transfer', 'line']):
        return 'Location'
    
    # 시간 관련
    if any(k in feature_lower for k in ['hour', 'time', '시간', '요일', 'week', 'month', 
                                         'day', 'quarter', 'season', 'rush', 'night', 
                                         'holiday', 'sin', 'cos', 'year']):
        return 'Time'
    
    # 날씨 관련
    if any(k in feature_lower for k in ['temp', 'rain', 'wind', 'humid', 'snow', 'weather',
                                         'discomfort', 'feels', 'cold', 'hot', 'di_']):
        return 'Weather'
    
    # 이벤트 관련
    if any(k in feature_lower for k in ['event', 'nearby']):
        return 'Event'
    
    # 교차 관련
    if '_x_' in feature_lower or 'x_' in feature_lower:
        return 'Interaction'
    
    return 'Other'


def create_feature_importance_top20():
    """상위 20개 Feature Importance"""
    data = load_feature_importances()
    
    # 모든 모델의 평균 계산
    all_features = set()
    for model_data in data.values():
        all_features.update(model_data.keys())
    
    avg_importance = {}
    for feature in all_features:
        values = []
        for model_data in data.values():
            if feature in model_data:
                values.append(model_data[feature])
        if values:
            avg_importance[feature] = np.mean(values)
    
    # 정렬
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    categories = [categorize_feature(f) for f in features]
    
    # 색상 매핑
    color_map = {
        'Time': COLORS['time'],
        'Location': COLORS['location'],
        'Weather': COLORS['weather'],
        'Event': COLORS['event'],
        'Interaction': COLORS['interaction'],
        'Other': '#95a5a6'
    }
    colors = [color_map.get(c, '#95a5a6') for c in categories]
    
    # 플롯
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Average across Optuna-optimized models)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importance (Optuna Optimized Models)', fontsize=14, fontweight='bold')
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['location'], edgecolor='black', label='Location'),
        Patch(facecolor=COLORS['time'], edgecolor='black', label='Time'),
        Patch(facecolor=COLORS['weather'], edgecolor='black', label='Weather'),
        Patch(facecolor=COLORS['event'], edgecolor='black', label='Event'),
        Patch(facecolor=COLORS['interaction'], edgecolor='black', label='Interaction')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # 값 표시
    for bar, imp in zip(bars, importances):
        ax.text(imp + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
               f'{imp:.1f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_top20.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: feature_importance_top20.png")
    
    return sorted_features


def create_group_importance():
    """그룹별 중요도"""
    data = load_feature_importances()
    
    # 모든 feature의 평균 importance
    all_features = set()
    for model_data in data.values():
        all_features.update(model_data.keys())
    
    avg_importance = {}
    for feature in all_features:
        values = []
        for model_data in data.values():
            if feature in model_data:
                values.append(model_data[feature])
        if values:
            avg_importance[feature] = np.mean(values)
    
    # 그룹별 합계
    group_totals = {'Time': 0, 'Location': 0, 'Weather': 0, 'Event': 0, 'Interaction': 0, 'Other': 0}
    
    for feature, importance in avg_importance.items():
        category = categorize_feature(feature)
        group_totals[category] = group_totals.get(category, 0) + importance
    
    # 비율 계산
    total = sum(group_totals.values())
    group_pct = {k: v/total*100 for k, v in group_totals.items() if v > 0}
    
    # 정렬
    sorted_groups = sorted(group_pct.items(), key=lambda x: x[1], reverse=True)
    groups = [g[0] for g in sorted_groups]
    percentages = [g[1] for g in sorted_groups]
    
    colors = [COLORS.get(g.lower(), '#95a5a6') for g in groups]
    
    # 플롯
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 막대 그래프
    ax1 = axes[0]
    bars = ax1.bar(groups, percentages, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Group Importance', fontsize=14, fontweight='bold')
    
    for bar, pct in zip(bars, percentages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # 파이 차트
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(percentages, labels=groups, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
                                        textprops={'fontsize': 11})
    ax2.set_title('Feature Group Distribution', fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'group_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: group_importance.png")
    
    return group_pct


def create_model_comparison_importance():
    """모델별 Feature Importance 비교"""
    data = load_feature_importances()
    
    # 상위 10개 feature (평균 기준)
    all_features = set()
    for model_data in data.values():
        all_features.update(model_data.keys())
    
    avg_importance = {}
    for feature in all_features:
        values = []
        for model_data in data.values():
            if feature in model_data:
                values.append(model_data[feature])
        if values:
            avg_importance[feature] = np.mean(values)
    
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    top_feature_names = [f[0] for f in top_features]
    
    # 모델별 importance
    models = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_feature_names))
    width = 0.25
    
    model_colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (model_name, model_data) in enumerate(data.items()):
        importances = [model_data.get(f, 0) for f in top_feature_names]
        # 정규화
        max_imp = max(importances) if max(importances) > 0 else 1
        normalized = [imp / max_imp * 100 for imp in importances]
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, normalized, width, label=model_name, 
                     color=model_colors[i], edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison Across Optuna-Optimized Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_feature_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: model_comparison_importance.png")


def create_cumulative_importance():
    """누적 중요도"""
    data = load_feature_importances()
    
    # 평균 importance
    all_features = set()
    for model_data in data.values():
        all_features.update(model_data.keys())
    
    avg_importance = {}
    for feature in all_features:
        values = []
        for model_data in data.values():
            if feature in model_data:
                values.append(model_data[feature])
        if values:
            avg_importance[feature] = np.mean(values)
    
    # 정렬
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 누적 계산
    total = sum(f[1] for f in sorted_features)
    cumulative = []
    running_sum = 0
    for feature, importance in sorted_features:
        running_sum += importance
        cumulative.append((running_sum / total) * 100)
    
    # 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(cumulative))
    ax.plot(x, cumulative, 'o-', color='#3498db', linewidth=2, markersize=3)
    ax.fill_between(x, cumulative, alpha=0.3, color='#3498db')
    
    # 주요 포인트 표시
    thresholds = [50, 80, 90, 95]
    for threshold in thresholds:
        idx = next((i for i, c in enumerate(cumulative) if c >= threshold), len(cumulative)-1)
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'{threshold}% at top {idx+1} features', 
                   xy=(idx, threshold), xytext=(idx+5, threshold+2),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    ax.set_xlabel('Number of Features (sorted by importance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Feature Importance (Optuna Optimized Models)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, len(cumulative)+5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cumulative_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: cumulative_importance.png")


def create_key_insights():
    """핵심 인사이트 시각화"""
    data = load_feature_importances()
    
    # 평균 importance
    all_features = set()
    for model_data in data.values():
        all_features.update(model_data.keys())
    
    avg_importance = {}
    for feature in all_features:
        values = []
        for model_data in data.values():
            if feature in model_data:
                values.append(model_data[feature])
        if values:
            avg_importance[feature] = np.mean(values)
    
    total = sum(avg_importance.values())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 위치 Feature
    ax1 = axes[0, 0]
    location_features = {k: v/total*100 for k, v in avg_importance.items() if categorize_feature(k) == 'Location'}
    sorted_loc = sorted(location_features.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_loc:
        names = [f[0] for f in sorted_loc]
        values = [f[1] for f in sorted_loc]
        ax1.barh(names, values, color=COLORS['location'], edgecolor='black')
        ax1.set_xlabel('Importance (%)', fontweight='bold')
        ax1.set_title('Location Features', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
    
    # 2. 시간 Feature (상위 10)
    ax2 = axes[0, 1]
    time_features = {k: v/total*100 for k, v in avg_importance.items() if categorize_feature(k) == 'Time'}
    sorted_time = sorted(time_features.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if sorted_time:
        names = [f[0] for f in sorted_time]
        values = [f[1] for f in sorted_time]
        ax2.barh(names, values, color=COLORS['time'], edgecolor='black')
        ax2.set_xlabel('Importance (%)', fontweight='bold')
        ax2.set_title('Top 10 Time Features', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
    
    # 3. 날씨 Feature
    ax3 = axes[1, 0]
    weather_features = {k: v/total*100 for k, v in avg_importance.items() if categorize_feature(k) == 'Weather'}
    sorted_weather = sorted(weather_features.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if sorted_weather:
        names = [f[0] for f in sorted_weather]
        values = [f[1] for f in sorted_weather]
        ax3.barh(names, values, color=COLORS['weather'], edgecolor='black')
        ax3.set_xlabel('Importance (%)', fontweight='bold')
        ax3.set_title('Top 10 Weather Features', fontsize=12, fontweight='bold')
        ax3.invert_yaxis()
    
    # 4. 이벤트 Feature
    ax4 = axes[1, 1]
    event_features = {k: v/total*100 for k, v in avg_importance.items() if categorize_feature(k) == 'Event'}
    sorted_event = sorted(event_features.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_event:
        names = [f[0] for f in sorted_event]
        values = [f[1] for f in sorted_event]
        ax4.barh(names, values, color=COLORS['event'], edgecolor='black')
        ax4.set_xlabel('Importance (%)', fontweight='bold')
        ax4.set_title('Event Features', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
    
    plt.suptitle('Key Feature Insights by Category (Optuna Optimized)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'key_insights.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: key_insights.png")


def main():
    """메인 실행"""
    print("=" * 50)
    print(" Creating Feature Importance Visualizations")
    print("=" * 50)
    
    top_features = create_feature_importance_top20()
    group_pct = create_group_importance()
    create_model_comparison_importance()
    create_cumulative_importance()
    create_key_insights()
    
    # 결과 출력
    print("\n" + "=" * 50)
    print(" Feature Importance Summary")
    print("=" * 50)
    
    print("\n[Top 5 Features]")
    for i, (feature, importance) in enumerate(top_features[:5]):
        print(f"  {i+1}. {feature}: {importance:.1f}")
    
    print("\n[Group Importance]")
    for group, pct in sorted(group_pct.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {group}: {pct:.1f}%")
    
    print("\n" + "=" * 50)
    print(" All visualizations created!")
    print("=" * 50)


if __name__ == "__main__":
    main()


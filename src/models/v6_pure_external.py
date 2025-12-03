"""
V6: Pure External Factors Model

타겟 정보를 전혀 사용하지 않는 순수 외부 요인 모델입니다.
- Lag/Rolling 완전 제외
- Target Encoding 완전 제외
- 오직 시간, 위치, 날씨, 이벤트 정보만 사용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor

from .base_model import load_featured_data


def get_pure_external_features(df: pd.DataFrame) -> List[str]:
    """
    순수 외부 요인 feature만 선별
    
    타겟(net_passengers, 승차, 하차) 정보가 들어간 모든 feature 제외:
    - lag_*, rolling_*, diff_*, pct_*
    - *_mean_*, *_std_*, *_median_* (타겟 통계)
    - Target Encoding feature
    """
    
    # 완전히 제외할 컬럼
    exclude_exact = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    
    # 제외할 키워드 (타겟 정보 포함 가능성)
    exclude_keywords = [
        # Lag/Rolling
        "lag_", "rolling_", "diff_", "pct_change",
        # 타겟 통계 기반
        "_mean_승차", "_mean_하차", "_mean_net",
        "_std_승차", "_std_하차", "_std_net",
        "time_mean", "weekday_mean", "month_mean",
        # Target Encoding
        "_te",
        # 기타 타겟 관련
        "승차", "하차", "net_passengers"
    ]
    
    selected_features = []
    
    for col in df.columns:
        # 정확히 제외할 컬럼
        if col in exclude_exact:
            continue
        
        # 키워드로 제외
        if any(kw in col for kw in exclude_keywords):
            continue
        
        selected_features.append(col)
    
    return selected_features


def train_v6_pure_external(
    data_path: Path,
    output_dir: Path,
    n_splits: int = 5,
    use_lightgbm: bool = True
) -> Dict:
    """
    V6 순수 외부 요인 모델 학습 (Time Series CV)
    
    타겟 정보가 전혀 없는 feature만 사용합니다:
    - 시간: Hour, 요일, Month, Season, Holiday 등
    - 위치: 순서, 호선, 환승역
    - 날씨: temperature, rainfall, humidity 등
    - 이벤트: event_count, is_event_nearby 등
    """
    print("=" * 70)
    print(" V6 Pure External Factors Model")
    print(" (타겟 정보 완전 제외 - 순수 외부 요인만)")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 시간순 정렬
    df = df.sort_values(["Date", "Hour", "역명"]).reset_index(drop=True)
    
    # 순수 외부 요인 feature 선별
    feature_cols = get_pure_external_features(df)
    
    print(f"\n[사용 Feature 목록]")
    print("-" * 50)
    
    # Feature 그룹별 출력
    time_features = [f for f in feature_cols if any(kw in f for kw in 
        ["Hour", "Time", "요일", "Day", "Month", "Week", "Season", "Rush", "Holiday", "Quarter", "Night", "시간"])]
    location_features = [f for f in feature_cols if any(kw in f for kw in 
        ["순서", "호선", "환승", "transfer", "order"])]
    weather_features = [f for f in feature_cols if any(kw in f for kw in 
        ["temp", "rain", "snow", "humid", "wind", "Weather", "DI", "Feels", "discomfort", "Precip", "Cold", "Hot", "Freez", "Extreme"])]
    event_features = [f for f in feature_cols if "event" in f.lower()]
    interaction_features = [f for f in feature_cols if "_x_" in f]
    
    other_features = [f for f in feature_cols if f not in 
        time_features + location_features + weather_features + event_features + interaction_features]
    
    print(f"  시간 관련: {len(time_features)}개")
    print(f"    → {time_features[:5]}...")
    print(f"  위치 관련: {len(location_features)}개")
    print(f"    → {location_features}")
    print(f"  날씨 관련: {len(weather_features)}개")
    print(f"    → {weather_features[:5]}...")
    print(f"  이벤트 관련: {len(event_features)}개")
    print(f"    → {event_features}")
    print(f"  교차 변수: {len(interaction_features)}개")
    print(f"    → {interaction_features[:5]}...")
    print(f"  기타: {len(other_features)}개")
    print(f"\n총 Feature 수: {len(feature_cols)}")
    
    # Target 확인
    target_col = "net_passengers"
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = []
    feature_importances_list = []
    
    print(f"\n{'='*50}")
    print(f" Time Series {n_splits}-Fold Cross Validation")
    print(f"{'='*50}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n[Fold {fold + 1}/{n_splits}]")
        print(f"  Train: {len(train_idx):,}행")
        print(f"  Val:   {len(val_idx):,}행")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        
        # 모델 학습
        if use_lightgbm and HAS_LIGHTGBM:
            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=63,
                max_depth=10,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=0.5,
                n_jobs=-1,
                verbose=-1,
                random_state=42
            )
            
            # Early stopping
            es_split = int(len(X_train) * 0.9)
            X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
            y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_val_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        
        print(f"  → RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, R²: {r2:.4f}")
        
        cv_results.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Feature Importance
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
                "fold": fold + 1
            })
            feature_importances_list.append(fi)
    
    # CV 결과 요약
    print(f"\n{'='*50}")
    print(" Cross Validation 결과 요약")
    print(f"{'='*50}")
    
    cv_df = pd.DataFrame(cv_results)
    
    mean_rmse = cv_df["rmse"].mean()
    std_rmse = cv_df["rmse"].std()
    mean_mae = cv_df["mae"].mean()
    std_mae = cv_df["mae"].std()
    mean_r2 = cv_df["r2"].mean()
    std_r2 = cv_df["r2"].std()
    
    print(f"\n  RMSE: {mean_rmse:,.2f} ± {std_rmse:,.2f}")
    print(f"  MAE:  {mean_mae:,.2f} ± {std_mae:,.2f}")
    print(f"  R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    
    # Feature Importance 평균
    if feature_importances_list:
        all_fi = pd.concat(feature_importances_list)
        mean_fi = all_fi.groupby("feature")["importance"].mean().sort_values(ascending=False)
        
        print(f"\n{'='*50}")
        print(" Feature Importance (평균)")
        print(f"{'='*50}")
        
        total_imp = mean_fi.sum()
        cumsum = 0
        for i, (feat, imp) in enumerate(mean_fi.head(30).items()):
            pct = imp / total_imp * 100
            cumsum += pct
            print(f"  {feat:40} {pct:6.2f}% (누적: {cumsum:5.1f}%)")
        
        # Feature Importance 저장
        fi_df = pd.DataFrame({
            "feature": mean_fi.index,
            "importance": mean_fi.values,
            "importance_pct": mean_fi.values / total_imp * 100
        })
        fi_df["cumulative_pct"] = fi_df["importance_pct"].cumsum()
        fi_df.to_csv(output_dir / "feature_importance.csv", index=False)
        
        # 그룹별 중요도
        print(f"\n[그룹별 Feature 중요도]")
        print("-" * 40)
        
        groups = {
            "시간": time_features,
            "위치": location_features,
            "날씨": weather_features,
            "이벤트": event_features,
            "교차변수": interaction_features,
            "기타": other_features,
        }
        
        for group_name, group_features in groups.items():
            group_imp = mean_fi[mean_fi.index.isin(group_features)].sum()
            group_pct = group_imp / total_imp * 100 if total_imp > 0 else 0
            print(f"  {group_name:15} {group_pct:6.2f}%")
    
    # 결과 저장
    cv_df.to_csv(output_dir / "cv_results.csv", index=False)
    
    summary = {
        "model": "V6_Pure_External",
        "n_splits": n_splits,
        "feature_count": len(feature_cols),
        "mean_rmse": float(mean_rmse),
        "std_rmse": float(std_rmse),
        "mean_mae": float(mean_mae),
        "std_mae": float(std_mae),
        "mean_r2": float(mean_r2),
        "std_r2": float(std_r2),
        "feature_groups": {
            "시간": len(time_features),
            "위치": len(location_features),
            "날씨": len(weather_features),
            "이벤트": len(event_features),
            "교차변수": len(interaction_features),
            "기타": len(other_features),
        }
    }
    
    import json
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(" V6 Pure External 학습 완료!")
    print(f"{'='*70}")
    
    return {
        "cv_results": cv_df,
        "summary": summary,
        "feature_importance": fi_df if feature_importances_list else None,
        "features_used": feature_cols
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v6_pure_external"
    
    results = train_v6_pure_external(data_path, output_dir, n_splits=5, use_lightgbm=True)








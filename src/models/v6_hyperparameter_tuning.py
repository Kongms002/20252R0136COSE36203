"""
V6: 하이퍼파라미터 튜닝

Optuna를 사용한 베이지안 최적화로 최적의 하이퍼파라미터를 탐색합니다.
TimeSeriesSplit 교차검증을 적용합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable
import json
from datetime import datetime

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb

from .base_model import load_featured_data, get_feature_columns


def create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3
) -> Callable:
    """Optuna objective 함수 생성"""
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42
        }
        
        r2_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            r2_scores.append(r2)
        
        return np.mean(r2_scores)
    
    return objective


def train_v6_hyperparameter_tuning(
    data_path: Path,
    output_dir: Path,
    n_trials: int = 50,
    n_splits: int = 3,
    sample_size: int = None
) -> Dict[str, Any]:
    """
    V6 하이퍼파라미터 튜닝을 실행합니다.
    
    Optuna로 베이지안 최적화를 수행합니다.
    """
    print("=" * 70)
    print(" V6: 하이퍼파라미터 튜닝 (Optuna)")
    print(f" {n_trials} trials, {n_splits}-Fold TimeSeriesSplit")
    print("=" * 70)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 샘플링
    if sample_size and sample_size < len(df):
        df = df.iloc[:sample_size]
        print(f"\n샘플링: {sample_size:,}행 사용")
    
    # Feature 추출
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["net_passengers"]
    
    print(f"\n데이터: {len(X):,}행, {len(feature_cols)} Features")
    
    # Optuna 스터디 생성
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",  # R² 최대화
        sampler=sampler,
        study_name="LightGBM_Tuning"
    )
    
    # 최적화 실행
    print("\n" + "=" * 70)
    print(" 하이퍼파라미터 탐색 중...")
    print("=" * 70)
    
    objective = create_objective(X, y, n_splits)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # LightGBM이 이미 병렬화되어 있음
    )
    
    # 최적 파라미터
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "=" * 70)
    print(" 최적 하이퍼파라미터")
    print("=" * 70)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\n  Best R² (CV): {best_value:.4f}")
    
    # 최적 파라미터로 최종 모델 학습
    print("\n" + "=" * 70)
    print(" 최종 모델 학습 (전체 데이터)")
    print("=" * 70)
    
    # Train/Test 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    final_params = {**best_params, "n_jobs": -1, "verbose": -1, "random_state": 42}
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X_train, y_train)
    
    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\n  Train R²: {train_r2:.4f}, RMSE: {train_rmse:,.2f}")
    print(f"  Test R²:  {test_r2:.4f}, RMSE: {test_rmse:,.2f}")
    
    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 최적 파라미터
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    
    # 모델 저장
    import joblib
    joblib.dump(final_model, output_dir / "best_model.joblib")
    
    # 스터디 결과
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optimization_history.csv", index=False)
    
    # 메트릭
    metrics = {
        "best_cv_r2": best_value,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    # 설정
    config = {
        "experiment": "V6_Hyperparameter_Tuning",
        "n_trials": n_trials,
        "n_splits": n_splits,
        "sample_size": sample_size or len(df),
        "best_params": best_params,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {output_dir}")
    
    return {
        "best_params": best_params,
        "best_cv_r2": best_value,
        "test_r2": test_r2,
        "test_rmse": test_rmse
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v6_hyperparameter_tuning"
    
    results = train_v6_hyperparameter_tuning(
        data_path, output_dir, 
        n_trials=30,  # 빠른 테스트
        sample_size=100000
    )








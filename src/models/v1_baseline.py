"""
V1: Baseline 모델

LightGBM을 사용한 기본 모델입니다.
최소한의 하이퍼파라미터 튜닝으로 빠르게 베이스라인 성능을 확인합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[경고] LightGBM이 설치되지 않았습니다. RandomForest를 대신 사용합니다.")

from sklearn.ensemble import RandomForestRegressor

from .base_model import (
    BaseModel, 
    ModelConfig, 
    get_feature_columns, 
    load_featured_data
)
from ..analysis.feature_importance import analyze_feature_importance, analyze_feature_groups


def train_v1_baseline(
    data_path: Path,
    output_dir: Path,
    use_lightgbm: bool = True
) -> BaseModel:
    """
    V1 Baseline 모델을 학습합니다.
    
    Parameters
    ----------
    data_path : Path
        featured_data.csv 경로
    output_dir : Path
        결과 저장 디렉토리
    use_lightgbm : bool
        LightGBM 사용 여부 (False면 RandomForest)
    
    Returns
    -------
    BaseModel
        학습된 모델
    """
    print("=" * 70)
    print(" V1 Baseline 모델 학습")
    print("=" * 70)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # Feature 컬럼 추출
    feature_cols = get_feature_columns(df)
    print(f"\n사용 Feature 수: {len(feature_cols)}")
    
    # 모델 설정
    if use_lightgbm and HAS_LIGHTGBM:
        config = ModelConfig(
            name="V1_Baseline_LightGBM",
            version="v1",
            description="LightGBM 기반 베이스라인 모델",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42
            }
        )
    else:
        config = ModelConfig(
            name="V1_Baseline_RandomForest",
            version="v1",
            description="RandomForest 기반 베이스라인 모델",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "n_jobs": -1,
                "random_state": 42
            }
        )
    
    # 모델 객체 생성
    model_wrapper = BaseModel(config)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = model_wrapper.prepare_data(df)
    
    # 모델 생성 및 학습
    print(f"\n모델 학습 시작: {config.name}")
    print("-" * 50)
    
    if use_lightgbm and HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(**config.hyperparameters)
        
        # Early stopping을 위한 validation set
        X_tr, X_val, y_tr, y_val = X_train.iloc[:-10000], X_train.iloc[-10000:], \
                                    y_train.iloc[:-10000], y_train.iloc[-10000:]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        print(f"  - Best iteration: {model.best_iteration_}")
    else:
        model = RandomForestRegressor(**config.hyperparameters)
        model.fit(X_train, y_train)
    
    model_wrapper.model = model
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 평가
    model_wrapper.metrics_train = model_wrapper.evaluate(y_train, y_train_pred)
    model_wrapper.metrics_test = model_wrapper.evaluate(y_test, y_test_pred)
    
    model_wrapper.print_metrics(model_wrapper.metrics_train, "Train")
    model_wrapper.print_metrics(model_wrapper.metrics_test, "Test")
    
    # Feature Importance 분석
    print("\n" + "=" * 50)
    model_wrapper.feature_importance = analyze_feature_importance(
        model, feature_cols, top_n=20, output_dir=output_dir
    )
    
    # 그룹별 중요도 분석
    if model_wrapper.feature_importance is not None:
        analyze_feature_groups(model_wrapper.feature_importance)
    
    # 저장
    print("\n" + "=" * 50)
    print("결과 저장 중...")
    model_wrapper.save(output_dir)
    
    # 예측 샘플 저장
    predictions_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_test_pred,
        "error": y_test.values - y_test_pred
    })
    predictions_df.head(1000).to_csv(output_dir / "predictions_sample.csv", index=False)
    
    print("\n" + "=" * 70)
    print(" V1 Baseline 학습 완료!")
    print("=" * 70)
    
    return model_wrapper


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v1"
    
    model = train_v1_baseline(data_path, output_dir, use_lightgbm=True)








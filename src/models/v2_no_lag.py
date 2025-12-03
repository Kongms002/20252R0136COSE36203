"""
V2: No-Lag 모델

Lag/Rolling feature를 제외하고 순수 외부 요인(시간, 날씨, 이벤트)만으로 학습합니다.
외부 요인이 승객 수요에 미치는 영향을 분석하기 위한 모델입니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import RandomForestRegressor

from .base_model import (
    BaseModel, 
    ModelConfig, 
    load_featured_data
)
from ..analysis.feature_importance import analyze_feature_importance, analyze_feature_groups


def get_no_lag_features(all_features: List[str]) -> List[str]:
    """Lag/Rolling feature를 제외한 feature 목록 반환"""
    lag_keywords = [
        "lag_", "rolling_", "diff_", "pct_", 
        "_mean_승차", "_mean_하차", "_mean_net"
    ]
    
    no_lag_features = []
    for f in all_features:
        if not any(kw in f for kw in lag_keywords):
            no_lag_features.append(f)
    
    return no_lag_features


def train_v2_no_lag(
    data_path: Path,
    output_dir: Path,
    use_lightgbm: bool = True
) -> BaseModel:
    """
    V2 No-Lag 모델을 학습합니다.
    
    Lag/Rolling feature 없이 순수 외부 요인만으로 학습하여
    시간, 날씨, 이벤트의 영향을 분석합니다.
    
    Parameters
    ----------
    data_path : Path
        featured_data.csv 경로
    output_dir : Path
        결과 저장 디렉토리
    use_lightgbm : bool
        LightGBM 사용 여부
    
    Returns
    -------
    BaseModel
        학습된 모델
    """
    print("=" * 70)
    print(" V2 No-Lag 모델 학습")
    print(" (Lag/Rolling feature 제외 - 순수 외부 요인 분석)")
    print("=" * 70)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 모든 Feature 컬럼 추출
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    all_features = [c for c in df.columns if c not in exclude_cols]
    
    # Lag feature 제외
    feature_cols = get_no_lag_features(all_features)
    
    print(f"\n전체 Feature 수: {len(all_features)}")
    print(f"사용 Feature 수 (Lag 제외): {len(feature_cols)}")
    print(f"제외된 Lag Feature 수: {len(all_features) - len(feature_cols)}")
    
    # 모델 설정
    if use_lightgbm and HAS_LIGHTGBM:
        config = ModelConfig(
            name="V2_NoLag_LightGBM",
            version="v2",
            description="Lag/Rolling feature 제외, 순수 외부 요인만 사용",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "max_depth": 10,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42
            }
        )
    else:
        config = ModelConfig(
            name="V2_NoLag_RandomForest",
            version="v2",
            description="Lag/Rolling feature 제외, 순수 외부 요인만 사용",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 500,
                "max_depth": 20,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
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
        
        # Early stopping
        X_tr, X_val = X_train.iloc[:-10000], X_train.iloc[-10000:]
        y_tr, y_val = y_train.iloc[:-10000], y_train.iloc[-10000:]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
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
        model, feature_cols, top_n=25, output_dir=output_dir
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
    print(" V2 No-Lag 학습 완료!")
    print("=" * 70)
    
    return model_wrapper


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v2"
    
    model = train_v2_no_lag(data_path, output_dir, use_lightgbm=True)








"""
모델 베이스 클래스 및 공통 유틸리티

모든 모델 버전에서 공통으로 사용하는 기능을 정의합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelConfig:
    """모델 설정"""
    name: str
    version: str
    description: str = ""
    features: List[str] = field(default_factory=list)
    target: str = "net_passengers"
    test_size: float = 0.2
    random_state: int = 42
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """모델 평가 지표"""
    rmse: float
    mae: float
    r2: float
    mape: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape
        }


class BaseModel:
    """모델 베이스 클래스"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.metrics_train: Optional[ModelMetrics] = None
        self.metrics_test: Optional[ModelMetrics] = None
        self.feature_importance: Optional[pd.DataFrame] = None
    
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """데이터를 학습/테스트로 분할"""
        
        feature_cols = self.config.features
        target_col = self.config.target
        
        # Feature 컬럼 확인
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"[경고] 누락된 Feature: {missing_features[:5]}...")
            feature_cols = [f for f in feature_cols if f in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 결측치 처리
        X = X.fillna(0)
        
        # 무한대 처리
        X = X.replace([np.inf, -np.inf], 0)
        
        # 시간순 분할 (시계열 데이터이므로)
        split_idx = int(len(X) * (1 - self.config.test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"데이터 분할 완료:")
        print(f"  - Train: {len(X_train):,}행")
        print(f"  - Test: {len(X_test):,}행")
        print(f"  - Feature 수: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> ModelMetrics:
        """모델 평가"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE 계산 (0이 아닌 값에 대해서만)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = None
        
        return ModelMetrics(rmse=rmse, mae=mae, r2=r2, mape=mape)
    
    def print_metrics(self, metrics: ModelMetrics, dataset_name: str = ""):
        """평가 결과 출력"""
        print(f"\n[{dataset_name} 평가 결과]")
        print("-" * 40)
        print(f"  RMSE: {metrics.rmse:,.2f}")
        print(f"  MAE:  {metrics.mae:,.2f}")
        print(f"  R²:   {metrics.r2:.4f}")
        if metrics.mape:
            print(f"  MAPE: {metrics.mape:.2f}%")
    
    def save(self, output_dir: Path):
        """모델 및 결과 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        if self.model:
            model_path = output_dir / "model.joblib"
            joblib.dump(self.model, model_path)
            print(f"모델 저장: {model_path}")
        
        # 설정 저장
        config_path = output_dir / "config.json"
        config_dict = {
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "target": self.config.target,
            "test_size": self.config.test_size,
            "random_state": self.config.random_state,
            "hyperparameters": self.config.hyperparameters,
            "feature_count": len(self.config.features),
            "timestamp": datetime.now().isoformat()
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        print(f"설정 저장: {config_path}")
        
        # 메트릭 저장
        metrics_path = output_dir / "metrics.json"
        metrics_dict = {
            "train": self.metrics_train.to_dict() if self.metrics_train else None,
            "test": self.metrics_test.to_dict() if self.metrics_test else None
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
        print(f"메트릭 저장: {metrics_path}")
        
        # Feature Importance 저장
        if self.feature_importance is not None:
            fi_path = output_dir / "feature_importance.csv"
            self.feature_importance.to_csv(fi_path, index=False)
            print(f"Feature Importance 저장: {fi_path}")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """데이터프레임에서 학습용 Feature 컬럼 추출"""
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    return [c for c in df.columns if c not in exclude_cols]


def load_featured_data(data_path: Path) -> pd.DataFrame:
    """Feature가 생성된 데이터 로드"""
    print(f"데이터 로드: {data_path}")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  - 행 수: {len(df):,}")
    print(f"  - 컬럼 수: {len(df.columns)}")
    return df








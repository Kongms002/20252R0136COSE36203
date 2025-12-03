"""
Feature Engineering 파이프라인

모든 파생변수 생성을 통합하여 실행합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from .time_features import create_time_features, get_time_feature_names
from .weather_features import create_weather_features, get_weather_feature_names
from .lag_features import create_lag_features, get_lag_feature_names
from .event_features import create_event_features, get_event_feature_names
from .interaction_features import create_interaction_features, get_interaction_feature_names


def create_all_features(
    df: pd.DataFrame,
    include_lag: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    모든 파생변수를 순차적으로 생성합니다.
    
    생성 순서:
    1. 시간 Feature (~22개)
    2. 날씨 Feature (~17개)
    3. Lag/Rolling Feature (~40개) - 선택적
    4. 이벤트 Feature (~8개)
    5. 교차 Feature (~14개)
    
    총 약 100+ 파생변수 생성
    
    Parameters
    ----------
    df : pd.DataFrame
        전처리된 데이터프레임 (merged_data.csv)
    include_lag : bool
        Lag feature 포함 여부 (대용량 데이터에서 느릴 수 있음)
    verbose : bool
        진행 상황 출력 여부
    
    Returns
    -------
    pd.DataFrame
        모든 feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    if verbose:
        print("=" * 70)
        print(" Feature Engineering 시작")
        print("=" * 70)
        print(f"입력 데이터: {result.shape[0]:,}행, {result.shape[1]}컬럼\n")
    
    # 1. 시간 Feature
    if verbose:
        print("[1/5] 시간 Feature 생성 중...")
    result = create_time_features(result)
    
    # 2. 날씨 Feature
    if verbose:
        print("[2/5] 날씨 Feature 생성 중...")
    result = create_weather_features(result)
    
    # 3. Lag/Rolling Feature (선택적)
    if include_lag:
        if verbose:
            print("[3/5] Lag/Rolling Feature 생성 중... (시간이 걸릴 수 있습니다)")
        result = create_lag_features(result)
    else:
        if verbose:
            print("[3/5] Lag Feature 건너뜀 (include_lag=False)")
    
    # 4. 이벤트 Feature
    if verbose:
        print("[4/5] 이벤트 Feature 생성 중...")
    result = create_event_features(result)
    
    # 5. 교차 Feature
    if verbose:
        print("[5/5] 교차 Feature 생성 중...")
    result = create_interaction_features(result)
    
    # 결과 요약
    if verbose:
        print()
        print("=" * 70)
        print(" Feature Engineering 완료")
        print("=" * 70)
        print(f"출력 데이터: {result.shape[0]:,}행, {result.shape[1]}컬럼")
        print(f"생성된 feature 수: {result.shape[1] - df.shape[1]}개")
        print(f"메모리 사용량: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return result


def get_all_feature_names(include_lag: bool = True) -> List[str]:
    """
    모든 생성된 feature의 컬럼명 목록을 반환합니다.
    
    Parameters
    ----------
    include_lag : bool
        Lag feature 포함 여부
    
    Returns
    -------
    List[str]
        Feature 컬럼명 목록
    """
    features = []
    features.extend(get_time_feature_names())
    features.extend(get_weather_feature_names())
    if include_lag:
        features.extend(get_lag_feature_names())
    features.extend(get_event_feature_names())
    features.extend(get_interaction_feature_names())
    return features


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    데이터프레임에서 Feature 컬럼과 Target 컬럼을 분리합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature가 생성된 데이터프레임
    
    Returns
    -------
    Tuple[List[str], str]
        (Feature 컬럼 목록, Target 컬럼명)
    """
    target_col = "net_passengers"
    
    # 제외할 컬럼 (ID성 컬럼, 원본 타겟 관련)
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return feature_cols, target_col


def prepare_train_data(
    data_path: Path,
    output_path: Optional[Path] = None,
    include_lag: bool = True
) -> pd.DataFrame:
    """
    학습용 데이터를 준비합니다.
    
    Parameters
    ----------
    data_path : Path
        merged_data.csv 경로
    output_path : Optional[Path]
        출력 경로 (지정 시 저장)
    include_lag : bool
        Lag feature 포함 여부
    
    Returns
    -------
    pd.DataFrame
        학습 준비가 완료된 데이터프레임
    """
    # 데이터 로드
    print(f"데이터 로드 중: {data_path}")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Feature 생성
    result = create_all_features(df, include_lag=include_lag)
    
    # 저장
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"저장 완료: {output_path}")
    
    return result


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "merged_data.csv"
    output_path = project_root / "outputs" / "featured_data.csv"
    
    # 전체 파이프라인 실행
    result = prepare_train_data(data_path, output_path, include_lag=True)
    
    # Feature 목록 출력
    print("\n" + "=" * 70)
    print(" 생성된 Feature 목록")
    print("=" * 70)
    
    feature_cols, target_col = get_feature_columns(result)
    print(f"Target: {target_col}")
    print(f"Feature 수: {len(feature_cols)}")
    print(f"Feature 목록:\n{feature_cols[:20]}...")  # 처음 20개만 출력








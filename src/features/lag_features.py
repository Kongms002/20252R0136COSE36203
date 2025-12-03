"""
Lag 및 Rolling Feature Engineering

시계열 자기상관성을 학습할 수 있도록 지연(lag) 및 이동평균(rolling) 변수를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_lag_features(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    sort_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Lag 및 Rolling 파생변수를 생성합니다.
    
    생성되는 변수 (~40개):
    - lag_1h/2h/3h_승차/하차/net: 1~3시간 전 승하차
    - lag_24h_*: 전날 같은 시간
    - lag_168h_*: 일주일 전 같은 요일/시간 (7일 * 24시간)
    - rolling_3h/6h/12h_mean_*: 이동평균
    - rolling_3h_std_*: 이동표준편차
    - rolling_3h_max/min_*: 이동 최대/최소
    - diff_1h_*: 1시간 변화량
    - pct_change_1h_*: 1시간 변화율
    - 통계 feature: 시간대/요일/월별 평균
    
    Parameters
    ----------
    df : pd.DataFrame
        승차, 하차, net_passengers 컬럼이 있는 데이터프레임
    group_cols : Optional[List[str]]
        그룹화 컬럼 (기본: ["역명", "호선"])
    sort_cols : Optional[List[str]]
        정렬 컬럼 (기본: ["Date", "Hour"])
    
    Returns
    -------
    pd.DataFrame
        Lag feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    if group_cols is None:
        group_cols = ["역명", "호선"]
    if sort_cols is None:
        sort_cols = ["Date", "Hour"]
    
    # 정렬 (시간순)
    result = result.sort_values(group_cols + sort_cols).reset_index(drop=True)
    
    # 대상 컬럼
    target_cols = ["승차", "하차", "net_passengers"]
    
    # 그룹별 처리
    for col in target_cols:
        if col not in result.columns:
            continue
            
        grouped = result.groupby(group_cols)[col]
        
        # 1. Lag Features (1h, 2h, 3h, 24h)
        result[f"lag_1h_{col}"] = grouped.shift(1)
        result[f"lag_2h_{col}"] = grouped.shift(2)
        result[f"lag_3h_{col}"] = grouped.shift(3)
        
        # 24시간 전 (하루 전 같은 시간) - 20개 시간대 기준
        # 실제로는 Date/Hour로 계산해야 정확하지만, 간단히 20으로 근사
        result[f"lag_24h_{col}"] = grouped.shift(20)
        
        # 2. Rolling Features
        result[f"rolling_3h_mean_{col}"] = grouped.transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        result[f"rolling_6h_mean_{col}"] = grouped.transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        result[f"rolling_12h_mean_{col}"] = grouped.transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        
        result[f"rolling_3h_std_{col}"] = grouped.transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        result[f"rolling_3h_max_{col}"] = grouped.transform(
            lambda x: x.rolling(window=3, min_periods=1).max()
        )
        result[f"rolling_3h_min_{col}"] = grouped.transform(
            lambda x: x.rolling(window=3, min_periods=1).min()
        )
        
        # 3. Diff Features
        result[f"diff_1h_{col}"] = grouped.diff(1)
        
        # 4. Percentage Change
        result[f"pct_change_1h_{col}"] = grouped.pct_change(1)
    
    # 5. 통계 기반 Feature (시간대별, 요일별 평균)
    if "Hour" in result.columns and "승차" in result.columns:
        hour_mean = result.groupby(["역명", "Hour"])["승차"].transform("mean")
        result["time_mean_승차"] = hour_mean
        
    if "요일" in result.columns and "승차" in result.columns:
        weekday_mean = result.groupby(["역명", "요일"])["승차"].transform("mean")
        result["weekday_mean_승차"] = weekday_mean
    
    if "Month" in result.columns and "승차" in result.columns:
        month_mean = result.groupby(["역명", "Month"])["승차"].transform("mean")
        result["month_mean_승차"] = month_mean
    
    # 6. 결측치 처리 (shift로 인한 NaN)
    # 첫 몇 행은 lag가 없으므로 0으로 대체하거나 제거
    lag_cols = [c for c in result.columns if c.startswith(("lag_", "rolling_", "diff_", "pct_"))]
    for col in lag_cols:
        result[col] = result[col].fillna(0)
    
    # inf 처리 (pct_change에서 0으로 나눌 때)
    result = result.replace([np.inf, -np.inf], 0)
    
    print(f"[Lag Feature] 생성 완료: {len(lag_cols)}개 변수 추가")
    
    return result


def create_weekly_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    일주일 전 같은 요일/시간대의 값을 가져옵니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        Date, Hour, 역명 컬럼이 있는 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        주간 lag feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 7일 전 날짜 계산
    result["Date_7d_ago"] = pd.to_datetime(result["Date"]) - pd.Timedelta(days=7)
    
    # 7일 전 데이터와 병합
    cols_to_merge = ["Date", "Hour", "역명", "호선", "승차", "하차", "net_passengers"]
    cols_to_merge = [c for c in cols_to_merge if c in result.columns]
    
    df_7d = result[cols_to_merge].copy()
    df_7d = df_7d.rename(columns={
        "Date": "Date_7d_ago",
        "승차": "lag_7d_승차",
        "하차": "lag_7d_하차",
        "net_passengers": "lag_7d_net_passengers"
    })
    
    result = result.merge(
        df_7d[["Date_7d_ago", "Hour", "역명", "호선", "lag_7d_승차", "lag_7d_하차", "lag_7d_net_passengers"]],
        on=["Date_7d_ago", "Hour", "역명", "호선"],
        how="left"
    )
    
    # 정리
    result = result.drop(columns=["Date_7d_ago"])
    
    # 결측치 처리
    for col in ["lag_7d_승차", "lag_7d_하차", "lag_7d_net_passengers"]:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    
    return result


def get_lag_feature_names() -> List[str]:
    """Lag feature 컬럼명 목록 반환"""
    features = []
    for col in ["승차", "하차", "net_passengers"]:
        features.extend([
            f"lag_1h_{col}", f"lag_2h_{col}", f"lag_3h_{col}", f"lag_24h_{col}",
            f"rolling_3h_mean_{col}", f"rolling_6h_mean_{col}", f"rolling_12h_mean_{col}",
            f"rolling_3h_std_{col}", f"rolling_3h_max_{col}", f"rolling_3h_min_{col}",
            f"diff_1h_{col}", f"pct_change_1h_{col}",
        ])
    features.extend(["time_mean_승차", "weekday_mean_승차", "month_mean_승차"])
    return features


if __name__ == "__main__":
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "outputs" / "merged_data.csv"
    df = pd.read_csv(data_path, nrows=5000)
    df["Date"] = pd.to_datetime(df["Date"])
    
    result = create_lag_features(df)
    print(f"\n결과 shape: {result.shape}")








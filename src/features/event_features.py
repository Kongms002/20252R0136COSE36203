"""
이벤트 관련 Feature Engineering

문화행사/이벤트가 승객 수요에 미치는 영향을 학습할 수 있도록 파생변수를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import List


def create_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    이벤트 관련 파생변수를 생성합니다.
    
    기존 컬럼 (이미 존재해야 함):
    - is_event_nearby: 해당 역에 이벤트 여부
    - event_count: 해당 역의 이벤트 수
    
    추가 생성되는 변수 (~6개):
    - event_intensity: 이벤트 강도 (count 기반)
    - event_nearby_weekend: 주말 + 이벤트 조합
    - event_nearby_evening: 저녁시간 + 이벤트 조합
    - event_nearby_rushhour: 러시아워 + 이벤트 조합
    - has_multiple_events: 복수 이벤트 여부
    - event_rolling_7d: 최근 7일간 이벤트 발생 빈도
    
    Parameters
    ----------
    df : pd.DataFrame
        is_event_nearby, event_count 컬럼이 있는 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        이벤트 feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 필수 컬럼 확인
    if "is_event_nearby" not in result.columns:
        print("[경고] is_event_nearby 컬럼이 없습니다. 이벤트 feature를 생성할 수 없습니다.")
        return result
    
    if "event_count" not in result.columns:
        result["event_count"] = result["is_event_nearby"].astype(int)
    
    # 1. 이벤트 강도 (로그 스케일)
    result["event_intensity"] = np.log1p(result["event_count"])
    
    # 2. 복수 이벤트 여부
    result["has_multiple_events"] = (result["event_count"] > 1).astype(int)
    
    # 3. 주말 + 이벤트 조합
    if "Is_Weekend" in result.columns:
        result["event_weekend"] = result["is_event_nearby"] * result["Is_Weekend"]
    
    # 4. 저녁시간 + 이벤트 조합 (18-22시)
    if "Hour" in result.columns:
        is_evening = ((result["Hour"] >= 18) & (result["Hour"] <= 22)).astype(int)
        result["event_evening"] = result["is_event_nearby"] * is_evening
    
    # 5. 러시아워 + 이벤트 조합
    if "Is_RushHour" in result.columns:
        result["event_rushhour"] = result["is_event_nearby"] * result["Is_RushHour"]
    
    # 6. 이벤트 타이밍 (시작/종료 전후)
    # 저녁 공연이 많으므로 17-19시를 이벤트 시작 시간대로 가정
    if "Hour" in result.columns:
        is_event_start_time = ((result["Hour"] >= 17) & (result["Hour"] <= 19)).astype(int)
        result["event_timing"] = result["is_event_nearby"] * is_event_start_time
        
        # 공연 종료 시간대 (21-23시)
        is_event_end_time = ((result["Hour"] >= 21) & (result["Hour"] <= 23)).astype(int)
        result["event_ending"] = result["is_event_nearby"] * is_event_end_time
    
    # 7. 역별 일일 이벤트 수 (그룹 통계)
    if "Date" in result.columns and "역명" in result.columns:
        daily_event = result.groupby(["Date", "역명"])["event_count"].transform("max")
        result["daily_event_count"] = daily_event
    
    print(f"[이벤트 Feature] 생성 완료: {len(get_event_feature_names())}개 변수 추가")
    
    return result


def get_event_feature_names() -> List[str]:
    """이벤트 feature 컬럼명 목록 반환"""
    return [
        "event_intensity",
        "has_multiple_events",
        "event_weekend",
        "event_evening",
        "event_rushhour",
        "event_timing",
        "event_ending",
        "daily_event_count",
    ]


if __name__ == "__main__":
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "outputs" / "merged_data.csv"
    df = pd.read_csv(data_path, nrows=1000)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 시간 feature 먼저 생성 (Is_Weekend, Is_RushHour 필요)
    from time_features import create_time_features
    df = create_time_features(df)
    
    result = create_event_features(df)
    print(f"\n결과 shape: {result.shape}")
    print(f"이벤트 feature: {get_event_feature_names()}")








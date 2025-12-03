"""
교차 변수 (Interaction Features) Engineering

변수 간 상호작용 효과를 학습할 수 있도록 교차 변수를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import List


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    교차 변수 (Interaction Features)를 생성합니다.
    
    생성되는 변수 (~12개):
    - Time_x_Weekday: 시간 × 요일
    - Hour_x_IsWeekend: 시간 × 주말여부
    - Rain_x_RushHour: 강수 × 러시아워
    - Rain_x_Weekend: 강수 × 주말
    - Temp_x_RushHour: 기온 × 러시아워
    - Snow_x_Weekend: 적설 × 주말
    - Event_x_Weekend: 이벤트 × 주말
    - Event_x_Evening: 이벤트 × 저녁시간
    - Transfer_x_RushHour: 환승역 × 러시아워
    - DI_x_RushHour: 불쾌지수 × 러시아워
    - Season_x_Weekend: 계절 × 주말
    - Holiday_x_Event: 공휴일 × 이벤트
    
    Parameters
    ----------
    df : pd.DataFrame
        필요한 기본 feature들이 있는 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        교차 feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 1. 시간 × 요일
    if "Hour" in result.columns and "요일" in result.columns:
        result["Time_x_Weekday"] = result["Hour"] * 10 + result["요일"]
    
    # 2. 시간 × 주말
    if "Hour" in result.columns and "Is_Weekend" in result.columns:
        result["Hour_x_IsWeekend"] = result["Hour"] * result["Is_Weekend"]
    
    # 3. 강수 × 러시아워
    if "Is_Rain" in result.columns and "Is_RushHour" in result.columns:
        result["Rain_x_RushHour"] = result["Is_Rain"] * result["Is_RushHour"]
    
    # 4. 강수 × 주말
    if "Is_Rain" in result.columns and "Is_Weekend" in result.columns:
        result["Rain_x_Weekend"] = result["Is_Rain"] * result["Is_Weekend"]
    
    # 5. 기온 × 러시아워
    if "temperature" in result.columns and "Is_RushHour" in result.columns:
        result["Temp_x_RushHour"] = result["temperature"] * result["Is_RushHour"]
    
    # 6. 적설 × 주말
    if "Is_Snow" in result.columns and "Is_Weekend" in result.columns:
        result["Snow_x_Weekend"] = result["Is_Snow"] * result["Is_Weekend"]
    
    # 7. 이벤트 × 주말
    if "is_event_nearby" in result.columns and "Is_Weekend" in result.columns:
        result["Event_x_Weekend"] = result["is_event_nearby"] * result["Is_Weekend"]
    
    # 8. 이벤트 × 저녁 (18-22시)
    if "is_event_nearby" in result.columns and "Hour" in result.columns:
        is_evening = ((result["Hour"] >= 18) & (result["Hour"] <= 22)).astype(int)
        result["Event_x_Evening"] = result["is_event_nearby"] * is_evening
    
    # 9. 환승역 × 러시아워
    if "is_transfer_station" in result.columns and "Is_RushHour" in result.columns:
        result["Transfer_x_RushHour"] = result["is_transfer_station"] * result["Is_RushHour"]
    
    # 10. 불쾌지수 × 러시아워
    if "discomfort_index" in result.columns and "Is_RushHour" in result.columns:
        result["DI_x_RushHour"] = result["discomfort_index"] * result["Is_RushHour"]
    
    # 11. 계절 × 주말
    if "Season" in result.columns and "Is_Weekend" in result.columns:
        result["Season_x_Weekend"] = result["Season"] * 10 + result["Is_Weekend"]
    
    # 12. 공휴일 × 이벤트
    if "Is_Holiday" in result.columns and "is_event_nearby" in result.columns:
        result["Holiday_x_Event"] = result["Is_Holiday"] * result["is_event_nearby"]
    
    # 13. 극단 날씨 × 러시아워
    if "Is_Extreme_Weather" in result.columns and "Is_RushHour" in result.columns:
        result["Extreme_x_RushHour"] = result["Is_Extreme_Weather"] * result["Is_RushHour"]
    
    # 14. 비 강도 × 러시아워
    if "rainfall" in result.columns and "Is_RushHour" in result.columns:
        result["RainIntensity_x_RushHour"] = result["rainfall"] * result["Is_RushHour"]
    
    print(f"[교차 Feature] 생성 완료: {len(get_interaction_feature_names())}개 변수 추가")
    
    return result


def get_interaction_feature_names() -> List[str]:
    """교차 feature 컬럼명 목록 반환"""
    return [
        "Time_x_Weekday",
        "Hour_x_IsWeekend",
        "Rain_x_RushHour",
        "Rain_x_Weekend",
        "Temp_x_RushHour",
        "Snow_x_Weekend",
        "Event_x_Weekend",
        "Event_x_Evening",
        "Transfer_x_RushHour",
        "DI_x_RushHour",
        "Season_x_Weekend",
        "Holiday_x_Event",
        "Extreme_x_RushHour",
        "RainIntensity_x_RushHour",
    ]


if __name__ == "__main__":
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "outputs" / "merged_data.csv"
    df = pd.read_csv(data_path, nrows=1000)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 필요한 feature 먼저 생성
    from time_features import create_time_features
    from weather_features import create_weather_features
    
    df = create_time_features(df)
    df = create_weather_features(df)
    
    result = create_interaction_features(df)
    print(f"\n결과 shape: {result.shape}")
    print(f"교차 feature: {get_interaction_feature_names()}")








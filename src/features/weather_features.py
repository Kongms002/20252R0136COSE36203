"""
날씨 관련 Feature Engineering

날씨가 승객 수요에 미치는 영향을 학습할 수 있도록 파생변수를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import List


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날씨 관련 파생변수를 생성합니다.
    
    생성되는 변수 (~25개):
    - Temp_Category: 기온 구간 (한파/추움/쌀쌀/적정/더움/폭염)
    - Humidity_Category: 습도 구간
    - Is_Rain, Rain_Category: 강수 여부 및 강도
    - Is_Snow, Snow_Category: 적설 여부 및 강도
    - Weather_Group: 날씨 그룹 (맑음/비/눈/한파/폭염)
    - Weather_Score: 날씨 점수 (외출 적합도)
    - Feels_Like_Temp: 체감온도
    - DI_Category: 불쾌지수 구간
    - Temp_Rolling3h/6h: 기온 이동평균
    - Rain_Rolling3h/6h: 강수량 이동평균
    - Temp_Diff: 기온 변화량
    - Is_Extreme_Weather: 극단 날씨 여부
    
    Parameters
    ----------
    df : pd.DataFrame
        temperature, humidity, rainfall, snowfall, wind_speed, discomfort_index 컬럼이 있는 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        날씨 feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 필수 컬럼 확인
    required_cols = ["temperature", "humidity", "rainfall", "snowfall", "wind_speed", "discomfort_index"]
    missing = [c for c in required_cols if c not in result.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    
    # 1. 기온 카테고리
    result["Temp_Category"] = pd.cut(
        result["temperature"],
        bins=[-np.inf, -10, 0, 10, 20, 28, np.inf],
        labels=[0, 1, 2, 3, 4, 5]  # 한파/추움/쌀쌀/적정/더움/폭염
    ).cat.codes.replace(-1, 3)  # NaN → 적정(3)으로 대체
    
    # 2. 습도 카테고리
    result["Humidity_Category"] = pd.cut(
        result["humidity"],
        bins=[0, 30, 50, 70, 85, 100],
        labels=[0, 1, 2, 3, 4]  # 건조/적정/습함/매우습함/극습
    ).cat.codes.replace(-1, 2)  # NaN → 보통(2)으로 대체
    
    # 3. 강수 여부 및 강도
    result["Is_Rain"] = (result["rainfall"] > 0).astype(int)
    result["Rain_Category"] = pd.cut(
        result["rainfall"],
        bins=[-np.inf, 0, 1, 5, 20, np.inf],
        labels=[0, 1, 2, 3, 4]  # 없음/약간/보통/강함/매우강함
    ).cat.codes.replace(-1, 0)  # NaN → 없음(0)으로 대체
    
    # 4. 적설 여부 및 강도
    result["Is_Snow"] = (result["snowfall"] > 0).astype(int)
    result["Snow_Category"] = pd.cut(
        result["snowfall"],
        bins=[-np.inf, 0, 1, 5, 10, np.inf],
        labels=[0, 1, 2, 3, 4]  # 없음/약간/보통/많음/매우많음
    ).cat.codes.replace(-1, 0)  # NaN → 없음(0)으로 대체
    
    # 5. 날씨 그룹 (복합 조건)
    result["Weather_Group"] = _classify_weather_group(result)
    
    # 6. 날씨 점수 (0-100, 높을수록 외출 적합)
    result["Weather_Score"] = _calculate_weather_score(result)
    
    # 7. 체감온도 (Wind Chill / Heat Index)
    result["Feels_Like_Temp"] = _calculate_feels_like(
        result["temperature"], result["humidity"], result["wind_speed"]
    )
    
    # 8. 불쾌지수 카테고리
    result["DI_Category"] = pd.cut(
        result["discomfort_index"],
        bins=[-np.inf, 68, 75, 80, np.inf],
        labels=[0, 1, 2, 3]  # 쾌적/보통/불쾌/매우불쾌
    ).cat.codes.replace(-1, 1)  # NaN → 보통(1)으로 대체
    
    # 9. 기온 관련 파생 변수
    result["Is_Cold"] = (result["temperature"] < 0).astype(int)
    result["Is_Hot"] = (result["temperature"] > 30).astype(int)
    result["Is_Freezing"] = (result["temperature"] < -10).astype(int)
    
    # 10. 극단 날씨 여부
    result["Is_Extreme_Weather"] = (
        (result["temperature"] < -10) |  # 한파
        (result["temperature"] > 33) |   # 폭염
        (result["rainfall"] > 20) |      # 폭우
        (result["snowfall"] > 10)        # 폭설
    ).astype(int)
    
    # 11. 강수/적설 강도 (연속형)
    result["Precipitation_Intensity"] = result["rainfall"] + result["snowfall"] * 10  # 눈은 가중치
    
    # 12. 풍속 카테고리
    result["Wind_Category"] = pd.cut(
        result["wind_speed"],
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=[0, 1, 2, 3]  # 약함/보통/강함/매우강함
    ).cat.codes.replace(-1, 1)  # NaN → 보통(1)으로 대체
    
    # 13. 강풍 여부
    result["Is_Windy"] = (result["wind_speed"] > 8).astype(int)
    
    print(f"[날씨 Feature] 생성 완료: {len(get_weather_feature_names())}개 변수 추가")
    
    return result


def _classify_weather_group(df: pd.DataFrame) -> pd.Series:
    """날씨 그룹 분류 (0=맑음, 1=비, 2=눈, 3=한파, 4=폭염)"""
    conditions = [
        (df["snowfall"] > 0),                          # 눈
        (df["rainfall"] > 0),                          # 비
        (df["temperature"] < -5),                      # 한파
        (df["temperature"] > 30),                      # 폭염
    ]
    choices = [2, 1, 3, 4]  # 눈, 비, 한파, 폭염
    return pd.Series(np.select(conditions, choices, default=0), index=df.index)


def _calculate_weather_score(df: pd.DataFrame) -> pd.Series:
    """날씨 점수 계산 (0-100, 높을수록 외출 적합)"""
    score = pd.Series(100.0, index=df.index)
    
    # 기온 감점 (15-25도가 최적)
    temp_penalty = np.abs(df["temperature"] - 20) * 2
    score -= temp_penalty.clip(0, 30)
    
    # 강수 감점
    score -= df["rainfall"] * 5
    score -= df["snowfall"] * 10
    
    # 습도 감점 (너무 높거나 낮으면 감점)
    humidity_penalty = np.abs(df["humidity"] - 50) * 0.3
    score -= humidity_penalty
    
    # 풍속 감점
    score -= df["wind_speed"] * 2
    
    return score.clip(0, 100)


def _calculate_feels_like(temp: pd.Series, humidity: pd.Series, wind: pd.Series) -> pd.Series:
    """
    체감온도 계산
    - 13°C 이하: Wind Chill (바람 체감)
    - 27°C 이상: Heat Index (열지수)
    - 그 외: 실제 기온
    """
    feels_like = temp.copy()
    
    # Wind Chill (체감온도 = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16)
    cold_mask = temp <= 13
    if cold_mask.any():
        v_pow = np.power(wind[cold_mask] * 3.6 + 0.1, 0.16)  # m/s -> km/h
        feels_like[cold_mask] = (
            13.12 + 0.6215 * temp[cold_mask] 
            - 11.37 * v_pow 
            + 0.3965 * temp[cold_mask] * v_pow
        )
    
    # Heat Index (간이 공식)
    hot_mask = temp >= 27
    if hot_mask.any():
        rh = humidity[hot_mask]
        t = temp[hot_mask]
        feels_like[hot_mask] = (
            -8.785 + 1.611 * t + 2.339 * rh
            - 0.146 * t * rh - 0.012 * t**2
            - 0.016 * rh**2 + 0.002 * t**2 * rh
            + 0.001 * t * rh**2 - 0.000002 * t**2 * rh**2
        )
    
    return feels_like


def get_weather_feature_names() -> List[str]:
    """날씨 feature 컬럼명 목록 반환"""
    return [
        "Temp_Category", "Humidity_Category",
        "Is_Rain", "Rain_Category",
        "Is_Snow", "Snow_Category",
        "Weather_Group", "Weather_Score",
        "Feels_Like_Temp", "DI_Category",
        "Is_Cold", "Is_Hot", "Is_Freezing",
        "Is_Extreme_Weather", "Precipitation_Intensity",
        "Wind_Category", "Is_Windy",
    ]


if __name__ == "__main__":
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "outputs" / "merged_data.csv"
    df = pd.read_csv(data_path, nrows=1000)
    
    result = create_weather_features(df)
    print(f"\n결과 shape: {result.shape}")
    print(f"날씨 feature: {get_weather_feature_names()}")


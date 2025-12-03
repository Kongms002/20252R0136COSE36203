"""
시간 관련 Feature Engineering

시간적 패턴을 학습할 수 있도록 다양한 시간 파생변수를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import List


# 2024년 대한민국 공휴일 목록
HOLIDAYS_2024 = [
    "2024-01-01",  # 신정
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",  # 설날
    "2024-03-01",  # 삼일절
    "2024-04-10",  # 국회의원선거
    "2024-05-05", "2024-05-06",  # 어린이날, 대체휴일
    "2024-05-15",  # 부처님오신날
    "2024-06-06",  # 현충일
    "2024-08-15",  # 광복절
    "2024-09-16", "2024-09-17", "2024-09-18",  # 추석
    "2024-10-03",  # 개천절
    "2024-10-09",  # 한글날
    "2024-12-25",  # 성탄절
]


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간 관련 파생변수를 생성합니다.
    
    생성되는 변수 (~20개):
    - Time_Sin, Time_Cos: 시간의 주기성 (24시간 주기)
    - 요일, 요일_Sin, 요일_Cos: 요일 패턴 (7일 주기)
    - Is_Weekend: 주말 여부
    - Month, 월_Sin, 월_Cos: 월 패턴 (12개월 주기)
    - DayOfMonth: 월 내 일자
    - WeekOfYear: 연간 주차
    - Quarter: 분기
    - Season: 계절
    - Is_RushHour: 출퇴근 시간
    - Is_LateNight: 심야 시간
    - Is_Holiday: 공휴일 여부
    - 시간대_그룹: 시간대 분류
    - DayOfYear: 연간 일차
    
    Parameters
    ----------
    df : pd.DataFrame
        Date, Hour 컬럼이 있는 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        시간 feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 기본 날짜/시간 컬럼 확인
    if "Date" not in result.columns:
        raise ValueError("Date 컬럼이 필요합니다.")
    if "Hour" not in result.columns:
        raise ValueError("Hour 컬럼이 필요합니다.")
    
    # Date를 datetime으로 변환
    result["Date"] = pd.to_datetime(result["Date"])
    
    # 1. 시간 주기성 (Sin/Cos 변환) - 24시간 주기
    result["Time_Sin"] = np.sin(2 * np.pi * result["Hour"] / 24)
    result["Time_Cos"] = np.cos(2 * np.pi * result["Hour"] / 24)
    
    # 2. 요일 (0=월요일, 6=일요일)
    result["요일"] = result["Date"].dt.dayofweek
    result["요일_Sin"] = np.sin(2 * np.pi * result["요일"] / 7)
    result["요일_Cos"] = np.cos(2 * np.pi * result["요일"] / 7)
    
    # 3. 주말 여부
    result["Is_Weekend"] = (result["요일"] >= 5).astype(int)
    
    # 4. 월 (1-12)
    result["Month"] = result["Date"].dt.month
    result["월_Sin"] = np.sin(2 * np.pi * result["Month"] / 12)
    result["월_Cos"] = np.cos(2 * np.pi * result["Month"] / 12)
    
    # 5. 월 내 일자 (1-31)
    result["DayOfMonth"] = result["Date"].dt.day
    
    # 6. 연간 주차 (1-53)
    result["WeekOfYear"] = result["Date"].dt.isocalendar().week.astype(int)
    
    # 7. 분기 (1-4)
    result["Quarter"] = result["Date"].dt.quarter
    
    # 8. 계절 (1=봄, 2=여름, 3=가을, 4=겨울)
    result["Season"] = result["Month"].apply(_get_season)
    
    # 9. 출퇴근 시간 (7-9, 18-20)
    result["Is_RushHour"] = result["Hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (18 <= h <= 20) else 0
    )
    
    # 10. 심야 시간 (23-5)
    result["Is_LateNight"] = result["Hour"].apply(
        lambda h: 1 if h >= 23 or h <= 5 else 0
    )
    
    # 11. 공휴일 여부
    holidays_set = set(pd.to_datetime(HOLIDAYS_2024))
    result["Is_Holiday"] = result["Date"].isin(holidays_set).astype(int)
    
    # 12. 시간대 그룹 (0=새벽, 1=출근, 2=오전, 3=점심, 4=오후, 5=퇴근, 6=저녁, 7=심야)
    result["시간대_그룹"] = result["Hour"].apply(_get_time_group)
    
    # 13. 연간 일차 (1-366)
    result["DayOfYear"] = result["Date"].dt.dayofyear
    result["DayOfYear_Sin"] = np.sin(2 * np.pi * result["DayOfYear"] / 365)
    result["DayOfYear_Cos"] = np.cos(2 * np.pi * result["DayOfYear"] / 365)
    
    # 14. 출퇴근 방향 추정 (아침=출근방향, 저녁=퇴근방향)
    result["Is_MorningRush"] = ((result["Hour"] >= 7) & (result["Hour"] <= 9)).astype(int)
    result["Is_EveningRush"] = ((result["Hour"] >= 18) & (result["Hour"] <= 20)).astype(int)
    
    print(f"[시간 Feature] 생성 완료: {_count_new_features(df, result)}개 변수 추가")
    
    return result


def _get_season(month: int) -> int:
    """월에서 계절을 반환 (1=봄, 2=여름, 3=가을, 4=겨울)"""
    if month in [3, 4, 5]:
        return 1  # 봄
    elif month in [6, 7, 8]:
        return 2  # 여름
    elif month in [9, 10, 11]:
        return 3  # 가을
    else:
        return 4  # 겨울


def _get_time_group(hour: int) -> int:
    """시간을 시간대 그룹으로 분류"""
    if 0 <= hour < 6:
        return 0  # 새벽
    elif 6 <= hour < 9:
        return 1  # 출근
    elif 9 <= hour < 12:
        return 2  # 오전
    elif 12 <= hour < 14:
        return 3  # 점심
    elif 14 <= hour < 18:
        return 4  # 오후
    elif 18 <= hour < 21:
        return 5  # 퇴근
    elif 21 <= hour < 24:
        return 6  # 저녁
    else:
        return 7  # 심야


def _count_new_features(original: pd.DataFrame, new: pd.DataFrame) -> int:
    """새로 추가된 컬럼 수 계산"""
    return len(new.columns) - len(original.columns)


def get_time_feature_names() -> List[str]:
    """시간 feature 컬럼명 목록 반환"""
    return [
        "Time_Sin", "Time_Cos",
        "요일", "요일_Sin", "요일_Cos",
        "Is_Weekend",
        "Month", "월_Sin", "월_Cos",
        "DayOfMonth", "WeekOfYear", "Quarter", "Season",
        "Is_RushHour", "Is_LateNight", "Is_Holiday",
        "시간대_그룹",
        "DayOfYear", "DayOfYear_Sin", "DayOfYear_Cos",
        "Is_MorningRush", "Is_EveningRush",
    ]


if __name__ == "__main__":
    # 테스트
    import pandas as pd
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "outputs" / "merged_data.csv"
    df = pd.read_csv(data_path, nrows=1000)
    df["Date"] = pd.to_datetime(df["Date"])
    
    result = create_time_features(df)
    print(f"\n결과 shape: {result.shape}")
    print(f"새로운 컬럼: {get_time_feature_names()}")








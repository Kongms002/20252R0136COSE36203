"""
문화행사 데이터 전처리 모듈

문화행사 데이터에서 역명을 추출하고, 날짜-역 매핑 테이블을 생성합니다.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Set, Tuple
from datetime import datetime, timedelta


def extract_station_names(subway_str: str) -> List[str]:
    """
    지하철 컬럼에서 역명을 추출합니다.
    
    예시:
    - '"월드컵경기장역"' → ['월드컵경기장']
    - '"안국역", "종로3가역"' → ['안국', '종로3가']
    - '3호선 경복궁 (5번출구) , 5호선 광화문 (2번출구)' → ['경복궁', '광화문']
    
    Parameters
    ----------
    subway_str : str
        지하철 컬럼 값
    
    Returns
    -------
    List[str]
        추출된 역명 리스트 ('역' 제거됨)
    """
    if pd.isna(subway_str) or not subway_str:
        return []
    
    stations = []
    
    # 패턴 1: "역명역" 형태 (큰따옴표로 감싸진 경우)
    pattern1 = r'"([^"]+역)"'
    matches1 = re.findall(pattern1, subway_str)
    for match in matches1:
        station = match.replace("역", "")  # '역' 제거
        if station:
            stations.append(station)
    
    # 패턴 2: 호선 역명 (출구) 형태
    pattern2 = r'\d호선\s+([가-힣0-9]+)\s*\('
    matches2 = re.findall(pattern2, subway_str)
    for match in matches2:
        station = match.replace("역", "")
        if station and station not in stations:
            stations.append(station)
    
    return stations


def preprocess_events(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    문화행사 데이터를 전처리합니다.
    
    전처리 내용:
    1. 지하철 컬럼에서 역명 추출
    2. 행사 기간 파싱 (startDate ~ EndDate)
    3. 날짜-역 매핑 테이블 생성 (날짜별, 역별 이벤트 여부)
    
    Parameters
    ----------
    event_df : pd.DataFrame
        원본 문화행사 데이터
    
    Returns
    -------
    pd.DataFrame
        날짜-역별 이벤트 정보 (Date, station_name, event_count)
    """
    df = event_df.copy()
    
    # 1. 역명 추출
    df["stations"] = df["지하철"].apply(extract_station_names)
    
    # 역이 있는 행만 필터링
    df_with_stations = df[df["stations"].apply(len) > 0].copy()
    print(f"[이벤트 전처리] 역 정보가 있는 행사: {len(df_with_stations):,} / {len(df):,}")
    
    # 2. 날짜 파싱
    df_with_stations["start_date"] = pd.to_datetime(
        df_with_stations["startDate"], errors="coerce"
    )
    df_with_stations["end_date"] = pd.to_datetime(
        df_with_stations["EndDate"], errors="coerce"
    )
    
    # 2024년 데이터만 필터링 (2024년과 겹치는 행사)
    year_start = pd.Timestamp("2024-01-01")
    year_end = pd.Timestamp("2024-12-31")
    
    df_2024 = df_with_stations[
        (df_with_stations["start_date"] <= year_end) & 
        (df_with_stations["end_date"] >= year_start)
    ].copy()
    
    # 날짜 범위를 2024년으로 클리핑
    df_2024["start_date"] = df_2024["start_date"].clip(lower=year_start)
    df_2024["end_date"] = df_2024["end_date"].clip(upper=year_end)
    
    print(f"  - 2024년 행사: {len(df_2024):,}")
    
    # 3. 날짜-역 매핑 테이블 생성
    date_station_events = []
    
    for _, row in df_2024.iterrows():
        start = row["start_date"]
        end = row["end_date"]
        stations = row["stations"]
        
        if pd.isna(start) or pd.isna(end):
            continue
        
        # 날짜 범위 생성
        date_range = pd.date_range(start=start, end=end, freq="D")
        
        for date in date_range:
            for station in stations:
                date_station_events.append({
                    "Date": date,
                    "station_name": station,
                    "event_count": 1
                })
    
    # DataFrame으로 변환
    events_expanded = pd.DataFrame(date_station_events)
    
    if len(events_expanded) == 0:
        print("  - 경고: 확장된 이벤트가 없습니다.")
        return pd.DataFrame(columns=["Date", "station_name", "event_count"])
    
    # 날짜-역별 이벤트 수 집계
    events_grouped = events_expanded.groupby(
        ["Date", "station_name"]
    )["event_count"].sum().reset_index()
    
    print(f"  - 날짜-역 조합: {len(events_grouped):,}")
    print(f"  - 고유 역 수: {events_grouped['station_name'].nunique()}")
    
    # 역명 목록 출력
    unique_stations = sorted(events_grouped["station_name"].unique())
    print(f"  - 이벤트가 있는 역: {unique_stations[:10]}{'...' if len(unique_stations) > 10 else ''}")
    
    return events_grouped


def create_event_features(
    boarding_df: pd.DataFrame,
    events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    승하차 데이터에 이벤트 feature를 추가합니다.
    
    추가되는 컬럼:
    - is_event_nearby: 해당 역에 이벤트가 있는지 여부 (0/1)
    - event_count: 해당 역의 이벤트 수
    
    Parameters
    ----------
    boarding_df : pd.DataFrame
        승하차 데이터 (Date, 역명 컬럼 필요)
    events_df : pd.DataFrame
        이벤트 데이터 (Date, station_name, event_count)
    
    Returns
    -------
    pd.DataFrame
        이벤트 feature가 추가된 승하차 데이터
    """
    df = boarding_df.copy()
    
    # 역명 정규화 (공백 제거, '역' 제거)
    df["station_name_norm"] = df["역명"].str.replace("역", "", regex=False).str.strip()
    
    events_norm = events_df.copy()
    events_norm["station_name_norm"] = events_norm["station_name"].str.strip()
    
    # 날짜-역명으로 병합
    df = df.merge(
        events_norm[["Date", "station_name_norm", "event_count"]],
        on=["Date", "station_name_norm"],
        how="left"
    )
    
    # 결측치 처리 (이벤트 없음 = 0)
    df["event_count"] = df["event_count"].fillna(0).astype(int)
    df["is_event_nearby"] = (df["event_count"] > 0).astype(int)
    
    # 정규화 컬럼 제거
    df = df.drop(columns=["station_name_norm"])
    
    event_count = df["is_event_nearby"].sum()
    print(f"[이벤트 Feature] 이벤트가 있는 행: {event_count:,} / {len(df):,} ({event_count/len(df)*100:.2f}%)")
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    from load_data import load_event_data, load_boarding_data
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("=" * 60)
    event_raw = load_event_data(data_dir)
    print()
    
    print("=" * 60)
    events_processed = preprocess_events(event_raw)
    print()
    print(events_processed.head(20))
    
    print("\n" + "=" * 60)
    # 테스트: 승하차 데이터와 병합
    boarding_df = load_boarding_data(data_dir)
    boarding_with_events = create_event_features(boarding_df, events_processed)
    print()
    print(boarding_with_events[boarding_with_events["is_event_nearby"] == 1].head(10))








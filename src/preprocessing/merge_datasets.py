"""
데이터 병합 모듈

승하차, 날씨, 이벤트 데이터를 하나의 학습용 데이터셋으로 병합합니다.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from .load_data import load_boarding_data, load_weather_data, load_event_data
from .weather_preprocessor import preprocess_weather
from .event_preprocessor import preprocess_events, create_event_features


def merge_all_datasets(
    data_dir: Path,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    모든 데이터셋을 병합하여 학습용 데이터를 생성합니다.
    
    병합 과정:
    1. 승하차 데이터 로드 및 기본 전처리
    2. 날씨 데이터 전처리 및 병합 (Date, Hour 기준)
    3. 이벤트 데이터 전처리 및 병합 (Date, 역명 기준)
    
    Parameters
    ----------
    data_dir : Path
        data 폴더 경로
    output_path : Optional[Path]
        결과 저장 경로 (지정 시 CSV로 저장)
    
    Returns
    -------
    pd.DataFrame
        병합된 학습용 데이터
    """
    print("=" * 70)
    print(" 데이터 병합 시작")
    print("=" * 70)
    print()
    
    # 1. 승하차 데이터 로드
    print("[Step 1] 승하차 데이터 로드")
    boarding_df = load_boarding_data(data_dir)
    print()
    
    # 2. 날씨 데이터 전처리 및 병합
    print("[Step 2] 날씨 데이터 전처리")
    weather_raw = load_weather_data(data_dir)
    weather_df = preprocess_weather(weather_raw)
    print()
    
    print("[Step 3] 승하차 + 날씨 병합")
    merged_df = boarding_df.merge(
        weather_df,
        on=["Date", "Hour"],
        how="left"
    )
    
    # 날씨 결측치 확인
    weather_cols = ["temperature", "rainfall", "humidity", "wind_speed", "snowfall", "discomfort_index"]
    missing_weather = merged_df[weather_cols].isna().sum()
    print(f"  - 병합 후 행 수: {len(merged_df):,}")
    print(f"  - 날씨 결측치:\n{missing_weather[missing_weather > 0]}")
    
    # 날씨 결측치 처리 (Date 범위 밖의 경우)
    for col in weather_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    print()
    
    # 3. 이벤트 데이터 전처리 및 병합
    print("[Step 4] 이벤트 데이터 전처리")
    event_raw = load_event_data(data_dir)
    events_df = preprocess_events(event_raw)
    print()
    
    print("[Step 5] 승하차 + 날씨 + 이벤트 병합")
    final_df = create_event_features(merged_df, events_df)
    print()
    
    # 최종 데이터 요약
    print("=" * 70)
    print(" 병합 완료")
    print("=" * 70)
    print(f"최종 데이터 형태: {final_df.shape}")
    print(f"메모리 사용량: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    print("컬럼 목록:")
    for i, col in enumerate(final_df.columns):
        dtype = final_df[col].dtype
        null_count = final_df[col].isna().sum()
        print(f"  {i+1:2}. {col}: {dtype} (결측: {null_count})")
    print()
    
    # 타겟 변수 통계
    print("타겟 변수 (net_passengers) 통계:")
    print(final_df["net_passengers"].describe())
    print()
    
    # 저장
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {output_path}")
    
    return final_df


def get_column_descriptions() -> dict:
    """
    최종 데이터셋의 컬럼 설명을 반환합니다.
    
    Returns
    -------
    dict
        컬럼명: 설명 매핑
    """
    return {
        # 원본 승하차 컬럼
        "날짜": "날짜 문자열 (YYYY-MM-DD)",
        "호선": "지하철 호선 (2, 3)",
        "역번호": "역 고유번호",
        "역명": "역 이름",
        "시간대": "시간대 코드 (530, 600, ..., 2300)",
        "승차": "승차 인원",
        "하차": "하차 인원",
        "순서": "노선 내 역 순서",
        "환승역": "환승 노선 수 (0=비환승역)",
        "요일구분": "요일 구분 (평일/토요일/일요일)",
        
        # 혼잡도 (참고용, 타겟 아님)
        "오름혼잡도": "오름차순 방향 혼잡도",
        "내림혼잡도": "내림차순 방향 혼잡도",
        "오름전역혼잡도": "오름차순 방향 전역 혼잡도",
        "내림전역혼잡도": "내림차순 방향 전역 혼잡도",
        
        # 생성된 컬럼
        "Date": "날짜 (datetime)",
        "Hour": "시간 (0-23)",
        "net_passengers": "순 승차인원 (승차 - 하차) [타겟 변수]",
        "is_transfer_station": "환승역 여부 (0/1)",
        
        # 날씨 컬럼
        "temperature": "기온 (°C)",
        "rainfall": "강수량 (mm)",
        "wind_speed": "풍속 (m/s)",
        "humidity": "습도 (%)",
        "snowfall": "적설 (cm)",
        "discomfort_index": "불쾌지수",
        
        # 이벤트 컬럼
        "is_event_nearby": "해당 역 이벤트 여부 (0/1)",
        "event_count": "해당 역 이벤트 수",
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    output_path = project_root / "outputs" / "merged_data.csv"
    
    merged_df = merge_all_datasets(data_dir, output_path)
    
    print("\n" + "=" * 70)
    print(" 샘플 데이터 (처음 5행)")
    print("=" * 70)
    print(merged_df.head())








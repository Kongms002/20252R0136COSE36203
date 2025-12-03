"""
데이터 로딩 유틸리티

Raw 데이터를 로드하고 기본적인 형식 변환을 수행합니다.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd


def try_read_csv(path: Path, encodings: Optional[List[str]] = None) -> pd.DataFrame:
    """
    여러 인코딩을 시도하여 CSV 파일을 읽습니다.
    
    Parameters
    ----------
    path : Path
        CSV 파일 경로
    encodings : Optional[List[str]]
        시도할 인코딩 목록 (기본: utf-8-sig, utf-8, cp949, euc-kr)
    
    Returns
    -------
    pd.DataFrame
        로드된 데이터프레임
    """
    if encodings is None:
        encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
    
    raise ValueError(f"파일을 읽을 수 없습니다: {path}. 마지막 에러: {last_error}")


def load_boarding_data(data_dir: Path) -> pd.DataFrame:
    """
    승하차 데이터를 로드하고 기본 전처리를 수행합니다.
    
    전처리 내용:
    1. 날짜 컬럼을 datetime으로 변환
    2. 시간대(530, 600, ...)를 정수 Hour(5, 6, ...)로 변환
    3. 타겟 변수 net_passengers 생성 (승차 - 하차)
    
    Parameters
    ----------
    data_dir : Path
        data 폴더 경로
    
    Returns
    -------
    pd.DataFrame
        전처리된 승하차 데이터
    """
    file_path = data_dir / "2024boarding.csv"
    df = try_read_csv(file_path)
    
    # 날짜 변환
    df["Date"] = pd.to_datetime(df["날짜"], format="%Y-%m-%d")
    
    # 시간대 변환: 530 → 5, 600 → 6, ..., 2300 → 23, 2400 → 23 (자정은 23시로 처리)
    # 시간대가 정수형 (530, 600, 700, ..., 2300, 2400)
    # 2400(자정)은 날씨 데이터가 0~23시만 있으므로 23시로 매핑
    df["Hour"] = df["시간대"] // 100
    df.loc[df["Hour"] == 24, "Hour"] = 23  # 자정 → 23시
    
    # 타겟 변수 생성: 순 승차인원 = 승차 - 하차
    df["net_passengers"] = df["승차"] - df["하차"]
    
    # 호선 정보 정리 (숫자 확인)
    df["호선"] = df["호선"].astype(int)
    
    # 환승역 정보 정리
    df["환승역"] = df["환승역"].astype(int)
    df["is_transfer_station"] = (df["환승역"] > 0).astype(int)
    
    print(f"[승하차 데이터] 로드 완료: {len(df):,}행")
    print(f"  - 날짜 범위: {df['Date'].min()} ~ {df['Date'].max()}")
    print(f"  - 호선: {sorted(df['호선'].unique())}")
    print(f"  - 역 수: {df['역명'].nunique()}")
    
    return df


def load_weather_data(data_dir: Path) -> pd.DataFrame:
    """
    날씨 데이터를 로드합니다.
    
    Parameters
    ----------
    data_dir : Path
        data 폴더 경로
    
    Returns
    -------
    pd.DataFrame
        날씨 데이터
    """
    file_path = data_dir / "2024weather.csv"
    df = try_read_csv(file_path)
    
    print(f"[날씨 데이터] 로드 완료: {len(df):,}행")
    print(f"  - 컬럼: {list(df.columns)}")
    
    return df


def load_event_data(data_dir: Path) -> pd.DataFrame:
    """
    문화행사 데이터를 로드합니다.
    
    Parameters
    ----------
    data_dir : Path
        data 폴더 경로
    
    Returns
    -------
    pd.DataFrame
        문화행사 데이터
    """
    file_path = data_dir / "2024seoul_culture_events.csv"
    df = try_read_csv(file_path)
    
    print(f"[문화행사 데이터] 로드 완료: {len(df):,}행")
    print(f"  - 컬럼: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    # 테스트
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("=" * 60)
    boarding_df = load_boarding_data(data_dir)
    print()
    
    print("=" * 60)
    weather_df = load_weather_data(data_dir)
    print()
    
    print("=" * 60)
    event_df = load_event_data(data_dir)


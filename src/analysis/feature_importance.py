"""
Feature Importance 분석 모듈

모델 기반 Feature 중요도를 분석하고 시각화합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json


def analyze_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 30,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    모델의 Feature Importance를 분석합니다.
    
    Parameters
    ----------
    model : Any
        학습된 모델 (feature_importances_ 속성 필요)
    feature_names : List[str]
        Feature 컬럼명 목록
    top_n : int
        상위 N개 feature 출력
    output_dir : Optional[Path]
        결과 저장 디렉토리
    
    Returns
    -------
    pd.DataFrame
        Feature importance 데이터프레임
    """
    if not hasattr(model, "feature_importances_"):
        print("[경고] 모델에 feature_importances_ 속성이 없습니다.")
        return pd.DataFrame()
    
    importances = model.feature_importances_
    
    # DataFrame 생성
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    # 정규화 (합=100%)
    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    )
    
    # 누적 합
    importance_df["cumulative_pct"] = importance_df["importance_pct"].cumsum()
    
    # 상위 N개 출력
    print(f"\n[Feature Importance] 상위 {top_n}개:")
    print("-" * 60)
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:<40} {row['importance_pct']:>6.2f}% (누적: {row['cumulative_pct']:>5.1f}%)")
    
    # 저장
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        print(f"\n저장: {output_dir / 'feature_importance.csv'}")
    
    return importance_df


def analyze_feature_groups(importance_df: pd.DataFrame) -> Dict[str, float]:
    """
    Feature 그룹별 중요도를 분석합니다.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance 데이터프레임
    
    Returns
    -------
    Dict[str, float]
        그룹별 중요도 합계
    """
    groups = {
        "시간": ["Time_", "Hour", "요일", "Month", "Season", "Week", "Day", "Rush", "Night", "Holiday"],
        "날씨": ["Temp", "Rain", "Snow", "Humid", "Wind", "Weather", "DI_", "Feel", "Extreme", "Precip"],
        "Lag/Rolling": ["lag_", "rolling_", "diff_", "pct_", "_mean_"],
        "이벤트": ["event", "Event"],
        "교차": ["_x_"],
        "기타": []
    }
    
    group_importance = {g: 0.0 for g in groups}
    
    for _, row in importance_df.iterrows():
        feature = row["feature"]
        importance = row["importance_pct"]
        
        assigned = False
        for group, keywords in groups.items():
            if group == "기타":
                continue
            if any(kw in feature for kw in keywords):
                group_importance[group] += importance
                assigned = True
                break
        
        if not assigned:
            group_importance["기타"] += importance
    
    print("\n[그룹별 Feature 중요도]:")
    print("-" * 40)
    for group, importance in sorted(group_importance.items(), key=lambda x: -x[1]):
        print(f"  {group:<15} {importance:>6.2f}%")
    
    return group_importance


def calculate_correlation_with_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "net_passengers",
    top_n: int = 20
) -> pd.DataFrame:
    """
    Feature와 Target 간 상관계수를 계산합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        데이터프레임
    feature_cols : List[str]
        Feature 컬럼 목록
    target_col : str
        Target 컬럼명
    top_n : int
        상위 N개 출력
    
    Returns
    -------
    pd.DataFrame
        상관계수 데이터프레임
    """
    correlations = []
    
    for col in feature_cols:
        if col in df.columns and target_col in df.columns:
            corr = df[col].corr(df[target_col])
            correlations.append({
                "feature": col,
                "correlation": corr,
                "abs_correlation": abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False)
    
    print(f"\n[Target과의 상관계수] 상위 {top_n}개:")
    print("-" * 50)
    for _, row in corr_df.head(top_n).iterrows():
        sign = "+" if row["correlation"] > 0 else ""
        print(f"  {row['feature']:<40} {sign}{row['correlation']:>7.4f}")
    
    return corr_df


if __name__ == "__main__":
    # 테스트
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "outputs" / "featured_data.csv"
    df = pd.read_csv(data_path, nrows=10000)
    
    # Feature 컬럼 추출
    exclude_cols = {"날짜", "Date", "역명", "역번호", "승차", "하차", "net_passengers",
                    "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도", "요일구분"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 상관계수 분석
    corr_df = calculate_correlation_with_target(df, feature_cols)








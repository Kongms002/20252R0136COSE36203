"""
Optuna 기반 하이퍼파라미터 최적화

V6 Pure External 모델을 기반으로 Optuna를 사용하여
LightGBM, XGBoost, CatBoost의 하이퍼파라미터를 최적화합니다.

평가 지표: MAE (Mean Absolute Error) - 도메인에 가장 적합
교차 검증: Time Series 5-Fold CV
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import json
import warnings
import time
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[ERROR] Optuna not installed. Run: pip install optuna")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent


def get_pure_external_features(df):
    """순수 외부 요인 feature만 선별 (V6와 동일)"""
    exclude_exact = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    
    exclude_keywords = [
        "lag_", "rolling_", "diff_", "pct_change",
        "_mean_승차", "_mean_하차", "_mean_net",
        "_std_승차", "_std_하차", "_std_net",
        "time_mean", "weekday_mean", "month_mean",
        "_te", "승차", "하차", "net_passengers"
    ]
    
    selected_features = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if any(kw in col for kw in exclude_keywords):
            continue
        selected_features.append(col)
    
    return selected_features


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    data_path = PROJECT_ROOT / "outputs" / "featured_data.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Hour", "역명"]).reset_index(drop=True)
    
    feature_cols = get_pure_external_features(df)
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["net_passengers"]
    
    return X, y, feature_cols


class OptunaOptimizer:
    """Optuna 기반 하이퍼파라미터 최적화 클래스"""
    
    def __init__(self, X, y, n_splits=5, n_trials=50):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.best_params = {}
        self.optimization_history = {}
        
    def optimize_lightgbm(self):
        """LightGBM 하이퍼파라미터 최적화"""
        print("\n" + "=" * 60)
        print(" LightGBM Hyperparameter Optimization")
        print("=" * 60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'verbose': -1,
                'random_state': 42
            }
            
            cv_scores = []
            for train_idx, val_idx in self.tscv.split(self.X):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                # Early stopping용 분할
                es_split = int(len(X_train) * 0.9)
                X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
                y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_es, y_es)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='lightgbm_optimization'
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['LightGBM'] = study.best_params
        self.optimization_history['LightGBM'] = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'history': [(t.number, t.value) for t in study.trials]
        }
        
        print(f"\n  Best MAE: {study.best_value:.2f}")
        print(f"  Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_xgboost(self):
        """XGBoost 하이퍼파라미터 최적화"""
        print("\n" + "=" * 60)
        print(" XGBoost Hyperparameter Optimization")
        print("=" * 60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'verbosity': 0,
                'random_state': 42
            }
            
            cv_scores = []
            for train_idx, val_idx in self.tscv.split(self.X):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                es_split = int(len(X_train) * 0.9)
                X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
                y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
                
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_es, y_es)],
                    verbose=False
                )
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='xgboost_optimization'
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['XGBoost'] = study.best_params
        self.optimization_history['XGBoost'] = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'history': [(t.number, t.value) for t in study.trials]
        }
        
        print(f"\n  Best MAE: {study.best_value:.2f}")
        print(f"  Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_catboost(self):
        """CatBoost 하이퍼파라미터 최적화"""
        print("\n" + "=" * 60)
        print(" CatBoost Hyperparameter Optimization")
        print("=" * 60)
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'random_seed': 42,
                'verbose': False
            }
            
            cv_scores = []
            for train_idx, val_idx in self.tscv.split(self.X):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                es_split = int(len(X_train) * 0.9)
                X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
                y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
                
                model = CatBoostRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_es, y_es),
                    early_stopping_rounds=30,
                    verbose=False
                )
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='catboost_optimization'
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['CatBoost'] = study.best_params
        self.optimization_history['CatBoost'] = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'history': [(t.number, t.value) for t in study.trials]
        }
        
        print(f"\n  Best MAE: {study.best_value:.2f}")
        print(f"  Best params: {study.best_params}")
        
        return study.best_params
    
    def run_all_optimizations(self):
        """모든 모델 최적화 실행"""
        print("\n" + "=" * 70)
        print(" Optuna Hyperparameter Optimization")
        print(f" Trials per model: {self.n_trials}")
        print(f" CV Folds: {self.n_splits}")
        print("=" * 70)
        
        start_time = time.time()
        
        if HAS_LIGHTGBM:
            self.optimize_lightgbm()
        
        if HAS_XGBOOST:
            self.optimize_xgboost()
        
        if HAS_CATBOOST:
            self.optimize_catboost()
        
        elapsed = time.time() - start_time
        print(f"\n총 최적화 시간: {elapsed/60:.1f}분")
        
        # 결과 저장
        with open(OUTPUT_DIR / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, ensure_ascii=False, indent=2)
        
        with open(OUTPUT_DIR / 'optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, ensure_ascii=False, indent=2)
        
        return self.best_params


def evaluate_with_optimized_params(X, y, best_params, feature_cols):
    """최적화된 파라미터로 최종 평가"""
    print("\n" + "=" * 70)
    print(" Final Evaluation with Optimized Parameters")
    print("=" * 70)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {}
    all_fold_results = {}
    feature_importances = {}
    learning_curves = {}
    
    models_to_evaluate = {}
    
    if 'LightGBM' in best_params and HAS_LIGHTGBM:
        params = best_params['LightGBM'].copy()
        params['n_jobs'] = -1
        params['verbose'] = -1
        params['random_state'] = 42
        models_to_evaluate['LightGBM_Optuna'] = ('lgb', params)
    
    if 'XGBoost' in best_params and HAS_XGBOOST:
        params = best_params['XGBoost'].copy()
        params['n_jobs'] = -1
        params['verbosity'] = 0
        params['random_state'] = 42
        models_to_evaluate['XGBoost_Optuna'] = ('xgb', params)
    
    if 'CatBoost' in best_params and HAS_CATBOOST:
        params = best_params['CatBoost'].copy()
        params['random_seed'] = 42
        params['verbose'] = False
        models_to_evaluate['CatBoost_Optuna'] = ('cat', params)
    
    # 기존 모델 (비교용)
    models_to_evaluate['LightGBM_Default'] = ('lgb', {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 10,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    })
    
    for model_name, (model_type, params) in models_to_evaluate.items():
        print(f"\n[{model_name}]")
        
        fold_results = []
        fold_importances = []
        train_losses = []
        val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Early stopping 분할
            es_split = int(len(X_train) * 0.9)
            X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
            y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
            
            if model_type == 'lgb':
                model = lgb.LGBMRegressor(**params)
                evals_result = {}
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_tr, y_tr), (X_es, y_es)],
                    eval_metric='mae',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.record_evaluation(evals_result)
                    ]
                )
                if fold == 4:  # 마지막 fold의 learning curve 저장
                    train_losses = evals_result['training']['l1']
                    val_losses = evals_result['valid_1']['l1']
                
                importance = model.feature_importances_
                
            elif model_type == 'xgb':
                model = xgb.XGBRegressor(**params)
                evals_result = {}
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_tr, y_tr), (X_es, y_es)],
                    verbose=False
                )
                importance = model.feature_importances_
                
            elif model_type == 'cat':
                model = CatBoostRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_es, y_es),
                    early_stopping_rounds=30,
                    verbose=False
                )
                importance = model.feature_importances_
            
            y_pred = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            fold_results.append({'rmse': rmse, 'mae': mae, 'r2': r2})
            fold_importances.append(importance)
            
            print(f"  Fold {fold+1}: MAE={mae:.2f}, R²={r2:.4f}")
        
        # 평균 결과
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        std_rmse = np.std([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        std_mae = np.std([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        std_r2 = np.std([r['r2'] for r in fold_results])
        
        results[model_name] = {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'mean_r2': mean_r2,
            'std_r2': std_r2
        }
        
        all_fold_results[model_name] = fold_results
        
        # Feature importance 평균
        mean_importance = np.mean(fold_importances, axis=0)
        feature_importances[model_name] = dict(zip(feature_cols, mean_importance))
        
        # Learning curves
        if train_losses and val_losses:
            learning_curves[model_name] = {
                'train': train_losses,
                'val': val_losses
            }
        
        print(f"  Mean: MAE={mean_mae:.2f}±{std_mae:.1f}, R²={mean_r2:.4f}±{std_r2:.4f}")
    
    return results, all_fold_results, feature_importances, learning_curves


def create_ensemble_with_optimized_models(X, y, best_params):
    """최적화된 모델로 앙상블 생성"""
    print("\n" + "=" * 70)
    print(" Ensemble with Optimized Models")
    print("=" * 70)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    ensemble_results = {}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        es_split = int(len(X_train) * 0.9)
        X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
        y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
        
        predictions = {}
        
        # LightGBM
        if 'LightGBM' in best_params and HAS_LIGHTGBM:
            params = best_params['LightGBM'].copy()
            params.update({'n_jobs': -1, 'verbose': -1, 'random_state': 42})
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])
            predictions['LightGBM'] = model.predict(X_val)
        
        # XGBoost
        if 'XGBoost' in best_params and HAS_XGBOOST:
            params = best_params['XGBoost'].copy()
            params.update({'n_jobs': -1, 'verbosity': 0, 'random_state': 42})
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
            predictions['XGBoost'] = model.predict(X_val)
        
        # CatBoost
        if 'CatBoost' in best_params and HAS_CATBOOST:
            params = best_params['CatBoost'].copy()
            params.update({'random_seed': 42, 'verbose': False})
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_es, y_es),
                     early_stopping_rounds=30, verbose=False)
            predictions['CatBoost'] = model.predict(X_val)
        
        if len(predictions) == 0:
            continue
        
        pred_values = np.array(list(predictions.values()))
        
        # Simple Average
        avg_pred = np.mean(pred_values, axis=0)
        if 'Simple_Average' not in ensemble_results:
            ensemble_results['Simple_Average'] = []
        ensemble_results['Simple_Average'].append({
            'mae': mean_absolute_error(y_val, avg_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, avg_pred)),
            'r2': r2_score(y_val, avg_pred)
        })
        
        # Weighted Average (R² 기반)
        weights = []
        for name, pred in predictions.items():
            r2 = r2_score(y_val, pred)
            weights.append(max(0, r2))
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        weighted_pred = np.average(pred_values, axis=0, weights=weights)
        if 'Weighted_Average' not in ensemble_results:
            ensemble_results['Weighted_Average'] = []
        ensemble_results['Weighted_Average'].append({
            'mae': mean_absolute_error(y_val, weighted_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, weighted_pred)),
            'r2': r2_score(y_val, weighted_pred)
        })
        
        # Median
        median_pred = np.median(pred_values, axis=0)
        if 'Median' not in ensemble_results:
            ensemble_results['Median'] = []
        ensemble_results['Median'].append({
            'mae': mean_absolute_error(y_val, median_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, median_pred)),
            'r2': r2_score(y_val, median_pred)
        })
    
    # 결과 요약
    summary = {}
    for ens_name, fold_results in ensemble_results.items():
        summary[ens_name] = {
            'mean_mae': np.mean([r['mae'] for r in fold_results]),
            'std_mae': np.std([r['mae'] for r in fold_results]),
            'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
            'mean_r2': np.mean([r['r2'] for r in fold_results]),
            'std_r2': np.std([r['r2'] for r in fold_results])
        }
        print(f"\n[{ens_name}]")
        print(f"  MAE: {summary[ens_name]['mean_mae']:.2f}±{summary[ens_name]['std_mae']:.1f}")
        print(f"  R²: {summary[ens_name]['mean_r2']:.4f}±{summary[ens_name]['std_r2']:.4f}")
    
    return summary


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print(" Optuna Hyperparameter Optimization Pipeline")
    print("=" * 70)
    
    # 데이터 로드
    print("\n[1] Loading data...")
    X, y, feature_cols = load_and_prepare_data()
    print(f"    - Samples: {len(X):,}")
    print(f"    - Features: {len(feature_cols)}")
    
    # Optuna 최적화
    print("\n[2] Running Optuna optimization...")
    optimizer = OptunaOptimizer(X, y, n_splits=5, n_trials=50)
    best_params = optimizer.run_all_optimizations()
    
    # 최적화된 파라미터로 평가
    print("\n[3] Evaluating with optimized parameters...")
    results, fold_results, feature_importances, learning_curves = evaluate_with_optimized_params(
        X, y, best_params, feature_cols
    )
    
    # 앙상블 평가
    print("\n[4] Evaluating ensemble methods...")
    ensemble_summary = create_ensemble_with_optimized_models(X, y, best_params)
    
    # 결과 저장
    print("\n[5] Saving results...")
    
    # 모델별 결과
    results_df = pd.DataFrame([
        {
            'model': name,
            'mean_mae': m['mean_mae'],
            'std_mae': m['std_mae'],
            'mean_rmse': m['mean_rmse'],
            'mean_r2': m['mean_r2'],
            'std_r2': m['std_r2']
        }
        for name, m in results.items()
    ])
    results_df = results_df.sort_values('mean_mae')
    results_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    
    # 앙상블 결과
    ensemble_df = pd.DataFrame([
        {
            'ensemble': name,
            'mean_mae': m['mean_mae'],
            'std_mae': m['std_mae'],
            'mean_rmse': m['mean_rmse'],
            'mean_r2': m['mean_r2'],
            'std_r2': m['std_r2']
        }
        for name, m in ensemble_summary.items()
    ])
    ensemble_df = ensemble_df.sort_values('mean_mae')
    ensemble_df.to_csv(OUTPUT_DIR / 'ensemble_comparison.csv', index=False)
    
    # Feature importance 저장 (float32 -> float 변환)
    feature_importances_serializable = {
        model_name: {k: float(v) for k, v in imp.items()}
        for model_name, imp in feature_importances.items()
    }
    with open(OUTPUT_DIR / 'feature_importances.json', 'w', encoding='utf-8') as f:
        json.dump(feature_importances_serializable, f, ensure_ascii=False, indent=2)
    
    # Learning curves 저장
    with open(OUTPUT_DIR / 'learning_curves.json', 'w', encoding='utf-8') as f:
        json.dump(learning_curves, f, ensure_ascii=False, indent=2)
    
    # 최종 요약
    print("\n" + "=" * 70)
    print(" Final Summary")
    print("=" * 70)
    
    # 최고 성능 모델 찾기
    all_results = {**results, **{f"Ensemble_{k}": {'mean_mae': v['mean_mae'], 'mean_r2': v['mean_r2']} 
                                  for k, v in ensemble_summary.items()}}
    best_model = min(all_results.items(), key=lambda x: x[1]['mean_mae'])
    
    default_mae = results.get('LightGBM_Default', {}).get('mean_mae', 193.33)
    best_mae = best_model[1]['mean_mae']
    improvement = (default_mae - best_mae) / default_mae * 100
    
    print(f"\n  Baseline (LightGBM Default): MAE = {default_mae:.2f}")
    print(f"  Best Model ({best_model[0]}): MAE = {best_mae:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # 최종 요약 저장
    final_summary = {
        'baseline': {
            'model': 'LightGBM_Default',
            'mae': default_mae
        },
        'best': {
            'model': best_model[0],
            'mae': best_mae,
            'r2': best_model[1]['mean_r2']
        },
        'improvement_pct': improvement,
        'best_params': best_params,
        'all_results': {k: {'mae': v['mean_mae'], 'r2': v.get('mean_r2', v.get('mean_r2'))} 
                       for k, v in all_results.items()}
    }
    
    with open(OUTPUT_DIR / 'final_summary.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    
    return final_summary


if __name__ == "__main__":
    main()

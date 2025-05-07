import numpy as np
import pandas as pd
import logging
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from data_preparation import prepare_prediction_data

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_temporal_features(df_full, X):
    """
    Add rolling, delta features, plus champ_mean and patch_mean.
    """
    df = df_full.copy()
    # Ordinalisation des patches
    patches = sorted(df['patch'].unique(), key=lambda x: [int(p) for p in x.split('.')])
    patch_map = {p: i for i, p in enumerate(patches)}
    df['patch_idx'] = df['patch'].map(patch_map)

    # Rolling et précédent
    df.sort_values(['champion_name', 'patch_idx'], inplace=True)
    df['champ_prev_win'] = df.groupby('champion_name')['winrate'].shift(1)
    df['champ_roll3']    = df.groupby('champion_name')['winrate'] \
                              .rolling(3, min_periods=1).mean() \
                              .reset_index(level=0, drop=True)

    # Moyennes
    champ_mean = df.groupby('champion_name')['winrate'].transform('mean')
    patch_mean = df.groupby('patch')['winrate'].transform('mean')
    global_mean = df['winrate'].mean()

    # Merge dans X
    X = X.copy()
    X['patch_idx']      = df.loc[X.index, 'patch_idx']
    X['champ_prev_win'] = df.loc[X.index, 'champ_prev_win'].fillna(global_mean)
    X['champ_roll3']    = df.loc[X.index, 'champ_roll3']
    X['champ_mean']     = champ_mean.loc[X.index]
    X['patch_mean']     = patch_mean.loc[X.index]
    X['champ_delta']    = df.loc[X.index, 'winrate'] - champ_mean.loc[X.index]

    return X

def main():
    logger.info("Loading and preparing data...")
    data = prepare_prediction_data()
    full_df    = data['full_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train, w_test = data['w_train'], data['w_test']

    # 1) Temporal + basic
    logger.info("Adding temporal & mean features...")
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test  = add_temporal_features(full_df.loc[X_test.index],  X_test)
    # Ajouter pickrate & total_games
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col]  = full_df.loc[X_test.index,  col]

    # 2) Polynômes degré 2
    logger.info("Generating polynomial features (degree=2)...")
    poly = PolynomialFeatures(2, include_bias=False)
    X_train_poly = pd.DataFrame(
        poly.fit_transform(X_train),
        index=X_train.index,
        columns=poly.get_feature_names_out(X_train.columns)
    )
    X_test_poly = pd.DataFrame(
        poly.transform(X_test),
        index=X_test.index,
        columns=X_train_poly.columns
    )
    logger.info(f"→ {X_train_poly.shape[1]} features after poly")

    # 3) VarianceThreshold
    logger.info("Applying VarianceThreshold...")
    vt = VarianceThreshold(0.01)
    X_train_vt = pd.DataFrame(
        vt.fit_transform(X_train_poly),
        index=X_train_poly.index,
        columns=X_train_poly.columns[vt.get_support()]
    )
    X_test_vt = pd.DataFrame(
        vt.transform(X_test_poly),
        index=X_test_poly.index,
        columns=X_train_vt.columns
    )
    logger.info(f"→ {X_train_vt.shape[1]} features after VT")

    # 4) Sélection par un modèle léger
    logger.info("Selecting features with preliminary XGBRegressor...")
    base = xgb.XGBRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1, reg_lambda=10,
        objective='reg:squarederror', random_state=42, verbosity=0
    )
    selector = SelectFromModel(base, threshold='median')
    selector.fit(X_train_vt, y_train, sample_weight=w_train)
    keep = selector.get_support(indices=True)
    X_train_sel = X_train_vt.iloc[:, keep]
    X_test_sel  = X_test_vt.iloc[:, keep]
    logger.info(f"→ {X_train_sel.shape[1]} features selected")

    # 5) DMatrix pour CV
    dtrain = xgb.DMatrix(X_train_sel, label=y_train, weight=w_train)

    # 6) CV XGBoost
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbosity': 0
    }
    logger.info("Running CV to find optimal rounds...")
    cvres = xgb.cv(
        params, dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=10,
        metrics='rmse',
        seed=42,
        as_pandas=True,
        verbose_eval=False
    )
    best_rounds = len(cvres)
    logger.info(f"→ Best rounds: {best_rounds}")

    # 7) Entraînement final
    model = xgb.XGBRegressor(
        n_estimators=best_rounds,
        **{k: params[k] for k in
           ['learning_rate','max_depth','subsample','colsample_bytree','gamma','reg_alpha','reg_lambda']},
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    logger.info("Training final model...")
    model.fit(X_train_sel, y_train, sample_weight=w_train, verbose=False)

    # 8) Évaluation
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test_sel)
    rmse = mean_squared_error(y_test, y_pred, squared=False, sample_weight=w_test)
    mae  = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
    r2   = r2_score(y_test, y_pred, sample_weight=w_test)
    logger.info(f"Final Results → RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

if __name__ == '__main__':
    main()

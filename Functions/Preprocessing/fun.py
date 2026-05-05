from ondil.estimators import OnlineDistributionalRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.constants import convert_temperature
from ondil.distributions import Normal
import xgboost as xgb
import pandas as pd
import numpy as np


# Function for adding calendar features to the dataframe.
def add_calendar_features(df:pd.DataFrame):

    # Season function
    def get_season(date):
        md = date.month * 100 + date.day
        if 321 <= md <= 620:
            return "spring"
        elif 621 <= md <= 922:
            return "summer"
        elif 923 <= md <= 1220:
            return "autumn"
        else:
            return "winter"

    idx = df.index
    df.copy()
    # Adding the new calendare features to the df.
    df['year']         = idx.year
    df['month']        = idx.month
    df['day']          = idx.day
    df['weekday']      = idx.weekday
    df['hour']         = idx.hour
    df['dayofyear']    = idx.dayofyear
    df['season']       = idx.to_series().apply(get_season)


    # Converting the categorical calendar features to float
    cat_cols = df.select_dtypes('object').columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes.astype(float)
        
    return df

def create_df(forecasts: pd.DataFrame, observations: pd.DataFrame, lead_times: list):

    # Extracting the chosen lead hours from the forecast df
    forecasts_leadtime = forecasts[forecasts.lead_hour.astype(int).isin(lead_times)].reset_index(names = 'valid_time').set_index(['run_time', 'valid_time']).sort_index().copy()
    forecasts_leadtime = forecasts_leadtime.reset_index(names = ['run_time', 'valid_time']).set_index('valid_time')
    
    # Both observations and forecasts are gathered at timezone UTC
    forecasts_leadtime.index = pd.to_datetime(forecasts_leadtime.index, utc=True)
    
    # Align observations hourly
        # The observations are recorded at minute 51 and forecasts at 00, is this a problem? 
    observations.index = pd.to_datetime(observations.index, utc=True).ceil('h')

    # Join hourly observations to hourly forecast rows
    df = forecasts_leadtime.join(observations)
    
    # Adding calendar features
    df = add_calendar_features(df)
    
    # Convert forecast temperatures
    df[['tmp_2m_c', 'dpt_2m_c']] = df[['tmp_2m_c', 'dpt_2m_c']].apply(convert_temperature, args=('k', 'f'))

    # Drop rows with missing values
    df = df.dropna(subset=["obs"]).reset_index(names="valid_time")




    return df


def feature_engineering(obs:pd.Series, df_og: pd.DataFrame):
    
    
    
    df_og['dayofyear_sin'] = np.sin(2 * np.pi * df_og['dayofyear'] / 365)
    df_og['dayofyear_cos'] = np.cos(2 * np.pi * df_og['dayofyear'] / 365)

    df_og['hour_sin'] = np.sin(2 * np.pi * df_og['hour'] / 24)
    df_og['hour_cos'] = np.cos(2 * np.pi * df_og['hour'] / 24)


    df_og['lead_hour_sin'] = np.sin(2 * np.pi * df_og['lead_hour'] / 24)
    df_og['lead_hour_cos'] = np.cos(2 * np.pi * df_og['lead_hour'] / 24)
    
    
    # Merging observations available at model run time with the just created df
    obs_runtime = obs.rename('obs_runtime').to_frame()

    # Creating rolling stats over the past 10 hours
        # OBS use autocorrelation to choose the best lagged features
    obs_runtime['obs_max'] = obs_runtime.obs_runtime.rolling('10h').max()
    obs_runtime['obs_min'] = obs_runtime.obs_runtime.rolling('10h').min()
    obs_runtime['obs_mean'] = obs_runtime.obs_runtime.rolling('10h').mean()

    # Creating lagged observations
    obs_runtime['obs_runtime_lag1'] = obs_runtime['obs_runtime'].shift(1)
    obs_runtime['obs_runtime_lag2'] = obs_runtime['obs_runtime'].shift(2)
    obs_runtime['obs_runtime_lag3'] = obs_runtime['obs_runtime'].shift(3)

    # Creating observations trends
    obs_runtime['obs_runtime_trend1'] = obs_runtime.obs_runtime - obs_runtime.obs_runtime_lag1
    obs_runtime['obs_runtime_trend2'] = obs_runtime.obs_runtime - obs_runtime.obs_runtime_lag2
    obs_runtime['obs_runtime_trend3'] = obs_runtime.obs_runtime - obs_runtime.obs_runtime_lag3


    df = df_og.join(obs_runtime).dropna()
    df_temp = df[df.lead_hour == 0].copy()
    runtime_error = (df_temp["obs_runtime"] - df_temp["tmp_2m_c"]).rename('runtime_error')
    return df.join(runtime_error).reset_index('run_time')




def get_maxtemps(df: pd.DataFrame):

    # Row where the forecast predicts the highest temp for each run
    df_max_temps = df.loc[df.groupby('run_time')['tmp_2m_c'].idxmax()].copy()

    # Observed max over the same window for each run
    obs_max = df.groupby('run_time')['obs'].max().rename('obs_temp')

    # Merge observations and max temperatures
    df_max_temps = df_max_temps.merge(obs_max, on='run_time', how='left').drop('obs',axis = 1).rename({'obs_temp': 'obs'}, axis = 1)
    df_max_temps = df_max_temps.set_index("run_time").drop('valid_time', axis = 1)

    return df_max_temps


def repeat_df_static(df, thresholds):

    # Repeating each index value length of thresholds amount of times
    df_repeated = df.loc[df.index.repeat(len(thresholds))].copy()

    # Making the threshold column by tileing the threshold values over each repeated section
    df_repeated["threshold"] = np.tile(thresholds, len(df))

    # Creating a binary target
    df_repeated["target"] = (df_repeated['obs'] <= df_repeated["threshold"]).astype(int)

    return df_repeated



def select_features(df_max):
    # Making a quick feature selection using XGBoost to also capture non linear relationships
    X_features_selection = df_max.drop('obs', axis=1)
    y_features_selection = df_max['obs']


    # Using only 20% of the data as training data for the feature selection and distributional regressor, to keep as much data as possible for the main model.
    # Shuffle is set to False, to avoid any data leakage.
    X_train, X_test, y_train, y_test = train_test_split(X_features_selection, y_features_selection, train_size=0.2, shuffle=False)


    # This model is only used for feature selection, not as the final forecasting model.
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        device='cuda'
    )

    xgb_model.fit(X_train, y_train)


    result = permutation_importance(
        xgb_model,
        X_test,
        y_test,
        scoring='neg_mean_squared_error',
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance = pd.Series(
        result.importances_mean,
        index=X_train.columns
    ).sort_values(ascending=False)


    # Selecting features with an importance above 0.
        # OBS: THIS IS VERY SIMPLE AND PROBABLY HAS TO BE REFINED
    selected_features = importance[importance > 0].index
    return selected_features.tolist() + ['obs']



def pred_uncertainty_features(df_max, features, drop_cols):

    train_bias, df_main = train_test_split(df_max[features].copy(), train_size = 0.20, shuffle = False)


    X_bias = train_bias.drop(['obs'] + drop_cols, axis = 1)
    y_bias = train_bias.obs


    equation = {
        0: "all", 
        1: "all",
    }

    # Create the estimator
    model = OnlineDistributionalRegression(
        distribution=Normal(),
        method="lasso",
        equation=equation,
        fit_intercept=True,
        ic="bic",
    )

    model.fit(X_bias, y_bias)

    df_main[['forecast_mean', 'forecast_scale']] = model.predict_distribution_parameters(df_main.drop(['obs'] + drop_cols , axis = 1))
    
    return df_main, model



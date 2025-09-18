from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta, MSTL
from .tsutils import mstl_season_selector
import pandas as pd

class BaseUnivariateForecaster:
    def __init__(self, freq: str, h: int, level: list[float] | None = None):
        """Initializes the SimpleUnivariateForecaster with specified parameters.
        Args:
            freq (str): Frequency of the time series. Options are 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'H' (half-yearly).
            h (int): Forecast horizon.
            level (list[float] | None, optional): Prediction interval levels. Defaults to [80, 95] if None.
        """
        self.freq = freq
        self.h = h
        self.level = level if level is not None else [80, 95]
        self.models = self._setup_models()

    def _setup_models(self) -> list[MSTL]:
        """Sets up the forecasting models with MSTL decomposition and returns a dictionary of models.
        Returns:
            dict: Dictionary containing univariate and multivariate models with MSTL decomposition.
        """

        return [AutoARIMA(), AutoCES(), AutoETS(), AutoTheta()]

    def fit(self, df: pd.DataFrame, id_col: str, time_col: str, target_col: str):
        raise NotImplementedError("Coming Soon!")

    def predict(self, df: pd.DataFrame, id_col: str, time_col: str, target_col: str) -> list[dict]:

        # initiate the Statsforecast object for the univariate models
        uvt_sf = StatsForecast(
            models=self.models, 
            freq=self.freq,
            n_jobs=-1,
        )

        # calle the forecast method to generate univariate predictions on the training samples and for the future
        uvt_fcst = uvt_sf.forecast(
            df=df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            h=self.h,
            level=self.level,
            fitted=False,
            )
        
        # convert uvt_fcst to a list of records (list of dicts)
        uvt_fcst_records = uvt_fcst.to_dict(orient="records")

        return uvt_fcst_records

class MSTLUnivariateForecaster(BaseUnivariateForecaster):
    def __init__(self, min_ts_len:int, freq: str, h: int, level: list[float] | None = None):
        """Initializes the MSTLUnivariateForecaster with specified parameters.
        Args:
            min_ts_len (int): Minimum length of the time series.
            freq (str): Frequency of the time series. Options are 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'H' (half-yearly).
            h (int): Forecast horizon.
            level (list[float] | None, optional): Prediction interval levels. Defaults to [80, 95] if None.
        """
        self.min_ts_len = min_ts_len
        self.season_lengths = mstl_season_selector(min_ts_len, freq)
        # Call parent constructor
        super().__init__(freq, h, level)

    def _setup_models(self) -> list[MSTL]:
        """Sets up the forecasting models with MSTL decomposition and returns a dictionary of models.
        Returns:
            dict: Dictionary containing univariate and multivariate models with MSTL decomposition.
        """

        # Define univariate models configuration
        uvt_dict_list = [
            {"alias": "MSTL_ARIMA", "model": AutoARIMA()},
            {"alias": "MSTL_CES", "model": AutoCES()},
            {"alias": "MSTL_ETS", "model": AutoETS()},
            {"alias": "MSTL_THETA", "model": AutoTheta()},
        ]

        # Initialize univariate models wrapped in MSTL
        uvt_models = []
        for each_dict in uvt_dict_list:
            if each_dict.get("alias") != "MSTL_ETS":
                uvt_models.append(MSTL(season_length=self.season_lengths, trend_forecaster=each_dict.get("model"), alias=each_dict.get("alias")))
            else:
                # For uvt_models like ETS which modify the seasons, we do not instantiate the object in the call to MSTL. An exception will be raised if we do so.
                uvt_models.append(MSTL(season_length=self.season_lengths, alias=each_dict.get("alias")))

        del uvt_dict_list
        return uvt_models
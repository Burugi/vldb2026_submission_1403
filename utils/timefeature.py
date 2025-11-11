import pandas as pd
import numpy as np

class TimeFeature:
    """시간 특성 생성 클래스"""
    @staticmethod
    def minute_of_hour(dt):
        return dt.dt.minute / 59.0 - 0.5
    
    @staticmethod
    def hour_of_day(dt):
        return dt.dt.hour / 23.0 - 0.5
    
    @staticmethod
    def day_of_week(dt):
        return dt.dt.dayofweek / 6.0 - 0.5
    
    @staticmethod
    def day_of_month(dt):
        return (dt.dt.day - 1) / 30.0 - 0.5
    
    @staticmethod
    def day_of_year(dt):
        return (dt.dt.dayofyear - 1) / 365.0 - 0.5
    
    @staticmethod
    def month_of_year(dt):
        return (dt.dt.month - 1) / 11.0 - 0.5

    @staticmethod
    def create_time_features(dates: pd.Series) -> np.ndarray:
        """시간 특성 생성"""
        dt_series = pd.to_datetime(dates)
        time_features = np.column_stack([
            TimeFeature.hour_of_day(dt_series),
            TimeFeature.day_of_week(dt_series),
            TimeFeature.day_of_month(dt_series),
            TimeFeature.month_of_year(dt_series)
        ])
        return time_features
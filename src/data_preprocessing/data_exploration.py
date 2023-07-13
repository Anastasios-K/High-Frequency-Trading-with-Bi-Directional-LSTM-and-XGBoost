import pandas as pd
import plotly.graph_objects as go
from ..config.config_loader import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker


class DataExplorator(object):

    def __init__(self, data: pd.DataFrame, config: ConfigLoader, info_tracker: InfoTracker):
        self.config = config
        self.data = data
        self.info_tracker = info_tracker

        self.__crate_candlestick_chart()

    def __crate_candlestick_chart(self):
        df = self.data
        config = self.config

        # create daily Timedelta
        daily_timedelta = pd.Timedelta("1 days")

        # calculate max date adding 100 daily timedeltas in the first date og data
        max_date = df.index[0] + daily_timedelta * 100

        # Filter out all dates higher than 100 * dailytimedelta
        df100 = df[df.index < max_date]

        # Resample the new data (100 days) by day
        df100 = df100.resample("D", label='right', closed='right').last()

        trace = go.Candlestick(
            x=df100.index,
            open=df100[config.dfstructure.open],
            high=df100[config.dfstructure.high],
            low=df100[config.dfstructure.low],
            close=df100[config.dfstructure.close]
        )

        layout = go.Layout(
            title='Candlestick Chart',
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(title='Price')
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

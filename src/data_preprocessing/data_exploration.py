import os
import pandas as pd
import plotly.graph_objects as go
from ydata_profiling import ProfileReport
from ..config.config_loading import ConfigLoader
from ..info_tracking.info_tracking import InfoTracker
from ..data_preprocessing.labels_creation import LabelCreator


class DataExplorator(object):
    """
    Class to explore the time-series data.
    It creates the following:
        1. Candlestick interactive chart
        2. Multi-resolution interactive line chart
        3. Explanatory Data Analysis report
    """

    def __init__(self, data: pd.DataFrame, config: ConfigLoader, info_tracker: InfoTracker):
        self.config = config
        self.data = data
        self.info_tracker = info_tracker

        os.makedirs(config.paths.path2save_exploration, exist_ok=True)
        self.__crate_candlestick_chart()
        self.__plot_multiple_data_resolutions()
        self.__create_eda_report()

    def __filter_data_in_interest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """ Filter the data based on the start and and end date. """

        df = self.data.copy()

        # convert start and end date to datetime
        start_date = pd.to_datetime(start_date, utc=True)
        end_date = pd.to_datetime(end_date, utc=True)

        # Check is start or end dates are equal to NaT after the datetime conversion.
        # That happens if empty string "" is given.
        start_checker = pd.notnull(start_date)
        end_checker = pd.notnull(end_date)

        # filter the data based on the given start and end dates
        if start_checker and end_checker:
            filtered_df = df.loc[(df.index > start_date) & (df.index < end_date)]
        elif start_checker and not end_checker:
            filtered_df = df.loc[df.index > start_date]
        elif not start_checker and end_checker:
            filtered_df = df.loc[df.index < end_date]
        else:
            filtered_df = df

        return filtered_df

    def __crate_candlestick_chart(self, start_date: str = "2016-01-01", end_date: str = "2016-04-01") -> None:

        dff = self.config.df_features

        # Filter data based on given start and end dates
        df = self.__filter_data_in_interest(start_date=start_date, end_date=end_date)

        # resample data to daily resolution anad drop rows with nan values
        df = df.resample("D", label='right', closed='right').last()
        df.dropna(inplace=True)

        # calculate difference between open and close
        df["diff_oc"] = df[dff.open] - df[dff.close]

        # calculate difference between high and low
        df["diff_hl"] = df[dff.high] - df[dff.low]

        # calculate average price of open close
        # we put the dots on the average price to design the box around it based on the open and close prices
        df["oc_avg"] = (df[dff.open] + df[dff.close]) / 2

        # calculate overall mean price of open close high and low
        df["close_avg"] = df["oc_avg"].mean()

        # classify the direction of the price (close high than open and vise versa)
        df["direction"] = 0
        df.loc[df["diff_oc"] < 0, 'direction'] = 1

        # make a plotly trace for open and close prices
        trace_oc = go.Scatter(
            mode="markers",
            x=df.index,
            y=df["oc_avg"],
            customdata=df[[dff.open, dff.close, dff.high, dff.low]],
            hovertemplate=
            "<br>Timestamp: %{x} <br>"
            "Open: %{customdata[0]}<br>"
            "Close: %{customdata[1]}<br>"
            "High: %{customdata[2]}<br>"
            "Low: %{customdata[3]}<br>"
            "<extra></extra>",
            hoverlabel=dict(
                bgcolor="#D0D0CE"
            ),
            marker=go.scatter.Marker(
                symbol="line-ns",  # A wide (width=5) line represents the box in the candlestick chart
                opacity=1,
                size=abs(df["diff_oc"]) * 2,
                line=dict(
                    width=5,
                    color=df["direction"],
                    colorscale=[[0, "#FF6347"], [1, "#3CB371"]]
                )
            )
        )

        # make a plotly trace for high and low
        trace_hl = go.Scatter(
            mode="markers",
            x=df.index,
            y=df["oc_avg"],
            customdata=df[[dff.open, dff.close, dff.high, dff.low]],
            hovertemplate=
            "<br>Timestamp: %{x} <br>"
            "Open: %{customdata[0]}<br>"
            "Close: %{customdata[1]}<br>"
            "High: %{customdata[2]}<br>"
            "Low: %{customdata[3]}<br>"
            "<extra></extra>",
            hoverlabel=dict(
                bgcolor="#D0D0CE"
            ),
            marker=go.scatter.Marker(
                symbol="line-ns",  # A thin (width=1) line represents the low-high interval in the candlestick chart
                size=abs(df["diff_hl"]) * 2,
                line=dict(
                    width=1,
                    color=df["direction"],
                    colorscale=[[0, "#DC143C"], [1, "#009A44"]]
                )
            )
        )

        # make a plotly trace for overall mean of Close-Open price
        trace_mean = go.Scatter(
            mode="lines",
            x=df.index,
            y=df["close_avg"],
            line=dict(
                color="black",
                width=1,
                dash="dot"
            ),
            hovertemplate=
            "Overall mean Close-Open: %{y} <extra></extra>",
            hoverlabel=dict(
                bgcolor="#D0D0CE"
            ),
        )

        # prepare the plotly layout object
        layout = go.Layout(
            title=dict(
                text=f"Candlestick Chart {str(start_date).split('+')[0]} - {str(end_date).split('+')[0]}",
                x=0.5,
                font=dict(
                    size=20,
                    family="Arial"
                )

            ),
            xaxis=dict(
                title=dict(
                    text="Dates",
                    font=dict(
                        size=20,
                        family="Arial"
                    )
                ),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor="black"
            ),
            yaxis=dict(
                title=dict(
                    text="Prices",
                    font=dict(
                        size=20,
                        family="Arial"
                    ),
                ),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor="black"
            ),
            showlegend=False,
            plot_bgcolor="#ffffff"
        )

        # create the figure and save it
        fig = go.Figure(data=[trace_oc, trace_hl, trace_mean], layout=layout)
        fig.write_html(os.path.join(self.config.paths.path2save_exploration, "candlestick_chart.html"))

    def __plot_multiple_data_resolutions(self,  start_date: str = "", end_date: str = "") -> None:
        """
        Create and save line plot with multiple resolutions of the given data.
        The resolutions are weekly, monthly, quarterly and yearly.
        It can focus on a specific time interval if start and end dates are given.
        """
        dff = self.config.df_features

        # Filter data based on given start and end dates
        df = self.__filter_data_in_interest(start_date=start_date, end_date=end_date)

        # Prepare weekly, monthly, quarterly and yearly resolutions
        weekly_res = df.resample('W').mean()
        monthly_res = df.resample('M').mean()
        quarter_res = df.resample('BQ').mean()
        yearly_res = df.resample('Y').mean()

        # Create weekly resolution trace
        weekly_trace = go.Scatter(
            mode="lines",
            x=weekly_res.index,
            y=weekly_res[dff.close],
            line=dict(
                color="#0076A8",
                width=1
            ),
            name="Weekly resolution"
        )

        # Create monthly resolution trace
        monthly_trace = go.Scatter(
            mode="lines",
            x=monthly_res.index,
            y=monthly_res[dff.close],
            line=dict(
                color="#43B02A",
                width=1
            ),
            name="Monthly resolution",
        )

        # Create quarterly resolution trace
        quarter_trace = go.Scatter(
            mode="lines",
            x=quarter_res.index,
            y=quarter_res[dff.close],
            line=dict(
                color="#DA291C",
                width=1
            ),
            name="Quarterly resolution",
        )

        # Create yearly resolution trace
        yearly_trace = go.Scatter(
            mode="lines",
            x=yearly_res.index,
            y=yearly_res[dff.close],
            line=dict(
                color="#000000",
                width=1
            ),
            name="Yearly resolution",
        )

        # prepare the plotly layout object
        layout = go.Layout(
            title=dict(
                text=f"Multiple Data Resolutions",
                x=0.5,
                font=dict(
                    size=20,
                    family="Arial"
                )

            ),
            xaxis=dict(
                title=dict(
                    text="Dates",
                    font=dict(
                        size=20,
                        family="Arial"
                    )
                ),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor="black"
            ),
            yaxis=dict(
                title=dict(
                    text="Prices",
                    font=dict(
                        size=20,
                        family="Arial"
                    ),
                ),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor="black"
            ),
            showlegend=True,
            plot_bgcolor="#ffffff"
        )

        # create the figure and save it
        fig = go.Figure(data=[weekly_trace, monthly_trace, quarter_trace, yearly_trace], layout=layout)
        fig.write_html(os.path.join(self.config.paths.path2save_exploration, "multiple_resolutions_chart.html"))

    def __create_eda_report(self) -> None:
        """ Create Exploratory Data Analysis. """
        df = self.data.copy()

        profile = ProfileReport(df, title="Pandas Profiling Report")

        profile.to_file(os.path.join(self.config.paths.path2save_exploration, "EDanalysis.html"))

    def label_creation(self):
        return LabelCreator(
            data=self.data,
            config=self.config,
            info_tracker=self.info_tracker
        )

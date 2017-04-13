from datetime import datetime, timedelta

import pandas as pd
import yahoo_finance


def get_stock_data_current(tick, days=50):
    """
    Get current data for tick.
    :param tick: Ticker for the stock.
    :param days: Number of days from current date.
    :return: 
    """
    _data = yahoo_finance.Share(tick)

    # today_date = datetime.today() - timedelta(days=1)
    today_date = datetime.today()
    today_date = str(today_date.date())
    # print today_date
    previous_date = datetime.today() - timedelta(days=days)
    previous_date = str(previous_date.date())
    # print previous_date

    df = _data.get_historical(start_date=previous_date, end_date=today_date)

    df = pd.DataFrame(df)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.index = df['Date']
    df.sort_index(inplace=True)

    # df = df[['Date', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']]
    df = df[['Open', 'High', 'Low', 'Adj_Close', 'Volume']]
    # Changing the data types of columns to numeric
    for cols in df:
        df[cols] = pd.to_numeric(df[cols], errors='coerce')

    return df


def get_historical_data(tick, start_date, end_date):
    """
    Get historical data for tick.
    :param tick: Ticker for the stock
    :param start_date: Format for date 'YYYY-MM-DD'
    :param end_date: Format for date 'YYYY-MM-DD'
    :return df: dataframe of values between start_date and end_date is ascending order of dates
    """
    _data = yahoo_finance.Share(tick)
    df = _data.get_historical(start_date=start_date, end_date=end_date)
    df = pd.DataFrame(df)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.index = df['Date']
    df.sort_index(inplace=True)

    # df = df[['Date', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']]
    df = df[['Open', 'High', 'Low', 'Adj_Close', 'Volume']]
    # Changing the data types of columns to numeric
    for cols in df:
        df[cols] = pd.to_numeric(df[cols], errors='coerce')

    return df
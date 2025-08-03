import os
import requests
import pandas as pd
import time
from datetime import datetime
from handlers.connection_manager import ensure_valid_tokens

class HistoricalDataHandler:
    def __init__(self):
        """
        Initialize the HistoricalDataHandler.
        """
        # print(f"DEBUG: Initialized HistoricalDataHandler. Instance ID: {id(self)}")

    def get_historical_data(self, symbol, periodType, period, frequencyType, freq, startDate=None, endDate=None, needExtendedHoursData=True):
        """Alias for fetch_historical_data for compatibility"""
        return self.fetch_historical_data(symbol, periodType, period, frequencyType, freq, startDate, endDate, needExtendedHoursData)

    def fetch_historical_data(self, symbol, periodType, period, frequencyType, freq, startDate=None, endDate=None, needExtendedHoursData=True):
        # print(f"DEBUG: [HistoricalDataHandler] fetch_historical_data called. Instance ID: {id(self)}")
        
        max_retries = 5  # Number of retries before giving up
        retry_delay = 2  # Initial retry delay in seconds

        for attempt in range(max_retries):
            try:
                data = self.get_hist_bars(symbol, periodType, period, frequencyType, freq, startDate, endDate, needExtendedHoursData)
                if data:
                    print(f"H")
                    return data
                else:
                    print("DEBUG: Fetch_historical_data_[HistoricalDataHandler] No data fetched.")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"ERROR: [HistoricalDataHandler] Request failed on attempt {attempt + 1} with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff  `                                                  `
                else:
                    print("ERROR: [HistoricalDataHandler] Maximum retry attempts reached. Aborting.")
                    return None

    def get_hist_bars(self, symbol, periodType, period, frequencyType, freq, startDate=None, endDate=None, needExtendedHoursData=True):
        """
        Retrieve historical bars from the Schwab API.

        Args:
            symbol (str): The stock symbol.
            periodType (str): The period type.
            period (int): The period length.
            frequencyType (str): The frequency type.
            freq (int): The frequency value.
            startDate (str): The start date.
            endDate (str): The end date.
            needExtendedHoursData (bool): Whether to include extended hours data.

        Returns:
            dict: The historical bar data.
        """
        tokens = ensure_valid_tokens()
        access_token = tokens["access_token"]

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }

        url = f"https://api.schwabapi.com/marketdata/v1/pricehistory?symbol={symbol}&periodType={periodType}&period={period}&frequencyType={frequencyType}&frequency={freq}&needExtendedHoursData={str(needExtendedHoursData).lower()}"

        if startDate:
            url += f"&startDate={startDate}"
        if endDate:
            url += f"&endDate={endDate}"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for non-200 responses

            data = response.json()

            if not data.get("empty", True):
                candles = [
                    {
                        "datetime": self.convert_timestamp(bar["datetime"]),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume")
                    }
                    for bar in data["candles"]
                ]
                return {
                    "symbol": symbol,
                    "candles": candles,
                    "previousClose": data.get("previousClose"),
                    "previousCloseDate": self.convert_timestamp(data.get("previousCloseDate"))
                }
            else:
                print("DEBUG: Get_hist_bars_[HistoricalDataHandler] No data returned.")
                return None
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            if response.status_code == 401:
                print("ERROR: Token expired, refreshing tokens.")
                ensure_valid_tokens(refresh=True)  # Force token refresh
            elif response.status_code == 429:
                print("ERROR: Rate limit exceeded, retrying after delay.")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                raise
        except Exception as e:
            print(f"ERROR: Failed to fetch historical data: {e}")
            raise

    def convert_timestamp(self, timestamp):
        """
        Convert a timestamp to a formatted datetime string.

        Args:
            timestamp (int): The timestamp to convert.

        Returns:
            str: The formatted datetime string.
        """
        if timestamp is not None:
            return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        return None

    def fetch_top_movers():
        tokens = ensure_valid_tokens()  # Get the token dictionary
        access_token = tokens["access_token"]  # Retrieve the access token

        url = 'https://api.schwabapi.com/marketdata/v1/movers/%24SPX'
        params = {
            'sort': 'VOLUME',
            'frequency': 5
        }
        headers = {
            'Authorization': f'Bearer {access_token}',
            'accept': 'application/json'
        }

        # Make the request
        response = requests.get(url, headers=headers, params=params)

        # Check for token expiration and retry if necessary
        if response.status_code == 401:
            print("Token expired, refreshing...")
            tokens = ensure_valid_tokens(refresh=True)
            access_token = tokens["access_token"]  # Get new token
            headers['Authorization'] = f'Bearer {access_token}'
            response = requests.get(url, headers=headers, params=params)  # Retry with refreshed token

        # Handle the response
        if response.status_code == 200:
            return response.json().get('screeners', [])
        else:
            print("Failed to fetch data", response.status_code)
            return []

    # Fetch top movers data
    top_movers = fetch_top_movers()

    # Filter for tickers with price < $150, then sort by volume in descending order and get top 5
    top_movers_filtered = sorted(
        [mover for mover in top_movers if mover['lastPrice'] < 150], 
        key=lambda x: x['volume'], 
        reverse=True
    )[:5]

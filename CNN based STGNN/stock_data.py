# Import required libraries: finnhub for stock data API, pandas for data manipulation,
# and datetime for handling date and time operations.
import finnhub
import pandas as pd
from datetime import datetime

# Define the API key for Finnhub, used to authenticate requests to the Finnhub API.
api_key = 'ckkos71r01qjc9bgfjt0ckkos71r01qjc9bgfjtg'

# Initialize a Finnhub client with the provided API key.
finnhub_client = finnhub.Client(api_key=api_key)

# Define a list of stock ticker symbols that you want to fetch data for.
comps = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]


# Define a function to fetch stock data from Finnhub.
# It takes a client object, stock symbol, start, and end date as inputs and returns stock candle data.
def fetch_stock_data(client, symbol, start_date, end_date):
    return client.stock_candles(symbol, 'D', start_date, end_date)


# Convert the start and end dates to Unix timestamp format.
# Here, data for the entire year of 2022 is fetched.
start_date = int(datetime(2022, 1, 1).timestamp())
end_date = int(datetime(2022, 12, 31).timestamp())

# Initialize an empty DataFrame to store the combined stock data.
all_data = pd.DataFrame()

# Loop over each stock symbol in the list 'comps'.
for symbol in comps:
    # Fetch stock data for the current symbol.
    data = fetch_stock_data(finnhub_client, symbol, start_date, end_date)

    # Check if data is available and contains the 'close' price key.
    if data and 'c' in data:
        # Convert the fetched data into a pandas DataFrame with appropriate column names.
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            f'{symbol}_Open': data['o'],
            f'{symbol}_High': data['h'],
            f'{symbol}_Low': data['l'],
            f'{symbol}_Close': data['c'],
            f'{symbol}_Volume': data['v']
        })

        # Merge this DataFrame with the 'all_data' DataFrame based on the 'Date' column.
        # If 'all_data' is empty, it gets initialized with the first stock's data.
        if all_data.empty:
            all_data = df
        else:
            all_data = pd.merge(all_data, df, on='Date', how='outer')

# After compiling data from all stocks, save it to a CSV file named 'stock.csv'.
all_data.to_csv('stock.csv', index=False)



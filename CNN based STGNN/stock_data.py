import finnhub
import pandas as pd
from datetime import datetime

api_key = 'ckkos71r01qjc9bgfjt0ckkos71r01qjc9bgfjtg'

finnhub_client = finnhub.Client(api_key=api_key)

comps = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

def fetch_stock_data(client, symbol, start_date, end_date):
    return client.stock_candles(symbol, 'D', start_date, end_date)

start_date = int(datetime(2022, 1, 1).timestamp())
end_date = int(datetime(2022, 12, 31).timestamp())

all_data = pd.DataFrame()

for symbol in comps:
    data = fetch_stock_data(finnhub_client, symbol, start_date, end_date)
    if data and 'c' in data:
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            f'{symbol}_Open': data['o'],
            f'{symbol}_High': data['h'],
            f'{symbol}_Low': data['l'],
            f'{symbol}_Close': data['c'],
            f'{symbol}_Volume': data['v']
        })
        if all_data.empty:
            all_data = df
        else:
            all_data = pd.merge(all_data, df, on='Date', how='outer')

all_data.to_csv('stock.csv', index=False)



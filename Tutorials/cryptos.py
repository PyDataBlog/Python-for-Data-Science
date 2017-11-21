import pandas as pd
import matplotlib.pyplot as plt

# load data of various cryptos from kaggle datasets:
# https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory/downloads/cryptocurrencypricehistory.zip

# if you run in jupyter, simply make sure that all CSV files are in the same folder as the (.ipynb) notebook file
# other IDEs should load the CSV filepath accordingly


def create_csv(name):
    file_path = f'{name}_price.csv'
    return pd.read_csv(file_path, parse_dates=True, index_col='Date')


dataset_to_create = ['bitcoin', 'bitcoin_cash', 'dash', 'ethereum_classic',
                     'bitconnect', 'litecoin', 'monero', 'nem',
                     'neo', 'numeraire', 'omisego',
                     'qtum', 'ripple', 'stratis', 'waves']

cryptos = [create_csv(currency) for currency in dataset_to_create]

bitcoin = cryptos[0]
bitcoin_cash = cryptos[1]
dash = cryptos[2]
ethereum_classic = cryptos[3]
bitconnect = cryptos[4]
litecoin = cryptos[5]
monero = cryptos[6]
nem = cryptos[7]
neo = cryptos[8]
numeraire = cryptos[9]
omisego = cryptos[10]
qtum = cryptos[11]
ripple = cryptos[12]
stratis = cryptos[13]
waves = cryptos[14]

dataset = [bitcoin, bitcoin_cash, dash,
           ethereum_classic, bitconnect,
           litecoin, monero, nem, neo,
           numeraire, omisego, qtum,
           ripple,stratis, waves]

for item in dataset:
    item.sort_index(inplace=True)
    item['30_day_mean'] = item['Close'].rolling(window=20).mean()
    item['30_day_volatility'] = item['Close'].rolling(window=20).std()

ethereum_classic[['Close','30_day_mean', '30_day_volatility']].plot(figsize=(10,8));
plt.title('Ethereum Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

print(bitcoin['Close'].resample('A').mean())
print(bitcoin['Close'].resample('A').apply(lambda x: x[-1]))

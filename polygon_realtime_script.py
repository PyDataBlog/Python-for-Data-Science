import time
import ast
import sqlite3
import pandas as pd
from typing import List
from polygon import WebSocketClient, STOCKS_CLUSTER, CRYPTO_CLUSTER, FOREX_CLUSTER


def my_custom_process_message(messages: List[str]):
    """
        Custom processing function for incoming streaming messages.
    """
    def add_message_to_list(message):
        """
            Simple function that parses dict objects from incoming message.
        """
        messages.append(ast.literal_eval(message))

    return add_message_to_list


def main(waiting_time = seconds):
    """
        Main function which connects to live stream data, and saves incoming data over
        some pre-determined time in an sqlite database.
    """
    key = 'YOUR-API-KEY-HERE'
    messages = []
    #my_client = WebSocketClient(STOCKS_CLUSTER, key, my_custom_process_message(messages))
    my_client = WebSocketClient(CRYPTO_CLUSTER, key, my_custom_process_message(messages))
    #my_client = WebSocketClient(FOREX_CLUSTER, key, my_custom_process_message(messages))
    my_client.run_async()

    #my_client.subscribe("T.MSFT", "T.AAPL", "T.AMD", "T.NVDA")  # Stock data
    my_client.subscribe("XA.BTC-USD", "XA.ETH-USD", "XA.LTC-USD")  # Crypto data
    #my_client.subscribe("C.USD/CNH", "C.USD/EUR")  # Forex data
    time.sleep(waiting_time)

    my_client.close_connection()

    df = pd.DataFrame(messages)

    df = df.iloc[5:, 0].to_frame()
    df.columns = ["data"]
    df["data"] = df["data"].astype("str")

    df = pd.json_normalize(df["data"].apply(lambda x : dict(eval(x))))

    # export data to sqlite
    with sqlite3.connect("realtime_crypto.sqlite") as conn:
        df.to_sql("data", con=conn, if_exists="append", index=False)


if __name__ == "__main__":
    main(waiting_time=60 * 420)

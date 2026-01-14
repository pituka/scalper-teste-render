from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
import pandas as pd

class CandleStream:
    def __init__(self):
        self.client = FuturesWebsocketClient()
        self.data = {}

    def start(self, symbol):
        self.client.start()

        self.client.kline(
            symbol=symbol.lower(),
            interval="5m",
            id=symbol,
            callback=self._on_kline
        )

    def _on_kline(self, msg):
        sym = msg["s"]
        k = msg["k"]

        candle = {
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "vol": float(k["v"]),
        }

        if sym not in self.data:
            self.data[sym] = []

        self.data[sym].append(candle)

        # manter só 300 candles
        self.data[sym] = self.data[sym][-300:]

    def get_df(self, symbol):
        if symbol not in self.data:
            return None
        return pd.DataFrame(self.data[symbol])


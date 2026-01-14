from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

class UserStream:
    def __init__(self, api_key, api_secret):
        self.client = UMFuturesWebsocketClient()
        self.positions = {}
        self.open_orders = {}
        self.last_trade = {}

        self.api_key = api_key
        self.api_secret = api_secret

    def start(self):
        self.client.start()

        # inicia user data stream
        self.client.user_data(
            api_key=self.api_key,
            api_secret=self.api_secret,
            callback=self._on_user_event
        )

    def _on_user_event(self, msg):
        event = msg.get("e")

        # ============================
        # POSIÇÕES (ACCOUNT_UPDATE)
        # ============================
        if event == "ACCOUNT_UPDATE":
            data = msg.get("a", {})
            positions = data.get("P", [])

            for p in positions:
                symbol = p["s"]
                amt = float(p["pa"])

                if amt != 0:
                    self.positions[symbol] = p
                else:
                    if symbol in self.positions:
                        del self.positions[symbol]

        # ============================
        # ORDENS E EXECUÇÕES (ORDER_TRADE_UPDATE)
        # ============================
        if event == "ORDER_TRADE_UPDATE":
            o = msg.get("o", {})
            symbol = o.get("s")

            if not symbol:
                return

            status = o.get("X")  # NEW, FILLED, CANCELED

            # ORDEM NOVA
            if status == "NEW":
                self.open_orders.setdefault(symbol, []).append(o)

            # ORDEM CANCELADA
            if status == "CANCELED":
                if symbol in self.open_orders:
                    self.open_orders[symbol] = [
                        x for x in self.open_orders[symbol] if x["i"] != o["i"]
                    ]

            # EXECUÇÃO (FILL)
            if status == "FILLED":
                self.last_trade[symbol] = o

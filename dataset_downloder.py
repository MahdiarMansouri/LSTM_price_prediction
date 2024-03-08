from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

# Configure your MT5 account details
account_number = 516701  # Replace with your account number
password = 'p&X7pbw#AFyK'  # Replace with your password
server = 'OtetGroup-MT5'  # Replace with your server
path = 'C:\Program Files\MetaTrader 5\\terminal64.exe'

# Initialize MT5 connection
account = mt5.initialize(path=path,
                         login=account_number,
                         password=password,
                         server=server,
                         portable=False)

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Print account information
account_info = mt5.account_info()
if account_info is not None:
    print(f"Balance: {account_info.balance}, Equity: {account_info.equity}")
    print('_' * 30)

symbol = 'EURUSD.ecn'


list = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 90000)
df = pd.DataFrame(list)
print(df.columns)
df.to_csv('XAUUSD-M15')
print(len(list))

mt5.shutdown()

import predict
import config
import argparse
import ccxt


binance = ccxt.binance(config={
    'apiKey': config.apiKey,
    'secret': config.apiSecurity,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})
def long_market(currency, price, amount):
    order = binance.create_limit_buy_order(
        symbol=currency,
        price=price,
        amount=amount
    )
    return order

def short_market(currency, price, amount):
    order = binance.create_market_sell_order(
        symbol=currency,
        price=price,
        amount=amount,
    )
    return order

def main(options):

    ##### CUSTOM OPTIONS #####
    amount = options["amount"] # BTC
    tpsl = options["tpsl"]
    lvrg = options["lvrg"]

    _, position = predict.main()


    ##### DEFAULT #####
    currency = 'BTCUSDT'



    if lvrg is True:
        lvrg = int(abs(gap / cur * 5000)) + 1
        client.futures_change_leverage(symbol=currency, leverage=lvrg)
    else:
        client.futures_change_leverage(symbol=currency, leverage=20)

    ################# CREATE ORDER ####################
    if test is False:
        if position == "long":
            long_market(currency, amount)
        else:
            short_market(currency, amount)

    return "HI"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs='+' ,help='Example) index.html', dest='filename')
    parser.add_argument('--optional', '-o', nargs='*', help='Example) save', default=[], dest='option')

    filename_list = parser.parse_args().filename
    option_list = parser.parse_args().option

    return filename_list, option_list

if __name__ == "__main__":
    options = get_arguments()
    main(options)
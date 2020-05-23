---
layout: post
title: "Bitcoin Futures Arbitrage Part 1"
categories:
  - Bitcoin Futures Series
tags:
  - bitcoin
  - futures
  - perpetual future
  - deribit
  - python
  - arbitrage
  - data science
  - investments
---

This is the first of a series about bitcoin futures on the exchange [Deribit.com](https://www.deribit.com). We will download the data we need to do our future analysis and get some background on how futures work.

Over the entire series I plan to (1) explore historical prices on bitcoin futures, (2) discover potential arbitrage opportunities and (3) do a full investment analysis of the potential profits from arbitrage strategies.

- [Part 1 - Getting the data]({% post_url 2019-05-11-bitcoin-futures-arbitrage-part-1 %})
- [Part 2 - Were there arbitrage profits in the past?]({% post_url 2019-05-12-bitcoin-futures-arbitrage-part-2 %})
- [Part 3 - Perpetual futures 101]({% post_url 2019-05-20-bitcoin-futures-arbitrage-part-3 %})
- [Part 4 - Arbitrage Profit Analysis]({% post_url 2019-05-24-bitcoin-futures-arbitrage-part-4 %})

## Background Information

Users on Deribit can trade in cryptocurrency derivates including options and futures. However, users are not able to buy or sell cryptocurrency on the platform. It doesn't accept traditional fiat currency; all deposits and withdrawals are done in crypto. Both the futures and options are cash-settled, simply transferring the profits and losses between the bitcoin accounts of the platform users. 

However, derivative prices are always based on an underlying asset price. The bitcoin futures and options at Deribit have payoffs based on the price of bitcoin in USD.

Thus, Deribit needs a way to track the fiat price of Bitcoin to determine the payoffs on options and futures. This brings us to our first important concept. 

### The Deribit Bitcoin Index

Deribit's Bitcoin index is **the** price of bitcoin by which all their derivatives profits are based on. It is an average of the current mid-price on 6 large cryptocurrency exchanges. Further information can be found [here](https://www.deribit.com/main#/indexes). Whenever I mention the *index* I mean this.

### Regular Futures vs Perpetual Futures

Deribit sells two types of futures, regular and perpetual. 

Any listings with an expiry date are just typical futures contracts. The specification can be found [here](https://www.deribit.com/pages/docs/futures). Wiki and Investopedia can provide good explanations of how they work. When I refer to a *future* in this document, I mean a typical future.

Perpetual futures are very different and have unusual characteristics. We will discuss them in length in part 2. When I refer to a *perpetual*, I mean this special type of perpetual future. 

## Historical Future Price Data

Let's download the data! Below is some python code that you can use to get all trade data for any instrument. I used this to get the sales price of every transaction for all Deribit futures in the last half of 2018 and 2019, which I will start exploring in part 2.


```python
import requests
from time import sleep

def download(instrument_names):
    """Downloads all past data for the provided instument names

    Parameters
    ----------
    instrument_names: iterable
        The list of instrument names to download.
    """
    for name in instrument_names:
        with open(f'downloads/{name}.txt', 'w') as txt:
            txt.write('timestamp,instrument_name,price,index_price\n')
            has_more = True
            seq = 1
            count = 1000
            while has_more:
                url = f'https://www.deribit.com/api/v2/public/get_last_trades_by_instrument?instrument_name={name}&start_seq={seq}&count={count}&include_old=true'
                r = None
                for _ in range(5):
                    while r is None:
                        try:
                            r = requests.get(url, timeout=5)
                        except Timeout:
                            sleep(2)
                            pass
                r = r.json()
                for trade in r['result']['trades']:
                    timestamp = trade['timestamp']
                    instrument_name = trade['instrument_name']
                    price = trade['price']
                    index_price = trade['index_price']
                    txt.write(f'{timestamp},{instrument_name},{price},{index_price}\n')
                seq += count
                has_more = r['result']['has_more']

def get_instrument_names(currency='BTC', kind='future', expired='true'):
    """Get instrument names
    
    Parameters
    ----------
    currency: string
        The currency of instrument names to download
    kind: string
        'future' or 'option'
    expired: bool
        past instruments too or only current ones
    """
    url = f'https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&kind={kind}&expired={expired}'
    r = None
    for x in range(5):
        while r is None:
            try:
                r = requests.get(url, timeout=5)
            except Timeout:
                sleep(2)
                pass
    r = r.json()
    for instrument in r['result']:
        yield instrument['instrument_name']

```

# import yfinance as yf
# import numpy as np
# from scipy.stats import norm
# import datetime
# import pandas as pd
# from scipy.interpolate import interp1d
# import time
# import requests
# from bs4 import BeautifulSoup

# def black_scholes(S, K, T, r, sigma, option_type='call'):
#     """블랙-숄즈 모델을 사용하여 옵션 가격을 계산합니다."""
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)

#     if option_type.lower() == 'call':
#         price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
#     elif option_type.lower() == 'put':
#         price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
#     else:
#         raise ValueError("옵션 유형은 'call' 또는 'put'이어야 합니다.")

#     return price

# def get_current_price(ticker):
#     """데이터 소스 순차적으로 시도하여 현재 주가를 가져옵니다. 각 단계별 출력 추가."""
#     print(f"{ticker} 주가 가져오기 시도 시작...")
#     price = None

#     # 1. yfinance 실시간 주가 시도
#     print("yfinance 실시간 주가 시도.")
#     try:
#         yf_ticker = yf.Ticker(ticker)
#         price = yf_ticker.info['regularMarketPrice']
#         print(f"yfinance 실시간 주가 사용! 주가: {price}")
#         return price
#     except Exception as e:
#         print(f"yfinance 오류 ({ticker}): {e}")

#     # 2. Yahoo Finance 웹 스크래핑 시도
#     if price is None:
#         print("yfinance 실패. Yahoo Finance 웹 스크래핑 시도.")
#         price = try_yahoo_finance(ticker)
#         if price is not None:
#             print(f"Yahoo Finance 웹 스크래핑 성공! 주가: {price}")
#             return price
#         else:
#             print("Yahoo Finance 웹 스크래핑 실패.")

#     # 3. Google Finance API 시도
#     if price is None:
#         print("Yahoo Finance 웹 스크래핑 실패. Google Finance API 시도.")
#         price = try_google_finance(ticker)
#         if price is not None:
#             print(f"Google Finance API 성공! 주가: {price}")
#             return price
#         else:
#             print("Google Finance API 실패.")

#     print(f"{ticker} 주가 가져오기 최종 실패.")
#     return price

# def try_yahoo_finance(ticker):
#     """Yahoo Finance에서 웹 스크래핑을 시도합니다."""
#     try:
#         url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()

#         soup = BeautifulSoup(response.content, "html.parser")
#         price_element = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
#         if price_element:
#             return float(price_element.text)
#         else:
#             print(f"Yahoo Finance에서 가격 요소를 찾을 수 없습니다. ({ticker})")
#             return None

#     except requests.exceptions.RequestException as e:
#         print(f"Yahoo Finance 웹 스크래핑 요청 오류 ({ticker}): {e}")
#         return None
#     except ValueError:
#         print(f"주가 변환 오류 (잘못된 형식) ({ticker})")
#         return None
#     except Exception as e:
#         print(f"기타 Yahoo Finance 웹 스크래핑 오류 ({ticker}): {e}")
#         return None

# def try_google_finance(ticker):
#     """Google Finance API를 사용하여 주가를 가져옵니다."""
#     try:
#         url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()

#         soup = BeautifulSoup(response.content, "html.parser")
#         price_element = soup.find("div", class_="YMlKec fxKbKc")
#         if price_element:
#             price_str = price_element.text.replace(",", "").replace("$", "")
#             return float(price_str)
#         else:
#             print(f"Google Finance에서 가격 요소를 찾을 수 없습니다. ({ticker})")
#             return None

#     except requests.exceptions.RequestException as e:
#         print(f"Google Finance API 요청 오류 ({ticker}): {e}")
#         return None
#     except ValueError:
#         print(f"주가 변환 오류 (잘못된 형식) ({ticker})")
#         return None
#     except Exception as e:
#         print(f"기타 Google Finance API 오류 ({ticker}): {e}")
#         return None

# def get_implied_volatility(ticker, expiry_date, underlying_price):
#     """yfinance에서 옵션 데이터를 가져오고 내재 변동성을 계산합니다."""
#     try:
#         stock = yf.Ticker(ticker)
#         option_chain = stock.option_chain(expiry_date)

#         if option_chain.calls.empty or option_chain.puts.empty:
#             raise ValueError(f"{ticker}의 {expiry_date} 옵션 데이터를 찾을 수 없습니다.")

#         call_options = option_chain.calls.dropna(subset=['strike', 'impliedVolatility'])

#         if call_options.empty:
#             raise ValueError(f"{ticker}의 유효한 콜옵션 데이터가 없습니다.")

#         strikes = call_options['strike'].values
#         ivs = call_options['impliedVolatility'].values

#         if strikes.size < 2: # 최소 2개 이상의 행사가격 필요
#             raise ValueError(f"{ticker}의 행사가격이 부족하여 내재변동성 보간이 불가능합니다.")

#         iv_interp = interp1d(strikes, ivs, kind='linear', fill_value="extrapolate")
#         return iv_interp

#     except Exception as e:
#         print(f"내재변동성 계산 오류 ({ticker}, {expiry_date}): {e}")
#         return None

# def calculate_us_option_price(ticker, expiry_date_str, strike_price, option_type, interest_rate=0.04):
#     """미국 옵션 가격을 계산합니다."""

#     underlying_price = get_current_price(ticker)

#     if underlying_price is None:
#         print(f"{ticker}의 주가를 가져오는데 실패하여 옵션 가격 계산을 중단합니다.") # 주가 획득 실패 시 메시지 추가
#         return None

#     implied_volatility_func = get_implied_volatility(ticker, expiry_date_str, underlying_price)

#     if implied_volatility_func is None:
#         print(f"{ticker}의 내재 변동성을 계산하는데 실패하여 옵션 가격 계산을 중단합니다.") # 내재 변동성 계산 실패 시 메시지 추가
#         return None

#     try:
#         expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
#         today = datetime.date.today()
#         T = (expiry_date - today).days / 365.25
#     except ValueError:
#         print("날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력해주세요.")
#         return None

#     # 행사가격을 사용하여 내재 변동성 추정
#     sigma = implied_volatility_func(strike_price)

#     # 블랙-숄즈 모델을 사용하여 옵션 가격 계산
#     option_price = black_scholes(underlying_price, strike_price, T, interest_rate, sigma, option_type)

#     print(f"옵션 가격 계산 완료: {option_price:.2f} USD")
#     return option_price

# # 예시 실행
# if __name__ == "__main__":
#     ticker = "IONQ"  # 예시 종목
#     expiry_date_str = "2024-12-20"  # 만기일 예시
#     strike_price = 40  # 행사가격 예시
#     option_type = "call"  # 옵션 유형 예시

#     price = calculate_us_option_price(ticker, expiry_date_str, strike_price, option_type)

import yfinance as yf
import numpy as np
from scipy.stats import norm
import datetime
import pandas as pd
import time

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """블랙-숄즈 모델을 사용하여 옵션 가격을 계산합니다."""
    try:  # 0으로 나누는 오류 방지
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    except ZeroDivisionError:
        print("ZeroDivisionError in black_scholes. Returning 0.")
        return 0.0

    if option_type.lower() == 'call':
        price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
    elif option_type.lower() == 'put':
        price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
    else:
        raise ValueError("옵션 유형은 'call' 또는 'put'이어야 합니다.")

    return price

def get_current_price(ticker):
    """yfinance를 사용하여 현재 주가를 가져옵니다."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if 'regularMarketPrice' in info:
            return info['regularMarketPrice']
        else:
            return yf_ticker.history(period="1d")['Close'][0] # 장외 시간일 경우 history에서 가져오기
    except Exception as e:
        print(f"주가 가져오기 오류 ({ticker}): {e}")
        return None

def implied_volatility(option_price, S, K, T, r, option_type):
    """이분법을 사용하여 내재 변동성을 계산합니다. 오류 처리 강화."""
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-4
    sigma_low = 0.01
    sigma_high = 2.0
    sigma = (sigma_low + sigma_high) / 2.0

    for i in range(MAX_ITERATIONS):
        try:
            price = black_scholes(S, K, T, r, sigma, option_type)
            diff = price - option_price

            if abs(diff) < PRECISION:
                return sigma

            if diff > 0:
                sigma_high = sigma
            else:
                sigma_low = sigma
            sigma = (sigma_low + sigma_high) / 2.0

        except OverflowError: # sigma가 너무 커서 계산 불가할 경우
            sigma_high = sigma
            sigma = (sigma_low + sigma_high) / 2.0
            print("OverflowError 발생. sigma 값을 조정합니다.")
        except ZeroDivisionError: # sigma가 0에 너무 가까워 계산 불가할 경우
            sigma_low = sigma
            sigma = (sigma_low + sigma_high) / 2.0
            print("ZeroDivisionError 발생. sigma 값을 조정합니다.")
        except Exception as e:
            print(f"내재 변동성 계산 중 예외 발생: {e}")
            return None

    print("내재 변동성 계산 실패 (최대 반복 횟수 초과).")
    return None

def calculate_option_price(ticker, expiry_date_str, strike_price, option_type, interest_rate=0.04):
    """옵션 가격을 계산합니다."""
    underlying_price = get_current_price(ticker)
    if underlying_price is None:
        print(f"{ticker}의 주가를 가져오는데 실패하여 옵션 가격 계산을 중단합니다.")
        return None

    try:
        expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        T = (expiry_date - today).days / 365.25
        if T <= 0 : # T가 0보다 작거나 같을 경우 오류 메세지 출력 후 None 반환
            print("만기일이 현재 날짜보다 이전입니다.")
            return None
    except ValueError:
        print("날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력해주세요.")
        return None
    
    try:
        yf_ticker = yf.Ticker(ticker)
        option_data = yf_ticker.option_chain(expiry_date_str)
        
        if option_type.lower() == 'call':
            options = option_data.calls
        elif option_type.lower() == 'put':
            options = option_data.puts
        else:
            print("옵션 유형이 잘못되었습니다. 'call' 또는 'put'을 입력해주세요.")
            return None
        
        target_option = options[options['strike'] == strike_price]
        if target_option.empty:
            print(f"해당 행사가격({strike_price})의 옵션이 존재하지 않습니다.")
            return None
        
        market_price = target_option['lastPrice'].iloc[0]

        iv = implied_volatility(market_price, underlying_price, strike_price, T, interest_rate, option_type)

        if iv is None:
            print(f"내재 변동성 계산에 실패했습니다.")
            return None
        
        calculated_price = black_scholes(underlying_price, strike_price, T, interest_rate, iv, option_type)

        print(f"시장 가격: {market_price:.2f} USD")
        print(f"계산된 가격: {calculated_price:.2f} USD")
        print(f"내재 변동성: {iv:.4f}")

        return calculated_price
        
    except Exception as e:
        print(f"옵션 정보 가져오기 오류: {e}")
        return None

# 예시 실행
if __name__ == "__main__":
    ticker = "AAPL"
    expiry_date_str = "2024-12-20"
    strike_price = 170
    option_type = "call"

    calculate_option_price(ticker, expiry_date_str, strike_price, option_type)

    ticker = "IONQ"
    expiry_date_str = "2024-12-20"
    strike_price = 42
    option_type = "call"
    calculate_option_price(ticker, expiry_date_str, strike_price, option_type)
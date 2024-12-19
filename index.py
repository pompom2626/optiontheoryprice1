import yfinance as yf
import numpy as np
from scipy.stats import norm
import datetime
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def get_current_price(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return info['regularMarketPrice']
        else:
            hist = yf_ticker.history(period="1d")
            if not hist.empty:
                return hist['Close'][-1]
            else:
                print(f"{ticker}의 1일 주가 기록이 없습니다.")
                return None
    except (IndexError, KeyError, TypeError):
        print(f"주가 정보가 없습니다. ({ticker})")
        return None
    except Exception as e:
        print(f"주가 가져오기 오류 ({ticker}): {e}")
        return None

def black_scholes(S, K, T, r, sigma, option_type='call'):
    try:
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

def binomial_option_price(S, K, T, r, sigma, n=100, option_type='call'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.zeros((n + 1, n + 1))
    prices[0, 0] = S
    for i in range(1, n + 1):
        for j in range(i + 1):
            prices[i, j] = S * (u ** j) * (d ** (i - j))

    option_values = np.zeros((n + 1, n + 1))
    if option_type.lower() == 'call':
        option_values[n, :] = np.maximum(prices[n, :] - K, 0)
    elif option_type.lower() == 'put':
        option_values[n, :] = np.maximum(K - prices[n, :], 0)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i + 1, j + 1] + (1 - p) * option_values[i + 1, j])

    return option_values[0, 0]

def implied_volatility(option_price, S, K, T, r, option_type):
    MAX_ITERATIONS = 500
    PRECISION = 1.0e-4
    sigma_low = 0.001
    sigma_high = 5.0
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

        except OverflowError:
            sigma_high = sigma
            sigma = (sigma_low + sigma_high) / 2.0
            print("OverflowError 발생. sigma 값을 조정합니다.")
        except ZeroDivisionError:
            sigma_low = sigma
            sigma = (sigma_low + sigma_high) / 2.0
            print("ZeroDivisionError 발생. sigma 값을 조정합니다.")
        except Exception as e:
            print(f"내재 변동성 계산 중 예외 발생: {e}")
            return None

    print("내재 변동성 계산 실패 (최대 반복 횟수 초과).")
    return None

def calculate_and_compare_option_prices(ticker, expiry_date_str, strike_price, option_type, interest_rate=0.04):
    underlying_price = get_current_price(ticker)
    if underlying_price is None:
        return None

    try:
        expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        T = (expiry_date - today).days / 365.25

        if T <= 0:
            print("만기일이 현재 날짜보다 이전입니다.")
            return None

    except ValueError:
        print("날짜 형식이 잘못되었습니다. %Y-%m-%d 형식으로 입력해주세요.")
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

        market_price = target_option['lastPrice'].iloc[0] if not target_option.empty and not pd.isna(target_option['lastPrice'].iloc[0]) else None
        bid = target_option['bid'].iloc[0] if not target_option.empty and not pd.isna(target_option['bid'].iloc[0]) else None
        ask = target_option['ask'].iloc[0] if not target_option.empty and not pd.isna(target_option['ask'].iloc[0]) else None

        if market_price is None or bid is None or ask is None:
            print("옵션 가격 정보가 없습니다.")
            return None

        iv = implied_volatility(market_price, underlying_price, strike_price, T, interest_rate, option_type)

        if iv is None:
            print(f"내재 변동성 계산에 실패했습니다.")
            return None

        bs_price = black_scholes(underlying_price, strike_price, T, interest_rate, iv, option_type)
        binomial_price = binomial_option_price(underlying_price, strike_price, T, interest_rate, iv, option_type=option_type)

        print(f"Underlying Price: {underlying_price:.2f}")
        print(f"Time to Expiry (T): {T:.4f}")
        print(f"Black-Scholes Price: {bs_price:.2f}")
        print(f"Binomial Price: {binomial_price:.2f}")

        return bs_price, binomial_price

    except Exception as e:
        print(f"옵션 정보 가져오기 오류: {e}")
        return None

# ------ 메인 실행 블록 (중요: 다른 함수들과 같은 들여쓰기 레벨에 위치) ------
if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "MSFT", "IONQ", "IONQ"]
    expiry_dates = ["2025-01-17", "2025-06-20", "2025-12-19", "2024-12-20", "2024-12-20"]
    strike_prices = [170, 180, 300, 38, 40]
    option_types = ["call", "put", "call", "call", "call"]

    results = []

    print("옵션 가격 계산 시작...\n")

    for ticker, expiry_date, strike_price, option_type in zip(tickers, expiry_dates, strike_prices, option_types):
        print(f"--- {ticker} {expiry_date} {strike_price} {option_type} ---")
        result = calculate_and_compare_option_prices(ticker, expiry_date, strike_price, option_type)
        if result:
            results.append({"Ticker": ticker, "Expiry Date": expiry_date, "Strike Price": strike_price, "Option Type": option_type, "BS Price": result[0], "Binomial Price": result[1]})
        print("-" * 30)

    # 추가 테스트 케이스
    print("\n--- 추가 테스트 ---")
    calculate_and_compare_option_prices("AAPL", "2025/01/17", 170, "call")  # 잘못된 날짜 형식
    calculate_and_compare_option_prices("AAPL", "2025-01-17", 999, "call")# 존재하지 않는 행사가격

    print("-" * 30)



    if results:
        df_results = pd.DataFrame(results)
        print("\n--- 요약 결과 ---")
        print(df_results)
    print("\n옵션 가격 계산 완료.")
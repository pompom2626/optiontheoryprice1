import yfinance as yf
import numpy as np
from scipy.stats import norm
import datetime

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """블랙-숄즈 모델을 사용하여 옵션 가격을 계산합니다."""
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

def get_current_price(ticker):
    """yfinance를 사용하여 현재 주가를 가져옵니다."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if 'regularMarketPrice' in info:
            return info['regularMarketPrice']
        else:
            return yf_ticker.history(period="1d")['Close'][0]
    except (IndexError, KeyError): # KeyError 추가
        print(f"주가 정보가 없습니다. ({ticker})")
        return None
    except Exception as e:
        print(f"주가 가져오기 오류 ({ticker}): {e}")
        return None


def implied_volatility(option_price, S, K, T, r, option_type):
    """이분법을 사용하여 내재 변동성을 계산합니다. 초기 범위 확장 및 최대 반복 횟수 증가."""
    MAX_ITERATIONS = 500
    PRECISION = 1.0e-4
    sigma_low = 0.001  # 초기 범위 확장
    sigma_high = 5.0   # 초기 범위 확장
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

def calculate_option_price(ticker, expiry_date_str, strike_price, option_type, interest_rate=0.04):
    """옵션 가격 및 관련 정보를 계산하고 출력합니다."""
    underlying_price = get_current_price(ticker)
    if underlying_price is None:
        print(f"{ticker}의 주가를 가져오는데 실패하여 옵션 가격 계산을 중단합니다.")
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

        market_price = target_option['lastPrice'].iloc[0]
        bid = target_option['bid'].iloc[0]
        ask = target_option['ask'].iloc[0]
        
        if np.isnan(market_price) or np.isnan(bid) or np.isnan(ask): # 추가: NaN 값 처리
            print("옵션 가격 정보가 없습니다.")
            return None

        iv = implied_volatility(market_price, underlying_price, strike_price, T, interest_rate, option_type)

        if iv is None:
            print(f"내재 변동성 계산에 실패했습니다.")
            print("Calculated Price 계산을 생략합니다.")
            return None

        calculated_price = black_scholes(underlying_price, strike_price, T, interest_rate, iv, option_type)

        print(f"Underlying Price: {underlying_price:.2f}")
        print(f"Time to Expiry (T): {T:.4f}")
        print(f"Calculated IV: {iv:.4f}")
        print(f"Bid: {bid:.2f} USD")
        print(f"Ask: {ask:.2f} USD")
        print(f"Market Price: {market_price:.2f} USD")
        print(f"Calculated Price: {calculated_price:.2f} USD")
        print(f"Mid Price: {(bid+ask)/2:.2f} USD") # Bid와 Ask의 중간값 추가

        return calculated_price

    except Exception as e:
        print(f"옵션 정보 가져오기 오류: {e}")
        return None

# 예시 실행
if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "MSFT"]
    expiry_dates = ["2025-01-17", "2025-06-20", "2025-12-19"]
    strike_prices = [170, 180, 300]
    option_types = ["call", "put", "call"]

    for ticker, expiry_date, strike_price, option_type in zip(tickers, expiry_dates, strike_prices, option_types):
        print(f"\n--- {ticker} {expiry_date} {strike_price} {option_type} ---")
        calculate_option_price(ticker, expiry_date, strike_price, option_type)

    # 잘못된 날짜 형식 테스트
    print("\n--- 잘못된 날짜 형식 테스트 ---")
    calculate_option_price("AAPL", "2025/01/17", 170, "call")

    # 존재하지 않는 행사가격 테스트
    print("\n--- 존재하지 않는 행사가격 테스트 ---")
    calculate_option_price("AAPL", "2025-01-17", 999, "call")  
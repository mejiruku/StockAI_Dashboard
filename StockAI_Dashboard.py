import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import tkinter as tk
from tkinter import ttk

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import webbrowser # アップデート時にブラウザを開くため追加

APP_VERSION = "0.0.1"
GITHUB_REPO = "mejiruku/StockAI_Dashboard"

# ================================================
# Splash Screen (Startup Loading)
# ================================================
class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.configure(bg='#0d1117')
        
        # Center on screen
        w, h = 400, 250
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Design
        tk.Label(self.root, text="📈 Stock AI System", bg='#0d1117', fg='#ffffff', 
                 font=('Segoe UI', 18, 'bold')).pack(pady=(50, 10))
        
        self.label_status = tk.Label(self.root, text="Initializing system...", bg='#0d1117', 
                                     fg='#58a6ff', font=('Segoe UI', 10))
        self.label_status.pack(pady=5)
        
        self.progress = ttk.Progressbar(self.root, length=300, mode='determinate')
        self.progress.pack(pady=20)
        self.progress['value'] = 0
        self.root.update()

    def update(self, text, val):
        self.label_status.config(text=text)
        self.progress['value'] = val
        self.root.update()

    def close(self):
        self.root.destroy()

# Show splash screen immediately
splash = SplashScreen()
splash.update("Loading modules...", 10)

import io
import base64
import contextlib
import threading
import glob
import requests
splash.update("Loading AI engines...", 30)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
splash.update("Preparing AI frameworks (LightGBM)...", 50)

import lightgbm as lgb
from sklearn.metrics import accuracy_score
import mplfinance as mpf
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
splash.update("Preparing RL engine (PPO)...", 70)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import japanize_matplotlib

import logging
import gc
import json

splash.update("Launching dashboard...", 90)

# フォルダの自動作成
for folder in ["models", "reports", "temp"]:
    os.makedirs(folder, exist_ok=True)


# フォントパスの設定
if sys.platform == "win32":
    PDF_FONT_PATH = "C:\\Windows\\Fonts\\msgothic.ttc"
else:
    PDF_FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
LOG_FILE_PATH        = "temp/current_log.txt"
LIST_FILE_PATH       = "temp/screener_lists.sai"
SCREENER_STATE_PATH  = "temp/screener_state.json"

# ログをColabの画面と一時ファイルの両方に出力するためのクラス
original_stdout = sys.stdout
class LiveLogger:
    def write(self, data):
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass
        if original_stdout is not None:
            try:
                original_stdout.write(data)
                original_stdout.flush()
            except Exception:
                pass
    def flush(self):
        if original_stdout is not None:
            try:
                original_stdout.flush()
            except Exception:
                pass

# ================================================
# 強化学習（RL）用の仮想トレード市場
# ================================================
class StockTradingEnv(gym.Env):
    def __init__(self, data_df, features):
        super(StockTradingEnv, self).__init__()
        self.df = data_df.reset_index(drop=True)
        self.features = features
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(features) + 2,), dtype=np.float32)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        self.initial_balance = 100000.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.avg_buy_price = 0.0

    def _get_obs(self):
        market_obs = self.df.loc[self.current_step, self.features].values.astype(np.float32)
        has_pos = 1.0 if self.shares_held > 0 else 0.0
        profit_ratio = 0.0
        if self.shares_held > 0 and self.avg_buy_price > 0:
            current_price = self.df.loc[self.current_step, 'Close']
            if pd.isna(current_price) and self.current_step > 0:
                current_price = self.df.loc[self.current_step - 1, 'Close']
            profit_ratio = (current_price - self.avg_buy_price) / self.avg_buy_price
        obs = np.append(market_obs, [has_pos, profit_ratio])
        return np.nan_to_num(obs).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.avg_buy_price = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        current_price = self.df.loc[self.current_step, 'Close']
        if pd.isna(current_price): current_price = self.df.loc[self.current_step - 1, 'Close'] if self.current_step > 0 else 1.0
        prev_price = self.df.loc[self.current_step - 1, 'Close'] if self.current_step > 0 else current_price
        prev_net_worth = self.net_worth
        trading_fee = 0.0000

        if action == 2:  # Buyい
            if self.balance > 0:
                buy_amount = self.balance * (1 - trading_fee)
                self.shares_held += buy_amount / max(current_price, 1e-8)
                self.balance = 0
                self.avg_buy_price = current_price
        elif action == 0:  # Sellり
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - trading_fee)
                self.shares_held = 0
                self.avg_buy_price = 0.0

        self.net_worth = self.balance + self.shares_held * current_price
        agent_return = ((self.net_worth - prev_net_worth) / max(prev_net_worth, 1.0)) * 100.0
        market_return = ((current_price - prev_price) / max(prev_price, 1.0)) * 100.0

        if self.shares_held > 0:
            reward = agent_return
        else:
            if market_return > 0: reward = -market_return
            elif market_return < 0: reward = abs(market_return)
            else: reward = 0.0

        return self._get_obs(), reward, done, False, {'net_worth': self.net_worth}

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_print_percent = -1

    def _on_step(self) -> bool:
        percent = int((self.num_timesteps / self.total_timesteps) * 20) * 5 # 5%刻み
        if percent > self.last_print_percent and percent <= 100:
            print(f"  ⏳ [RL学習中] {percent}% Done... ({self.num_timesteps}/{self.total_timesteps})")
            self.last_print_percent = percent
        return True

_screener_flags = {'stop': False, 'pause': False}
import time as _time_mod

class InterruptCallback(BaseCallback):
    def __init__(self, flags, verbose=0):
        super().__init__(verbose)
        self.flags = flags
    def _on_step(self) -> bool:
        return not self.flags['stop']

def classify_signal_5(p_c, action):
    if   p_c == 2 and action == 2: return "Strong Buy",  5
    elif p_c == 2 or  action == 2: return "Buy",         4
    elif p_c == 0 and action == 0: return "Strong Sell", 1
    elif p_c == 0 or  action == 0: return "Sell",        2
    else:                          return "Neutral",      3

# ================================================
# アプリケーションロジック本体
# ================================================
class StockAILogic:
    def is_jp_ticker(self, code):
        clean = str(code).upper().strip().replace('.T', '')
        return len(clean) >= 4 and sum(c.isdigit() for c in clean) >= 3

    def show_plot(self, chart_data, candle_days, stock_code, bb_u, bb_l, rsi, macd, msig):
        try:
            print("\n📊 Rendering technical chart...")
            bb_u  = bb_u.reindex(chart_data.index)
            bb_l  = bb_l.reindex(chart_data.index)
            rsi   = rsi.reindex(chart_data.index)
            macd  = macd.reindex(chart_data.index)
            msig  = msig.reindex(chart_data.index)
            add_plots = [
                mpf.make_addplot(bb_u, panel=0, color='#888888', linestyle='--', alpha=0.7),
                mpf.make_addplot(bb_l, panel=0, color='#888888', linestyle='--', alpha=0.7),
                mpf.make_addplot(rsi,  panel=2, color='darkorange', ylabel='RSI(14)'),
                mpf.make_addplot(macd, panel=3, color='royalblue',  ylabel='MACD'),
                mpf.make_addplot(msig, panel=3, color='tomato'),
            ]
            mc = mpf.make_marketcolors(up='#e84040', down='#4488ff', edge='inherit', wick='inherit', volume='inherit')
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', base_mpf_style='nightclouds', facecolor='#1a1a2e', figcolor='#1a1a2e', rc={'axes.labelcolor': '#cccccc', 'xtick.color': '#aaaaaa', 'ytick.color': '#aaaaaa', 'font.family': 'IPAexGothic'})
            fig, axes = mpf.plot(chart_data, type='candle', style=style, title=f'\n  Stock {stock_code}  — AI Candlestick Chart (Last {candle_days} Days)', ylabel='Price (JPY)', volume=True, addplot=add_plots, figsize=(15, 11), panel_ratios=(4, 1.2, 1.5, 1.5), returnfig=True)

            legend_elements = [mpatches.Patch(color='#e84040', label='Bullish (Up)'), mpatches.Patch(color='#4488ff', label='Bearish (Down)'), mpatches.Patch(color='#888888', label='Bollinger Band (2σ)')]
            axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10, facecolor='#2a2a4a', edgecolor='#555555', labelcolor='#cccccc')
            fig.savefig(f"temp/chart_{stock_code}.png", bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        except Exception as e: print(f"\nChart Display Error: {e}")

    def fetch_latest_close(self, code):
        if not self.is_jp_ticker(code): return None
        try:
            ticker = code + ".T" if str(code).isdigit() else code
            res = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5d&interval=1d",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=10
            )
            result = res.json()["chart"]["result"][0]
            closes = result["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if closes:
                return float(closes[-1])
        except Exception:
            pass
        return None

    def prepare_data(self, stock_code):
        CANDLE_DAYS = 120
        def fetch_yahoo_data(t):
            try:
                res = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=max&interval=1d", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if res.status_code != 200: return pd.DataFrame()
                js = res.json()
                if not js.get("chart",{}).get("result"): return pd.DataFrame()
                result = js["chart"]["result"][0]
                if not result.get("timestamp"): return pd.DataFrame()
                dates = (pd.to_datetime(result["timestamp"], unit='s').tz_localize('UTC').tz_convert('Asia/Tokyo').normalize())
                q = result["indicators"]["quote"][0]
                return pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]}, index=dates).sort_index().dropna()
            except Exception:
                return pd.DataFrame()

        data = fetch_yahoo_data(f"{stock_code}.T")
        if data.empty: return None, None, 0
        if len(data) > 2500: data = data.iloc[-2500:]

        try:
            data = data.join(fetch_yahoo_data("JPY=X")[['Close']].rename(columns={'Close': 'MACRO_USDJPY'}).shift(1), how='left')
            data = data.join(fetch_yahoo_data("^GSPC")[['Close']].rename(columns={'Close': 'MACRO_SP500'}).shift(1), how='left')
            data = data.join(fetch_yahoo_data("^N225")[['Close']].rename(columns={'Close': 'MACRO_N225'}).shift(1), how='left')
            data = data.join(fetch_yahoo_data("^VIX")[['Close']].rename(columns={'Close': 'MACRO_VIX'}).shift(1), how='left')
            data[['MACRO_USDJPY', 'MACRO_SP500', 'MACRO_N225', 'MACRO_VIX']] = data[['MACRO_USDJPY', 'MACRO_SP500', 'MACRO_N225', 'MACRO_VIX']].ffill()
        except: pass
        data = data.dropna()
        if len(data) < 50:
            return None, None, len(data)  # データ日数を第3戻り値で返す

        for n in [1, 2, 3, 5, 10, 20]: data[f'Return_{n}'] = data['Close'].pct_change(n)
        for n in [1, 2, 3, 5]: data[f'Lag_Return_{n}'] = data['Return_1'].shift(n)
        for w in [5, 10, 20, 60, 200]: data[f'SMA_ratio_{w}'] = data['Close'] / data['Close'].rolling(w).mean() - 1

        bb_mid = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_upper'], data['BB_lower'] = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
        data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / bb_mid

        tr = pd.concat([data['High'] - data['Low'], (data['High'] - data['Close'].shift()).abs(), (data['Low'] - data['Close'].shift()).abs()], axis=1).max(axis=1)
        data['ATR_ratio'] = tr.rolling(14).mean() / data['Close']

        delta = data['Close'].diff()
        gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
        for p in [9, 14]: data[f'RSI_{p}'] = (100 - 100 / (1 + gain.rolling(p).mean() / loss.rolling(p).mean().replace(0, np.nan))).fillna(50)

        l14, h14 = data['Low'].rolling(14).min(), data['High'].rolling(14).max()
        data['Stoch_K'] = (data['Close'] - l14) / (h14 - l14) * 100
        data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

        data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_sig'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_sig']

        for n in [5, 10, 20]: data[f'Mom_{n}'] = data['Close'] / data['Close'].shift(n) - 1
        data['Vol_ratio_5'] = data['Volume'] / data['Volume'].rolling(5).mean()
        data['Vol_ratio_20'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['HV_10'] = data['Return_1'].rolling(10).std() * np.sqrt(252)
        data['HV_20'] = data['Return_1'].rolling(20).std() * np.sqrt(252)
        data['Gap'] = (data['Open'] - data['Close'].shift()) / data['Close'].shift()

        cr = (data['High'] - data['Low']).replace(0, np.nan)
        data['Body_ratio'], data['Upper_wick_ratio'], data['Lower_wick_ratio'] = (data['Close'] - data['Open']) / cr, (data['High'] - data[['Close', 'Open']].max(axis=1)) / cr, (data[['Close', 'Open']].min(axis=1) - data['Low']) / cr
        data['Is_bullish'] = (data['Close'] > data['Open']).astype(int)
        data['DayOfWeek'], data['Month'] = data.index.dayofweek, data.index.month

        macro_features = []
        if 'MACRO_USDJPY' in data.columns:
            data['MACRO_USDJPY_Ret1'] = data['MACRO_USDJPY'].pct_change(1)
            macro_features += ['MACRO_USDJPY', 'MACRO_USDJPY_Ret1']
        if 'MACRO_SP500' in data.columns:
            data['MACRO_SP500_Ret1'] = data['MACRO_SP500'].pct_change(1)
            macro_features += ['MACRO_SP500_Ret1']
        if 'MACRO_N225' in data.columns:
            data['MACRO_N225_Ret1'] = data['MACRO_N225'].pct_change(1)
            macro_features += ['MACRO_N225_Ret1']
        if 'MACRO_VIX' in data.columns:
            data['MACRO_VIX_Level'] = data['MACRO_VIX']
            macro_features += ['MACRO_VIX_Level']

        data = data.bfill().ffill().fillna(0)
        features = ['Return_1', 'Return_2', 'Return_3', 'Return_5', 'Return_10', 'Return_20', 'Lag_Return_1', 'Lag_Return_2', 'Lag_Return_3', 'Lag_Return_5', 'SMA_ratio_5', 'SMA_ratio_10', 'SMA_ratio_20', 'SMA_ratio_60', 'SMA_ratio_200', 'BB_position', 'BB_width', 'ATR_ratio', 'HV_10', 'HV_20', 'RSI_9', 'RSI_14', 'Stoch_K', 'Stoch_D', 'MACD', 'MACD_sig', 'MACD_hist', 'Mom_5', 'Mom_10', 'Mom_20', 'Vol_ratio_5', 'Vol_ratio_20', 'Gap', 'Body_ratio', 'Upper_wick_ratio', 'Lower_wick_ratio', 'Is_bullish', 'DayOfWeek', 'Month'] + macro_features
        return data, features, CANDLE_DAYS

    def run_ai(self, stock_code):
        if not self.is_jp_ticker(stock_code):
            print(f"⚠️ 🇺🇸 US and foreign stocks are not supported. Prediction skipped for {stock_code}.")
            return

        print(f"📡 Fetching data for ticker {stock_code}...\n")
        data, features, CANDLE_DAYS = self.prepare_data(stock_code)
        if data is None:
            if CANDLE_DAYS > 0:
                print(f"⚠️ Insufficient Data ({CANDLE_DAYS} days) - Cannot analyze newly listed stocks.")
            else:
                print("⚠️ Data Fetch Failed - Check the ticker code.")
            return

        print("🧠 Building AI learning model...\n")
        actual_days = min(CANDLE_DAYS, len(data))
        cd = data.iloc[-actual_days:].copy(); cd.index = cd.index.tz_localize(None)
        self.show_plot(cd[['Open', 'High', 'Low', 'Close', 'Volume']], actual_days, stock_code, cd['BB_upper'], cd['BB_lower'], cd['RSI_14'], cd['MACD'], cd['MACD_sig'])

        for i in range(1, 21):
            threshold = 0.01 * np.sqrt(i)
            data[f'Target_{i}'] = np.select([(data['Close'].shift(-i) - data['Close']) / data['Close'] <= -threshold, (data['Close'].shift(-i) - data['Close']) / data['Close'] >= threshold], [0, 2], default=1).astype(float)
        predict_data = data.iloc[-1:].copy()
        data = data.dropna(subset=[f'Target_{i}' for i in range(1, 21)])

        test_size = 200 if len(data) > 250 else max(5, int(len(data) * 0.2))
        train, test = data.iloc[:-test_size], data.iloc[-test_size:]
        X_latest = predict_data[features].fillna(0)

        def make_model(seed, c_weight): return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, max_depth=6, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.2, random_state=seed, class_weight=c_weight, verbose=-1)
        direction_map = {0: "↓ Down", 1: "→ Flat", 2: "↑ Up"}

        print("=" * 66 + f"\n  Ticker {stock_code} AI Prediction Results\n" + "=" * 66)
        print(f"  {'Period':>4}  {'Accuracy':>6}  {'Prediction':<10}  {'Confidence':>5}  {'Verdict'}\n" + "-" * 66)

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        report_data = []
        for i in range(1, 21):
            y_train, y_test = train[f'Target_{i}'].astype(int).values, test[f'Target_{i}'].astype(int).values
            counts = np.bincount(y_train)
            c_weight = {c: len(y_train)/(len(counts)*count) for c, count in enumerate(counts) if count > 0}

            probas = [make_model(s, c_weight).fit(train[features], y_train).predict_proba(X_latest)[0] for s in [42, 123, 456]]
            pred_class = int(np.argmax(np.mean(probas, axis=0)))
            confidence = np.mean(probas, axis=0)[pred_class] * 100

            acc_scores = []
            for tr_idx, te_idx in tscv.split(train):
                cv_train_X, cv_test_X = train[features].iloc[tr_idx], train[features].iloc[te_idx]
                cv_train_y, cv_test_y = y_train[tr_idx], y_train[te_idx]
                cv_counts = np.bincount(cv_train_y)
                cv_weight = {c: len(cv_train_y)/(len(cv_counts)*count) for c, count in enumerate(cv_counts) if count > 0}
                try:
                    m_cv = make_model(42, cv_weight).fit(cv_train_X, cv_train_y)
                    acc_scores.append(accuracy_score(cv_test_y, m_cv.predict(cv_test_X)))
                except:
                    pass
            
            m_eval = make_model(42, c_weight).fit(train[features], y_train)
            acc_test = accuracy_score(y_test, m_eval.predict(test[features]))
            acc = (np.mean(acc_scores) + acc_test) / 2 if acc_scores else acc_test

            star = "★" if confidence >= 55 else " "
            print(f"  {i:>2} Days Later  {acc*100:>5.1f}%  {direction_map[pred_class]:<10}  {confidence:>4.0f}%  {star}")
            report_data.append([f"{i} Days Later", f"{acc*100:.1f}%", direction_map[pred_class], f"{confidence:.0f}%", star])

        print("\n" + "=" * 66 + "\n  💪 Reinforcement Learning AI (PPO) Simulation...\n" + "=" * 66)
        vec_env = make_vec_env(StockTradingEnv, n_envs=1, env_kwargs={'data_df': train, 'features': features}, vec_env_cls=DummyVecEnv)
        try:
            rl_model = PPO.load(f"models/ppo_{stock_code}.zip", env=vec_env) if os.path.exists(f"models/ppo_{stock_code}.zip") else PPO("MlpPolicy", vec_env, verbose=0, learning_rate=0.0003, n_steps=512)
            rl_model.learn(total_timesteps=20000, callback=ProgressCallback(total_timesteps=20000))
            rl_model.save(f"models/ppo_{stock_code}.zip")
            obs_latest = np.append(X_latest.values[0].astype(np.float32), [0.0, 0.0])
            action, _ = rl_model.predict(np.nan_to_num(obs_latest), deterministic=True)
            rec = {0: "[Downtrend Alert] Sell Immediately 📉", 1: "[Wait & See] Hold Position ⏸️", 2: "[Uptrend Opportunity] Strong Buy 📈"}[int(action)]
            print(f"\n  🤖 Recommended Action ⇒  {rec}\n" + "*" * 66)
            self._create_pdf_report(stock_code, report_data, rec)
        finally:
            try: vec_env.close()
            except: pass
            gc.collect()

    def _create_pdf_report(self, stock_code, report_data, rec):
        try:
            print("\n📄 Generating PDF...")
            path = f"reports/Stock_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            c = canvas.Canvas(path, pagesize=A4)
            pdf_font = 'Helvetica'
            if os.path.exists(PDF_FONT_PATH):
                try:
                    pdfmetrics.registerFont(TTFont('JapaneseFont', PDF_FONT_PATH))
                    pdf_font = 'JapaneseFont'
                except: pass
            
            c.setFont(pdf_font, 18); c.drawString(30, A4[1] - 50, f"Ticker {stock_code} : AI Prediction Report")
            c.setFont(pdf_font, 10); c.drawString(30, A4[1] - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if os.path.exists(f"temp/chart_{stock_code}.png"): c.drawImage(f"temp/chart_{stock_code}.png", 30, A4[1] - 370, width=A4[0]-60, height=280, preserveAspectRatio=True)
            c.setFont(pdf_font, 14); c.drawString(30, A4[1] - 410, "■ Today's AI Recommended Action"); c.setFont(pdf_font, 13); c.drawString(40, A4[1] - 440, rec)
            c.setFont(pdf_font, 14); c.drawString(30, A4[1] - 490, "■ Prediction by Period")
            y = A4[1] - 520; c.setFont(pdf_font, 10)
            x_pos = [40, 100, 180, 320, 380]
            for i, h in enumerate(["Period", "Accuracy", "Direction", "Confidence", "Verdict"]): c.drawString(x_pos[i], y, h)
            c.line(30, y-5, A4[0]-30, y-5); y -= 25
            for row in report_data:
                for i, col in enumerate(row): c.drawString(x_pos[i], y, str(col))
                y -= 20
                if y < 50: c.showPage(); c.setFont(pdf_font, 10); y = A4[1] - 50
            c.save(); print(f"✨ Report saved: {path}")
        except Exception as e: print(f"❌ PDF Error: {e}")

    def fetch_stock_name(self, stock_code):
        try:
            name = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_code}.T?range=1d&interval=1d", headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()["chart"]["result"][0].get("meta", {}).get("longName", "Unknown")
            return name if name else "Unknown"
        except: return "Unknown"

    def run_screener(self, mode, tickers, resume_done=None, resume_results=None):
        global _screener_flags
        _screener_flags['stop'] = False; _screener_flags['pause'] = False
        results_map = {5: [], 4: [], 3: [], 2: [], 1: []}
        
        if resume_done is None:
            resume_done = set()
            with open(SCREENER_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"mode": mode, "tickers": tickers, "done": [], "results": [], "last_updated": None}, f, ensure_ascii=False, indent=2)
        else:
            # resume_resultsをresults_mapに復元し、これまでのログを表示する
            if resume_results:
                print("="*40)
                print("⏪ 前回の中断地点から再開します (過去のログを復元中...)")
                print("="*40)
                for item in resume_results:
                    rk = item.get('rank')
                    sc = item.get('score')
                    row = item.get('row')
                    if rk in results_map:
                        results_map[rk].append((sc, row))
                        # ログの簡易再現
                        c, n, p, lab = row[0], row[1], row[2], row[3]
                        extra = f" (Conf: {row[4]})" if len(row) > 4 else ""
                        print(f"--- [{c}] {n} ---")
                        print(f"  {lab}{extra}")
                print("-" * 40)
                print("✅ ログの復元が完了しました。続きを開始します。\n")

        try:
            print(f"Starting Screener mode={mode}  {len(tickers)} stocks\n")
            done_codes  = list(resume_done)
            for code in tickers:
                if code in resume_done: print(f"  [{code}] Skipped"); continue
                if _screener_flags['pause']:
                    print("\nPaused...")
                    while _screener_flags['pause'] and not _screener_flags['stop']: _time_mod.sleep(0.4)
                    if _screener_flags['stop']: break
                    print("Resumed\n")
                if _screener_flags['stop']: break
                name = self.fetch_stock_name(code)
                print(f"--- [{code}] {name} ---")
                data, features, days = self.prepare_data(code)
                if data is None:
                    if days > 0:
                        print(f"  ⚠️ [{code}] データ不足({days} days) - Newly listed stocks will be skipped")
                    else:
                        print(f"  ⚠️ [{code}] Data Fetch Failed (possibly delisted, wrong code, or network error)")
                    done_codes.append(code); continue
                latest = data.iloc[-1]
                label, rank = "Neutral", 3
                if mode == "fast":
                    rsi, macd, sig = latest['RSI_14'], latest['MACD'], latest['MACD_sig']
                    b, s = 0, 0
                    if rsi < 30:  b += 3
                    _prev = data.iloc[-2] if len(data) >= 2 else data.iloc[-1]
                    if macd > sig and _prev['MACD'] <= _prev['MACD_sig']: b += 3
                    if rsi > 70:  s += 3
                    if macd < sig and _prev['MACD'] >= _prev['MACD_sig']: s += 3
                    if   b >= 6: label, rank = "Strong Buy",  5
                    elif b >= 3: label, rank = "Buy",         4
                    elif s >= 6: label, rank = "Strong Sell", 1
                    elif s >= 3: label, rank = "Sell",        2
                    print(f"  {label}")
                    results_map[rank].append((b if b>0 else s, [code, name, f"{latest['Close']:.1f}", label]))
                elif mode == "ai":
                    for i in range(1, 21):
                        threshold = 0.01 * np.sqrt(i)
                        data[f'Target_{i}'] = np.select(
                            [(data['Close'].shift(-i)-data['Close'])/data['Close']<= -threshold,
                             (data['Close'].shift(-i)-data['Close'])/data['Close']>= threshold],
                            [0, 2], default=1).astype(float)
                    p_data = data.iloc[-1:].copy()
                    data   = data.dropna(subset=[f'Target_{i}' for i in range(1,21)])
                    if len(data) < 50:
                        print(f"  ⚠️ [{code}] データ不足({len(data)} days) - Newly listed stocks will be skipped")
                        done_codes.append(code); continue
                    train   = data.iloc[:-max(5, int(len(data)*0.2))]
                    y_train = train['Target_1'].astype(int).values
                    counts  = np.bincount(y_train)
                    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                        class_weight={c: len(y_train)/(len(counts)*ct) for c,ct in enumerate(counts) if ct>0},
                        verbose=-1)
                    proba = m.fit(train[features], y_train).predict_proba(p_data[features])[0]
                    p_c   = int(np.argmax(proba))
                    vec_env = make_vec_env(StockTradingEnv, n_envs=1,
                        env_kwargs={'data_df': train, 'features': features}, vec_env_cls=DummyVecEnv)
                    try:
                        rl = PPO("MlpPolicy", vec_env, verbose=0, n_steps=256)
                        rl.learn(total_timesteps=3000, callback=InterruptCallback(_screener_flags))
                        if not _screener_flags['stop']:
                            obs_latest = np.append(p_data[features].values[0].astype(np.float32), [0.0, 0.0])
                            action, _ = rl.predict(np.nan_to_num(obs_latest), deterministic=True)
                            label, rank = classify_signal_5(p_c, int(action))
                            ppo_map = {0:"Sell",1:"Hold",2:"Buy"}[int(action)]
                            dir_map = {0:"down",1:"flat",2:"up"}[p_c]
                            print(f"  AI: {dir_map} {proba[p_c]*100:.0f}%  PPO:{ppo_map}  => {label}")
                            results_map[rank].append((int(proba[p_c]*100),
                                [code, name, f"{latest['Close']:.1f}", label, f"{proba[p_c]*100:.0f}%"]))
                    finally:
                        try: vec_env.close()
                        except: pass
                        gc.collect()
                # 中断チェック：分析が完了する前に中断された場合は、完了リストに入れない
                if _screener_flags['stop']: break

                done_codes.append(code)
                with open(SCREENER_STATE_PATH, "w", encoding="utf-8") as sf:
                    flat = [{"rank":rk,"score":sc,"row":row}
                            for rk,items in results_map.items() for sc,row in items]
                    # 日時を確実に文字列で保存
                    now_str = datetime.now().strftime("%m/%d %H:%M")
                    # tickers（全銘柄リスト）も一緒に保存して、再起動後にUIが空でも再開できるようにする
                    json.dump({"mode":mode, "tickers":tickers, "done":done_codes, "results":flat, "last_updated": now_str}, sf, ensure_ascii=False, indent=2)
                
                # 次の銘柄に行く前に再度チェック（念のため）
                if _screener_flags['stop']: break
            if not _screener_flags['stop']: print("\nCompleted!")
            ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path   = self._create_screener_pdf(mode, results_map, ts)
            resai_path = self._create_resai(mode, results_map, ts)
            return pdf_path, resai_path
        except Exception as e:
            print(f"Error: {e}"); return "", ""

    def _create_screener_pdf(self, mode, results_map, ts=None):
        try:
            ts   = ts or datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"reports/Screener_{mode}_{ts}.pdf"
            c    = canvas.Canvas(path, pagesize=A4); W, H = A4
            pdf_font = 'Helvetica'
            if os.path.exists(PDF_FONT_PATH):
                try:
                    pdfmetrics.registerFont(TTFont('JapaneseFont', PDF_FONT_PATH))
                    pdf_font = 'JapaneseFont'
                except: pass

            def new_page():
                c.showPage(); c.setFont(pdf_font, 9); return H - 50
            c.setFont(pdf_font,18); c.drawString(30,H-50,f"Screener [{mode.upper()}]")
            c.setFont(pdf_font,10); c.drawString(30,H-72,datetime.now().strftime('%Y-%m-%d %H:%M'))
            c.line(30,H-80,W-30,H-80); y = H-100
            for rank,cat,color in [(5,"Strong Buy",(0.9,0.2,0.1)),(4,"Buy",(0.1,0.6,0.2)),
                                    (3,"Neutral",(0.5,0.5,0.5)),(2,"Sell",(0.1,0.3,0.9)),
                                    (1,"Strong Sell",(0.0,0.1,0.6))]:
                items = sorted(results_map.get(rank,[]), key=lambda x:x[0], reverse=True)
                if not items: continue
                if y < 120: y = new_page()
                c.setFillColorRGB(*color); c.rect(28,y-4,W-56,20,fill=1,stroke=0)
                c.setFillColorRGB(1,1,1); c.setFont(pdf_font,13)
                c.drawString(35,y+2,f"{cat}  ({len(items)})")
                c.setFillColorRGB(0,0,0); y -= 28; c.setFont(pdf_font,9)
                sc_x = [35, 100, 310, 390]
                for i,h in enumerate(["Code","Name","Close","Signal"]):
                    c.drawString(sc_x[i],y,h)
                c.line(30,y-4,W-30,y-4); y -= 18
                for sc,row in items:
                    if y < 50: y = new_page(); c.setFont(pdf_font,9)
                    disp = list(row[:4])
                    # Name列(index 1)が長い場合は切り詰め
                    if len(str(disp[1])) > 28: disp[1] = str(disp[1])[:26] + ".."
                    for i,col in enumerate(disp): c.drawString(sc_x[i],y,str(col))
                    y -= 16
                y -= 8
            c.save(); print(f"PDF: {path}"); return os.path.basename(path)
        except Exception as e: print(f"PDF err: {e}"); return ""

    def _create_resai(self, mode, results_map, ts=None):
        try:
            ts   = ts or datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"reports/Screener_{mode}_{ts}.resai"
            lmap = {5:"Strong Buy",4:"Buy",3:"Neutral",2:"Sell",1:"Strong Sell"}
            records = [{"code":r[0],"name":r[1],"close":r[2],
                "signal":lmap.get(rank,"Neutral"),"rank":rank,"score":sc,
                "predicted_at":ts,"mode":mode,"verified":False,"actual_close":None}
                for rank,items in results_map.items() for sc,r in items]
            with open(path,"w",encoding="utf-8") as f:
                json.dump({"version":1,"mode":mode,"predicted_at":ts,"records":records},
                          f, ensure_ascii=False, indent=2)
            print(f"resai: {path}"); return os.path.basename(path)
        except Exception as e: print(f"resai err: {e}"); return ""


def _load_lists():
    if os.path.exists(LIST_FILE_PATH):
        try:
            with open(LIST_FILE_PATH,"r",encoding="utf-8") as f: return json.load(f).get("lists",{})
        except: pass
    return {}
def _save_lists(d):
    held = _load_stocks_held()
    with open(LIST_FILE_PATH,"w",encoding="utf-8") as f:
        json.dump({"version":1,"lists":d, "stocks_held": held},f,ensure_ascii=False,indent=2)

STOCKS_HELD_PATH = "temp/stocks_held.json"
def _load_stocks_held():
    if os.path.exists(STOCKS_HELD_PATH):
        try:
            with open(STOCKS_HELD_PATH,"r",encoding="utf-8") as f: return json.load(f).get("stocks",[])
        except: pass
    return []
def _save_stocks_held(stocks):
    with open(STOCKS_HELD_PATH,"w",encoding="utf-8") as f:
        json.dump({"version":1,"stocks":list(stocks)},f,ensure_ascii=False,indent=2)
    # .saiファイル側にも同期させる
    _save_lists(_load_lists())

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ttkbootstrap as ttkbs
from PIL import Image, ImageTk
import queue, threading, contextlib, json, glob, os, sys

# QueueLogger for Tkinter
LOG_FILE_PATH = "temp/current_log.txt"

class QueueLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.q = queue.Queue()
    
    def write(self, data):
        self.q.put(data)
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass
        if original_stdout is not None:
            try:
                original_stdout.write(data)
            except Exception:
                pass

    def flush(self):
        if original_stdout is not None:
            try:
                original_stdout.flush()
            except Exception:
                pass

sys_stdout_logger = None

class StockAIApp(ttkbs.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title(f"📈 Stock AI Dashboard v{APP_VERSION} (Beta)")
        self.geometry("1200x800") # 初期サイズを少し小さくして起動を速める
        self.style.theme_use("darkly") # テーマを再適用
        
        # 最大化処理は表示後に遅延させることで、起動時の「固まり」を減らす
        self.after(100, lambda: self.state('zoomed'))
        self.logic = StockAILogic()
        self._routine_mode = None

        self.notebook = ttkbs.Notebook(self, bootstyle="dark")
        self.notebook.pack(expand=True, fill='both', padx=8, pady=8)

        # Tab: Batch Screener (最左)
        self.tab_screener = ttkbs.Frame(self.notebook)
        self.notebook.add(self.tab_screener, text='  🔍 Batch Screener  ')
        self.build_screener_tab()

        # Tab: Single Analysis
        self.tab_single = ttkbs.Frame(self.notebook)
        self.notebook.add(self.tab_single, text='  📈 Single AI Analysis  ')
        self.build_single_tab()

        # Tab: Verify
        self.tab_verify = ttkbs.Frame(self.notebook)
        self.notebook.add(self.tab_verify, text='  📊 Prediction Verify  ')
        self.build_verify_tab()

        self.after(100, self.poll_log_queue)
        self.after(1000, self.check_for_updates) # 起動1秒後にアップデートチェック

    def check_for_updates(self):
        """GitHubの最新リリースをチェックする"""
        def worker():
            try:
                # GitHub APIを使用して最新リリースを取得
                url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    latest_ver = data.get("tag_name", "").replace("v", "")
                    current_ver = APP_VERSION.replace("v", "")
                    
                    # 単純な文字列比較ではなく、必要に応じてパッケージ管理ライブラリなどを使うこともありますが、
                    # ここではシンプルにタグ名の違いをチェックします。
                    if latest_ver != current_ver:
                        self.after(0, lambda: self.show_update_dialog(data))
            except:
                pass # 通信エラーなどは無視して続行
        
        threading.Thread(target=worker, daemon=True).start()

    def show_update_dialog(self, data):
        latest_ver = data.get("tag_name")
        body = data.get("body", "新バージョンがリリースされました。")
        msg = f"新しいバージョン ({latest_ver}) が見つかりました。\n\n"
        msg += f"【更新内容】:\n{body[:200]}...\n\n"
        msg += "GitHubのリリースページを開いて最新版をダウンロードしますか？"
        
        if messagebox.askyesno("Update Available", msg):
            webbrowser.open(data.get("html_url"))

    def build_single_tab(self):
        ft = ttkbs.Frame(self.tab_single); ft.pack(fill='x', pady=10, padx=10)
        ttkbs.Label(ft, text="Ticker Code (e.g. 7203):", font=('Segoe UI',11)).pack(side='left', padx=8)
        self.entry_code = ttkbs.Entry(ft, width=15, font=('Segoe UI',12)); self.entry_code.pack(side='left', padx=8)
        self.btn_run_single = ttkbs.Button(ft, text="▶ Run AI Prediction", command=self.run_single_click, bootstyle="success")
        self.btn_run_single.pack(side='left', padx=6)
        self.btn_open_pdf = ttkbs.Button(ft, text="📄 Open PDF", command=self.open_single_pdf, bootstyle="secondary")
        self.btn_open_pdf.pack(side='left', padx=6)
        self.lbl_status = ttkbs.Label(self.tab_single, text="", font=('Segoe UI',10), bootstyle="info")
        self.lbl_status.pack(pady=4)
        self.single_paned = ttkbs.Panedwindow(self.tab_single, orient='horizontal')
        self.single_paned.pack(expand=True, fill='both', padx=10, pady=5)
        self.text_log_single = tk.Text(self.single_paned, bg="#1a1f2e", fg="#c9d1d9", font=('Consolas',10), width=60, insertbackground='white')
        self.single_paned.add(self.text_log_single)
        self.chart_label = tk.Label(self.single_paned, bg="#0d1117"); self.single_paned.add(self.chart_label)
        self.current_chart_code = ""

    def build_screener_tab(self):
        ctrl = ttkbs.Frame(self.tab_screener); ctrl.pack(fill='x', padx=10, pady=(8,4))
        # リスト管理パネル
        lf = ttk.LabelFrame(ctrl, text=" 📂 Screener List ")
        lf.pack(side='left', fill='both', expand=True, padx=(0,6))
        lfi = ttkbs.Frame(lf); lfi.pack(fill='both', expand=True, padx=6, pady=4)
        r1 = ttkbs.Frame(lfi); r1.pack(fill='x', pady=2)
        self.combo_lists = ttkbs.Combobox(r1, state="readonly", width=28, font=('Segoe UI',10))
        self.combo_lists.pack(side='left', padx=4)
        self.combo_lists.bind("<<ComboboxSelected>>", self.load_list_to_textarea)
        ttkbs.Button(r1, text="📋 Manage Lists", command=self.open_list_manager, bootstyle="info-outline").pack(side='left', padx=4)
        ttkbs.Button(r1, text="💾 Save As", command=self.save_list, bootstyle="secondary-outline").pack(side='left', padx=4)
        ttkbs.Button(r1, text="🗑️ Delete", command=self.delete_list, bootstyle="danger-outline").pack(side='left', padx=4)
        r2 = ttkbs.Frame(lfi); r2.pack(fill='x', pady=2)
        ttkbs.Button(r2, text="📂 Import .sai", command=self.import_lists, bootstyle="secondary-outline").pack(side='left', padx=4)
        ttkbs.Button(r2, text="📤 Export .sai", command=self.export_lists, bootstyle="secondary-outline").pack(side='left', padx=4)
        # Stocks Heldパネル
        hf = ttk.LabelFrame(ctrl, text=" 💼 Stocks Held ")
        hf.pack(side='left', fill='both', expand=True)
        hfi = ttkbs.Frame(hf); hfi.pack(fill='both', expand=True, padx=6, pady=4)
        self.lbl_held_count = ttkbs.Label(hfi, text="0 stocks held", font=('Segoe UI',10), bootstyle="warning")
        self.lbl_held_count.pack(anchor='w')
        
        sb_h = ttkbs.Scrollbar(hfi, orient='vertical', bootstyle="warning-round")
        self.tv_held = ttkbs.Treeview(hfi, columns=("code", "name", "close"), show="headings", height=10, yscrollcommand=sb_h.set, bootstyle="warning")
        sb_h.config(command=self.tv_held.yview); sb_h.pack(side='right', fill='y')
        self.tv_held.heading("code", text="Ticker"); self.tv_held.column("code", width=60, anchor='center')
        self.tv_held.heading("name", text="Name"); self.tv_held.column("name", width=120, anchor='w')
        self.tv_held.heading("close", text="Close"); self.tv_held.column("close", width=70, anchor='e')
        self.tv_held.pack(fill='both', expand=True, pady=2)
        
        hbtn = ttkbs.Frame(hfi); hbtn.pack(fill='x', pady=2)
        ttkbs.Button(hbtn, text="➕ Add", command=self.add_manual_held, bootstyle="success-outline").pack(side='left', padx=2)
        ttkbs.Button(hbtn, text="📋 Manage List", command=self.open_held_list_manager, bootstyle="info-outline").pack(side='left', padx=2)
        ttkbs.Button(hbtn, text="🗑️ Remove", command=self.remove_from_held, bootstyle="danger-outline").pack(side='left', padx=2)
        ttkbs.Button(hbtn, text="🧠 Analyze Held", command=self.run_held_analysis, bootstyle="warning").pack(side='left', padx=2)
        ttkbs.Button(hbtn, text="🔄", command=self.refresh_held_list, bootstyle="secondary-outline").pack(side='left', padx=2)
        self.refresh_selector(); self.refresh_held_list()
        # ティッカーリスト (UI上は非表示にし、内部変数としてのみ保持)
        tf = ttkbs.Frame(self.tab_screener)
        # tf.pack(fill='x', padx=10, pady=(4,0))
        # ttkbs.Label(tf, text="Ticker List (one per line):", font=('Segoe UI',10)).pack(anchor='w', pady=(0,2))
        self.text_tickers = tk.Text(tf, height=5, font=('Consolas',11), bg="#1a1f2e", fg="#c9d1d9", insertbackground='white')
        # self.text_tickers.pack(fill='x')
        # ボタン行
        fb = ttkbs.Frame(self.tab_screener); fb.pack(fill='x', padx=10, pady=6)
        self.btn_fast = ttkbs.Button(fb, text="⚡ Fast Technical Scan", command=lambda: self.run_screener_click('fast'), bootstyle="primary")
        self.btn_fast.pack(side='left', padx=4)
        self.btn_ai = ttkbs.Button(fb, text="🧠 Full AI Screener", command=lambda: self.run_screener_click('ai'), bootstyle="info")
        self.btn_ai.pack(side='left', padx=4)
        self.btn_routine = ttkbs.Button(fb, text="🔁 Routine", command=self.run_routine_click, bootstyle="success")
        self.btn_routine.pack(side='left', padx=4)
        ttkbs.Separator(fb, orient='vertical').pack(side='left', padx=8, fill='y')
        self.btn_resume = ttkbs.Button(fb, text="⏪ Resume", command=self.run_screener_resume, bootstyle="secondary-outline")
        self.btn_resume.pack(side='left', padx=4)
        self.btn_pause = ttkbs.Button(fb, text="⏸️ Pause", command=self.toggle_pause, bootstyle="warning-outline")
        self.btn_pause.pack(side='left', padx=4)
        self.btn_stop = ttkbs.Button(fb, text="⏹️ Stop", command=self.stop_screener, bootstyle="danger")
        self.btn_stop.pack(side='left', padx=4)
        self.btn_open_report = ttkbs.Button(fb, text="📄 Open Reports", command=self.open_screener_report, bootstyle="light-outline")
        self.btn_open_report.pack(side='right', padx=4)
        self.btn_backup = ttkbs.Button(fb, text="💾 Backup App", command=self.backup_app, bootstyle="secondary-outline")
        self.btn_backup.pack(side='right', padx=4)
        self.text_log_screen = tk.Text(self.tab_screener, bg="#0d1117", fg="#c9d1d9", font=('Consolas',10), insertbackground='white')
        self.text_log_screen.pack(expand=True, fill='both', padx=10, pady=6)
        self.check_resume_availability()

    def build_verify_tab(self):
        f = ttkbs.Frame(self.tab_verify); f.pack(fill='x', pady=10, padx=10)
        ttkbs.Button(f, text="📂 Select .resai File", command=self.load_and_verify_resai, bootstyle="info").pack(side='left', padx=8)
        self.btn_export_verify_pdf = ttkbs.Button(f, text="📄 Save as PDF", command=self.export_verify_pdf, bootstyle="secondary")
        self.btn_export_verify_pdf.pack(side='left', padx=8)
        self.lbl_verify_stat = ttkbs.Label(f, text="", font=('Segoe UI',10)); self.lbl_verify_stat.pack(side='left', padx=12)
        vbox = ttkbs.Frame(self.tab_verify); vbox.pack(expand=True, fill='both', padx=10, pady=10)
        sy = ttkbs.Scrollbar(vbox, orient="vertical", bootstyle="info-round")
        self.tree_verify = ttkbs.Treeview(vbox, columns=("code","name","signal","pred","actual","chg","hit"),
                                           show="headings", yscrollcommand=sy.set, bootstyle="info")
        sy.config(command=self.tree_verify.yview); sy.pack(side='right', fill='y')
        for col,text,w,anc in [("code","Ticker",80,'center'),("name","Name",200,'w'),
                                ("signal","Signal",100,'center'),("pred","Pred Close",90,'e'),
                                ("actual","Cur Close",90,'e'),("chg","Change %",80,'center'),("hit","Verdict",80,'center')]:
            self.tree_verify.heading(col, text=text); self.tree_verify.column(col, width=w, anchor=anc)
        self.tree_verify.tag_configure('hit', foreground='#3fb950')
        self.tree_verify.tag_configure('miss', foreground='#f85149')
        self.tree_verify.pack(side='left', expand=True, fill='both')

    # === Stocks Held ===
    def refresh_held_list(self):
        stocks = _load_stocks_held()
        # 既存の項目を一度クリア
        for item in self.tv_held.get_children(): self.tv_held.delete(item)
        
        # まずはコードだけを挿入して、UIを即座に更新する
        for s in stocks:
            # iidをコードにすることで、後で個別に更新しやすくする
            self.tv_held.insert('', 'end', iid=s, values=(s, "⏳ Loading...", "-"))
        
        self.lbl_held_count.config(text=f"{len(stocks)} stocks held")
        
        # 非同期で名前と株価を取得する
        def worker():
            for s in stocks:
                try:
                    name = self.logic.fetch_stock_name(s)
                    close = self.logic.fetch_latest_close(s)
                    close_str = f"{close:.1f}" if close else "-"
                    # 取得できたら、該当する行を更新する
                    if self.tv_held.exists(s):
                        self.after(0, lambda c=s, n=name, cl=close_str: 
                                   self.tv_held.item(c, values=(c, n, cl)) if self.tv_held.exists(c) else None)
                except:
                    pass
        
        threading.Thread(target=worker, daemon=True).start()

    def remove_from_held(self):
        sel = self.tv_held.selection()
        if not sel: return
        stocks = _load_stocks_held()
        to_rm = {self.tv_held.item(item, "values")[0] for item in sel}
        _save_stocks_held([s for s in stocks if s not in to_rm])
        self.refresh_held_list()

    def open_held_list_manager(self):
        p = tk.Toplevel(self); p.title("Held Stocks List Manager"); p.geometry("400x500")
        p.configure(bg="#0d1117")
        ttk.Label(p, text="Enter Ticker Codes (one per line):", font=('Segoe UI', 10, 'bold')).pack(pady=10)
        txt = tk.Text(p, font=('Consolas', 11), bg="#161b22", fg="#c9d1d9", insertbackground='white')
        txt.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Load current
        stocks = _load_stocks_held()
        txt.insert('end', "\n".join(stocks))
        
        def save():
            codes = [c.strip() for c in txt.get('1.0', 'end').splitlines() if c.strip()]
            _save_stocks_held(codes)
            self.refresh_held_list()
            p.destroy()
            messagebox.showinfo("Success", "Held stocks updated.")
            
        ttkbs.Button(p, text="💾 Save Changes", command=save, bootstyle="success").pack(pady=10)

    def run_held_analysis(self):
        stocks = _load_stocks_held()
        if not stocks:
            messagebox.showinfo("Info", "No stocks in Held list.")
            return
        # スクリーナーのリストを書き換えて開始
        self.text_tickers.delete('1.0', 'end')
        self.text_tickers.insert('end', "\n".join(stocks))
        self.run_screener_click('ai')

    def add_to_stocks_held(self, codes):
        stocks = _load_stocks_held()
        for c in codes:
            if c not in stocks: stocks.append(c)
        _save_stocks_held(stocks)
        self.refresh_held_list()
        
    def add_manual_held(self):
        from tkinter import simpledialog
        code = simpledialog.askstring("Add to Stocks Held", "Enter Ticker Code:", parent=self)
        if code and code.strip():
            self.add_to_stocks_held([code.strip()])
            
    def backup_app(self):
        import shutil
        import zipfile
        import time
        from datetime import datetime
        
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_name = f"AppBackup_{ts}.zip"
        zip_path = os.path.join(backup_dir, zip_name)
        
        # 必要なデータディレクトリだけをバックアップする（容量削減のため）
        targets = ["temp", "reports"]
        files = ["株予想_AI.py", "app_icon.ico", "build.py"]
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for target in targets:
                    if os.path.isdir(target):
                        for root, _, filenames in os.walk(target):
                            for name in filenames:
                                file_path = os.path.join(root, name)
                                zf.write(file_path, arcname=file_path)
                for f in files:
                    if os.path.exists(f):
                        zf.write(f, arcname=f)
            messagebox.showinfo("Backup", f"Backup created successfully:\n{zip_path}")
        except Exception as e:
            messagebox.showerror("Backup Error", f"Failed to create backup:\n{e}")

    # === List Manager Modal ===
    def open_list_manager(self):
        m = ttkbs.Toplevel(self); m.title("📋 List Manager"); m.geometry("860x560"); m.grab_set()
        m.update_idletasks(); m.tk.eval(f'tk::PlaceWindow {str(m)} center')
        pw = ttkbs.Panedwindow(m, orient='horizontal'); pw.pack(expand=True, fill='both', padx=8, pady=8)
        left = ttkbs.Frame(pw); pw.add(left, weight=1)
        ttkbs.Label(left, text="Lists", font=('Segoe UI',11,'bold')).pack(anchor='w', padx=4, pady=4)
        self._lm_lb = tk.Listbox(left, bg="#1a1f2e", fg="#c9d1d9", font=('Segoe UI',10), selectmode='single', exportselection=False)
        self._lm_lb.pack(expand=True, fill='both', padx=4)
        self._lm_lb.bind("<<ListboxSelect>>", self._lm_sel)
        lbtns = ttkbs.Frame(left); lbtns.pack(fill='x', padx=4, pady=4)
        ttkbs.Button(lbtns, text="➕ New", command=lambda: self._lm_new(m), bootstyle="success-outline").pack(side='left', padx=2)
        ttkbs.Button(lbtns, text="✏️ Rename", command=lambda: self._lm_rename(m), bootstyle="info-outline").pack(side='left', padx=2)
        ttkbs.Button(lbtns, text="🗑️ Del", command=self._lm_del, bootstyle="danger-outline").pack(side='left', padx=2)
        right = ttkbs.Frame(pw); pw.add(right, weight=2)
        ttkbs.Label(right, text="Tickers (one per line)", font=('Segoe UI',11,'bold')).pack(anchor='w', padx=4, pady=4)
        self._lm_txt = tk.Text(right, bg="#1a1f2e", fg="#c9d1d9", font=('Consolas',11), insertbackground='white')
        self._lm_txt.pack(expand=True, fill='both', padx=4, pady=4)
        rbtns = ttkbs.Frame(right); rbtns.pack(fill='x', padx=4, pady=4)
        ttkbs.Button(rbtns, text="💾 Save", command=self._lm_save, bootstyle="success").pack(side='left', padx=2)
        ttkbs.Button(rbtns, text="📂 Import", command=lambda: self._lm_import(m), bootstyle="secondary-outline").pack(side='left', padx=2)
        ttkbs.Button(rbtns, text="📤 Export", command=self.export_lists, bootstyle="secondary-outline").pack(side='left', padx=2)
        ttkbs.Button(rbtns, text="❌ Close", command=m.destroy, bootstyle="danger-outline").pack(side='right', padx=2)
        self._lm_current = None; self._lm_refresh()

    def _lm_refresh(self):
        self._lm_lb.delete(0, 'end')
        for k, v in _load_lists().items(): self._lm_lb.insert('end', f"{k} ({len(v.strip().splitlines())})")

    def _lm_sel(self, e=None):
        sel = self._lm_lb.curselection()
        if not sel: return
        name = self._lm_lb.get(sel[0]).split(' (')[0]; self._lm_current = name
        self._lm_txt.delete('1.0', 'end'); self._lm_txt.insert('end', _load_lists().get(name, ''))

    def _lm_new(self, p):
        from tkinter import simpledialog
        n = simpledialog.askstring("New List", "List name:", parent=p)
        if not n: return
        d = _load_lists(); d[n] = ""; _save_lists(d); self._lm_refresh(); self.refresh_selector()

    def _lm_rename(self, p):
        if not self._lm_current: return
        from tkinter import simpledialog
        n = simpledialog.askstring("Rename", f"New name for '{self._lm_current}':", parent=p)
        if not n: return
        d = _load_lists(); d[n] = d.pop(self._lm_current, ""); _save_lists(d)
        self._lm_current = n; self._lm_refresh(); self.refresh_selector()

    def _lm_del(self):
        if not self._lm_current: return
        if messagebox.askyesno("Delete", f"Delete '{self._lm_current}'?"):
            d = _load_lists(); d.pop(self._lm_current, None); _save_lists(d)
            self._lm_current = None; self._lm_txt.delete('1.0', 'end'); self._lm_refresh(); self.refresh_selector()

    def _lm_save(self):
        if not self._lm_current:
            messagebox.showwarning("Warning", "No list selected."); return
        d = _load_lists(); d[self._lm_current] = self._lm_txt.get('1.0', 'end').strip()
        _save_lists(d); self._lm_refresh(); self.refresh_selector()
        messagebox.showinfo("Saved", f"List '{self._lm_current}' saved.")

    def _lm_import(self, p):
        fp = filedialog.askopenfilename(defaultextension=".sai", filetypes=[("SAI Files","*.sai")], parent=p)
        if not fp: return
        try:
            with open(fp,"r",encoding="utf-8") as f: nd = json.load(f)
            d = _load_lists(); d.update(nd.get('lists', {})); _save_lists(d)
            
            # 保有銘柄のマージ
            new_held = nd.get('stocks_held', [])
            if new_held:
                current_held = _load_stocks_held()
                merged = list(current_held)
                for c in new_held:
                    if c not in merged: merged.append(c)
                _save_stocks_held(merged)

            self._lm_refresh(); self.refresh_selector()
            messagebox.showinfo("Import", "Imported successfully (including Stocks Held).")
        except Exception as e: messagebox.showerror("Error", str(e))

    # === Routine Flow ===
    def run_routine_click(self):
        tickers = [t.strip() for t in self.text_tickers.get('1.0','end').splitlines() if t.strip()]
        held = _load_stocks_held()
        
        # 結合リスト（重複排除しつつ順序を保持）
        combined = []
        for t in tickers + held:
            if t not in combined: combined.append(t)
            
        if not combined:
            messagebox.showwarning("Warning", "Both Ticker list and Stocks Held are empty."); return
            
        # モード選択ダイアログ
        dlg = ttkbs.Toplevel(self); dlg.title("Routine – Select Mode"); dlg.geometry("360x170"); dlg.grab_set()
        dlg.update_idletasks(); dlg.tk.eval(f'tk::PlaceWindow {str(dlg)} center')
        ttkbs.Label(dlg, text="Select screening mode:", font=('Segoe UI',12,'bold')).pack(pady=18)
        bf = ttkbs.Frame(dlg); bf.pack()
        self._routine_mode = None
        def pick(m): self._routine_mode = m; dlg.destroy()
        ttkbs.Button(bf, text="⚡ Fast Technical", command=lambda: pick('fast'), bootstyle="primary", width=18).pack(side='left', padx=8)
        ttkbs.Button(bf, text="🧠 Full AI", command=lambda: pick('ai'), bootstyle="info", width=14).pack(side='left', padx=8)
        self.wait_window(dlg)
        if not self._routine_mode: return
        self._set_screener_btns('disabled')
        self.text_log_screen.delete('1.0','end')
        global _screener_flags; _screener_flags['stop'] = False; _screener_flags['pause'] = False
        global sys_stdout_logger; sys_stdout_logger = QueueLogger(self.text_log_screen)
        mode = self._routine_mode
        def worker():
            with contextlib.redirect_stdout(sys_stdout_logger):
                self.logic.run_screener(mode, combined)
            try:
                with open(SCREENER_STATE_PATH,"r",encoding="utf-8") as f: st = json.load(f)
                all_results = st.get("results",[])
            except: all_results = []
            
            # 対象リスト（tickers）の中から購入候補を抽出
            buys = [r for r in all_results if r.get("rank",0) >= 4 and r.get("row",[""])[0] in tickers]
            self.after(0, lambda: self._routine_buy_dialog(buys, mode, all_results))
        threading.Thread(target=worker, daemon=True).start()

    def _routine_buy_dialog(self, buys, mode, all_results):
        """Step2: 買い候補選択"""
        if not buys:
            messagebox.showinfo("Routine", "No Buy candidates found in the target list.\nProceeding to Sell selection.")
            self._routine_sell_prepare(all_results); return
            
        dlg = ttkbs.Toplevel(self); dlg.title("🛒 Select Stocks to Buy"); dlg.geometry("860x520"); dlg.grab_set()
        dlg.update_idletasks(); dlg.tk.eval(f'tk::PlaceWindow {str(dlg)} center')
        ttkbs.Label(dlg, text="Select stocks to add to Stocks Held:", font=('Segoe UI',14,'bold')).pack(pady=8, padx=12, anchor='w')
        ttkbs.Label(dlg, text="Click on rows to select multiple stocks. (Leave all unselected to skip)", font=('Segoe UI',10), bootstyle="secondary").pack(anchor='w', padx=14)
        
        fr = ttkbs.Frame(dlg); fr.pack(expand=True, fill='both', padx=10, pady=6)
        sb = ttkbs.Scrollbar(fr, orient='vertical', bootstyle="success-round")
        # ソート（Rank降順、Score降順）
        buys.sort(key=lambda x: (x.get("rank",0), x.get("score",0.0) if isinstance(x.get("score"), (int,float)) else 0.0), reverse=True)
        
        tv = ttkbs.Treeview(fr, columns=("check","code","name","signal","close","score"), show="headings", yscrollcommand=sb.set, bootstyle="success")
        sb.config(command=tv.yview); sb.pack(side='right', fill='y'); tv.pack(expand=True, fill='both')
        
        tv.heading("check", text="✔"); tv.column("check", width=40, anchor='center')
        tv.heading("code", text="Ticker"); tv.column("code", width=80, anchor='center')
        tv.heading("name", text="Name"); tv.column("name", width=220, anchor='w')
        tv.heading("signal", text="Signal"); tv.column("signal", width=120, anchor='center')
        tv.heading("close", text="Close"); tv.column("close", width=90, anchor='e')
        tv.heading("score", text="Score/Conf"); tv.column("score", width=100, anchor='center')

        lmap = {5:"🔥 Strong Buy", 4:"📈 Buy"}
        
        current_rank = None
        for r in buys:
            rank = r.get("rank", 4)
            if current_rank != rank:
                current_rank = rank
                group_name = lmap.get(rank, "Buy")
                iid = tv.insert('', 'end', values=("", f"--- {group_name} ---", "", "", "", ""))
                tv.item(iid, tags=('header',))
                
            row = r.get("row", [])
            code = row[0] if row else ""
            name = row[1] if len(row)>1 else ""
            close = row[2] if len(row)>2 else "-"
            sig = lmap.get(rank, "Buy")
            score = row[4] if len(row)>4 else str(r.get("score",""))
            iid = tv.insert('', 'end', values=("☐", code, name, sig, close, score))
            tv.item(iid, tags=('selectable',))
            
        tv.tag_configure('header', background='#2c3e50', foreground='white')
        
        def toggle_check(event):
            region = tv.identify("region", event.x, event.y)
            if region != "cell": return
            iid = tv.focus()
            if not iid or 'selectable' not in tv.item(iid, 'tags'): return
            col = tv.identify_column(event.x)
            if col == '#1': # check column
                vals = list(tv.item(iid, "values"))
                vals[0] = "☑" if vals[0] == "☐" else "☐"
                tv.item(iid, values=vals)
                
        tv.bind('<ButtonRelease-1>', toggle_check)
            
        bf = ttkbs.Frame(dlg); bf.pack(fill='x', padx=10, pady=10)
        def confirm():
            codes = []
            selected_records = []
            for iid in tv.get_children():
                if 'selectable' in tv.item(iid, 'tags'):
                    vals = tv.item(iid, "values")
                    if vals[0] == "☑":
                        codes.append(vals[1])
                        # 対応するrを取得
                        rec = next((r for r in buys if r.get("row",[""])[0] == vals[1]), None)
                        if rec: selected_records.append(rec)
            dlg.destroy()
            if codes: self.add_to_stocks_held(codes)
            self._routine_sell_prepare(all_results, selected_records)
            
        def skip(): dlg.destroy(); self._routine_sell_prepare(all_results, [])
        ttkbs.Button(bf, text="✅ Add Checked to Stocks Held", command=confirm, bootstyle="success").pack(side='left', padx=6)
        ttkbs.Button(bf, text="⏭️ Skip", command=skip, bootstyle="secondary-outline").pack(side='left', padx=6)

    def _routine_sell_prepare(self, all_results, selected_buys_data=None):
        if selected_buys_data is None: selected_buys_data = []
        """Step3: 売却候補の準備（スクリーニングは不要）"""
        held = _load_stocks_held()
        if not held:
            messagebox.showinfo("Routine", "Stocks Held is empty. Routine complete.")
            self._set_screener_btns('normal'); return
            
        # Stocks Held に存在する銘柄のみを抽出
        sells = [r for r in all_results if r.get("row",[""])[0] in held]
        self._routine_sell_dialog(sells, selected_buys_data)

    def _routine_sell_dialog(self, sells, selected_buys_data):
        """Step4: 売却候補選択"""
        self._set_screener_btns('normal')
        if not sells:
            messagebox.showinfo("Routine", "No valid data for Stocks Held.\nRoutine complete!")
            self._finish_routine_pdf(selected_buys_data, [])
            return
            
        dlg = ttkbs.Toplevel(self); dlg.title("💰 Select Stocks to Sell"); dlg.geometry("860x520"); dlg.grab_set()
        dlg.update_idletasks(); dlg.tk.eval(f'tk::PlaceWindow {str(dlg)} center')
        ttkbs.Label(dlg, text="Select stocks to remove from Stocks Held:", font=('Segoe UI',14,'bold')).pack(pady=8, padx=12, anchor='w')
        ttkbs.Label(dlg, text="Even non-sell signals are shown so you can manually select. (Leave all unselected to skip)", font=('Segoe UI',10), bootstyle="secondary").pack(anchor='w', padx=14)
        
        fr = ttkbs.Frame(dlg); fr.pack(expand=True, fill='both', padx=10, pady=6)
        sb = ttkbs.Scrollbar(fr, orient='vertical', bootstyle="danger-round")
        # ソート（Rank昇順、Score昇順(小さい方がSellとして強い場合)）
        sells.sort(key=lambda x: (x.get("rank",3), x.get("score",0.0) if isinstance(x.get("score"), (int,float)) else 0.0))
        
        tv = ttkbs.Treeview(fr, columns=("check","code","name","signal","close","score"), show="headings", yscrollcommand=sb.set, bootstyle="danger")
        sb.config(command=tv.yview); sb.pack(side='right', fill='y'); tv.pack(expand=True, fill='both')

        tv.heading("check", text="✔ (Full/Part)"); tv.column("check", width=80, anchor='center')
        tv.heading("code", text="Ticker"); tv.column("code", width=80, anchor='center')
        tv.heading("name", text="Name"); tv.column("name", width=220, anchor='w')
        tv.heading("signal", text="Signal"); tv.column("signal", width=120, anchor='center')
        tv.heading("close", text="Close"); tv.column("close", width=90, anchor='e')
        tv.heading("score", text="Score/Conf"); tv.column("score", width=100, anchor='center')

        lmap = {1:"🔻 Strong Sell", 2:"📉 Sell", 3:"➖ Neutral", 4:"📈 Buy", 5:"🔥 Strong Buy"}
        
        current_rank = None
        for r in sells:
            rank = r.get("rank", 3)
            if current_rank != rank:
                current_rank = rank
                group_name = lmap.get(rank, "Neutral")
                iid = tv.insert('', 'end', values=("", f"--- {group_name} ---", "", "", "", ""))
                tv.item(iid, tags=('header',))
                
            row = r.get("row",[])
            code = row[0] if row else ""
            name = row[1] if len(row)>1 else ""
            close = row[2] if len(row)>2 else "-"
            sig = lmap.get(rank, "Neutral")
            score = row[4] if len(row)>4 else str(r.get("score",""))
            iid = tv.insert('', 'end', values=("☐", code, name, sig, close, score))
            tv.item(iid, tags=('selectable',))
            
            # 売却シグナルなら赤文字にするタグ設定
            if rank <= 2:
                tv.item(iid, tags=('selectable', 'sell_sig'))
                
        tv.tag_configure('header', background='#2c3e50', foreground='white')
        tv.tag_configure('sell_sig', foreground='#ff6b6b')
        
        def toggle_check(event):
            region = tv.identify("region", event.x, event.y)
            if region != "cell": return
            iid = tv.focus()
            if not iid or 'selectable' not in tv.item(iid, 'tags'): return
            col = tv.identify_column(event.x)
            if col == '#1': # check column
                vals = list(tv.item(iid, "values"))
                if vals[0] == "☐": vals[0] = "☑ Full"
                elif vals[0] == "☑ Full": vals[0] = "☑ Part"
                else: vals[0] = "☐"
                tv.item(iid, values=vals)
                
        tv.bind('<ButtonRelease-1>', toggle_check)
        
        bf = ttkbs.Frame(dlg); bf.pack(fill='x', padx=10, pady=10)
        def confirm():
            codes_to_remove = []
            selected_sells_data = []
            for iid in tv.get_children():
                if 'selectable' in tv.item(iid, 'tags'):
                    vals = tv.item(iid, "values")
                    if vals[0] in ("☑ Full", "☑ Part"):
                        if vals[0] == "☑ Full": codes_to_remove.append(vals[1])
                        rec = next((r for r in sells if r.get("row",[""])[0] == vals[1]), None)
                        if rec: selected_sells_data.append(rec)
            dlg.destroy()
            if codes_to_remove:
                held = _load_stocks_held()
                _save_stocks_held([s for s in held if s not in codes_to_remove])
                self.refresh_held_list()
                messagebox.showinfo("Routine", f"Removed {len(codes_to_remove)} stock(s) (Full Sell) from Stocks Held.\nRoutine complete!")
            else:
                messagebox.showinfo("Routine", "No full sell stocks removed.\nRoutine complete!")
                
            self._finish_routine_pdf(selected_buys_data, selected_sells_data)
                
        def skip():
            dlg.destroy()
            messagebox.showinfo("Routine", "Skipped.\nRoutine complete!")
            self._finish_routine_pdf(selected_buys_data, [])
            
        ttkbs.Button(bf, text="🗑️ Remove Selected from Stocks Held", command=confirm, bootstyle="danger").pack(side='left', padx=6)
        ttkbs.Button(bf, text="⏭️ Skip", command=skip, bootstyle="secondary-outline").pack(side='left', padx=6)
        
    def _finish_routine_pdf(self, selected_buys, selected_sells):
        """ルーティン終了後に、購入と売却の選択銘柄を分けてPDFに出力して表示する"""
        if not selected_buys and not selected_sells:
            return 
            
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f"reports/Routine_Actions_{ts}.pdf"
        
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import os, sys
            
            c = canvas.Canvas(path, pagesize=A4); W, H = A4
            pdf_font = 'Helvetica'
            if sys.platform == "win32":
                font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
            else:
                font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('JapaneseFont', font_path))
                    pdf_font = 'JapaneseFont'
                except: pass

            def new_page():
                c.showPage(); c.setFont(pdf_font, 9); return H - 50
                
            c.setFont(pdf_font, 18); c.drawString(30, H - 50, "Routine Actions Report")
            c.setFont(pdf_font, 10); c.drawString(30, H - 72, datetime.now().strftime('%Y-%m-%d %H:%M'))
            c.line(30, H - 80, W - 30, H - 80); y = H - 100
            
            groups = [
                ("🛒 Stocks Added to Held (BUY)", (0.1, 0.6, 0.2), selected_buys),
                ("💰 Stocks Removed from Held (SELL)", (0.9, 0.2, 0.1), selected_sells)
            ]
            
            for title, color, items in groups:
                if not items: continue
                if y < 120: y = new_page()
                c.setFillColorRGB(*color); c.rect(28, y - 4, W - 56, 20, fill=1, stroke=0)
                c.setFillColorRGB(1, 1, 1); c.setFont(pdf_font, 13)
                c.drawString(35, y + 2, f"{title}  ({len(items)})")
                c.setFillColorRGB(0, 0, 0); y -= 28; c.setFont(pdf_font, 9)
                sc_x = [35, 100, 310, 390]
                for i, h in enumerate(["Code", "Name", "Close", "Signal"]):
                    c.drawString(sc_x[i], y, h)
                c.line(30, y - 4, W - 30, y - 4); y -= 18
                
                for rec in items:
                    if y < 50: y = new_page()
                    row = rec.get("row", [])
                    disp = list(row[:4]) if len(row) >= 4 else list(row) + [""] * (4 - len(row))
                    if len(str(disp[1])) > 28: disp[1] = str(disp[1])[:26] + ".."
                    for i, col in enumerate(disp[:4]): c.drawString(sc_x[i], y, str(col))
                    y -= 16
                y -= 8
                
            c.save()
            print(f"PDF: {path}")
            full_path = os.path.abspath(path)
            if os.path.exists(full_path):
                os.startfile(full_path)
                
        except Exception as e:
            print(f"PDF err: {e}")

    def _set_screener_btns(self, state):
        for b in [self.btn_fast, self.btn_ai, self.btn_routine]: b.config(state=state)

    # --- Single Actions ---
    def run_single_click(self):
        code = self.entry_code.get().strip()
        if not code: return
        self.current_chart_code = code
        self.btn_run_single.config(state='disabled')
        self.btn_open_pdf.config(state='disabled')
        self.lbl_status.config(text="AI Analyzing...")
        self.text_log_single.delete('1.0', 'end')
        self.chart_label.config(image='')
        
        global sys_stdout_logger
        sys_stdout_logger = QueueLogger(self.text_log_single)

        def worker():
            with contextlib.redirect_stdout(sys_stdout_logger):
                self.logic.run_ai(code)
            self.after(0, self.finish_single)
            
        threading.Thread(target=worker, daemon=True).start()

    def finish_single(self):
        self.lbl_status.config(text="Done")
        self.btn_run_single.config(state='normal')
        self.btn_open_pdf.config(state='normal')
        sys_stdout_logger.flush()
        
        # Load Chart Image
        chart_path = f"temp/chart_{self.current_chart_code}.png"
        if os.path.exists(chart_path):
            try:
                img = Image.open(chart_path)
                img.thumbnail((500, 700), Image.Resampling.LANCZOS)
                self.chart_photo = ImageTk.PhotoImage(img) # Keep ref
                self.chart_label.config(image=self.chart_photo)
            except Exception as e:
                print(f"Chart Load Error: {e}")

    def open_single_pdf(self):
        code = self.current_chart_code
        if not code:
            messagebox.showinfo("Information", "Please run AI prediction first to generate a report.")
            return
        pdfs = glob.glob(f"reports/Stock_{code}_*.pdf")
        if pdfs:
            latest = max(pdfs, key=os.path.getctime)
            os.startfile(os.path.abspath(latest))
        else:
            messagebox.showinfo("PDF Not Found", f"{code} report is not available yet.")

    # --- Lists Management ---
    def refresh_selector(self):
        lists = _load_lists()
        self.combo_lists['values'] = [f"{k} ({len(v.strip().splitlines())} stocks)" for k,v in lists.items()]
        self.combo_lists.set('')
        
    def load_list_to_textarea(self, event=None):
        sel = self.combo_lists.get()
        if not sel: return
        name = sel.split(' (')[0]
        lists = _load_lists()
        if name in lists:
            self.text_tickers.delete('1.0', 'end')
            self.text_tickers.insert('end', lists[name])

    def save_list(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Save", "Enter a list name:")
        if not name: return
        tickers = self.text_tickers.get('1.0', 'end').strip()
        if not tickers: return
        lists = _load_lists()
        lists[name] = tickers
        _save_lists(lists)
        self.refresh_selector()
        messagebox.showinfo("Save", f"List '{name}' was saved.")

    def delete_list(self):
        sel = self.combo_lists.get()
        if not sel: return
        name = sel.split(' (')[0]
        if messagebox.askyesno("Delete", f"Are you sure you want to delete '{name}'?"):
            lists = _load_lists()
            lists.pop(name, None)
            _save_lists(lists)
            self.refresh_selector()

    def export_lists(self):
        fp = filedialog.asksaveasfilename(defaultextension=".sai", initialfile="screener_lists.sai", title="Export List")
        if fp:
            import shutil
            if os.path.exists(LIST_FILE_PATH):
                shutil.copy(LIST_FILE_PATH, fp)
                messagebox.showinfo("Export", f"Exported to: {fp}")

    def import_lists(self):
        fp = filedialog.askopenfilename(defaultextension=".sai", filetypes=[("SAI Files", "*.sai")])
        if fp:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    new_data = json.load(f)
                lists = _load_lists()
                lists.update(new_data.get('lists', {}))
                _save_lists(lists)
                
                # 保有銘柄のマージ
                new_held = new_data.get('stocks_held', [])
                if new_held:
                    current_held = _load_stocks_held()
                    merged = list(current_held)
                    for c in new_held:
                        if c not in merged: merged.append(c)
                    _save_stocks_held(merged)
                    self.refresh_held_list()

                self.refresh_selector()
                messagebox.showinfo("Import", "List and Stocks Held imported successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # --- Screener Actions ---
    def check_resume_availability(self):
        self.btn_resume.config(state='normal')
        if os.path.exists(SCREENER_STATE_PATH):
            try:
                with open(SCREENER_STATE_PATH, "r", encoding="utf-8") as f:
                    st = json.load(f)
                done = st.get("done", [])
                ts = st.get("last_updated")
                if done:
                    ts_str = f" ({ts})" if ts else ""
                    self.btn_resume.config(text=f"⏪ Resume {len(done)} Done{ts_str}")
                    return
            except Exception as e:
                print(f"Resume check error: {e}")
        self.btn_resume.config(text="⏪ Resume")

    def run_screener_click(self, mode):
        self._run_screener_internal(mode, False)
        
    def run_screener_resume(self):
        if not os.path.exists(SCREENER_STATE_PATH):
            messagebox.showinfo("Information", "No previous screening state found to resume.")
            return
        try:
            with open(SCREENER_STATE_PATH,"r",encoding="utf-8") as f:
                st = json.load(f)
            mode = st.get('mode', 'fast')
            self._run_screener_internal(mode, True)
        except Exception as e:
            messagebox.showerror("Error", f"Resume failed: {e}")

    def _run_screener_internal(self, mode, resume):
        # UI上の入力
        ui_tickers = [t.strip() for t in self.text_tickers.get('1.0', 'end').splitlines() if t.strip()]
        
        self._set_screener_btns('disabled')
        self.btn_resume.config(state='disabled')
        self.text_log_screen.delete('1.0', 'end')
        
        global _screener_flags
        _screener_flags['stop'] = False
        _screener_flags['pause'] = False
        self.btn_pause.config(text="⏸️ Pause")
        
        global sys_stdout_logger
        sys_stdout_logger = QueueLogger(self.text_log_screen)

        def worker():
            resume_done = set()
            resume_results = []
            final_tickers = ui_tickers
            
            if resume and os.path.exists(SCREENER_STATE_PATH):
                try:
                    with open(SCREENER_STATE_PATH,"r",encoding="utf-8") as f: st=json.load(f)
                    resume_done = set(st.get("done",[]))
                    resume_results = st.get("results", [])
                    # 保存されていた銘柄リストがあれば、それを優先して使用する
                    saved_tickers = st.get("tickers")
                    if saved_tickers:
                        final_tickers = saved_tickers
                        # 画面上のテキストボックスにも復元する（空の場合のみではなく、再開時は前回のリストを表示）
                        self.after(0, lambda: self.text_tickers.delete('1.0', 'end'))
                        self.after(0, lambda: self.text_tickers.insert('end', "\n".join(saved_tickers)))
                except Exception: pass
            
            if not final_tickers:
                # 銘柄がない場合はログに出力して終了
                print("⚠️ Error: 銘柄リストが空です。リストを選択するか入力してください。")
                self.after(0, self.finish_screener)
                return

            with contextlib.redirect_stdout(sys_stdout_logger):
                self.logic.run_screener(mode, final_tickers, resume_done, resume_results)
            self.after(0, self.finish_screener)
            
        threading.Thread(target=worker, daemon=True).start()

    def toggle_pause(self):
        global _screener_flags
        _screener_flags['pause'] = not _screener_flags['pause']
        self.btn_pause.config(text="▶️ Resumed" if _screener_flags['pause'] else "⏸️ Pause")

    def stop_screener(self):
        if messagebox.askyesno("Stop", "スクリーニングをStopしますか？(進捗はSaveされます)"):
            global _screener_flags
            _screener_flags['stop'] = True

    def finish_screener(self):
        self._set_screener_btns('normal')
        self.check_resume_availability()
        sys_stdout_logger.flush()
        
    def open_screener_report(self):
        # Open reports folder directly
        os.startfile(os.path.abspath("reports"))

    # --- Verify Actions ---
    def load_and_verify_resai(self):
        fp = filedialog.askopenfilename(defaultextension=".resai", filetypes=[("RESAI Files", "*.resai")])
        if not fp: return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get('records', [])
            
            for item in self.tree_verify.get_children():
                self.tree_verify.delete(item)
            self.lbl_verify_stat.config(text="⏳ Fetching current prices and verifying...")
            
            def worker():
                total_len = len(records)
                results = []
                for idx, rec in enumerate(records):
                    self.after(0, lambda i=idx+1, t=total_len: self.lbl_verify_stat.config(text=f"⏳ Verifying {i} / {t}..."))
                    try:
                        cur = self.logic.fetch_latest_close(rec.get('code',''))
                        if cur is not None:
                            pred = float(rec.get('close',0))
                            rank = rec.get('rank',3)
                            pct = ((cur-pred)/pred*100) if pred else 0
                            hit = (rank>=4 and pct>0) or (rank<=2 and pct<0) or (rank==3 and abs(pct)<=1.0)
                            results.append({**rec,'actual_close':round(cur,1),'change_pct':round(pct,2),'hit':hit,'verified':True})
                        else:
                            results.append({**rec,'actual_close':None,'change_pct':None,'hit':None,'verified':False})
                    except:
                        results.append({**rec,'actual_close':None,'change_pct':None,'hit':None,'verified':False})
                self.after(0, lambda: self.finish_verify(results, data))
                
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")

    def finish_verify(self, results, orig_data):
        try:
            total = len([r for r in results if r.get('verified')])
            correct = len([r for r in results if r.get('hit')])
            acc = round(correct/total*100,1) if total>0 else 0
            
            self.lbl_verify_stat.config(text=f"🎯 Prediction Accuracy: {acc}% ({correct}/{total}) | {orig_data.get('predicted_at','')}")
            self.current_verify_data = {'results': results, 'orig': orig_data}
            if hasattr(self, 'btn_export_verify_pdf'):
                self.btn_export_verify_pdf.config(state='normal')
            
            for r in results:
                chg = f"{r['change_pct']}%" if r['change_pct'] is not None else "-"
                if r['change_pct'] is not None and r['change_pct'] > 0: chg = "+" + chg
                
                hit_str = ""
                tag = ""
                if r['hit'] is True:
                    hit_str = "✅ Hit"
                    tag = "hit"
                elif r['hit'] is False:
                    hit_str = "❌ Miss"
                    tag = "miss"
                    
                self.tree_verify.insert("", "end", values=(
                    r['code'], r['name'], r['signal'], r['close'], r['actual_close'] or '-', chg, hit_str
                ), tags=(tag,))
        except Exception as e:
            import traceback
            print(f"Verify UI Error: {e}")
            traceback.print_exc()

    def export_verify_pdf(self):
        if not hasattr(self, 'current_verify_data'):
            messagebox.showinfo("Information", "Please load and verify a report first.")
            return
        data = self.current_verify_data
        orig = data['orig']
        results = data['results']
        
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        import datetime
        import os
        
        pdf_path = f"reports/Verify_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        os.makedirs("reports", exist_ok=True)
        
        pdf_font = 'Helvetica'
        if os.path.exists(PDF_FONT_PATH):
            try:
                pdfmetrics.registerFont(TTFont('JapaneseFont', PDF_FONT_PATH))
                pdf_font = 'JapaneseFont'
            except: pass

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Update styles to use Japanese font
        styles['Title'].fontName = pdf_font
        styles['Normal'].fontName = pdf_font

        elements.append(Paragraph(f"Prediction Verify Report", styles['Title']))
        elements.append(Paragraph(f"Target: {orig.get('predicted_at','')}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        tdata = [['Ticker', 'Name', 'Signal', 'Pred', 'Actual', 'Change%', 'Verdict']]
        for r in results:
            chg = f"{r.get('change_pct','-')}%" if r.get('change_pct') is not None else "-"
            verdict = "Hit" if r.get('hit') is True else ("Miss" if r.get('hit') is False else "-")
            name = str(r.get('name',''))
            # Use full name since we have Japanese font
            tdata.append([
                str(r.get('code','')), 
                (name[:15] + '..' if len(name)>15 else name), 
                str(r.get('signal','')), 
                str(r.get('close','-')), 
                str(r.get('actual_close','-')), 
                chg, 
                verdict
            ])
            
        t = Table(tdata, colWidths=[60, 140, 60, 60, 60, 60, 60])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,-1), pdf_font),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black)
        ]))
        
        for i, r in enumerate(results):
            if r.get('hit') is True:
                t.setStyle(TableStyle([('TEXTCOLOR', (6, i+1), (6, i+1), colors.darkgreen)]))
            elif r.get('hit') is False:
                t.setStyle(TableStyle([('TEXTCOLOR', (6, i+1), (6, i+1), colors.red)]))
        
        elements.append(t)
        doc.build(elements)
        os.startfile(os.path.abspath(pdf_path))

    # --- Poll Queue ---

    def poll_log_queue(self):
        try:
            while True:
                if 'sys_stdout_logger' in globals() and sys_stdout_logger:
                    msg = sys_stdout_logger.q.get_nowait()
                    if sys_stdout_logger.text_widget:
                        sys_stdout_logger.text_widget.insert('end', msg)
                        sys_stdout_logger.text_widget.see('end')
                else:
                    break
        except queue.Empty:
            pass
        self.after(50, self.poll_log_queue)

if __name__ == "__main__":
    # システムの準備が整ったらロード画面を閉じる（テーマの競合を避けるため、作成前に破棄）
    if 'splash' in globals():
        try: splash.close()
        except: pass
    
    # メインウィンドウを作成
    app = StockAIApp()
    app.mainloop()

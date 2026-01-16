
import pandas as pd
import numpy as np
import json
import re
import time
import logging
import random
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入核心类
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer

# 尝试导入 OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Backtest")

class LLMBacktester(EnhancedWebStockAnalyzer):
    """
    LLM 回测器 V4.3：多线程极速版 + V3.2经典高胜率Prompt
    """
    def __init__(self, config_file='config.json'):
        super().__init__(config_file)
        self.full_price_history = {} 
        self.lock = threading.Lock()
        self._init_api_key_pool()

    def _init_api_key_pool(self):
        """初始化 API Key 池"""
        api_conf = self.config.get('api_keys', {}).get('openai', {})
        # raw_key = api_conf.get('api_key')
        
        raw_key = ["sk-S4WWKpUB7KRldSwQoCrJRCmoxR9i0I5gVIeGZNbuk2RrW8vl"]
        self.api_keys = []
        if isinstance(raw_key, list):
            self.api_keys = raw_key
        elif isinstance(raw_key, str) and raw_key:
            self.api_keys = [raw_key]
            
        self.base_url = 'https://api.vectorengine.ai/v1'
        self.model = 'deepseek-v3.2'
        
        if self.api_keys:
            print(f"🚀 API Key 池已加载: {len(self.api_keys)} 个 Key 待命")
        else:
            print("⚠️ 未找到有效的 API Key")

    def _get_random_api_key(self):
        if not self.api_keys: return None
        return random.choice(self.api_keys)

    def _get_data_slice(self, stock_code, simulation_date):
        """获取数据切片 (强化数据清洗)"""
        if stock_code not in self.full_price_history:
            with self.lock:
                if stock_code not in self.full_price_history:
                    df = self.get_stock_data(stock_code)
                    if not df.empty:
                        # 强力索引清洗
                        if not isinstance(df.index, pd.DatetimeIndex):
                            if 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                                df.set_index('date', inplace=True)
                            elif '日期' in df.columns:
                                df['日期'] = pd.to_datetime(df['日期'])
                                df.set_index('日期', inplace=True)
                            else:
                                try: df.index = pd.to_datetime(df.index)
                                except: pass
                    df = df.sort_index()
                    self.full_price_history[stock_code] = df
        
        full_df = self.full_price_history.get(stock_code)
        if full_df is None or full_df.empty:
            return pd.DataFrame()

        sim_dt = pd.to_datetime(simulation_date)
        mask = full_df.index <= sim_dt
        return full_df.loc[mask].copy()

    # === 独立计算模块 (绕过父类 Bug) ===
    
    def calculate_technical_indicators(self, df):
        """独立计算技术指标 (新增布林带支持)"""
        try:
            if df.empty: return {'rsi': 50, 'ma5': 0, 'ma20': 0, 'bb_pos': 0.5}
            close = df['close']
            
            indicators = {}
            indicators['ma5'] = close.rolling(5).mean().iloc[-1]
            indicators['ma20'] = close.rolling(20).mean().iloc[-1]
            
            # --- 新增：布林带计算 ---
            # 布林带是震荡市的神器
            std = close.rolling(20).std().iloc[-1]
            mid = indicators['ma20']
            upper = mid + 2 * std
            lower = mid - 2 * std
            # 计算当前价格在布林带的位置 (0=下轨, 0.5=中轨, 1=上轨)
            # 如果跌破下轨，数值会 < 0
            if upper != lower:
                indicators['bb_pos'] = (close.iloc[-1] - lower) / (upper - lower)
            else:
                indicators['bb_pos'] = 0.5
            # -----------------------

            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            return indicators
        except:
            return {'rsi': 50, 'ma5': 0, 'ma20': 0, 'bb_pos': 0.5}

    def analyze_smart_money_flow(self, df):
        """独立计算资金流"""
        try:
            if df.empty: return {'money_flow_score': 50, 'flow_status': '中性'}
            score = 50
            recent = df.tail(5)
            vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
            for _, row in recent.iterrows():
                pct = row.get('change_pct', 0)
                vol = row['volume']
                if pct > 0: score += 5 if vol > vol_ma5 else 2
                elif pct < 0: score -= 5 if vol > vol_ma5 else 1
            status = "流入" if score >= 60 else "流出" if score <= 40 else "中性"
            return {'money_flow_score': min(100, max(0, score)), 'flow_status': status}
        except:
            return {'money_flow_score': 50, 'flow_status': '中性'}

    def _calculate_trend_data(self, df):
        try:
            df = df.copy()
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            
            # =========== V5.3 新增：计算 MA20 斜率 ===========
            # 逻辑：今天的MA20 减去 昨天的MA20
            # 如果结果 > 0，说明均线向上；如果 < 0，说明还在跌
            df['MA20_slope'] = df['MA20'].diff()
            # ===============================================
            
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = exp12 - exp26
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD_Bar'] = 2 * (df['DIF'] - df['DEA'])
            vol_ma20 = df['volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['volume'] / vol_ma20
            return df
        except:
            return df

    def _generate_market_data_table(self, df, days=15):
        recent_df = df.tail(days).copy()
        table_str = "| 日期 | 收盘 | 涨跌% | MA5状态 | MACD | 量比 |\n|---|---|---|---|---|---|\n"
        for date, row in recent_df.iterrows():
            date_str = date.strftime('%m-%d')
            close = row['close']
            pct = row.get('change_pct', 0)
            if pd.isna(pct) and 'p_change' in row: pct = row['p_change']
            ma5 = row.get('MA5', 0)
            ma_flag = "⬆️" if close > ma5 else "⬇️"
            macd = row.get('MACD_Bar', 0)
            macd_icon = "🔴" if macd > 0 else "🟢"
            vol = row.get('Vol_Ratio', 0)
            vol_str = f"{vol:.1f}"
            if vol > 1.8: vol_str += "🔥"
            table_str += f"| {date_str} | {close:.2f} | {pct:+.2f} | {ma_flag} | {macd_icon}{macd:.2f} | {vol_str} |\n"
        return table_str

    def check_hard_rules(self, df_slice, tech, money):
        """
        🛑 V5.3 终极风控：增加均线斜率过滤
        """
        rsi = tech.get('rsi', 50)
        if rsi > 70: return False, f"RSI过热({rsi:.1f})"

        close = df_slice.iloc[-1]['close']
        ma5 = tech.get('ma5', 0)
        ma20 = tech.get('ma20', 0)
        bb_pos = tech.get('bb_pos', 0.5)
        
        # 获取斜率 (如果没有这一列，默认给0)
        ma20_slope = df_slice.iloc[-1].get('MA20_slope', 0)
        
        # [A] 右侧顺势 (价格 > MA20)
        if close > ma20:
            # === 新增过滤 ===
            # === V5.4 补丁：增加乖离率过滤 (防止追高接盘) ===
            bias = (close - ma20) / ma20 * 100
            if bias > 8: # 如果偏离均线超过8%，认为乖离率过大，风险高
                return False, f"乖离率过大({bias:.1f}%)，防止追高"
            # ============================================
            # 如果 MA20 还在下行 (斜率 < -0.01)，说明是假突破，或者是均线压制
            if ma20_slope < -0.01:
                return False, f"MA20趋势向下(斜率{ma20_slope:.3f})"
            # ===============
            
            if close < ma5:
                pct = df_slice.iloc[-1].get('change_pct', 0)
                if pct < -3: return False, "趋势中大阴线破位"
                
        # [B] 左侧震荡 (价格 < MA20)
        else:
            is_oversold = rsi < 35 
            is_boll_low = bb_pos < 0.15
            is_shrink = df_slice.iloc[-1].get('Vol_Ratio', 1.0) < 0.8
            
            if not ((is_oversold or is_boll_low) and is_shrink):
                return False, "左侧要求：(超跌或布林下轨) + 缩量"

        # MACD 通用过滤
        try:
            macd_val = df_slice.iloc[-1].get('MACD_Bar', 0)
            prev_macd = df_slice.iloc[-2].get('MACD_Bar', 0)
            if macd_val < -0.2 and macd_val < prev_macd: 
                return False, "MACD绿柱大幅发散"
        except: pass

        return True, "通过"   

    def _call_ai_api_pool(self, prompt):
        try:
            current_key = self._get_random_api_key()
            if not current_key: return '{"action": "HOLD", "confidence": 0}'
            
            client = OpenAI(api_key=current_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional trader. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                timeout=20
            )
            return response.choices[0].message.content
        except Exception as e:
            # 打印详细错误方便调试
            print(f"\n❌ API Error ({str(current_key)[-4:]}): {str(e)}")
            return f'{{"action": "HOLD", "confidence": 0, "reason": "API Error"}}'

    def _build_prompt(self, history_table, tech, money):
        """
        V5.1 全天候版 Prompt：教AI做震荡低吸
        """
        return f"""
你是一名**全能型**交易员。请根据当前形态选择策略（趋势跟随 或 震荡低吸）。

【个股数据】
{history_table}

【指标快照】
MA20(生命线):{tech.get('ma20',0):.2f} | 布林位置:{tech.get('bb_pos',0.5):.2f} (0=下轨, 1=上轨) | RSI:{tech.get('rsi',50):.1f}

【决策逻辑】
请判断当前是**趋势**还是**震荡**：

👉 **场景 A：趋势向上 (现价 > MA20)**
   - 策略：**顺势买入**。
   - 信号：缩量回踩 MA5/MA20 不破，或放量突破前高。

👉 **场景 B：震荡/超跌 (现价 < MA20)**
   - 策略：**箱体低吸** (Mean Reversion)。
   - 信号：
     1. **布林下轨支撑**：布林位置 < 0.15，且出现止跌K线（小阳线/十字星）。
     2. **缩量企稳**：下跌缩量，表明抛压枯竭。
   - 警告：如果是放量长阴杀跌，坚决 HOLD。

【输出格式】
JSON Only:
{{
    "action": "BUY" or "HOLD" or "SELL",
    "confidence": 75,
    "reason": "震荡触及布林下轨，RSI处于低位，且今日缩量收阳，博弈反弹..."
}}
"""             

    def process_single_stock_day(self, stock_code, date_val):
        try:
            df_slice = self._get_data_slice(stock_code, date_val)
            if df_slice.empty or len(df_slice) < 30: return None

            df_enriched = self._calculate_trend_data(df_slice)
            tech = self.calculate_technical_indicators(df_slice)
            money = self.analyze_smart_money_flow(df_slice)

            # === 👇👇👇 修改开始 👇👇👇 ===
            # 打印被过滤的原因 (Debug 全量模式)
            is_valid, reject_reason = self.check_hard_rules(df_enriched, tech, money)
            if not is_valid: 
                # 原代码有 if random.random() < 0.05: ...
                # 现在直接打印，这样你就能看到每一只为什么被拒了
                print(f"🚫 {stock_code} {date_val.strftime('%m-%d')} 过滤: {reject_reason}")
                return None
            # === 👆👆👆 修改结束 👆👆👆 ===    

            history_table = self._generate_market_data_table(df_enriched, days=15)
            prompt = self._build_prompt(history_table, tech, money)

            ai_res = self._call_ai_api_pool(prompt)
            
            try:
                match = re.search(r"(\{.*\})", ai_res, re.DOTALL)
                decision = json.loads(match.group(1)) if match else json.loads(ai_res)
            except:
                decision = {'action': 'HOLD', 'confidence': 0}

            action = decision.get('action', 'HOLD').upper()
            
            return {
                'date': date_val.strftime('%Y-%m-%d'),
                'stock': stock_code,
                'action': action,
                'confidence': decision.get('confidence', 0),
                'reason': decision.get('reason', 'None'),
                'close': float(df_slice.iloc[-1]['close'])
            }

        except Exception:
            return None

def run_concurrent_backtest(stock_list, backtest_days=10, max_workers=20):
    print("="*60)
    print(f"🚀 LLM 极速回测 V4.3 (高胜率版) | 股票: {len(stock_list)} | 线程: {max_workers}")
    print("="*60)
    
    tester = LLMBacktester()
    all_results = []
    
    print("📥 正在预加载数据...")
    for code in stock_list:
        tester._get_data_slice(code, datetime.now())
        print(".", end="", flush=True)
    print("\n✅ 数据准备就绪，开始并发分析...")

    tasks = []
    for stock_code in stock_list:
        full_data = tester.full_price_history.get(stock_code)
        if full_data is None: continue
        available_dates = full_data.index.sort_values()
        test_dates = available_dates[-(backtest_days+1):-1] 
        for date_val in test_dates:
            tasks.append((stock_code, date_val))

    start_time = time.time()
    total_tasks = len(tasks)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(tester.process_single_stock_day, code, date): (code, date) 
            for code, date in tasks
        }
        
        for future in as_completed(future_to_task):
            completed += 1
            res = future.result()
            
            if completed % 10 == 0 or completed == total_tasks:
                print(f"\r⏳ 进度: {completed}/{total_tasks} ({(completed/total_tasks)*100:.1f}%)", end="")
            
            if res:
                code = res['stock']
                date_str = res['date']
                action = res['action']
                
                full_df = tester.full_price_history.get(code)
                try:
                    curr_idx = full_df.index.get_loc(pd.to_datetime(date_str))
                    next_close = full_df.iloc[curr_idx + 1]['close']
                    actual_ret = (next_close - res['close']) / res['close'] * 100
                    res['return'] = actual_ret
                    
                    if action == 'BUY':
                        all_results.append(res)
                        print(f"\n🔥 {date_str} {code} BUY! (信:{res['confidence']}) -> 次日: {actual_ret:.2f}%")
                            
                except:
                    pass

    print(f"\n\n🏁 回测完成! 耗时: {time.time()-start_time:.1f}秒")
    
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print(f"🟢 总买入次数: {len(df)}")
        win_rate = len(df[df['return'] > 0]) / len(df) * 100
        avg_ret = df['return'].mean()
        print(f"🏆 胜率: {win_rate:.1f}%")
        print(f"💰 平均收益: {avg_ret:.2f}%")
        print("="*60)
        
        print("\n📝 详细清单:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df[['date', 'stock', 'return', 'reason']].to_string(index=False))
    else:
        print("未触发任何买入信号")


if __name__ == "__main__":
    # 在这里填入你想回测的股票
    target_stocks = [
"601888",
"601600",
"300750",
"603993",
"600498",
"000630",
"002460",
"300475",
"002326",
"688110",
"688158",
"300118",
"000792",
"601696",
"002466",
"002709",
"300568",  
"000737",
"601168",
"600219",
"300390",
"002497",
"300803"
]





    run_concurrent_backtest(target_stocks, backtest_days=20, max_workers=20)

    # run_backtest_simulation(target_stocks, backtest_days=20)
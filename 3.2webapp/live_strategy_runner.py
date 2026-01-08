# live_strategy_runner.py
import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import threading

# 引入核心组件
from global_scanner import GlobalMarketScanner
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer

# ==========================================
# 🛠️ 实盘配置区
# ==========================================
# 在这里填入你当前持有的股票代码（用于程序帮你判断卖点）
MY_HOLDINGS = [
    # 格式: {'code': '股票代码', 'buy_price': 买入价, 'hold_days': 持有天数, 'market': 'us_stock'}
    # 示例:
    # {'code': 'AAPL', 'buy_price': 185.5, 'hold_days': 5, 'market': 'us_stock'},
    # {'code': '00700', 'buy_price': 280.0, 'hold_days': 2, 'market': 'hk_stock'},
]

# ==========================================

class LiveTrader:
    def __init__(self):
        self.scanner = GlobalMarketScanner()
        self.analyzer = EnhancedWebStockAnalyzer()
        print("✅ 实盘交易系统已启动...")
        print(f"📅 当前日期: {datetime.now().strftime('%Y-%m-%d')}")

    def _get_market_from_code(self, stock_code):
        if stock_code.isdigit():
            if len(stock_code) == 5: return 'hk_stock'
            if len(stock_code) == 6: return 'a_stock'
        return 'us_stock'

    def analyze_single_stock(self, stock_code, mode='buy_check', holding_info=None):
        """
        分析单只股票
        mode='buy_check': 扫描买入机会
        mode='sell_check': 检查持仓卖出信号
        """
        try:
            # 1. 识别市场
            _, market = self.analyzer.validate_stock_code(stock_code)
            if market == 'UNKNOWN':
                market = self._get_market_from_code(stock_code)

            # 2. 获取实时数据 (获取最近3个月数据以计算指标)
            # 注意：实盘时，最后一行 close 通常是当前最新价
            # df = self.analyzer.get_stock_data(stock_code, period='3mo')
            # 2. 获取数据 (⚠️ 修改点：改为 1y 以确保 MA200 能计算)
            df = self.analyzer.get_stock_data(stock_code, period='1y')
            
            if df.empty or len(df) < 60:
                print(f"⚠️ [{stock_code}] 数据不足，跳过")
                return None

            # 3. 计算指标
            tech = self.analyzer.calculate_technical_indicators(df)
            money = self.analyzer.analyze_smart_money_flow(df)
            
            # 获取最新切片
            curr_close = df.iloc[-1]['close']
            vol_ratio = df.iloc[-1]['volume'] / (df['volume'].rolling(20).mean().iloc[-1] + 1)
            rsi = tech.get('rsi', 50)
            ma20 = tech.get('ma20', 0)
            ma20_slope = tech.get('ma20_slope', 0) # 确保 calculate_technical_indicators 里算了这个
            
            # 手动补算 slope 如果没有
            if ma20_slope == 0 and len(df) > 5:
                ma20_series = df['close'].rolling(20).mean()
                recent = ma20_series.iloc[-5:].values
                if not np.isnan(recent).any():
                    slope, _ = np.polyfit(np.arange(len(recent)), recent, 1)
                    ma20_slope = slope

            # ==========================================
            # 🔵 卖出逻辑检查 (仅针对持仓)
            # ==========================================
            if mode == 'sell_check' and holding_info:
                buy_price = holding_info['buy_price']
                highest_price = holding_info.get('highest_price', buy_price) # 获取历史最高
                hold_days = holding_info['hold_days']
                
                # 更新最高价 (模拟盘中创新高)
                if curr_close > highest_price:
                    highest_price = curr_close
                    print(f"📈 [{stock_code}] 创新高! 最高价更新为: {highest_price}")

                # 预估收益率
                profit_pct = (curr_close - buy_price) / buy_price * 100
                
                sell_reason = ""
                should_sell = False

                # 1. 硬止损
                if profit_pct < -5.0:
                    should_sell = True; sell_reason = f"硬止损触发 (当前{profit_pct:.2f}%)"
                # >>> 卖出规则 2: 移动止盈 (回测核心逻辑) <<<
                # 逻辑：如果曾经赚超过 10%，现在从最高点回撤超过 3%，就走人
                elif highest_price > buy_price * 1.10:
                    drawdown = (curr_close - highest_price) / highest_price * 100
                    if drawdown < -3.0:
                        should_sell = True; sell_reason = f"移动止盈触发 (高点回撤 {drawdown:.2f}%)"

                # 2. 时间止损/动能止损
                elif market == 'hk_stock' and hold_days > 12 and profit_pct < 0.5:
                    should_sell = True; sell_reason = "港股动能耗尽(>12天滞涨)"
                elif market == 'us_stock' and hold_days > 5 and profit_pct < 0:
                    should_sell = True; sell_reason = "美股动能耗尽(>5天亏损)"
                elif market == 'a_stock' and hold_days > 5 and profit_pct < -2:
                    should_sell = True; sell_reason = "A股弱势整理"

                # 输出结果
                color = "🔴" if should_sell else "🟢"
                print(f"{color} [持仓分析] {stock_code} | 现价:{curr_close} | 累计:{profit_pct:.2f}% | 持有:{hold_days}天")
                if should_sell:
                    print(f"   🚨 建议卖出! 原因: {sell_reason}")
                else:
                    # 如果没卖，检查是否有高风险提示
                    if rsi > 80: print(f"   ⚠️ 警告: RSI过高({rsi:.1f})，注意随时止盈")
                    else: print(f"   ✅ 状态健康，继续持有")
                
                return # 卖出检查结束
                
                # return # 卖出检查结束

            # ==========================================
            # 🔴 买入逻辑检查 (仅针对新机会)
            # ==========================================
            if mode == 'buy_check':
                # >>> 1. 差异化初筛 (完全复用回测逻辑) <<<
                potential_signal = False
                
                if market == 'a_stock':
                    # A股逻辑: 趋势向上 + 放量 OR 超跌
                    trend_ok = (curr_close > ma20) or (ma20_slope > -0.0005)
                    vol_ok = vol_ratio > 0.8
                    oversold = (rsi < 35)
                    if (trend_ok and vol_ok) or oversold: potential_signal = True
                
                elif market == 'hk_stock':
                    # 港股逻辑: 价格>2 + 有流动性
                    if (curr_close > 2.0) and (vol_ratio > 0.6): potential_signal = True
                
                elif market == 'us_stock':
                    # 美股逻辑: 趋势多头 + 动量不灭 OR 超跌
                    trend_ok = (curr_close > ma20) or (ma20_slope > 0)
                    momentum_ok = (rsi > 40) and (tech.get('kdj_signal') != '死叉')
                    oversold = (rsi < 30)
                    if (trend_ok and momentum_ok) or oversold: potential_signal = True

                if not potential_signal:
                    print(f"   💤 {stock_code} 初筛未过")
                    return None

                # >>> 2. 准备 AI 数据 <<<
                strategy_hint = ""
                if market == 'a_stock': strategy_hint = "A股(T+1)，极大重视安全性，拒绝下降趋势。"
                elif market == 'hk_stock': strategy_hint = "港股(T+0)，流动性第一，拒绝低成交量。"
                elif market == 'us_stock': strategy_hint = "美股(T+0)，顺势为主，允许RSI略高。"

                price_info = {
                    "close": round(curr_close, 2),
                    "change_pct": round(df.iloc[-1]['change_pct'], 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "market_hint": strategy_hint
                }

                print(f"🤖 呼叫AI分析: {stock_code}...", end="", flush=True)
                
                # >>> 3. 调用 LLM <<<
                ai_result = self.analyzer.get_llm_trade_decision(
                    stock_code, datetime.now().strftime('%Y-%m-%d'), 
                    price_info, tech, money
                )
                
                action = ai_result.get('action', 'HOLD')
                phase = ai_result.get('market_phase', '未知')
                reason = ai_result.get('reason', '无')
                
                print(f" -> {action}")

                if action == "BUY":
                    # >>> 4. 差异化风控 <<<
                    risk_pass = True
                    risk_msg = ""
                    
                    if market == 'a_stock':
                        if price_info['change_pct'] > 9.5: risk_pass = False; risk_msg = "涨停风险"
                        if ma20_slope < -0.05: risk_pass = False; risk_msg = "趋势极差"
                    elif market == 'hk_stock':
                        if curr_close < 1.0: risk_pass = False; risk_msg = "仙股风险"
                        if vol_ratio < 0.5: risk_pass = False; risk_msg = "无流动性"
                    elif market == 'us_stock':
                        if rsi > 85: risk_pass = False; risk_msg = "极度超买"

                    if risk_pass:
                        print(f"\n🔥🔥🔥 [发现机会] {stock_code} 🔥🔥🔥")
                        print(f"   💰 现价: {curr_close}")
                        print(f"   🌊 阶段: {phase}")
                        print(f"   💡 理由: {reason}")
                        print(f"   ⚠️ 风险: {ai_result.get('risk_warning')}")
                        print(f"   📊 资金流: {money.get('flow_status', '未知')}")
                        print(f"   🛑 建议止损位: {curr_close * 0.95:.2f} (-5%)")
                        print("-" * 40)
                    else:
                        print(f"   🛑 风控拦截: {risk_msg}")

        except Exception as e:
            print(f"❌ 分析出错 {stock_code}: {e}")

    def run_daily_scan(self, market='a_stock', top_n=30):
        """运行每日扫描"""
        print(f"\n🌍 开始扫描市场: {market.upper()} (Top {top_n})")
        print("=" * 50)
        
        # 1. 获取候选名单
        stock_list = []
        if market == 'a_stock': stock_list = self.scanner.get_a_candidates(top_n)
        elif market == 'hk_stock': stock_list = self.scanner.get_hk_candidates(top_n)
        elif market == 'us_stock': stock_list = self.scanner.get_us_candidates(top_n)
        
        if not stock_list:
            print("⚠️ 未扫描到股票，请检查网络或现在是否休市。")
            return

        # print(f"📋 候选名单: {stock_list}\n")
        print(f"📋 候选名单({len(stock_list)}): {stock_list}\n")

        # 2. 逐个分析
        for i, stock in enumerate(stock_list):
            print(f"[{i+1}/{len(stock_list)}] ", end="")
            self.analyze_single_stock(stock, mode='buy_check')
            time.sleep(1.5) # 给 API 喘息时间

    def check_my_holdings(self):
        """检查当前持仓"""
        if not MY_HOLDINGS:
            print("📭 当前无持仓记录 (请在代码顶部 MY_HOLDINGS 填写)")
            return

        print(f"\n💼 开始检查持仓 ({len(MY_HOLDINGS)}只)")
        print("=" * 50)
        for holding in MY_HOLDINGS:
            # 补全 highest_price 字段 (防止用户没填报错)
            if 'highest_price' not in holding:
                holding['highest_price'] = holding['buy_price']
                
            self.analyze_single_stock(holding['code'], mode='sell_check', holding_info=holding)
            time.sleep(1)

# ==========================================
# ▶️ 程序入口
# ==========================================
if __name__ == "__main__":
    trader = LiveTrader()
    
    # 1. 先检查持仓 (优先级最高)
    trader.check_my_holdings()
    
    # 2. 扫描新机会 (请取消注释你想跑的市场)
    
    # --- A股 (下午 14:45 跑) ---
    # trader.run_daily_scan(market='a_stock', top_n=20)
    
    # --- 港股 15:45 - 15:55（收盘前），或者 10:30（早盘消化后）。 ---
    # trader.run_daily_scan(market='hk_stock', top_n=20)
    
    # --- 美股 北京时间 04:30（收盘前半小时），或者 23:00（开盘半小时后） ---
    trader.run_daily_scan(market='us_stock', top_n=20)
# strategy_core.py
# 🧠 策略核心层：统一管理回测与实盘的交易逻辑

import pandas as pd
import logging
import threading
import json
import time
from datetime import datetime

# 引入基础组件
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
from global_scanner import GlobalMarketScanner

# 配置独立日志（默认日志，兼容旧代码）
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

# 初始化三个默认日志（保持向后兼容）
ai_logger = setup_logger("AI_Log", "logs_ai_analysis.log")
trade_logger = setup_logger("Trade_Log", "logs_trade_action.log")
pnl_logger = setup_logger("PnL_Log", "logs_pnl_tracking.log")

class StrategyEngine:
    def __init__(self, ai_logger=None, trade_logger=None, pnl_logger=None):
        """
        初始化策略引擎
        :param ai_logger: 用于记录AI思考过程的日志对象
        :param trade_logger: 用于记录买卖操作的日志对象
        :param pnl_logger: 用于记录盈亏资产的日志对象
        """
        self.analyzer = EnhancedWebStockAnalyzer()
        self.scanner = GlobalMarketScanner()
        self.download_lock = threading.Lock()
        
        # 如果调用者没传日志对象，就用默认的（防止报错）
        self.ai_logger = ai_logger or logging.getLogger("Dummy_AI")
        self.trade_logger = trade_logger or logging.getLogger("Dummy_Trade")
        self.pnl_logger = pnl_logger or logging.getLogger("Dummy_PnL")

    def get_candidates(self, market, limit=20):
        """统一选股接口"""
        if market == 'hk_stock':
            return self.scanner.get_hk_candidates(top_n=limit)
        elif market == 'us_stock':
            return self.scanner.get_us_candidates(top_n=limit)
        elif market == 'a_stock':
            return self.scanner.get_a_candidates(top_n=limit)
        return []

    def analyze_ticker(self, code, current_date_str, data_slice=None):
        """
        核心分析函数：输入代码和数据，输出 AI 决策
        """
        # 1. 准备数据
        if data_slice is None:
            # 实盘模式：下载最新数据
            with self.download_lock:
                data_slice = self.analyzer.get_stock_data(code, period='3mo')
            
        if data_slice.empty or len(data_slice) < 30:
            return None
    
        # 2. 计算指标
        tech = self.analyzer.calculate_technical_indicators(data_slice)
        money = self.analyzer.analyze_smart_money_flow(data_slice)
            
        curr_row = data_slice.iloc[-1]
        close_price = curr_row['close']
            
        # 3. 🔍 调用 analyzer 的搜索功能
        # 构造搜索关键词，例如 "腾讯控股 00700 最新新闻 利好 利空"
        search_query = f"{code} stock latest news analysis sentiment"
        news_context = self.analyzer.search_market_news(search_query)
    
        # 4. 构建 AI 输入
        price_info = {
            "close": round(close_price, 2),
            "change_pct": round(curr_row['change_pct'], 2),
            "vol_ratio": round(tech.get('vol_ratio_20', 1.0), 2),
            "news_summary": news_context  # ✅ 注入新闻
        }
    
        # 5. 调用 AI
        try:
            ai_res = self.analyzer.get_llm_trade_decision(
                code, current_date_str, price_info, tech, money
            )
                
            # ✅ 使用传入的 logger 记录，互不干扰
            self.ai_logger.info(f"[{current_date_str}] {code} | {ai_res.get('action')} | 信:{ai_res.get('confidence')} | {ai_res.get('reason')}")
                
            return {
                "code": code,
                "price": close_price,
                "confidence": ai_res.get('confidence', 0),
                "action": ai_res.get('action', 'HOLD'),
                "reason": ai_res.get('reason', '无'),
                "tech": tech  # 传递RSI等指标用于后续计算
            }
        except Exception as e:
            self.ai_logger.error(f"{code} AI分析错误: {e}")
            return None

    def calculate_holding_score(self, holding_info, current_price, current_date, tech):
        """
        核心换仓评分逻辑 (Smart Swap Score)
        """
        cost = holding_info['cost_price']
        profit_pct = (current_price - cost) / cost * 100
        
        # ==========================================
        # 🛠️ 修复点：兼容日期格式 (YYYY-MM-DD 和 YYYY-MM-DD HH:MM:SS)
        # ==========================================
        if isinstance(holding_info['buy_date'], str):
            try:
                # 优先尝试带时间的格式 (实盘用)
                buy_dt = datetime.strptime(holding_info['buy_date'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # 如果报错，尝试只带日期的格式 (回测用)
                buy_dt = datetime.strptime(holding_info['buy_date'], '%Y-%m-%d')
        else:
            # 如果已经是 datetime 对象，直接用
            buy_dt = holding_info['buy_date']
            
        if isinstance(current_date, str):
            curr_dt = datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S') if ':' in current_date else datetime.strptime(current_date, '%Y-%m-%d')
        else:
            curr_dt = current_date

        hold_days = (curr_dt - buy_dt).days
        
        rsi = tech.get('rsi', 50)
        ma20 = tech.get('ma20', 0)
        ma20_slope = tech.get('ma20_slope', 0)

        # === 评分公式 (与 V3 一致) ===
        score = profit_pct
        
        # 👇👇👇 [补全1] 趋势破位 (一票否决) 👇👇👇
        if ma20 > 0 and current_price < ma20:
            score -= 50 
            return score, f"破位(跌破MA20)|盈{profit_pct:.1f}%"

        # 👇👇👇 [补全2] 利润回吐保护 (防止坐电梯) 👇👇👇
        highest = holding_info.get('highest_price', cost)
        max_profit_pct = (highest - cost) / cost * 100
        if max_profit_pct > 5:
            profit_retain_ratio = profit_pct / max_profit_pct
            if profit_retain_ratio < 0.6: # 利润回吐了 40% 以上
                score -= 20

        # 1. 僵尸股惩罚
        if hold_days > 5 and profit_pct < 2:
            score -= (hold_days - 5) * 1.5
            if abs(profit_pct) < 1: score -= 5
            
        # 2. 反弹保护
        if rsi < 30: score += 20
        elif rsi < 40: score += 5
        
        # 3. 趋势破位
        if ma20_slope < -0.005 and current_price < ma20:
            score -= 15
            
        return score, f"盈{profit_pct:.1f}%|天{hold_days}|RSI{rsi:.0f}"
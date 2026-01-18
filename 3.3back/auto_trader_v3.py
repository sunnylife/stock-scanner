# auto_trader_v3.py
# 修复版：包含强制减仓逻辑 + 市场标签标准化 + 并发AI扫描

import sys
import os
import json
import time
import math
import schedule
import requests
import pandas as pd
import logging
import concurrent.futures
import csv
import threading
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_FLOOR
import socket

# 🛡️【保命补丁】全局网络超时设置
# 任何网络请求（包括下载数据、AI请求）如果超过 30 秒没反应，强制报错断开
# 这样线程就会报错释放锁，不会把整个程序拖死
socket.setdefaulttimeout(30)

# 引入核心组件
try:
    from global_scanner import GlobalMarketScanner
    from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
except ImportError as e:
    print(f"❌ 核心模块导入失败: {e}")
    sys.exit(1)

# ==========================================
# 🔒 线程锁
# ==========================================
download_lock = threading.Lock() # 保护数据下载接口
csv_lock = threading.Lock()      # 保护文件写入

# ==========================================
# 🔌 长桥 SDK 导入
# ==========================================
try:
    from longport.openapi import TradeContext, Config, OrderSide, OrderType, TimeInForceType
    LONGBRIDGE_INSTALLED = True
except ImportError:
    try:
        from longport.openapi import TradeContext, Config, OrderSide, OrderType, TimeInForceType as TimeInForce
        LONGBRIDGE_INSTALLED = True
    except ImportError:
        LONGBRIDGE_INSTALLED = False
        print("⚠️ 未检测到 longport 库，实盘功能不可用")

def beijing_converter(*args):
    """将日志时间强制转换为北京时间 (UTC+8)"""
    utc_dt = datetime.now(timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    return bj_dt.timetuple()

logging.Formatter.converter = beijing_converter

# ==========================================
# ⚙️ 全局配置区
# ==========================================
CONFIG = {
    "SIMULATION_MODE": False,  # ⚠️ 全局总开关：True=全模拟，False=读取各市场配置

    # 请填入你的长桥 API Key
    "LB_APP_KEY": "f1bcf06adc2989210ab7caa4fd9101f6",
    "LB_APP_SECRET": "5e62e4155b17eba48c7e56e93045d5ea44e130dd411433c859b5da0db4d36cd1",
    # "LB_APP_KEY": "f1bcf06",
    # "LB_APP_SECRET": "5e62e4155b",
    "LB_ACCESS_TOKEN": "m_eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJsb25nYnJpZGdlIiwic3ViIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNzc1MTA0NzAzLCJpYXQiOjE3NjczMjg3MDMsImFrIjoiZjFiY2YwNmFkYzI5ODkyMTBhYjdjYWE0ZmQ5MTAxZjYiLCJhYWlkIjoyMDkzNjY2MywiYWMiOiJsYiIsIm1pZCI6MjI3MDg1ODMsInNpZCI6IkxPS2JaU01wVHkwUEp2K2x0dnd0Y1E9PSIsImJsIjozLCJ1bCI6MCwiaWsiOiJsYl8yMDkzNjY2MyJ9.nbd303ne7BLpCURukhpmv0awPoVvHNBiCpqvV68XkIMufs3dYzSCW2QNBWAlX92o8S5aBQQOGko6AB7l6CQiRLtKIefRUfCl0MxVAFm395GjAb7tCsOHcTqToOpfbyt1UrfqYl69NxWT4b2RMEnuPMa5Bn_tYVJiX-MNWYtn7ycdlWQbOfF4rTtWfyN-NlimUj_br7eTDLUImnJFYWSezwE7Vw09Tl-L6H-h4xvYYCrkUlmE_e1ECFFvwn9EQrmtyYTtVBz8mU_LJrVhOuZaRAcGn3Ew4CRtt3-L2Y2Ajox3LKMhhsDqr2FnhPdcFxOfZgvRIt-XunVE3ZZShwW9b-BGnLRrwi_X8pAsXkKUCeszMSi2HVY2iYoRwCDJWqmI1qE8nWPjuo9giX9kpeBu8Uv70FrRqX5WJMPlJXNO-CXeS2j44WSH1jOzDEqwLayL-YzH8PQBbplCSN8GDMXnwRX_PllL8Mk5R2s2UGA_2x9E6s8rmUUpBU9V2N9krPl35z7eaZxhOPEhvaKQhYxBYgrrxmY9gAtxFfo-TRYRQcw2C3DngP84NogJjMyVlYgtFTiSgawMQ1XnH-VpmcqBOMbN2kLcO6WHePZDj3uUEj0um0zctIystgqyIR-fCq_68AEc41r6DRocWruFkEKqy0PuV48U9U8Ewy23eemzXEc",
    
    "WECHAT_BOT_URL": "https://sctapi.ftqq.com/SCT308357T9fdH4QuIfs0J7h0tp4AW6xXu.send", 
    "HOLDINGS_FILE": "holdings.json",
    
    # --- 🇺🇸 美股配置 ---
    "US_SETTINGS": {
        "ENABLED": True,
        "SIMULATION": False, # 🟢 False = 实盘
        "MAX_HOLDINGS": 2,
        "ALLOCATED_CAPITAL": 1800, 
        "MIN_TRADE_AMT": 50
    },
    
    # --- 🇭🇰 港股配置 ---
    "HK_SETTINGS": {
        "ENABLED": True,
        "SIMULATION": True,  # 🔵 True = 模拟 (虚拟盘)
        "MAX_HOLDINGS": 2,
        "ALLOCATED_CAPITAL": 10000,
        "MIN_TRADE_AMT": 3000
    }
}

# ==========================================
# 📝 日志与数据记录
# ==========================================
log_filename = f'trader_log_{datetime.now().strftime("%Y%m%d")}.txt'

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(log_filename)
sys.stderr = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

csv_filename = "trade_history_data.csv"
csv_headers = [
    "time", "action", "market", "code", "price", "shares", 
    "cost", "profit_amt", "profit_pct", 
    "confidence", "market_phase", "reason", 
    "rsi", "kdj_k", "ma20_slope", "vol_ratio", "atr", "money_flow", 
    "total_asset", "cash_left", "hold_days"
]

def log_trade_data(data_dict):
    file_exists = os.path.isfile(csv_filename)
    with csv_lock: # 加锁写入
        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            if not file_exists: writer.writeheader()
            safe_data = {k: v for k, v in data_dict.items() if k in csv_headers}
            writer.writerow(safe_data)

# ==========================================
# 💰 辅助类
# ==========================================
class CostCalculator:
    @staticmethod
    def calculate_cost(market, shares, price):
        if market == 'hk': 
            val = shares * price
            stamp = math.ceil(val * 0.001)
            return round(stamp + 20, 2)
        else:
            fee = max(1.0, 0.005 * shares) + 0.01
            return round(fee, 2)

class NotificationManager:
    def send(self, title, content):
        logger.info(f"📨 [微信推送] {title}...")
        if not CONFIG["WECHAT_BOT_URL"]: return
        try:
            payload = {"title": title, "desp": f"【AutoTrader】\n{datetime.now()}\n---\n{content}"}
            requests.post(CONFIG["WECHAT_BOT_URL"], json=payload, timeout=10)
        except Exception: pass

class HoldingsManager:
    def __init__(self, filename):
        self.filename = filename
        self.holdings = self._load()

    def _load(self):
        if not os.path.exists(self.filename): return {}
        try:
            with open(self.filename, 'r') as f: return json.load(f)
        except: return {}

    def save(self):
        try:
            with open(self.filename, 'w') as f: json.dump(self.holdings, f, indent=2)
        except: pass

    def add_holding(self, code, price, shares, cost, market):
        # 存入时统一清洗 market 标签，避免后续混乱
        clean_market = 'hk' if market in ['hk', 'hk_stock'] else 'us'
        self.holdings[code] = {
            "code": code, "market": clean_market,
            "buy_price": float(price), "shares": float(shares),
            "cost_price": float((price * shares + cost) / shares),
            "highest_price": float(price),
            "buy_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save()

    def remove_holding(self, code):
        if code in self.holdings: del self.holdings[code]; self.save()

    def update_highest(self, code, price):
        if code in self.holdings and price > self.holdings[code]['highest_price']:
            self.holdings[code]['highest_price'] = float(price)
            self.save()

    def get_all(self): return self.holdings.values()
    
    # 获取市场持仓数量 (增加了标签清洗逻辑)
    def get_market_count(self, market):
        target_market = 'hk' if market in ['hk', 'hk_stock'] else 'us'
        count = 0
        for h in self.holdings.values():
            h_market = h.get('market', 'us')
            # 兼容旧标签
            if h_market in ['hk', 'hk_stock']: h_market = 'hk'
            else: h_market = 'us'
            
            if h_market == target_market:
                count += 1
        return count

class LongbridgeExecutor:
    def __init__(self):
        self.ctx = None
        
        # 判断是否需要连接实盘 API
        # 逻辑：全局模拟为False 且 (美股是实盘 OR 港股是实盘)
        is_global_sim = CONFIG["SIMULATION_MODE"]
        us_need_real = CONFIG["US_SETTINGS"]["ENABLED"] and not CONFIG["US_SETTINGS"].get("SIMULATION", False)
        hk_need_real = CONFIG["HK_SETTINGS"]["ENABLED"] and not CONFIG["HK_SETTINGS"].get("SIMULATION", False)
        
        need_connection = not is_global_sim and (us_need_real or hk_need_real)

        if LONGBRIDGE_INSTALLED and need_connection:
            try:
                conf = Config(CONFIG["LB_APP_KEY"], CONFIG["LB_APP_SECRET"], CONFIG["LB_ACCESS_TOKEN"])
                self.ctx = TradeContext(conf)
                logger.info("🔌 长桥 API 已连接 (混合模式)")
            except Exception as e:
                logger.error(f"❌ API 连接失败: {e}")

    def get_symbol_suffix(self, code, market):
        return f"{code}.HK" if market == 'hk' else f"{code}.US"

    def _is_simulated(self, market):
        """判断指定市场是否处于模拟模式"""
        if CONFIG["SIMULATION_MODE"]: return True
        if market == 'hk' and CONFIG["HK_SETTINGS"].get("SIMULATION", False): return True
        if market == 'us' and CONFIG["US_SETTINGS"].get("SIMULATION", False): return True
        return False

    def get_lot_size(self, code, market):
        if market == 'us': return 1
        if self._is_simulated(market): return 100
        try:
            info = self.ctx.static_info([self.get_symbol_suffix(code, market)])
            return int(info[0].board_lot) if info else 100
        except: return 100

    def estimate_max_buy(self, code, price, market):
        if self._is_simulated(market): return 99999
        if not self.ctx: return 0
        try:
            resp = self.ctx.estimate_max_purchase_quantity(
                symbol=self.get_symbol_suffix(code, market), 
                order_type=OrderType.LO, side=OrderSide.Buy, price=str(price)
            )
            return float(resp.data.cash_max_qty)
        except: return 0

    def execute_order(self, code, side, price, shares, market):
        symbol = self.get_symbol_suffix(code, market)
        
        # 1. 检查是否为模拟模式 (分市场)
        if self._is_simulated(market):
            logger.info(f"🛠️ [{market.upper()}模拟] {symbol} {side} {shares}股 @ {price}")
            return True
            
        if not self.ctx: return False
        
        # 定义变量防止 except 访问报错
        qty_str = "0"
        price_str = "0.00"

        try:
            # 2. 🛡️ 价格处理：买入挂高1%，卖出挂低1%，并强制保留2位小数转字符串
            # 这样可以避免 25.00000001 这种奇怪的精度导致 API 报错
            limit_price = price * 1.01 if side == OrderSide.Buy else price * 0.99
            price_str = f"{limit_price:.2f}"

            # 3. 🛡️ 数量处理：美股强制转整数
            # 即使 _execute_buy 里已经取整了，这里再保险一次
            if market == 'us':
                final_shares = int(shares) # 强制去掉小数位
                qty_str = str(final_shares)
            else:
                # 港股通常是整手，也可以强转 int 只要你确定不是碎股交易
                qty_str = str(int(shares)) 

            # 4. 提交订单
            self.ctx.submit_order(
                symbol=symbol, 
                order_type=OrderType.LO, # 限价单
                side=side,
                submitted_quantity=Decimal(qty_str),  # ✅ 传干净的字符串
                submitted_price=Decimal(price_str),   # ✅ 传干净的字符串
                time_in_force=TimeInForceType.Day
            )
            
            logger.info(f"✅ 订单提交成功: {symbol} {side} {qty_str}股 @ {price_str}")
            return True

        except Exception as e:
            # 5. 打印详细参数，方便 Debug
            logger.error(f"❌ 下单失败: {e} | 尝试提交: Symbol={symbol}, Qty={qty_str}, Price={price_str}")
            return False

    def get_cash_balance(self):
        # 只要连上了 API，就返回真实资金，否则返回模拟资金
        if self.ctx:
            try:
                resp = self.ctx.account_balance()
                for acc in resp:
                    for cash in acc.cash_infos:
                        if cash.currency == 'USD': return float(cash.available_cash)
                return 0.0
            except: return 0.0
        
        return 100000.0

# ==========================================
# 🧠 策略控制器 (完整版)
# ==========================================
class AutoTrader:
    def __init__(self):
        # 1. 先定义好今天的日志文件路径
        today = datetime.now().strftime('%Y%m%d')
        log_dir = "live_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 2. 创建 logger 工具函数
        def make_logger(name, file):
            l = logging.getLogger(name)
            l.setLevel(logging.INFO)
            l.handlers.clear()  # 清空已有的 handler
            h = logging.FileHandler(file, encoding='utf-8')
            h.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            l.addHandler(h)
            return l
        
        # 3. 从 strategy_core 导入并注入到 StrategyEngine
        from strategy_core import StrategyEngine
        self.strategy = StrategyEngine(
            ai_logger=make_logger("LiveAI", f"{log_dir}/ai_{today}.log"),
            trade_logger=make_logger("LiveTrade", f"{log_dir}/trade_{today}.log"),
            pnl_logger=make_logger("LivePnL", f"{log_dir}/pnl_{today}.log")
        )
        
        self.scanner = GlobalMarketScanner()
        # 注意：AutoTrader 自己的 analyzer 可以复用 strategy 里的，节省内存
        self.analyzer = self.strategy.analyzer
        self.holdings = HoldingsManager(CONFIG["HOLDINGS_FILE"])
        self.executor = LongbridgeExecutor()
        self.notifier = NotificationManager()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # ----------------------------------------------------
    # 🏃‍♂️ 扫描与买入逻辑 (保持 V3 的 AI 并发)
    # ----------------------------------------------------
    def _worker_analyze_stock(self, code, market):
        if code in [h['code'] for h in self.holdings.get_all()]: return None
        try:
            with download_lock:
                df = self.analyzer.get_stock_data(code, period='3mo')
                time.sleep(0.1)
            if df.empty or len(df) < 30: return None

            tech = self.analyzer.calculate_technical_indicators(df)
            money = self.analyzer.analyze_smart_money_flow(df)
            curr_row = df.iloc[-1]
            close_price = curr_row['close']
            vol_ratio = tech.get('vol_ratio_20', 1.0)
            ma20 = tech.get('ma20', 0)
            ma20_slope = tech.get('ma20_slope', 0)
            rsi = tech.get('rsi', 50)

            # 快速初筛
            potential_signal = False
            if market == 'hk': 
                if close_price > 1.0 and vol_ratio > 0.6: potential_signal = True
            elif market == 'us': 
                if (close_price > ma20) or (rsi > 40): potential_signal = True
            
            if not potential_signal: return None

            price_info = {
                "close": round(close_price, 2),
                "change_pct": round(curr_row['change_pct'], 2),
                "vol_ratio": round(vol_ratio, 2),
                "market_hint": f"{market}实盘"
            }
            
            logger.info(f"🤖 [AI思考中] {code} ...")
            ai_res = self.analyzer.get_llm_trade_decision(code, "today", price_info, tech, money)
            
            threshold = 80 if market == 'hk' else 75
            # 🛡️ [新增] 物理熔断风控 (防止 AI 上头)
            if ai_res.get('action') == "BUY":
                # 1. 拒绝极度超买
                if rsi > 85: 
                    logger.info(f"🛑 风控拦截 {code}: RSI {rsi:.1f} 过高，强制取消买入")
                    return None
                
                # 2. 拒绝垃圾股暴涨 (基本面分 < 50 且 涨幅 > 10%)
                # (需要你把 fundamental_score 传进来，或者简单判断)
                if close_price < 2.0 and vol_ratio > 5.0: # 举例：仙股巨量
                     logger.info(f"🛑 风控拦截 {code}: 仙股异常放量")
                     return None
                     
            if ai_res.get('action') == "BUY" and ai_res.get('confidence', 0) >= threshold:
                return {
                    "code": code, "price": close_price,
                    "confidence": ai_res['confidence'],
                    "reason": ai_res.get('reason', '无'),
                    "ai_raw": ai_res,
                    "tech_snapshot": {
                        "rsi": rsi, "kdj_k": tech.get("kdj_k", 0), 
                        "ma20_slope": ma20_slope, "vol_ratio": vol_ratio,
                        "atr": tech.get("atr", 0), "money_flow": money.get("money_flow_score", 0)
                    }
                }
            return None
        except Exception as e:
            logger.error(f"❌ 分析 {code} 异常: {e}")
            return None
    def _find_weakest_holding(self, market):
        """
        [智能版] 寻找该市场中'性价比最低'的持仓
        综合考虑：盈亏、持仓时间、是否超卖(反弹潜力)、趋势状态
        """
        market_holdings = [
            h for h in self.holdings.get_all() 
            if self._normalize_market(h.get('market')) == market
        ]
        
        if not market_holdings: return None
        
        candidates = []
        
        # 使用线程池并发获取持仓的最新技术指标
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_pos = {}
            for pos in market_holdings:
                # 提交任务：获取最近 60 天数据(为了算 MA20 和 RSI)
                future = executor.submit(self.analyzer.get_stock_data, pos['code'], period='3mo')
                future_to_pos[future] = pos
            
            for future in concurrent.futures.as_completed(future_to_pos):
                pos = future_to_pos[future]
                try:
                    df = future.result()
                    if df.empty or len(df) < 30: continue
                    
                    # 1. 计算基础数据
                    curr_price = df.iloc[-1]['close']
                    cost = pos.get('cost_price', pos.get('buy_price'))
                    profit_pct = (curr_price - cost) / cost * 100
                    
                    buy_time = datetime.strptime(pos['buy_date'], '%Y-%m-%d %H:%M:%S')
                    hold_days = (datetime.now() - buy_time).days
                    
                    # 2. 计算技术指标
                    tech = self.analyzer.calculate_technical_indicators(df)
                    rsi = tech.get('rsi', 50)
                    ma20 = tech.get('ma20', 0)
                    ma20_slope = tech.get('ma20_slope', 0)

                    # 👇👇👇 [核心修改] 调用 StrategyEngine 进行评分，而不是自己算 👇👇👇
                    # 这样保证了回测和实盘用的是完全同一套“裁决标准”
                    score, reason = self.strategy.calculate_holding_score(
                        holding_info=pos, 
                        current_price=curr_price, 
                        current_date=datetime.now(), 
                        tech=tech
                    )  
                        
                    candidates.append({
                        'pos': pos,
                        'score': score,
                        'reason': f"盈亏{profit_pct:.1f}%|RSI{rsi:.0f}|天数{hold_days}"
                    })
                    
                except Exception as e:
                    logger.error(f"❌ 评估持仓 {pos['code']} 失败: {e}")
                    continue
        
        if not candidates: return None
        
        # 按分数排序，分数最低的排前面（最该卖的）
        candidates.sort(key=lambda x: x['score'])
        
        worst_candidate = candidates[0]
        logger.info(f"🧐 持仓评估结果 (最差): {worst_candidate['pos']['code']} 得分 {worst_candidate['score']:.1f} [{worst_candidate['reason']}]")
        
        return worst_candidate

    def job_scan_market(self, market='us'):
        settings = CONFIG["HK_SETTINGS"] if market == 'hk' else CONFIG["US_SETTINGS"]
        if not settings["ENABLED"]: return
        
        # 1. 获取当前仓位状态
        current_count = self.holdings.get_market_count(market)
        max_holdings = settings["MAX_HOLDINGS"]
        is_full = current_count >= max_holdings

        logger.info(f"\n🚀 [启动扫描] {market.upper()} 市场 | 仓位: {current_count}/{max_holdings}")
        
        # ⚠️ 修改点1：如果满了，不直接 Return，而是进入“换仓模式”检查
        # 但为了节省资源，如果满了，我们要求必须扫描到“极好”的机会才换
        
        # 2. 获取候选池
        candidates = []
        if market == 'hk': candidates = self.scanner.get_hk_candidates(top_n=20)
        else: candidates = self.scanner.get_us_candidates(top_n=20)
        
        if not candidates:
            logger.info("⚠️ 无候选股票")
            return

        # 3. 启动 AI 分析
        potential_buys = []
        future_to_code = {self.thread_pool.submit(self._worker_analyze_stock, code, market): code for code in candidates}

        for future in concurrent.futures.as_completed(future_to_code):
            res = future.result()
            if res: potential_buys.append(res)

        # 按置信度排序
        potential_buys.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not potential_buys:
            logger.info("💤 无高置信度机会")
            return

        # 4. 决策环节：买入 或 换仓
        best_target = potential_buys[0] # 只看最好的一个
        logger.info(f"🔥 发现最佳机会: {best_target['code']} (信心:{best_target['confidence']})")

        buy_logs = []

        if not is_full:
            # A. 仓位没满 -> 直接买
            self._execute_buy(best_target, market, settings, buy_logs)
            
        else:
            # B. 仓位已满 -> 触发 PK 逻辑 (优胜劣汰)
            logger.info(f"🤔 仓位已满，正在评估是否值得换仓...")
            
            # 门槛：新机会的信心必须非常高 (比如 > 85) 才有资格替换手里的票
            # 否则频繁换仓会亏手续费
            swap_threshold = 85 
            if best_target['confidence'] < swap_threshold:
                logger.info(f"🛑 新机会信心 ({best_target['confidence']}) 未达到换仓门槛 ({swap_threshold})，放弃。")
                return

            # 寻找最弱的持仓
            weakest = self._find_weakest_holding(market)
            if not weakest: return 

            w_pos = weakest['pos']
            w_score = weakest['score']
            
            # 2. 换仓阈值判断 (关键逻辑)
            # 只有当新机会的信心极高，且旧持仓真的很烂时，才换
            
            should_swap = False
            swap_reason = ""
            
            # 情况一：旧持仓严重破位或死气沉沉 (分数很低)
            # 这种情况下，只要新机会还可以 (信心>75)，就止损换仓
            if w_score < -10: 
                if best_target['confidence'] >= 75:
                    should_swap = True
                    swap_reason = f"止损换强 (持仓分 {w_score:.1f} 低于 -10)"

            # 情况二：旧持仓一般般 (分数在 0 左右震荡)
            # 这种情况下，新机会必须非常强 (信心>85)，才值得覆盖手续费
            elif w_score < 5:
                if best_target['confidence'] >= 85:
                    should_swap = True
                    swap_reason = f"择优汰劣 (新机会信心 {best_target['confidence']} 高)"
            
            # 情况三：旧持仓表现很好 (分数 > 10，比如正在主升浪)
            # 坚决不换！哪怕新机会也是 90 分也不换，避免卖飞牛股
            else:
                logger.info(f"🛡️ 最差持仓 {w_pos['code']} 得分 {w_score:.1f} 依然健康，拒绝换仓")
        
                
            if should_swap:
                logger.info(f"🔄 [执行换仓] 卖出 {w_pos['code']} -> 买入 {best_target['code']}")
                
                # 1. 先卖
                sell_log = []
                self._execute_sell({
                    'pos': w_pos, 
                    'price': w_pos['buy_price'], # 这里应该传最新价，但在 _execute_sell 里只是用来记日志，近似一下没关系，或者你再 fetch 一次
                    'pct': w_profit,
                    'reason': f"被 {best_target['code']} 替换"
                }, sell_log)
                
                # 2. 再买
                # 稍微等一下确保资金释放（如果是T+0资金）
                time.sleep(1) 
                self._execute_buy(best_target, market, settings, buy_logs)
                
                if sell_log: self.notifier.send("🔄 换仓-卖出", "\n".join(sell_log))
            else:
                logger.info("🛡️ 手中持仓表现尚可，暂不替换。")

        if buy_logs:
            self.notifier.send(f"🔵 {market.upper()} 买入/换仓", "\n".join(buy_logs))

    def _execute_buy(self, target, market, settings, buy_logs):
        code = target['code']
        price = target['price']
        
        curr_count = self.holdings.get_market_count(market)
        slots_left = settings["MAX_HOLDINGS"] - curr_count
        if slots_left <= 0: return

        budget = settings["ALLOCATED_CAPITAL"] / settings["MAX_HOLDINGS"]
        raw_shares = budget / price
        
        final_shares = 0
        # 👇👇👇 [核心修改点] 👇👇👇
        if market == 'us':
            # ❌ 原代码: final_shares = round(raw_shares, 2)
            # ✅ 新代码: 强制向下取整，保证成交稳定性
            final_shares = int(raw_shares) 
            
        elif market == 'hk':
            lot_size = self.executor.get_lot_size(code, market)
            final_shares = int(raw_shares // lot_size) * lot_size
        
        # 🛡️ 增加最小股数检查
        if final_shares < 1:
            logger.info(f"⚠️ {code} 资金不足购买 1 股，跳过")
            return
        # 👆👆👆 [修改结束] 👆👆👆
        
        if final_shares * price < settings["MIN_TRADE_AMT"]: return

        fee = CostCalculator.calculate_cost(market, final_shares, price)
        if self.executor.execute_order(code, OrderSide.Buy, price, final_shares, market):
            self.holdings.add_holding(code, price, final_shares, fee, market)
            msg = f"🚀 买入 {code}: {final_shares}股 @ {price}"
            buy_logs.append(msg)
            logger.info(msg)
            
            curr_cash = self.executor.get_cash_balance()
            total_asset = curr_cash + sum(h['shares']*h['buy_price'] for h in self.holdings.get_all())
            log_trade_data({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "action": "BUY", "market": market, "code": code,
                "price": price, "shares": final_shares, "cost": fee,
                "confidence": target['confidence'],
                "reason": target['reason'],
                "total_asset": round(total_asset, 2), "cash_left": round(curr_cash, 2)
            })

    # ----------------------------------------------------
    # 🛡️ 监控与卖出 (核心修复部分)
    # ----------------------------------------------------
    def _normalize_market(self, tag):
        """统一市场标签"""
        if tag in ['hk', 'hk_stock']: return 'hk'
        if tag in ['us', 'us_stock']: return 'us'
        return 'us'

    def job_monitor(self):
        logger.info("\n⏰ [开始监控] 检查持仓...")
        all_holdings = list(self.holdings.get_all())
        if not all_holdings: return

        sell_logs = []
        
        # ==========================================
        # 1. 强制瘦身逻辑 (Force Reduce) - 修复版
        # ==========================================
        for market in ['hk', 'us']:
            # 使用 normalize 修复标签匹配问题
            market_holdings = [h for h in all_holdings if self._normalize_market(h.get('market')) == market]
            settings = CONFIG["HK_SETTINGS"] if market == 'hk' else CONFIG["US_SETTINGS"]
            max_limit = settings["MAX_HOLDINGS"]
            excess_count = len(market_holdings) - max_limit

            if excess_count > 0:
                logger.warning(f"⚠️ {market.upper()} 持仓超标 ({len(market_holdings)}/{max_limit})，计算强制减仓...")
                
                # 获取当前收益率以便排序
                candidates_to_sort = []
                for pos in market_holdings:
                    res = self._check_single_pos(pos, force_check_price=True) # 只查价格，不查策略
                    if res:
                        candidates_to_sort.append(res)
                
                # 按收益率从小到大排序（先卖亏得多的）
                candidates_to_sort.sort(key=lambda x: x['pct'])
                
                # 取出最差的 N 个
                to_sell = candidates_to_sort[:excess_count]
                
                for item in to_sell:
                    # 强制改为 SELL 指令
                    item['reason'] = f"强制瘦身(排名倒数第{to_sell.index(item)+1})"
                    self._execute_sell(item, sell_logs)
                    
                    # 从 all_holdings 中移除，避免后续重复检查
                    # 注意：这里只是从内存列表移除，防止下一步 check 又卖一次
                    # 实际上 execute_sell 已经操作了 holdingsManager
                    original_pos = item['pos']
                    if original_pos in all_holdings:
                        all_holdings.remove(original_pos)

        # ==========================================
        # 2. 常规监控逻辑 (并发检查)
        # ==========================================
        if all_holdings:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_pos = {executor.submit(self._check_single_pos, pos): pos for pos in all_holdings}
                for future in concurrent.futures.as_completed(future_to_pos):
                    res = future.result()
                    if res and res.get('action') == 'SELL': # 只有明确 SELL 才卖
                        self._execute_sell(res, sell_logs)

        if sell_logs:
            self.notifier.send("🔴 卖出报告", "\n".join(sell_logs))
        logger.info(f"✅ [监控结束] 本轮检查完毕，持仓无恙，继续持有。")

    def _check_single_pos(self, pos, force_check_price=False):
        """
        检查单只持仓 (包含：保本、分级止盈、均线破位、硬止损)
        """
        code = pos['code']
        try:
            with download_lock:
                df = self.analyzer.get_stock_data(code, period='3mo') # 获取足够长的数据算均线
            if df.empty or len(df) < 30: return None
            
            # 1. 基础数据准备
            curr_price = df.iloc[-1]['close']
            cost = pos.get('cost_price', pos.get('buy_price'))
            
            # 自动修复最高价 (防止历史数据缺失导致 highest 为 0)
            highest = pos.get('highest_price', cost)
            if curr_price > highest:
                highest = curr_price
                self.holdings.update_highest(code, curr_price)
            
            # 计算核心指标
            pct = (curr_price - cost) / cost * 100               # 当前盈亏%
            max_profit_pct = (highest - cost) / cost * 100       # 历史最大盈亏%
            drawdown = (curr_price - highest) / highest * 100    # 当前回撤%
            
            # 如果是强制查询模式(用于换仓排序)，直接返回数据，不触发卖出
            if force_check_price:
                 return {"pos": pos, "price": curr_price, "pct": pct, "action": "INFO"}

            # ====================================================
            # 🛡️ 卖出策略核心 (优先级从高到低)
            # ====================================================

            # 策略A: 保本止损 (防止“由赚变亏”)
            # 逻辑：如果曾经赚过 4% 以上，绝不允许跌穿成本价 (+0.5%是留给手续费的)
            if max_profit_pct > 4 and pct < 0.5:
                return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"保本离场(曾盈{max_profit_pct:.1f}%)", "pct": pct}

            # 策略B: 分级移动止盈 (Trailing Stop) - 替代了旧版的回撤止盈
            # 1. 既然赚了 >8%，就别太贪，回撤 3% 锁定利润
            if max_profit_pct > 8 and drawdown < -3:
                return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"止盈(高位回撤3%)", "pct": pct}
            # 2. 如果赚了 >4% (微利)，回撤 5% 就走 (防止深套)
            elif max_profit_pct > 4 and drawdown < -5:
                return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"止盈(回撤保护)", "pct": pct}

            # 策略C: 均线破位 (技术面离场)
            # 即使没亏多少，但如果跌破 MA20 生命线，说明趋势坏了，先出来观望
            tech = self.analyzer.calculate_technical_indicators(df)
            ma20 = tech.get('ma20', 0)
            if ma20 > 0 and curr_price < ma20 * 0.995: # 跌破 0.5% 确认有效跌破
                 return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"趋势破位(跌破MA20)", "pct": pct}

            # 策略D: 硬性止损 (底线)
            # 不管什么原因，亏了 7% 必须无脑砍，防止爆仓
            if pct < -7: 
                return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"硬止损(-7%)", "pct": pct}
            
            # ====================================================
            # 🧠 AI 诊断 (只有未触发硬规则时才咨询 AI)
            # ====================================================
            # 只有在微盈微亏 (-5% ~ 10%) 的尴尬区间，才让 AI 决定去留
            if -5 < pct < 10:
                money = self.analyzer.analyze_smart_money_flow(df)
                price_info = {"close": curr_price, "change_pct": df.iloc[-1]['change_pct'], "vol_ratio": 1.0}
                
                # 调用 AI
                ai_res = self.analyzer.get_llm_trade_decision(code, "today", price_info, tech, money)
                
                # 如果 AI 强烈建议卖出 (信心 > 75)，则卖出
                if ai_res.get('action') == "SELL" and ai_res.get('confidence', 0) > 75:
                    return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"AI建议卖出: {ai_res.get('reason')}", "pct": pct}

            return None # 继续持有

        except Exception as e:
            logger.error(f"❌ 监控 {code} 异常: {e}")
            return None

    def _execute_sell(self, res, sell_logs):
        pos = res['pos']
        code = pos['code']
        market = pos['market'] # 这里的 market 已经是清洗过的 'hk' 或 'us'
        
        # 再次确保市场标签正确
        clean_market = self._normalize_market(market)

        if self.executor.execute_order(code, OrderSide.Sell, res['price'], pos['shares'], clean_market):
            self.holdings.remove_holding(code)
            msg = f"🔴 卖出 {code}: 盈亏 {res['pct']:.1f}% | 原因: {res['reason']}"
            sell_logs.append(msg)
            logger.info(msg)
            
            log_trade_data({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "action": "SELL", "market": clean_market, "code": code,
                "price": res['price'], "shares": pos['shares'],
                "profit_pct": res['pct'], "reason": res['reason']
            })

    def run(self):
        logger.info("⏳ AutoTrader V3 (修复版) 启动...")
        mode_str = []
        if CONFIG["US_SETTINGS"]["ENABLED"]:
            us_mode = "模拟" if CONFIG["US_SETTINGS"].get("SIMULATION", False) or CONFIG["SIMULATION_MODE"] else "实盘"
            mode_str.append(f"美股:{us_mode}")
        if CONFIG["HK_SETTINGS"]["ENABLED"]:
            hk_mode = "模拟" if CONFIG["HK_SETTINGS"].get("SIMULATION", False) or CONFIG["SIMULATION_MODE"] else "实盘"
            mode_str.append(f"港股:{hk_mode}")
            
        logger.info(f"模式: {' | '.join(mode_str)}")
        
        # 调度任务
        schedule.every().day.at("09:40").do(self.job_scan_market, market='hk')
        schedule.every().day.at("11:30").do(self.job_monitor)
        schedule.every().day.at("13:35").do(self.job_scan_market, market='hk')
        schedule.every().day.at("15:39").do(self.job_monitor)
        
        schedule.every().day.at("22:35").do(self.job_scan_market, market='us')
        schedule.every().day.at("02:00").do(self.job_monitor)
        schedule.every().day.at("03:00").do(self.job_scan_market, market='us')
        
        # 启动时立即执行一次监控，处理积压的持仓
        self.job_monitor()
        
        # 启动时立即执行一次港股扫描（如果启用）
        if CONFIG["HK_SETTINGS"]["ENABLED"]:
            logger.info("🔍 启动时触发港股市场扫描...")
            self.job_scan_market(market='hk')

        logger.info("💤 系统进入待机模式，等待下一次调度任务...")
        
        # 👇👇👇 [修改] 下面这块 while 循环 👇👇👇
        last_heartbeat = datetime.now()

        while True:
            schedule.run_pending()
            # 每分钟打印一次心跳，证明没死
            if (datetime.now() - last_heartbeat).seconds > 60:
                print(f"[{datetime.now().strftime('%H:%M')}] .", end='', flush=True) # 打印一个小点
                last_heartbeat = datetime.now()
                
            # time.sleep(1) # 改成 1 秒检查一次，响应更灵敏
            time.sleep(30)

if __name__ == "__main__":
    trader = AutoTrader()
    trader.run()
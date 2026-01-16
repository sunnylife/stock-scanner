# auto_trader_v3.py
# 基于 StrategyEngine 的实盘交易系统

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🛡️【保命补丁】全局网络超时设置
socket.setdefaulttimeout(30)

# 引入核心组件
try:
    from strategy_core import StrategyEngine, trade_logger, pnl_logger, ai_logger  # 导入策略核心
except ImportError as e:
    print(f"❌ 策略核心模块导入失败: {e}")
    sys.exit(1)

# ==========================================
# 🔒 线程锁
# ==========================================
csv_lock = threading.Lock()      # 保护文件写入
# 注意：download_lock 已经在 StrategyEngine 内部管理，这里不需要重复定义

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
    "SIMULATION_MODE": True,  # ⚠️ 调试为True，实盘请改为 False
    
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
        "MAX_HOLDINGS": 3,
        "ALLOCATED_CAPITAL": 3000, 
        "MIN_TRADE_AMT": 50
    },
    
    # --- 🇭🇰 港股配置 ---
    "HK_SETTINGS": {
        "ENABLED": True,
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
        if LONGBRIDGE_INSTALLED and not CONFIG["SIMULATION_MODE"]:
            try:
                conf = Config(CONFIG["LB_APP_KEY"], CONFIG["LB_APP_SECRET"], CONFIG["LB_ACCESS_TOKEN"])
                self.ctx = TradeContext(conf)
                logger.info("🔌 长桥 API 已连接")
            except Exception as e:
                logger.error(f"❌ API 连接失败: {e}")

    def get_symbol_suffix(self, code, market):
        return f"{code}.HK" if market == 'hk' else f"{code}.US"

    def get_lot_size(self, code, market):
        if market == 'us': return 1
        if CONFIG["SIMULATION_MODE"]: return 100
        try:
            info = self.ctx.static_info([self.get_symbol_suffix(code, market)])
            return int(info[0].board_lot) if info else 100
        except: return 100

    def estimate_max_buy(self, code, price, market):
        if CONFIG["SIMULATION_MODE"]: return 99999
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
        if CONFIG["SIMULATION_MODE"]:
            logger.info(f"🛠️ [模拟交易] {symbol} {side} {shares}股 @ {price}")
            return True
        if not self.ctx: return False
        try:
            limit_price = price * 1.01 if side == OrderSide.Buy else price * 0.99
            self.ctx.submit_order(
                symbol=symbol, order_type=OrderType.LO, side=side,
                submitted_quantity=Decimal(str(shares)), 
                submitted_price=Decimal(f"{limit_price:.2f}"),
                time_in_force=TimeInForceType.Day
            )
            return True
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False

    def get_cash_balance(self):
        if CONFIG["SIMULATION_MODE"]: return 100000.0
        if not self.ctx: return 0.0
        try:
            resp = self.ctx.account_balance()
            for acc in resp:
                for cash in acc.cash_infos:
                    if cash.currency == 'USD': return float(cash.available_cash)
            return 0.0
        except: return 0.0

# ==========================================
# 🧠 策略控制器 (基于 StrategyEngine)
# ==========================================
class AutoTrader:
    def __init__(self):
        self.strategy = StrategyEngine()  # 策略大脑
        self.holdings = HoldingsManager(CONFIG["HOLDINGS_FILE"])  # 账本
        self.executor = LongbridgeExecutor()  # 交易执行器（手）
        self.notifier = NotificationManager()  # 通知器
        self.thread_pool = ThreadPoolExecutor(max_workers=5)  # 线程池

    # ----------------------------------------------------
    # 🏃‍♂️ 扫描与买入逻辑 (基于 StrategyEngine)
    # ----------------------------------------------------
    def _worker_analyze_stock(self, code, market):
        """
        使用 StrategyEngine 分析单只股票
        注意：实盘模式下 data_slice=None，策略层会自动下载最新数据
        """
        if code in [h['code'] for h in self.holdings.get_all()]: 
            return None
        
        try:
            logger.info(f"🤖 [AI思考中] {code} ...")
            
            # 调用策略核心的分析函数
            # data_slice=None 表示实盘模式，策略层会自动下载最新数据
            result = self.strategy.analyze_ticker(code, "today", data_slice=None)
            
            if not result:
                return None
            
            # 设置信心阈值
            threshold = 80 if market == 'hk' else 75
            
            if result['action'] == 'BUY' and result['confidence'] >= threshold:
                return {
                    "code": result['code'],
                    "price": result['price'],
                    "confidence": result['confidence'],
                    "reason": result['reason'],
                    "tech_snapshot": result.get('tech', {})  # 技术指标快照
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 分析 {code} 异常: {e}")
            return None
    def _find_weakest_holding(self, market):
        """
        使用 StrategyEngine 的评分系统寻找最弱持仓
        """
        market_holdings = [
            h for h in self.holdings.get_all() 
            if self._normalize_market(h.get('market')) == market
        ]
        
        if not market_holdings: 
            return None
        
        candidates = []
        
        # 使用线程池并发获取持仓的最新数据和评分
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_pos = {}
            for pos in market_holdings:
                # 提交任务：获取最新数据
                future = executor.submit(self.strategy.analyzer.get_stock_data, pos['code'], period='3mo')
                future_to_pos[future] = pos
            
            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                try:
                    df = future.result()
                    if df.empty or len(df) < 30: 
                        continue
                    
                    # 获取当前价格
                    curr_price = df.iloc[-1]['close']
                    
                    # 计算技术指标
                    tech = self.strategy.analyzer.calculate_technical_indicators(df)
                    
                    # 调用策略核心的持仓评分函数
                    score, reason = self.strategy.calculate_holding_score(
                        pos, curr_price, datetime.now(), tech
                    )
                    
                    candidates.append({
                        'pos': pos,
                        'score': score,
                        'reason': reason,
                        'curr_price': curr_price
                    })
                    
                except Exception as e:
                    logger.error(f"❌ 评估持仓 {pos['code']} 失败: {e}")
                    continue
        
        if not candidates: 
            return None
        
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
        
        # ⚠️ 说明：如果仓位满了，依然会扫描，用于换仓评估
                
        # 2. 使用 StrategyEngine 获取候选池
        market_key = market + '_stock' if market in ['hk', 'us'] else market
        candidates = self.strategy.get_candidates(market_key, limit=20)
                
        if not candidates:
            logger.info("⚠️ 无候选股票")
            return
        
        logger.info(f"📋 获得候选股票: {len(candidates)} 只")
        
        # 3. 并发分析候选股票
        potential_buys = []
        future_to_code = {
            self.thread_pool.submit(self._worker_analyze_stock, code, market): code 
            for code in candidates
        }
        
        for future in as_completed(future_to_code):
            res = future.result()
            if res: 
                potential_buys.append(res)
        
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
            
            # 门槛：新机会的信心必须非常高才有资格替换
            swap_threshold = 85 
            if best_target['confidence'] < swap_threshold:
                logger.info(f"🛑 新机会信心 ({best_target['confidence']}) 未达到换仓门槛 ({swap_threshold})，放弃。")
                return

            # 使用策略引擎评分系统寻找最弱持仓
            weakest = self._find_weakest_holding(market)
            if not weakest: 
                return 

            w_pos = weakest['pos']
            w_score = weakest['score']
            w_price = weakest.get('curr_price', w_pos.get('buy_price'))
            
            # 计算当前盈亏
            cost = w_pos.get('cost_price', w_pos.get('buy_price'))
            w_profit = (w_price - cost) / cost * 100
            
            # 换仓阈值判断 (关键逻辑)
            should_swap = False
            swap_reason = ""
            
            # 情况一：旧持仓严重破位或死气沉沉 (分数很低)
            if w_score < -10: 
                if best_target['confidence'] >= 75:
                    should_swap = True
                    swap_reason = f"止损换强 (持仓分 {w_score:.1f} 低于 -10)"

            # 情况二：旧持仓一般般 (分数在 0 左右震荡)
            elif w_score < 5:
                if best_target['confidence'] >= 85:
                    should_swap = True
                    swap_reason = f"择优汰劣 (新机会信心 {best_target['confidence']} 高)"
            
            # 情况三：旧持仓表现很好 (分数 > 10)
            else:
                logger.info(f"🛡️ 最差持仓 {w_pos['code']} 得分 {w_score:.1f} 依然健康，拒绝换仓")
        
                
            if should_swap:
                logger.info(f"🔄 [执行换仓] 卖出 {w_pos['code']} -> 买入 {best_target['code']}")
                
                # 1. 先卖
                sell_log = []
                self._execute_sell({
                    'pos': w_pos, 
                    'price': w_price,
                    'pct': w_profit,
                    'reason': f"被 {best_target['code']} 替换 ({swap_reason})"
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
        if market == 'us': final_shares = round(raw_shares, 2)
        elif market == 'hk':
            lot_size = self.executor.get_lot_size(code, market)
            final_shares = int(raw_shares // lot_size) * lot_size
        
        if final_shares * price < settings["MIN_TRADE_AMT"]: return

        fee = CostCalculator.calculate_cost(market, final_shares, price)
        if self.executor.execute_order(code, OrderSide.Buy, price, final_shares, market):
            self.holdings.add_holding(code, price, final_shares, fee, market)
            msg = f"🚀 买入 {code}: {final_shares}股 @ {price}"
            buy_logs.append(msg)
            logger.info(msg)
            
            curr_cash = self.executor.get_cash_balance()
            total_asset = curr_cash + sum(h['shares']*h['buy_price'] for h in self.holdings.get_all())
            
            # 记录到 CSV
            log_trade_data({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "action": "BUY", "market": market, "code": code,
                "price": price, "shares": final_shares, "cost": fee,
                "confidence": target['confidence'],
                "reason": target['reason'],
                "total_asset": round(total_asset, 2), "cash_left": round(curr_cash, 2)
            })
            
            # 同时记录到策略核心的交易日志
            trade_logger.info(f"REAL_TRADE | BUY {code} | {final_shares}股 @ {price} | 信心:{target['confidence']} | {target['reason'][:30]}")

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
        检查单只持仓
        force_check_price=True 时，只返回当前价格和盈亏信息，不进行卖出判断
        """
        code = pos['code']
        try:
            # 使用策略引擎的分析器获取数据
            df = self.strategy.analyzer.get_stock_data(code, period='1mo')
            if df.empty: 
                return None
            
            curr_price = df.iloc[-1]['close']
            cost = pos.get('cost_price', pos.get('buy_price'))
            highest = pos.get('highest_price', cost)
            
            if curr_price > highest:
                highest = curr_price
                self.holdings.update_highest(code, curr_price)
            
            pct = (curr_price - cost) / cost * 100
            
            # 如果是强制查询模式，直接返回数据
            if force_check_price:
                return {"pos": pos, "price": curr_price, "pct": pct, "action": "INFO"}

            # 常规检查模式
            drawdown = (curr_price - highest) / highest * 100
            
            # 1. 硬止损
            if pct < -7: 
                return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"硬止损 {pct:.1f}%", "pct": pct}
            
            # 2. 回撤止盈
            if pct > 10 and drawdown < -4:
                return {"action": "SELL", "pos": pos, "price": curr_price, "reason": f"回撤止盈", "pct": pct}

            # 3. AI 诊断（使用策略引擎）
            # 简单策略：只有微盈微亏时才问AI
            if -5 < pct < 10:
                # 使用策略引擎的 analyze_ticker 进行诊断
                result = self.strategy.analyze_ticker(code, "today", data_slice=df)
                
                if result and result['action'] == 'SELL' and result['confidence'] > 75:
                    return {
                        "action": "SELL", 
                        "pos": pos, 
                        "price": curr_price, 
                        "reason": f"AI卖出: {result['reason'][:30]}", 
                        "pct": pct
                    }

            return None
            
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
            
            # 记录到 CSV
            log_trade_data({
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "action": "SELL", "market": clean_market, "code": code,
                "price": res['price'], "shares": pos['shares'],
                "profit_pct": res['pct'], "reason": res['reason']
            })
            
            # 同时记录到策略核心的交易日志
            trade_logger.info(f"REAL_TRADE | SELL {code} | {pos['shares']}股 @ {res['price']} | 盈亏:{res['pct']:.1f}% | {res['reason']}")

    def run(self):
        logger.info("⏳ AutoTrader V3 (基于 StrategyEngine) 启动...")
        logger.info(f"模式: {'🛠️ 模拟' if CONFIG['SIMULATION_MODE'] else '💸 实盘'}")
        logger.info("🧠 策略引擎已加载，使用统一的分析和评分系统")
        
        # 调度任务
        schedule.every().day.at("09:40").do(self.job_scan_market, market='hk')
        schedule.every().day.at("11:30").do(self.job_monitor)
        schedule.every().day.at("13:35").do(self.job_scan_market, market='hk')
        schedule.every().day.at("15:30").do(self.job_monitor)
        
        schedule.every().day.at("22:35").do(self.job_scan_market, market='us')
        schedule.every().day.at("02:00").do(self.job_monitor)
        schedule.every().day.at("03:00").do(self.job_scan_market, market='us')
        
        # 启动时立即执行一次监控，处理积压的持仓
        self.job_monitor()

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
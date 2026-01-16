# auto_trader_v2.py
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
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR

# ==========================================
# 🕒 强制修复日志时区 (UTC -> 北京时间)
# ==========================================
from datetime import timezone, timedelta

# 引入之前的核心组件
from global_scanner import GlobalMarketScanner
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer


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
        print("⚠️ 未检测到 longport 库，实盘功能不可用。请运行: pip install longport")

def beijing_converter(*args):
    """将日志时间强制转换为北京时间 (UTC+8)"""
    utc_dt = datetime.now(timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    return bj_dt.timetuple()

# 覆盖 logging 的默认时间转换器
logging.Formatter.converter = beijing_converter

# ==========================================
# ⚙️ 全局配置区 (分市场配置)
# ==========================================
CONFIG = {
    "SIMULATION_MODE": True,  # ⚠️ 调试为True，实盘为False
    
    "LB_APP_KEY": "f1bcf06adc2989210ab7caa4fd9101f6",
    "LB_APP_SECRET": "5e62e4155b17eba48c7e56e93045d5ea44e130dd411433c859b5da0db4d36cd1",
    # "LB_APP_KEY": "f1bcf06",
    # "LB_APP_SECRET": "5e62e4155b",
    "LB_ACCESS_TOKEN": "m_eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJsb25nYnJpZGdlIiwic3ViIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNzc1MTA0NzAzLCJpYXQiOjE3NjczMjg3MDMsImFrIjoiZjFiY2YwNmFkYzI5ODkyMTBhYjdjYWE0ZmQ5MTAxZjYiLCJhYWlkIjoyMDkzNjY2MywiYWMiOiJsYiIsIm1pZCI6MjI3MDg1ODMsInNpZCI6IkxPS2JaU01wVHkwUEp2K2x0dnd0Y1E9PSIsImJsIjozLCJ1bCI6MCwiaWsiOiJsYl8yMDkzNjY2MyJ9.nbd303ne7BLpCURukhpmv0awPoVvHNBiCpqvV68XkIMufs3dYzSCW2QNBWAlX92o8S5aBQQOGko6AB7l6CQiRLtKIefRUfCl0MxVAFm395GjAb7tCsOHcTqToOpfbyt1UrfqYl69NxWT4b2RMEnuPMa5Bn_tYVJiX-MNWYtn7ycdlWQbOfF4rTtWfyN-NlimUj_br7eTDLUImnJFYWSezwE7Vw09Tl-L6H-h4xvYYCrkUlmE_e1ECFFvwn9EQrmtyYTtVBz8mU_LJrVhOuZaRAcGn3Ew4CRtt3-L2Y2Ajox3LKMhhsDqr2FnhPdcFxOfZgvRIt-XunVE3ZZShwW9b-BGnLRrwi_X8pAsXkKUCeszMSi2HVY2iYoRwCDJWqmI1qE8nWPjuo9giX9kpeBu8Uv70FrRqX5WJMPlJXNO-CXeS2j44WSH1jOzDEqwLayL-YzH8PQBbplCSN8GDMXnwRX_PllL8Mk5R2s2UGA_2x9E6s8rmUUpBU9V2N9krPl35z7eaZxhOPEhvaKQhYxBYgrrxmY9gAtxFfo-TRYRQcw2C3DngP84NogJjMyVlYgtFTiSgawMQ1XnH-VpmcqBOMbN2kLcO6WHePZDj3uUEj0um0zctIystgqyIR-fCq_68AEc41r6DRocWruFkEKqy0PuV48U9U8Ewy23eemzXEc",
    
    "WECHAT_BOT_URL": "https://sctapi.ftqq.com/SCT308357T9fdH4QuIfs0J7h0tp4AW6xXu.send", 
    "HOLDINGS_FILE": "holdings.json",
    
    # --- 🇺🇸 美股配置 ---
    "US_SETTINGS": {
        "ENABLED": True,           # 是否开启美股交易
        "MAX_HOLDINGS": 2,         # 美股最大持仓数
        "ALLOCATED_CAPITAL": 1200, # 美股专用资金 (美元)
        "MIN_TRADE_AMT": 50        # 最小交易额
    },
    
    # --- 🇭🇰 港股配置 ---
    "HK_SETTINGS": {
        "ENABLED": False,          # ⚠️ 如果不想买港股，把这里改成 False 即可
        "MAX_HOLDINGS": 2,         # 港股最大持仓数
        "ALLOCATED_CAPITAL": 10000,# 港股专用资金 (港币) - 注意单位！
        "MIN_TRADE_AMT": 3000      # 港股一手通常较贵，门槛设高点
    }
}

# ==========================================
# 📝 日志与数据记录 (增强版)
# ==========================================
# 1. 运行日志 (文本)
log_filename = f'trader_log_{datetime.now().strftime("%Y%m%d")}.txt'
class DualLogger:
    """
    黑科技：将屏幕输出同时写入文件
    解决 analyzer 内部 print/log 无法写入日志的问题
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        # 1. 输出到屏幕
        self.terminal.write(message)
        # 2. 输出到文件
        self.log.write(message)
        self.log.flush() # 强制立即写入，防止丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 🚀 立即接管系统标准输出
sys.stdout = DualLogger(log_filename)
sys.stderr = sys.stdout # 错误信息也抓取

# 配置 logging (让它只输出到 stdout，然后由 DualLogger 接管写入文件)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # 关键：只往屏幕打，DualLogger 会自动存文件
    ]
)
logger = logging.getLogger(__name__)

# 2. 交易数据日志 (CSV, 用于后续分析)
csv_filename = "trade_history_data.csv"
csv_headers = [
    "time", "action", "market", "code", "price", "shares", 
    "cost", "profit_amt", "profit_pct", # 盈亏数据
    "confidence", "market_phase", "reason", # AI判断
    "rsi", "kdj_k", "ma20_slope", "vol_ratio", "atr", "money_flow", # 核心技术指标
    "total_asset", "cash_left", "hold_days" # 账户状态
]

def log_trade_data(data_dict):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers) # 使用新的表头
        if not file_exists:
            writer.writeheader()
        # 过滤掉不在表头里的多余字段，防止报错
        safe_data = {k: v for k, v in data_dict.items() if k in csv_headers}
        writer.writerow(safe_data)

# ==========================================
# 💰 费率计算器
# ==========================================
class CostCalculator:
    @staticmethod
    def calculate_cost(market, shares, price):
        if market == 'hk': return CostCalculator._hk_cost(shares, price)
        return CostCalculator._us_cost(shares, price)

    @staticmethod
    def _us_cost(shares, price):
        # 美股: 平台费 + 佣金
        fee = max(1.0, 0.005 * shares) + (0.003 * shares) + 0.01
        return round(fee, 2)

    @staticmethod
    def _hk_cost(shares, price):
        # 港股: 印花税0.1% + 佣金等 (估算值)
        # 注意: 传入的 price 是港币
        val = shares * price
        stamp = math.ceil(val * 0.001)
        return round(stamp + 15 + 5, 2) # 估算 20 HKD 杂费

# ==========================================
# 📢 消息通知
# ==========================================
class NotificationManager:
    def send(self, title, content):
        logger.info(f"📨 [微信推送] {title}...")
        if not CONFIG["WECHAT_BOT_URL"]: return
        try:
            payload = {"title": title, "desp": f"【量化交易提醒】\n{datetime.now()}\n---\n{content}"}
            requests.post(CONFIG["WECHAT_BOT_URL"], json=payload, timeout=10)
        except Exception as e:
            logger.error(f"❌ 微信发送失败: {e}")

# ==========================================
# 💾 持仓管理
# ==========================================
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
        # 存入持仓
        self.holdings[code] = {
            "code": code, "market": market,
            "buy_price": float(price), "shares": float(shares), # 支持小数
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
    
    def get_market_count(self, market):
        """获取某市场的当前持仓数"""
        return len([h for h in self.holdings.values() if h.get('market') == market])

# ==========================================
# 🔌 长桥 API 执行器 (增强版)
# ==========================================
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
        """
        [新增] 获取港股每手股数
        例如: 腾讯(00700) -> 100, 比亚迪(01211) -> 500
        """
        if market == 'us': return 1 # 美股没有手限制
        if CONFIG["SIMULATION_MODE"]: return 100 # 模拟默认100
        
        try:
            symbol = self.get_symbol_suffix(code, market)
            # 调用静态信息接口
            info = self.ctx.static_info([symbol])
            if info and len(info) > 0:
                lot = int(info[0].board_lot)
                logger.info(f"ℹ️ {code} 每手股数: {lot}")
                return lot
        except Exception as e:
            logger.warning(f"⚠️ 获取每手股数失败 {code}: {e}, 默认100")
        return 100

    def estimate_max_buy(self, code, price, market):
        """查询券商允许的最大购买力"""
        symbol = self.get_symbol_suffix(code, market)
        if CONFIG["SIMULATION_MODE"]: return 99999
        if not self.ctx: return 0
        try:
            resp = self.ctx.estimate_max_purchase_quantity(
                symbol=symbol, order_type=OrderType.LO, side=OrderSide.Buy, price=str(price)
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
            # 提交订单
            # [修改] 计算限价单价格 (买入挂高1%，卖出挂低1%，确保成交但防飞单)
            # 注意小数位精度：美股2位，港股3位(简化处理都2-3位)
            limit_price = price * 1.01 if side == OrderSide.Buy else price * 0.99
            limit_price_str = f"{limit_price:.2f}" # 转字符串

            resp = self.ctx.submit_order(
                symbol=symbol, 
                order_type=OrderType.LO, # 改为限价单 (Limit Order)
                side=side,
                submitted_quantity=Decimal(str(shares)), 
                submitted_price=Decimal(limit_price_str), # 必须填价格
                time_in_force=TimeInForceType.Day 
            )

            logger.info(f"✅ 订单提交成功: {resp.order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False

    def get_cash_balance(self):
        """
        获取当前账户的可用现金余额。
        优先返回美元(USD)，如果没美元则尝试返回港币(HKD)，都没有则返回0。
        """
        # 1. 模拟模式直接返回假数据
        if CONFIG["SIMULATION_MODE"]:
            return 100000.0
            
        # 2. 实盘模式调用API
        if not self.ctx: return 0.0
        
        try:
            # 调用长桥 API: 获取资产总览
            resp = self.ctx.account_balance()
            
            # 遍历返回的账户信息
            for acc_balance in resp:
                # 遍历该账户下的多币种现金详情
                for cash_info in acc_balance.cash_infos:
                    # 优先获取美元可用现金 (available_cash)
                    if cash_info.currency == 'USD':
                        return float(cash_info.available_cash)
                    # 备选：如果是港股交易为主，也可以改逻辑优先取 HKD
                    # elif cash_info.currency == 'HKD':
                    #     return float(cash_info.available_cash)
            
            # 如果没找到 USD，尝试随便返回一个非零的可用现金，或者就返回 0
            if resp and len(resp) > 0 and resp[0].cash_infos:
                return float(resp[0].cash_infos[0].available_cash)
                
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 获取余额失败: {e}")
            return 0.0
# ==========================================
# 🧠 策略控制器
# ==========================================
class AutoTrader:
    def __init__(self):
        self.scanner = GlobalMarketScanner()
        self.analyzer = EnhancedWebStockAnalyzer()
        self.holdings = HoldingsManager(CONFIG["HOLDINGS_FILE"])
        self.executor = LongbridgeExecutor()
        self.notifier = NotificationManager()
        self._ai_cache = {} 

    def _calc_buy_size(self, code, price, market):
        """
        [核心逻辑升级] 针对不同市场计算买入数量
        """
        if price <= 0: return 0
        
        # 1. 读取对应市场的配置
        settings = CONFIG["HK_SETTINGS"] if market == 'hk' else CONFIG["US_SETTINGS"]
        
        # 2. 检查该市场的持仓上限
        curr_count = self.holdings.get_market_count(market)
        slots_left = settings["MAX_HOLDINGS"] - curr_count
        if slots_left <= 0: return 0
        
        # 3. 资金分配 (总配额 / 剩余坑位)
        # 注意: 这里的钱是"虚拟配额"，不是账户总现金。
        # 如果你想用账户真实剩余现金，需要调用 executor.get_cash_balance 并区分币种(较复杂)
        # 简单起见，我们假设你账户里钱够，按配额买。
        budget_per_stock = settings["ALLOCATED_CAPITAL"] / settings["MAX_HOLDINGS"]
        
        if budget_per_stock < settings["MIN_TRADE_AMT"]: return 0
        
        # 4. 计算股数 (区分市场)
        raw_shares = budget_per_stock / price
        
        if market == 'us':
            # 美股: 支持碎股，保留2位小数 (比如买 1.5 股 NVDA)
            final_shares = round(raw_shares, 2)
            if final_shares < 0.01: return 0
            
        elif market == 'hk':
            # 港股: 必须整手买
            lot_size = self.executor.get_lot_size(code, market)
            # 向下取整到整数手 (例如算出来 250 股，每手 100，只能买 200)
            final_shares = int(raw_shares // lot_size) * lot_size
            if final_shares < lot_size: return 0 # 连一手都买不起
            
        # 5. 券商风控检查 (防止保证金不够)
        max_allowed = self.executor.estimate_max_buy(code, price, market)
        
        # 如果是港股，max_allowed 也得向下取整到 lot_size
        if market == 'hk':
            lot_size = self.executor.get_lot_size(code, market) # 再次获取防止变量没传
            max_allowed = int(max_allowed // lot_size) * lot_size
            
        final_shares = min(final_shares, max_allowed)
        
        return final_shares

    def job_scan_market(self, market='us'):
        # 1. 检查开关
        settings = CONFIG["HK_SETTINGS"] if market == 'hk' else CONFIG["US_SETTINGS"]
        if not settings["ENABLED"]:
            return # 该市场已关闭

        logger.info(f"⏰ 开始扫描 {market.upper()} 市场...")
        
        # 2. 检查仓位
        if self.holdings.get_market_count(market) >= settings["MAX_HOLDINGS"]:
            logger.info(f"   🚫 {market.upper()} 仓位已满")
            return

        # 3. 扫描
        candidates = []
        if market == 'hk': candidates = self.scanner.get_hk_candidates(top_n=20)
        else: candidates = self.scanner.get_us_candidates(top_n=20)
        
        potential_buys = []
        
        def analyze(code):
            # 过滤已持仓
            if code in [h['code'] for h in self.holdings.get_all()]: return None
            try:
                # 简单缓存
                cache_key = f"{market}_{code}_{datetime.now().strftime('%Y%m%d')}"
                if cache_key in self._ai_cache: return self._ai_cache[cache_key]

                df = self.analyzer.get_stock_data(code, period='3mo')
                if df.empty or len(df) < 20: return None
                
                # 技术与AI分析
                tech = self.analyzer.calculate_technical_indicators(df)
                money = self.analyzer.analyze_smart_money_flow(df)
                price_info = {"close": df.iloc[-1]['close'], "change_pct": df.iloc[-1]['change_pct'], "vol_ratio": 1.0}
                
                # ==========================================
                # 👇 新增：在这里加日志，证明程序没死，只是在等 AI
                # ==========================================
                start_time = time.time()
                logger.info(f"🧠 [AI] 正在请求 DeepSeek 分析 {code} ...")
                
                # 调用 AI (这是最耗时的一步)
                ai_res = self.analyzer.get_llm_trade_decision(code, "today", price_info, tech, money)
                
                duration = time.time() - start_time
                logger.info(f"⚡ [AI] {code} 分析完成，耗时 {duration:.2f}秒")
                # ==========================================
                
                # 阈值设置: 港股要求更高
                threshold = 80 if market == 'hk' else 75
                
                if ai_res.get('action') == "BUY" and ai_res.get('confidence', 0) >= threshold:
                    return {
                        "code": code,
                        "price": df.iloc[-1]['close'],
                        "confidence": ai_res['confidence'],
                        "reason": ai_res.get('reason'),
                        "ai_raw": ai_res,
                        # 👇 新增：把技术指标打包带走
                        "tech_snapshot": {
                            "rsi": tech.get("rsi", 0),
                            "kdj_k": tech.get("kdj_k", 0),
                            "ma20_slope": tech.get("ma20_slope", 0),
                            "vol_ratio": price_info.get("vol_ratio", 0),
                            "atr": tech.get("atr", 0),
                            "money_flow": money.get("money_flow_score", 0)
                        },
                        "tech_score": tech_score
                    }
            except: return None

        # 并发执行
        for code in candidates:
            logger.info(f"🔎 正在分析 {code} ...")
            try:
                # 直接调用，不通过线程池
                res = analyze(code)
                if res:
                    potential_buys.append(res)
                    logger.info(f"✅ {code} 命中策略！")
            except Exception as e:
                logger.error(f"❌ {code} 出错: {e}")

        # 排序
        potential_buys.sort(key=lambda x: x['confidence'], reverse=True)
        
        buy_logs = []
        
        # 4. 执行买入
        for target in potential_buys:
            # 再次检查仓位 (防止循环中途满了)
            if self.holdings.get_market_count(market) >= settings["MAX_HOLDINGS"]: break
            
            code = target['code']
            price = target['price']
            
            # 计算股数 (区分市场逻辑)
            shares = self._calc_buy_size(code, price, market)
            
            if shares > 0:
                fee = CostCalculator.calculate_cost(market, shares, price)
                if self.executor.execute_order(code, OrderSide.Buy, price, shares, market):
                    self.holdings.add_holding(code, price, shares, fee, market)
                    
                    buy_logs.append(f"🚀 买入 {code}({market}): {shares}股 @ {price} | AI:{target['confidence']}")
                    
                    # 获取当前资产状态
                    curr_cash = self.executor.get_cash_balance()
                    # 估算总资产 (简易版)
                    total_asset = curr_cash + sum(h['shares']*h['buy_price'] for h in self.holdings.get_all())

                    log_trade_data({
                        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "action": "BUY",
                        "market": market,
                        "code": code,
                        "price": price,
                        "shares": shares,
                        "cost": fee, # 记录手续费
                        "profit_amt": 0, "profit_pct": 0, "hold_days": 0,

                        # AI 与 理由
                        "confidence": target['confidence'],
                        "market_phase": target['ai_raw'].get('market_phase', 'unknown'),
                        "reason": target['reason'],

                        # 👇 关键：写入快照指标
                        "rsi": round(target['tech_snapshot']['rsi'], 2),
                        "kdj_k": round(target['tech_snapshot']['kdj_k'], 2),
                        "ma20_slope": round(target['tech_snapshot']['ma20_slope'], 5),
                        "vol_ratio": round(target['tech_snapshot']['vol_ratio'], 2),
                        "atr": round(target['tech_snapshot']['atr'], 3),
                        "money_flow": round(target['tech_snapshot']['money_flow'], 1),

                        # 账户状态
                        "total_asset": round(total_asset, 2),
                        "cash_left": round(curr_cash, 2)
                    })
            else:
                logger.info(f"   ⚠️ {code} 资金不足或不足一手")

        if buy_logs:
            self.notifier.send(f"🔵 {market.upper()} 买入汇总", "\n".join(buy_logs))

    def job_monitor(self):
        """监控卖出 (通用) - 终极防卡死版"""
        logger.info("⏰ 监控持仓... (开始扫描)")
        
        # 1. 获取所有持仓
        all_holdings = list(self.holdings.get_all())
        if not all_holdings: 
            logger.info("✅ 当前无持仓，监控结束")
            return

        sell_logs = []

        # ==========================================
        # 🛡️ 辅助函数：带超时保护的数据获取
        # ==========================================
        def _safe_get_data(code):
            """专门用于安全获取数据，带超时控制"""
            try:
                # 这里的 timeout 是给 thread.result() 用的，不是给 get_stock_data 用的
                # 所以我们需要把 get_stock_data 放进线程池里跑
                return self.analyzer.get_stock_data(code, period='1mo')
            except Exception:
                return None

        # ==========================================
        # 1. 强制瘦身逻辑 (Force Reduce) - 已升级防卡死
        # ==========================================
        try:
            for market in ['us', 'hk']:
                market_holdings = [h for h in all_holdings if h.get('market', 'us') == market]
                settings = CONFIG["HK_SETTINGS"] if market == 'hk' else CONFIG["US_SETTINGS"]
                max_limit = settings["MAX_HOLDINGS"]
                excess_count = len(market_holdings) - max_limit
                
                if excess_count > 0:
                    logger.warning(f"⚠️ {market.upper()} 持仓超标 ({len(market_holdings)}/{max_limit})，计算盈亏排序中...")
                    
                    # 🎯 关键修改：使用线程池来获取排序用的数据，防止卡死
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        # 提交任务
                        future_to_pos = {executor.submit(_safe_get_data, pos['code']): pos for pos in market_holdings}
                        
                        for future in concurrent.futures.as_completed(future_to_pos):
                            pos = future_to_pos[future]
                            code = pos['code']
                            try:
                                # ⏳ 强制等待 10 秒，拿不到数据就当做 -999 处理，绝不卡死
                                df = future.result(timeout=10)
                                if df is not None and not df.empty:
                                    curr = df.iloc[-1]['close']
                                    # 兼容旧数据 cost_price
                                    cost = pos.get('cost_price', pos.get('buy_price', 0.01))
                                    pos['current_profit_pct'] = (curr - cost) / cost * 100
                                else:
                                    pos['current_profit_pct'] = -999
                            except concurrent.futures.TimeoutError:
                                logger.error(f"⏳ [瘦身检查] 获取 {code} 数据超时，跳过，标记为优先卖出")
                                pos['current_profit_pct'] = -999
                            except Exception as e:
                                pos['current_profit_pct'] = -999

                    # 排序并卖出
                    market_holdings.sort(key=lambda x: x['current_profit_pct'])
                    targets_to_remove = market_holdings[:excess_count]

                    for pos in targets_to_remove:
                        code = pos['code']
                        logger.info(f"📉 [强制减仓] 卖出: {code} (盈亏 {pos.get('current_profit_pct', 0):.1f}%)")
                        
                        # 获取卖出价格 (同样需要防卡死，这里简单处理，若卡住用买入价)
                        sell_price = pos['buy_price']
                        try:
                            # 尝试快速获取一下最新价
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                f = executor.submit(_safe_get_data, code)
                                df = f.result(timeout=5)
                                if df is not None and not df.empty:
                                    sell_price = df.iloc[-1]['close']
                        except: pass
                        
                        if self.executor.execute_order(code, OrderSide.Sell, sell_price, pos['shares'], market):
                            self.holdings.remove_holding(code)
                            logger.info(f"✅ [强制减仓] {code} 卖出指令已提交")

        except Exception as e:
            logger.error(f"❌ 强制瘦身逻辑出错: {e}")

        # ==========================================
        # 2. 常规监控逻辑 (Regular Monitor) - 保持防卡死
        # ==========================================
        # 重新获取持仓（因为刚才可能卖掉了一些）
        all_holdings = list(self.holdings.get_all())
        if not all_holdings: return

        # 定义单只股票检查逻辑
        def _check_single_pos(pos):
            code = pos['code']
            # 👇 关键：打印正在检查谁，卡死也能知道是谁
            logger.info(f"🔍 正在检查持仓: {code} ...")
            
            # 1. 获取数据 (如果卡，通常是卡在这里)
            df = self.analyzer.get_stock_data(code, period='1mo')
            if df.empty: return None

            curr_price = df.iloc[-1]['close']
            
            # 2. 兼容旧数据的字段 (防止 KeyError)
            cost = pos.get('cost_price', pos.get('buy_price', 0.01))
            highest = pos.get('highest_price', cost)
            
            # 更新最高价
            if curr_price > highest:
                highest = curr_price
                self.holdings.update_highest(code, curr_price)
            
            # 计算指标
            pct = (curr_price - cost) / cost * 100
            drawdown = (curr_price - highest) / highest * 100

            # 3. 卖出判断
            reason = ""
            should_sell = False
            if pct < -7: should_sell=True; reason="止损"
            elif pct > 15 and drawdown < -4: should_sell=True; reason="回撤止盈"

            if should_sell:
                return {
                    "action": "SELL", "pos": pos, 
                    "price": curr_price, "reason": reason, "profit_pct": pct
                }
            return None

        # 启动单线程池 (Timeout=20s)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_pos = {executor.submit(_check_single_pos, pos): pos for pos in all_holdings}
            
            for future in concurrent.futures.as_completed(future_to_pos):
                pos = future_to_pos[future]
                code = pos['code']
                try:
                    # ⏳ 超时保护：每只股票最多给20秒
                    result = future.result(timeout=20)
                    
                    if result and result['action'] == "SELL":
                        target_pos = result['pos']
                        market = target_pos.get('market', 'us')
                        if self.executor.execute_order(code, OrderSide.Sell, result['price'], target_pos['shares'], market):
                            self.holdings.remove_holding(code)
                            
                            # 记录日志
                            sell_logs.append(f"🔴 卖出 {code}: 盈亏 {result['profit_pct']:.1f}% ({result['reason']})")
                            
                            # 尝试记录CSV (如果获取余额失败也不崩)
                            try:
                                cash = self.executor.get_cash_balance()
                                log_trade_data({
                                    "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "action": "SELL", "market": market, "code": code,
                                    "price": result['price'], "shares": target_pos['shares'],
                                    "profit_amt": 0, "profit_pct": result['profit_pct'],
                                    "reason": result['reason'], "cash_left": cash
                                })
                            except Exception as log_err:
                                logger.error(f"⚠️ 交易记录写入失败(不影响交易): {log_err}")

                except concurrent.futures.TimeoutError:
                    logger.error(f"⏳ 监控 {code} 超时 (超过20s)，强制跳过！")
                except Exception as e:
                    logger.error(f"❌ 监控 {code} 出错: {e}")

        logger.info("✅ 监控持仓完成")
        if sell_logs:
            self.notifier.send("🔴 卖出汇总", "\n".join(sell_logs))

        

    def run(self):
        logger.info("⏳ 交易系统启动...")
        
        # 港股
        schedule.every().day.at("09:45").do(self.job_scan_market, market='hk')
        schedule.every().day.at("12:17").do(self.job_monitor)
        schedule.every().day.at("13:15").do(self.job_scan_market, market='hk')
        schedule.every().day.at("15:30").do(self.job_monitor)
        
        # 美股
        schedule.every().day.at("22:35").do(self.job_scan_market, market='us')
        schedule.every().day.at("02:00").do(self.job_scan_market, market='us')
        schedule.every().day.at("04:50").do(self.job_monitor)
        
        while True:
            schedule.run_pending()
            time.sleep(30)

if __name__ == "__main__":
    trader = AutoTrader()
    


    # # 恢复正常的调度运行
    trader.run()
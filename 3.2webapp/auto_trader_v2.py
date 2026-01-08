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

# ==========================================
# ⚙️ 全局配置区 (分市场配置)
# ==========================================
CONFIG = {
    "SIMULATION_MODE": True,  # ⚠️ 调试为True，实盘为False
    
    "LB_APP_KEY": "请填入你的AppKey",
    "LB_APP_SECRET": "请填入你的AppSecret",
    "LB_ACCESS_TOKEN": "请填入你的AccessToken",
    
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
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
            resp = self.ctx.submit_order(
                symbol=symbol, order_type=OrderType.Market, side=side,
                submitted_quantity=Decimal(str(shares)), time_in_force=TimeInForceType.Day
            )
            logger.info(f"✅ 订单提交成功: {resp.order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False

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
                
                ai_res = self.analyzer.get_llm_trade_decision(code, "today", price_info, tech, money)
                
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze, c) for c in candidates]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res: potential_buys.append(res)

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
        """监控卖出 (通用)"""
        logger.info("⏰ 监控持仓...")
        all_holdings = list(self.holdings.get_all())
        if not all_holdings: return

        sell_logs = []
        for pos in all_holdings:
            code = pos['code']
            market = pos.get('market', 'us')
            
            try:
                df = self.analyzer.get_stock_data(code, period='1mo')
                if df.empty: continue
                curr_price = df.iloc[-1]['close']
                self.holdings.update_highest(code, curr_price)
                
                cost = pos['cost_price']
                pct = (curr_price - cost) / cost * 100
                drawdown = (curr_price - pos['highest_price']) / pos['highest_price'] * 100
                
                # 卖出逻辑
                sell = False
                reason = ""
                if pct < -7: sell=True; reason="止损"
                elif pct > 15 and drawdown < -4: sell=True; reason="回撤止盈"
                
                if sell:
                    if self.executor.execute_order(code, OrderSide.Sell, curr_price, pos['shares'], market):
                        self.holdings.remove_holding(code)
                        sell_logs.append(f"🔴 卖出 {code}: 盈亏 {pct:.1f}% ({reason})")
                        
                        # 重新获取资产
                        curr_cash = self.executor.get_cash_balance()
                        total_asset = curr_cash # 此时现金已增加，粗略计算即可

                        log_trade_data({
                            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "action": "SELL",
                            "market": market,
                            "code": code,
                            "price": curr_price,
                            "shares": pos['shares'],
                            "cost": fee, # 卖出手续费
                            
                            # 👇 核心盈亏数据
                            "profit_amt": round(realized_pnl, 2),
                            "profit_pct": round(profit_pct, 2),
                            "hold_days": days_held,
                            
                            "reason": reason, # 卖出原因 (止损/止盈)
                            
                            # 卖出时技术指标可以留空，或者也记录当时的 RSI 看看是否超买
                            "total_asset": round(total_asset, 2),
                            "cash_left": round(curr_cash, 2)
                        })
            except: pass
            
        if sell_logs:
            self.notifier.send("🔴 卖出汇总", "\n".join(sell_logs))

    def run(self):
        logger.info("⏳ 交易系统启动...")
        
        # 港股
        schedule.every().day.at("09:45").do(self.job_scan_market, market='hk')
        schedule.every().day.at("11:30").do(self.job_monitor)
        schedule.every().day.at("13:15").do(self.job_scan_market, market='hk')
        schedule.every().day.at("15:50").do(self.job_monitor)
        
        # 美股
        schedule.every().day.at("22:35").do(self.job_scan_market, market='us')
        schedule.every().day.at("02:00").do(self.job_scan_market, market='us')
        schedule.every().day.at("04:50").do(self.job_monitor)
        
        while True:
            schedule.run_pending()
            time.sleep(30)

if __name__ == "__main__":
    AutoTrader().run()
# auto_trader_pro.py
import sys
import os
import json
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
from decimal import Decimal

# 引入之前的核心组件 (确保这两个文件在同一目录下)
from global_scanner import GlobalMarketScanner
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer

# 尝试导入长桥SDK，如果没有安装则优雅降级
try:
    from longport.openapi import TradeContext, Config, OrderSide, OrderType, TimeInForce
    LONGBRIDGE_INSTALLED = True
except ImportError:
    LONGBRIDGE_INSTALLED = False
    print("⚠️ 未检测到 longport 库，实盘交易功能将无法执行。请运行: pip install longport")

# ==========================================
# ⚙️ 全局配置区 (USER CONFIG)
# ==========================================
CONFIG = {
    # --- 交易开关 ---
    "SIMULATION_MODE": True,  # 【重要】True=只打印不交易; False=真金白银自动买卖
    
    # --- 长桥 API 配置 (请去长桥开发者中心申请) ---
    "LB_APP_KEY": "f1bcf09101f6",
    "LB_APP_SECRET": "5e62e415",
    
    # --- 资金管理 ---
    "MAX_POSITION_PER_STOCK": 1000,  # 单只股票最大买入金额 (例如2万元)
    
    # --- 文件路径 ---
    "HOLDINGS_FILE": "holdings.json"
}

# ==========================================
# 💾 持仓管理器 (JSON持久化)
# ==========================================
class HoldingsManager:
    """负责将持仓数据保存到硬盘，防止程序重启丢失 '最高价' 等关键信息"""
    def __init__(self, filename):
        self.filename = filename
        self.holdings = self._load()

    def _load(self):
        if not os.path.exists(self.filename):
            return {}
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 读取持仓文件失败: {e}")
            return {}

    def save(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.holdings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存持仓文件失败: {e}")

    def add_holding(self, stock_code, price, shares, market):
        """记录新买入"""
        self.holdings[stock_code] = {
            "code": stock_code,
            "market": market,
            "buy_price": float(price),
            "shares": int(shares),
            "highest_price": float(price), # 初始最高价 = 买入价
            "buy_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "hold_days": 0
        }
        self.save()

    def remove_holding(self, stock_code):
        """卖出后删除"""
        if stock_code in self.holdings:
            del self.holdings[stock_code]
            self.save()

    def update_highest_price(self, stock_code, current_price):
        """更新移动止盈的锚点"""
        if stock_code in self.holdings:
            if current_price > self.holdings[stock_code]['highest_price']:
                old = self.holdings[stock_code]['highest_price']
                self.holdings[stock_code]['highest_price'] = float(current_price)
                print(f"📈 [{stock_code}] 创新高! {old} -> {current_price} (已存档)")
                self.save()

    def get_all(self):
        return self.holdings.values()

# ==========================================
# 🔌 长桥 API 执行器
# ==========================================
class LongbridgeExecutor:
    def __init__(self):
        self.ctx = None
        if LONGBRIDGE_INSTALLED and not CONFIG["SIMULATION_MODE"]:
            try:
                conf = Config(
                    app_key=CONFIG["LB_APP_KEY"],
                    app_secret=CONFIG["LB_APP_SECRET"],
                    access_token=CONFIG["LB_ACCESS_TOKEN"]
                )
                self.ctx = TradeContext(conf)
                print("🔌 长桥 API 连接成功")
            except Exception as e:
                print(f"❌ 长桥 API 连接失败: {e}")

    def _format_symbol(self, stock_code, market):
        """将内部代码转换为长桥代码格式 (如 700 -> 700.HK)"""
        if market == 'hk_stock':
            # 移除前导0，长桥港股通常是 700.HK
            return f"{int(stock_code)}.HK"
        elif market == 'us_stock':
            return f"{stock_code}.US"
        elif market == 'a_stock':
            # A股需要判断深沪，这里简单处理，实际需更严谨
            if stock_code.startswith('6'): return f"{stock_code}.SH"
            return f"{stock_code}.SZ"
        return stock_code

    def execute_buy(self, stock_code, market, price, shares):
        """执行买入"""
        symbol = self._format_symbol(stock_code, market)
        msg = f"🔵 [API买入] {symbol} | 价格:{price} | 股数:{shares}"
        
        if CONFIG["SIMULATION_MODE"]:
            print(f"🛠️ (模拟模式) {msg}")
            return True # 模拟买入成功
        
        if not self.ctx: return False

        try:
            # 市价单买入
            resp = self.ctx.submit_order(
                symbol=symbol,
                order_type=OrderType.Market, # 市价单确保成交
                side=OrderSide.Buy,
                submitted_quantity=shares,
                time_in_force=TimeInForce.Day
            )
            print(f"✅ {msg} | 订单ID: {resp.order_id}")
            return True
        except Exception as e:
            print(f"❌ 买入失败: {e}")
            return False

    def execute_sell(self, stock_code, market, price, shares):
        """执行卖出"""
        symbol = self._format_symbol(stock_code, market)
        msg = f"🔴 [API卖出] {symbol} | 价格:{price} | 股数:{shares}"
        
        if CONFIG["SIMULATION_MODE"]:
            print(f"🛠️ (模拟模式) {msg}")
            return True
        
        if not self.ctx: return False

        try:
            resp = self.ctx.submit_order(
                symbol=symbol,
                order_type=OrderType.Market,
                side=OrderSide.Sell,
                submitted_quantity=shares,
                time_in_force=TimeInForce.Day
            )
            print(f"✅ {msg} | 订单ID: {resp.order_id}")
            return True
        except Exception as e:
            print(f"❌ 卖出失败: {e}")
            return False

# ==========================================
# 🧠 核心策略控制器
# ==========================================
class AutoTrader:
    def __init__(self):
        self.scanner = GlobalMarketScanner()
        self.analyzer = EnhancedWebStockAnalyzer()
        self.holdings_manager = HoldingsManager(CONFIG["HOLDINGS_FILE"])
        self.executor = LongbridgeExecutor()
        
        print("\n" + "="*50)
        print(f"🚀 自动交易机器人已启动")
        print(f"📂 持仓文件: {CONFIG['HOLDINGS_FILE']}")
        print(f"🛡️ 模式: {'🛠️ 模拟 (只看)' if CONFIG['SIMULATION_MODE'] else '💸 实盘 (自动下单)'}")
        print("="*50 + "\n")

    def _calculate_shares(self, price, market):
        """计算买入股数 (向下取整到手数)"""
        target_amount = CONFIG["MAX_POSITION_PER_STOCK"]
        if price <= 0: return 0
        
        raw_shares = target_amount / price
        
        if market == 'a_stock' or market == 'hk_stock':
            # 港股A股通常一手100股
            shares = int(raw_shares // 100) * 100
            return max(100, shares) # 至少买1手
        else:
            # 美股可以买1股
            return max(1, int(raw_shares))

    def _get_market_from_code(self, stock_code):
        if stock_code.isdigit():
            if len(stock_code) == 5: return 'hk_stock'
            if len(stock_code) == 6: return 'a_stock'
        return 'us_stock'

    # ----------------------------------------------------------------
    # 逻辑核心 1: 监控与卖出 (Monitor & Sell)
    # ----------------------------------------------------------------
    def check_holdings_logic(self):
        holdings = list(self.holdings_manager.get_all())
        if not holdings:
            print("📭 当前无持仓，无需监控。")
            return

        print(f"💼 正在监控 {len(holdings)} 个持仓...")
        
        for pos in holdings:
            code = pos['code']
            market = pos['market']
            buy_price = pos['buy_price']
            highest = pos['highest_price']
            shares = pos['shares']
            
            # 1. 获取最新行情
            try:
                df = self.analyzer.get_stock_data(code, period='1y')
                if df.empty: continue
                curr_close = df.iloc[-1]['close']
                
                # 2. 更新最高价 (移动止盈的关键)
                self.holdings_manager.update_highest_price(code, curr_close)
                highest = max(highest, curr_close) # 确保变量也是最新的

                # 3. 计算指标
                profit_pct = (curr_close - buy_price) / buy_price * 100
                drawdown = (curr_close - highest) / highest * 100
                
                # 计算持有天数 (粗略计算)
                buy_date = datetime.strptime(pos['buy_date'], '%Y-%m-%d %H:%M:%S')
                hold_days = (datetime.now() - buy_date).days

                # 4. 判断卖出信号
                sell_signal = False
                reason = ""

                # A. 硬止损 (-5%)
                if profit_pct < -5.0:
                    sell_signal = True; reason = f"硬止损 (亏损{profit_pct:.1f}%)"
                
                # B. 移动止盈 (赚过10%且回撤3%)
                elif highest > buy_price * 1.10 and drawdown < -3.0:
                    sell_signal = True; reason = f"移动止盈 (高点回撤{drawdown:.1f}%)"
                
                # C. 时间止损 (短期不动)
                elif market == 'us_stock' and hold_days > 3 and profit_pct < 1:
                    sell_signal = True; reason = "美股动能耗尽"

                # 5. 执行操作
                print(f"   🔎 {code}: 现价{curr_close} | 盈亏{profit_pct:.1f}% | 回撤{drawdown:.1f}%", end="")
                
                if sell_signal:
                    print(f" -> 🚨 触发卖出: {reason}")
                    # 调用 API
                    success = self.executor.execute_sell(code, market, curr_close, shares)
                    if success:
                        self.holdings_manager.remove_holding(code)
                        print(f"   🗑️ 已从持仓列表中移除")
                else:
                    print(" -> ✅ 持有")
                
                time.sleep(1) # 防封

            except Exception as e:
                print(f"   ❌ 监控异常 {code}: {e}")

    # ----------------------------------------------------------------
    # 逻辑核心 2: 扫描与买入 (Scan & Buy)
    # ----------------------------------------------------------------
    def run_scan_logic(self, market='us_stock', top_n=20):
        print(f"\n🌍 开始扫描买入机会: {market} (Top {top_n})")
        
        # 1. 获取名单
        candidates = []
        try:
            if market == 'us_stock': candidates = self.scanner.get_us_candidates(top_n)
            elif market == 'hk_stock': candidates = self.scanner.get_hk_candidates(top_n)
            elif market == 'a_stock': candidates = self.scanner.get_a_candidates(top_n)
        except:
            print("❌ 扫描器连接失败")
            return

        # 2. 过滤已持仓
        current_holdings = [h['code'] for h in self.holdings_manager.get_all()]
        candidates = [c for c in candidates if c not in current_holdings]

        # 3. 逐个分析
        for code in candidates:
            try:
                # 获取数据
                df = self.analyzer.get_stock_data(code, period='1y')
                if df.empty or len(df) < 60: continue
                
                # 初筛 (快速规则)
                curr_close = df.iloc[-1]['close']
                ma20 = df['close'].rolling(20).mean().iloc[-1]
                
                # 简单的趋势过滤，节省 AI Token
                if curr_close < ma20: continue 

                # AI 深度分析
                print(f"🤖 分析 {code} ... ", end="")
                tech = self.analyzer.calculate_technical_indicators(df)
                money = self.analyzer.analyze_smart_money_flow(df)
                price_info = {"close": curr_close, "change_pct": df.iloc[-1]['change_pct'], "vol_ratio": 1.0}
                
                ai_res = self.analyzer.get_llm_trade_decision(
                    code, datetime.now().strftime('%Y-%m-%d'), price_info, tech, money
                )
                
                action = ai_res.get('action', 'HOLD')
                print(f"{action} ({ai_res.get('confidence',0)}%)")

                # 如果 AI 强烈建议买入
                if action == "BUY" and ai_res.get('confidence', 0) >= 75:
                    # 再次检查风控 (最后一道防线)
                    if tech.get('rsi', 50) > 80:
                        print("   🛑 RSI过高，放弃追高")
                        continue

                    print(f"   🚀 正在执行买入程序...")
                    shares = self._calculate_shares(curr_close, market)
                    
                    # 调用 API
                    success = self.executor.execute_buy(code, market, curr_close, shares)
                    
                    if success:
                        self.holdings_manager.add_holding(code, curr_close, shares, market)
                        print(f"   📝 已写入持仓记录")
                
                time.sleep(2) # 遵守 API 频率限制

            except Exception as e:
                print(f"Err {code}: {e}")

# ==========================================
# ▶️ 运行入口
# ==========================================
if __name__ == "__main__":
    bot = AutoTrader()
    
    # --- 步骤 1: 监控现有持仓 (先卖出止损，释放资金) ---
    # bot.check_holdings_logic()
    
    # --- 步骤 2: 扫描新机会 (再买入) ---
    # 根据当前时间自动判断跑哪个市场，或者手动指定
    bot.run_scan_logic(market='hk_stock', top_n=30) 
    # bot.run_scan_logic(market='us_stock', top_n=20)

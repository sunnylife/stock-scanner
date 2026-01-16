# auto_backtest_runner_v3.py
import sys
import os
import pandas as pd
import time
import threading
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from strategy_core import StrategyEngine # 导入策略核心
from strategy_backtest import TimeTravelAnalyzer

# 配置
BACKTEST_CONFIG = {
    "INITIAL_CAPITAL": 100000,
    "MAX_HOLDINGS": 3,
    "SWAP_THRESHOLD": 85,
    "BACKTEST_DAYS": 30,
    "MARKET": "us_stock", # ✅ 可以在这里改测美股"us_stock""hk_stock""a_stock"
    
    # ==========================================
    # 🕒 时间设置（三种模式任选其一）
    # ==========================================
    # 【默认模式】使用天数（从最新数据往前推）
    # "BACKTEST_DAYS": 30,  # 回测最近30天
    
    # 【精确模式】指定具体的开始和结束日期（优先级高于 BACKTEST_DAYS）
    "START_DATE": None,  # 例如: "2024-01-01" 或 None
    "END_DATE": None,    # 例如: "2024-12-31" 或 None（None表示使用数据的最新日期）
    
    # ==========================================
    # 📋 使用示例
    # ==========================================
    # 示例1: 回测最近60天
    # "BACKTEST_DAYS": 60,
    # "START_DATE": None,
    # "END_DATE": None,
    
    # 示例2: 回测2024年全年
    # "START_DATE": "2024-01-01",
    # "END_DATE": "2024-12-31",
    
    # 示例3: 回测从2024年1月到现在
    # "START_DATE": "2024-01-01",
    # "END_DATE": None,  # None会使用数据的最新日期
    
    # 示例4: 回测2024年最后60天
    # "BACKTEST_DAYS": 60,
    # "END_DATE": "2024-12-31",
    # "START_DATE": None,  # 会自动计算为 END_DATE - 60天
    
    # ==========================================
    # 💡 逻辑说明
    # ==========================================
    # - 如果 START_DATE 和 END_DATE 都设置了，则使用指定区间
    # - 如果只设置 START_DATE，则从该日期到数据最新日期
    # - 如果只设置 END_DATE，则从（END_DATE - BACKTEST_DAYS天）到 END_DATE
    # - 如果都不设置，则使用 BACKTEST_DAYS（从数据最新日期往前推）
}

# ==========================================
# 👇 核心新增：双向日志记录器 (保持你原有的逻辑)
# ==========================================
class DualLogger:
    """
    将控制台输出同时重定向到文件和屏幕
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        # 写到屏幕
        self.terminal.write(message)
        # 写到文件
        self.log.write(message)  
        self.log.flush() # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 设置日志目录
LOG_DIR = "backtest_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 生成带时间戳的日志文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(LOG_DIR, f"run_{timestamp}.log")

# 🚨 立即接管系统输出
sys.stdout = DualLogger(log_filename)
sys.stderr = sys.stdout

print(f"📝 本次运行日志将存储于: {log_filename}")

# ==========================================
# 🛠️ 并行工具准备
# ==========================================
print_lock = threading.Lock()  # 打印锁,防止多线程输出乱码

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AutoBacktest")

# ==========================================
# 📝 回测专用 Logger 创建函数
# ==========================================
def create_backtest_loggers():
    """
    为每次回测创建独立的 logger 实例
    日志路径: backtest_logs/[timestamp]_*.log
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "backtest_logs"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    def make_logger(name, file):
        l = logging.getLogger(f"{name}_{timestamp}")  # 加时间戳避免冲突
        l.setLevel(logging.INFO)
        l.handlers.clear()  # 清空已有的 handler
        h = logging.FileHandler(file, encoding='utf-8')
        h.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        l.addHandler(h)
        return l
    
    return {
        'ai': make_logger("BacktestAI", f"{log_dir}/{timestamp}_ai.log"),
        'trade': make_logger("BacktestTrade", f"{log_dir}/{timestamp}_trade.log"),
        'pnl': make_logger("BacktestPnL", f"{log_dir}/{timestamp}_pnl.log")
    }


# ==========================================
# 📊 组合回测核心类
# ==========================================
class PortfolioBacktester:
    def __init__(self, stock_list, loggers):
        # ✅ 注入回测专用的 logger
        self.strategy = StrategyEngine(
            ai_logger=loggers['ai'],
            trade_logger=loggers['trade'],
            pnl_logger=loggers['pnl']
        )
        self.stock_list = stock_list
        self.cash = BACKTEST_CONFIG["INITIAL_CAPITAL"]
        self.holdings = {}  # {code: {shares, cost_price, buy_date, last_price, last_score}}
        self.data_cache = {}  # 数据缓存
        self.market = BACKTEST_CONFIG["MARKET"]
        
        # 保存 logger 引用，方便后续记录
        self.trade_logger = loggers['trade']
        self.pnl_logger = loggers['pnl']
        
        self._preload_data()

    def _preload_data(self):
        """预加载所有股票数据"""
        safe_print("📥 预加载数据中...")
        analyzer = TimeTravelAnalyzer()
        
        for code in self.stock_list:
            try:
                # 重置时间穿越状态，下载全量数据
                analyzer.set_simulation_date(None)
                df = analyzer.get_stock_data(code)
                
                if not df.empty:
                    # 验证核心列
                    required_cols = ['open', 'close', 'high', 'low', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        if 'change_pct' not in df.columns and 'close' in df.columns:
                            df['change_pct'] = df['close'].pct_change() * 100
                    
                    self.data_cache[code] = df
                    safe_print(f"✅ {code} 数据加载成功 ({len(df)} 条)")
                else:
                    safe_print(f"⚠️ {code} 数据为空")
                    
            except Exception as e:
                safe_print(f"❌ {code} 加载失败: {e}")
                
            time.sleep(0.3)  # 防止API限流
        
        safe_print(f"📥 数据加载完成，共 {len(self.data_cache)} 只股票\n")

    def _get_data_slice(self, code, target_date):
        """获取指定日期之前的数据切片"""
        if code not in self.data_cache:
            return pd.DataFrame()
        
        full_df = self.data_cache[code]
        target_dt = pd.to_datetime(target_date)
        
        # 只保留目标日期及之前的数据
        mask = full_df.index <= target_dt
        return full_df.loc[mask].copy()
    def run(self):
        """运行组合回测"""
        safe_print(f"\n🚀 开始回测 {self.market.upper()}...")
        safe_print("="*60)
        
        # ==========================================
        # 🕒 确定回测时间范围（支持三种模式）
        # ==========================================
        all_dates = set()
        for df in self.data_cache.values():
            all_dates.update(df.index)
        
        if not all_dates:
            safe_print("❌ 无可用数据")
            return
        
        sorted_dates = sorted(all_dates)
        
        # 获取配置的时间参数
        start_date_str = BACKTEST_CONFIG.get("START_DATE")
        end_date_str = BACKTEST_CONFIG.get("END_DATE")
        backtest_days = BACKTEST_CONFIG.get("BACKTEST_DAYS", 30)
        
        # 解析结束日期
        if end_date_str:
            end_date = pd.to_datetime(end_date_str)
        else:
            end_date = sorted_dates[-1]  # 使用数据中的最新日期
        
        # 解析开始日期
        if start_date_str:
            start_date = pd.to_datetime(start_date_str)
        else:
            # 如果没有指定开始日期，则从结束日期往前推 backtest_days 天
            start_date = end_date - timedelta(days=backtest_days)
        
        # 筛选出在指定区间内的日期
        sim_dates = [d for d in sorted_dates if start_date <= d <= end_date]
        
        if not sim_dates:
            safe_print(f"❌ 指定区间 [{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}] 内无数据")
            return
        
        if len(sim_dates) < 10:
            safe_print(f"⚠️ 数据较少（仅 {len(sim_dates)} 天），回测结果可能不准确")
        
        safe_print(f"📅 回测区间: {sim_dates[0].strftime('%Y-%m-%d')} 至 {sim_dates[-1].strftime('%Y-%m-%d')} (共 {len(sim_dates)} 个交易日)")
        safe_print(f"💰 初始资金: {self.cash:,.0f}")
        safe_print("="*60 + "\n")
        
        # 日志记录 (使用注入的 logger)
        self.pnl_logger.info(f"\n{'='*60}")
        self.pnl_logger.info(f"新回测开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.pnl_logger.info(f"市场: {self.market} | 初始资金: {self.cash}")
        self.pnl_logger.info(f"{'='*60}")
        
        # 每日循环
        for current_date in sim_dates:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # 1. 监控持仓
            for code in list(self.holdings.keys()):
                df_slice = self._get_data_slice(code, current_date)
                
                if df_slice.empty or len(df_slice) < 30:
                    continue
                
                curr_row = df_slice.iloc[-1]
                curr_price = curr_row['close']
                self.holdings[code]['last_price'] = curr_price
                
                # 计算技术指标
                try:
                    tech = self.strategy.analyzer.calculate_technical_indicators(df_slice)
                except:
                    continue
                
                # 调用策略层评分
                score, reason = self.strategy.calculate_holding_score(
                    self.holdings[code], curr_price, current_date, tech
                )
                
                self.holdings[code]['last_score'] = score
                
                # 记录持仓状态
                profit_pct = (curr_price - self.holdings[code]['cost_price']) / self.holdings[code]['cost_price'] * 100
                safe_print(f"📌 [{date_str}] {code} | 价格:{curr_price:.2f} | 盈亏:{profit_pct:+.1f}% | 评分:{score:.1f} | {reason}")
            
            # 2. 扫描新机会 (只在还有仓位时)
            if len(self.holdings) < BACKTEST_CONFIG["MAX_HOLDINGS"]:
                candidates = []
                
                for code in self.stock_list:
                    if code in self.holdings:
                        continue
                    
                    df_slice = self._get_data_slice(code, current_date)
                    
                    if df_slice.empty or len(df_slice) < 30:
                        continue
                    
                    # 调用策略层分析 (注意：这里关闭新闻获取以提速)
                    # # 临时 mock 新闻功能
                    # original_fetch = self.strategy.fetch_market_news
                    # self.strategy.fetch_market_news = lambda x: "回测模式:新闻已禁用"
                    # ✅ 正确写法 (Mock analyzer 里的搜索方法)
                    original_search = self.strategy.analyzer.search_market_news
                    self.strategy.analyzer.search_market_news = lambda x: "回测模式:新闻已禁用"
                    
                    try:
                        res = self.strategy.analyze_ticker(code, date_str, df_slice)
                        if res and res['action'] == 'BUY':
                            candidates.append(res)
                    except Exception as e:
                        safe_print(f"⚠️ {code} 分析失败: {e}")
                    finally:
                        self.strategy.analyzer.search_market_news = original_search
                
                # 按信心度排序
                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                
                # 3. 换仓博弈逻辑
                for candidate in candidates:
                    if len(self.holdings) >= BACKTEST_CONFIG["MAX_HOLDINGS"]:
                        # 满仓，检查是否换仓
                        weakest_code = min(self.holdings.keys(), key=lambda c: self.holdings[c]['last_score'])
                        weakest_score = self.holdings[weakest_code]['last_score']
                        
                        # 新股票信心度需要超过阈值才能换
                        if candidate['confidence'] > BACKTEST_CONFIG["SWAP_THRESHOLD"] and weakest_score < 60:
                            # 执行换仓
                            old_pos = self.holdings[weakest_code]
                            sell_price = old_pos['last_price']
                            sell_value = old_pos['shares'] * sell_price
                            profit_pct = (sell_price - old_pos['cost_price']) / old_pos['cost_price'] * 100
                            
                            self.cash += sell_value
                            
                            safe_print(f"🔄 [{date_str}] 换仓: 卖出 {weakest_code} (评分:{weakest_score:.1f}, 盈亏:{profit_pct:+.1f}%)")
                            self.trade_logger.info(f"[{date_str}] 卖出 {weakest_code} | 价格:{sell_price:.2f} | 盈亏:{profit_pct:+.1f}% | 原因:换仓")
                            
                            del self.holdings[weakest_code]
                    
                    # 买入新股票
                    if len(self.holdings) < BACKTEST_CONFIG["MAX_HOLDINGS"]:
                        buy_price = candidate['price']
                        position_size = self.cash / (BACKTEST_CONFIG["MAX_HOLDINGS"] - len(self.holdings))
                        shares = int(position_size / buy_price)
                        
                        if shares > 0:
                            cost = shares * buy_price
                            self.cash -= cost
                            
                            self.holdings[candidate['code']] = {
                                'shares': shares,
                                'cost_price': buy_price,
                                'buy_date': date_str,
                                'last_price': buy_price,
                                'last_score': 100
                            }
                            
                            safe_print(f"✅ [{date_str}] 买入 {candidate['code']} | 价格:{buy_price:.2f} | 数量:{shares} | 信心:{candidate['confidence']}")
                            self.trade_logger.info(f"[{date_str}] 买入 {candidate['code']} | 价格:{buy_price:.2f} | 数量:{shares} | 理由:{candidate['reason'][:30]}")
            
            # 4. 记录每日资产 (使用注入的 logger)
            total_asset = self.cash + sum([h['shares'] * h['last_price'] for h in self.holdings.values()])
            self.pnl_logger.info(f"{date_str} | 现金:{self.cash:,.0f} | 持仓市值:{total_asset - self.cash:,.0f} | 总资产:{total_asset:,.0f}")
        
        # 回测结束，输出报告
        self._print_report()


    def _print_report(self):
        """输出回测报告"""
        safe_print("\n" + "="*60)
        safe_print("📊 组合回测报告")
        safe_print("="*60)
        
        # 计算总资产
        total_market_value = sum([h['shares'] * h['last_price'] for h in self.holdings.values()])
        total_asset = self.cash + total_market_value
        total_return = (total_asset - BACKTEST_CONFIG["INITIAL_CAPITAL"]) / BACKTEST_CONFIG["INITIAL_CAPITAL"] * 100
        
        safe_print(f"💰 初始资金: {BACKTEST_CONFIG['INITIAL_CAPITAL']:,.0f}")
        safe_print(f"💵 剩余现金: {self.cash:,.0f}")
        safe_print(f"📈 持仓市值: {total_market_value:,.0f}")
        safe_print(f"🎯 总资产: {total_asset:,.0f}")
        safe_print(f"📊 总收益率: {total_return:+.2f}%")
        safe_print("\n当前持仓:")
        
        if self.holdings:
            for code, pos in self.holdings.items():
                profit_pct = (pos['last_price'] - pos['cost_price']) / pos['cost_price'] * 100
                safe_print(f"  {code} | 成本:{pos['cost_price']:.2f} | 现价:{pos['last_price']:.2f} | 盈亏:{profit_pct:+.1f}% | 数量:{pos['shares']}")
        else:
            safe_print("  (空仓)")
        
        safe_print("="*60 + "\n")
        
        # 记录到日志 (使用注入的 logger)
        self.pnl_logger.info(f"\n{'='*60}")
        self.pnl_logger.info(f"回测结束 | 总收益率: {total_return:+.2f}%")
        self.pnl_logger.info(f"{'='*60}\n")

def main():
    safe_print("\n" + "="*60)
    safe_print("🚀 组合回测系统 V3 (基于 StrategyEngine)")
    safe_print("="*60)
    
    try:
        # ✅ 1. 先创建回测专用的 logger
        loggers = create_backtest_loggers()
        safe_print(f"📝 回测日志已准备就绪\n")
        
        # 2. 使用 StrategyEngine 的扫描器获取候选股票池
        # 注意：这里创建一个临时的 engine 仅用于获取候选池
        # 真正的回测引擎会在 PortfolioBacktester 中创建
        temp_engine = StrategyEngine()
        market = BACKTEST_CONFIG["MARKET"]
        
        safe_print(f"\n🔍 正在扫描 {market.upper()} 市场...")
        target_stocks = temp_engine.get_candidates(market, limit=20)
        
        if not target_stocks:
            safe_print("❌ 未扫描到有效股票")
            return
        
        safe_print(f"✅ 扫描完成，共 {len(target_stocks)} 只候选股票")
        safe_print(f"📋 候选池: {target_stocks}\n")
        
        # 3. 创建回测器并运行 (传入 loggers)
        runner = PortfolioBacktester(target_stocks, loggers)
        runner.run()
        
    except KeyboardInterrupt:
        safe_print("\n⚠️ 用户中断")
    except Exception as e:
        safe_print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
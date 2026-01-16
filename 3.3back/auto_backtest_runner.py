# auto_backtest_runner.py
import logging
import sys
import os
import numpy as np
from datetime import datetime
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
print_lock = threading.Lock()  # 打印锁，防止多线程输出乱码

download_lock = threading.Lock()
def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)

print("DEBUG: 正在加载模块...")

try:
    from global_scanner import GlobalMarketScanner
    from strategy_backtest import TimeTravelAnalyzer
    from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
except ImportError as e:
    safe_print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AutoBacktest")


# ==========================================
# 🧵 核心逻辑封装：单只股票回测任务 (提取出来以便并行)
# ==========================================
def process_single_stock_task(stock_code, backtest_days):
    """
    单个股票的处理逻辑，将在独立线程中运行
    """
    # ⚠️ 关键：每个线程必须创建自己独立的 Analyzer 实例！
    # 因为 TimeTravelAnalyzer 内部有 simulation_date 状态，共享会冲突
    local_analyzer = TimeTravelAnalyzer()
    
    # 结果统计容器（返回给主线程汇总）
    result = {
        'market': 'UNKNOWN',
        'signals': 0,      # 总触发信号数
        'ai_approved': 0,  # AI批准数
        'winning': 0,      # 盈利数
        'total_return': 0.0,
        'logs': []         # 暂存日志（可选，这里直接用 safe_print）
    }

    # 为了模拟人类操作，稍微随机休眠一点点（并行模式下可以缩短时间）
    # time.sleep(random.uniform(0.5, 1.5)) 

    safe_print(f"\n🔍 线程启动检查: {stock_code}", end="")

    try:
        # 1. 识别股票所属市场
        normalized_code, market = local_analyzer.normalize_stock_code(stock_code)
        result['market'] = market
        
        # 2. 获取数据 (先重置时间下载全量)
        local_analyzer.set_simulation_date(None)
        # 👇👇👇 [修改] 给下载过程加锁 👇👇👇
        # 注意：Baostock 必须串行下载，不能并行！
        safe_print(f"🔒 [{stock_code}] 排队下载数据中...")
        with download_lock:
            # 在这个缩进块里的代码，同一时间只能有一个线程运行
            try:
                df_temp = local_analyzer.get_stock_data(normalized_code)
                # 为了防止 Baostock 服务器因为请求太快封IP，稍微缓一下
                time.sleep(0.5) 
            except Exception as e:
                safe_print(f"❌ [{stock_code}] 下载失败: {e}")
                df_temp = pd.DataFrame() # 返回空
        # 👆👆👆 [结束] 锁自动释放 👆👆👆

        if df_temp.empty or len(df_temp) < 60:
            safe_print(f" | {stock_code} 数据不足/为空")
            return result

        # 3. 验证核心列
        required_cols = ['open', 'close', 'high', 'low', 'volume', 'change_pct']
        missing_cols = [col for col in required_cols if col not in df_temp.columns]
        if missing_cols:
            if 'change_pct' in missing_cols and 'close' in df_temp.columns:
                df_temp['change_pct'] = df_temp['close'].pct_change() * 100
            else:
                safe_print(f" | {stock_code} 缺失核心列: {missing_cols}")
                return result

        # 4. 筛选日期
        available_dates = df_temp.index.sort_values()
        if len(available_dates) < backtest_days + 1:
            safe_print(f" | {stock_code} 日期不足")
            return result
        test_dates = available_dates[-(backtest_days + 1):]
        
        # 定义持仓状态
        position = {
            'holding': False,      # 是否持仓
            'buy_price': 0.0,      # 买入价格
            'buy_date': None,      # 买入日期
            'highest_price': 0.0,  # 持仓期间最高价（用于移动止损）
            'hold_days': 0         # 持有天数
        }
        # ==========================================
        # 🔄 核心循环：每天做一次决策
        # ==========================================
        for i in range(len(test_dates) - 1):
            curr_date = test_dates[i]
            
            # 必须设置时间穿越，否则计算指标会用到未来的数据
            local_analyzer.set_simulation_date(curr_date)
            df_slice = local_analyzer.get_stock_data(normalized_code)
            
            if df_slice.empty or len(df_slice) < 60: continue

            # 获取当日关键数据
            curr_close = df_slice.iloc[-1]['close']
            #    ------------------------------------------
            # 🟢 [卖出逻辑] 如果持有持仓，检查是否要卖
            # ------------------------------------------
            if position['holding']:
                position['hold_days'] += 1
                if curr_close > position['highest_price']:
                    position['highest_price'] = curr_close # 更新最高价

                sell_signal = False
                sell_reason = ""

                # 计算浮动盈亏
                # 优化后 (模拟 印花税+佣金):
                cost = 0.2 if market == 'a_stock' else 0.1 # A股印花税贵一点
                profit_pct = ((curr_close - position['buy_price']) / position['buy_price'] * 100) - cost

                # >>> 卖出规则 A: A股 T+1 限制 (优化点) <<<
                can_sell = True
                if market == 'a_stock' and position['hold_days'] < 1:
                    can_sell = False # 刚买不到1天，不能卖

                if can_sell:
                    # 1. 硬止损 (通用)
                    if profit_pct < -5.0:
                        sell_signal = True; sell_reason = f"硬止损({profit_pct:.1f}%)"
                    
                    # 2. 移动止盈 (通用)
                    # 曾赚10%以上，回撤超过3%就跑
                    elif position['highest_price'] > position['buy_price'] * 1.10:
                        drawdown = (curr_close - position['highest_price']) / position['highest_price'] * 100
                        if drawdown < -3.0:
                            sell_signal = True; sell_reason = "移动止盈(回撤保护)"
                    
                    # 3. 时间止损 (美股/港股优化)
                    # 如果是T+0市场，买了3天还没涨，说明动能不足，跑
                    # elif market in ['us_stock', 'hk_stock'] and position['hold_days'] > 3 and profit_pct < 1:
                    #     sell_signal = True; sell_reason = "动能耗尽(时间止损)"
                    elif market == 'hk_stock' and position['hold_days'] > 8 and profit_pct < 0.5:
                        sell_signal = True; sell_reason = "港股磨叽(时间止损)"
                    elif market == 'us_stock' and position['hold_days'] > 3 and profit_pct < 1:
                        sell_signal = True; sell_reason = "美股动能耗尽"

                # 执行卖出
                if sell_signal:
                    result['signals'] += 1
                    result['total_return'] += profit_pct
                    if profit_pct > 0: result['winning'] += 1
                    
                    safe_print(f"   💰 [卖出] {curr_date.strftime('%m-%d')} | 收益: {profit_pct:+.2f}% | 持有{position['hold_days']}天 | 原因: {sell_reason}")
                    
                    # 清空状态
                    position = {'holding': False, 'buy_price': 0, 'buy_date': None, 'highest_price': 0, 'hold_days': 0}
                    continue # 卖出当天不买回

            # ------------------------------------------
            # 🔴 [买入逻辑] 如果空仓，检查是否要买 (这里融合你的差异化策略)
            # ------------------------------------------
            if not position['holding']:
                try:
                    tech = local_analyzer.calculate_technical_indicators(df_slice)
                    money = local_analyzer.analyze_smart_money_flow(df_slice)
                except: continue

                # 提取指标
                rsi = tech.get('rsi', 50)
                ma20 = tech.get('ma20', 0)
                ma20_slope = tech.get('ma20_slope', 0)
                vol_ratio = df_slice.iloc[-1]['volume'] / (df_slice['volume'].rolling(20).mean().iloc[-1] + 1)
                
                # >>> 融合 1: 差异化初筛 (Pre-screening) <<<
                potential_signal = False
                debug_reject_reason = "" # 新增：记录被筛掉的原因，方便调试
                
                # A股: 稳健趋势 + 放量
                if market == 'a_stock':
                    trend_ok = (curr_close > ma20) or (ma20_slope > -0.001)
                    vol_ok = vol_ratio > 1.0
                    oversold = (rsi < 30) and (curr_close < ma20 * 0.85)
                    potential_signal = (trend_ok and vol_ok) or oversold
                    if (trend_ok and vol_ok) or oversold:
                        potential_signal = True
                    else:
                        debug_reject_reason = f"趋势/量能不符 (Slope:{ma20_slope:.4f}, Vol:{vol_ratio:.2f})"
                
                # 港股: 避仙股 + 资金流
                elif market == 'hk_stock':
                    liquidity_ok = (curr_close > 2.0) and (vol_ratio > 0.6)
                    # money_ok = money.get('money_flow_score', 50) > 60
                    # potential_signal = liquidity_ok and money_ok
                    if liquidity_ok:
                        potential_signal = True
                    else:
                        debug_reject_reason = f"流动性不足 (Price:{curr_close}, Vol:{vol_ratio:.2f})"
                
                # 美股: 强动量 + 顺势
                elif market == 'us_stock':
                    # 1. 趋势多头: 价格在均线上方 OR 均线斜率向上
                    trend_ok = (curr_close > ma20) or (ma20_slope > 0)
                    
                    # 2. 动量正常: RSI 没有死掉，KDJ 没有死叉 (不要求必须金叉，只要不是死叉就行)
                    momentum_ok = (rsi > 40) and (tech.get('kdj_signal') != '死叉')
                    
                    # 3. 超跌反弹: RSI 极低
                    oversold = (rsi < 30)

                    if (trend_ok and momentum_ok) or oversold:
                        potential_signal = True
                    else:
                        debug_reject_reason = f"趋势/动量不佳 (RSI:{rsi:.1f}, Trend:{trend_ok})"

                if not potential_signal: continue

                # >>> 融合 2: 差异化 Prompt 提示 <<<
                strategy_hint = ""
                if market == 'a_stock': strategy_hint = "A股(T+1)，极大重视安全性，拒绝下降趋势。"
                elif market == 'hk_stock': strategy_hint = "港股(T+0)，流动性第一，拒绝低成交量。"
                elif market == 'us_stock': strategy_hint = "美股(T+0)，顺势为主，允许RSI略高。"

                price_info = {
                    "close": round(curr_close, 2),
                    "change_pct": round(df_slice.iloc[-1]['change_pct'], 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "market_hint": strategy_hint # 传给AI
                }

                # 调用 AI
                try:
                    ai_result = local_analyzer.get_llm_trade_decision(
                        normalized_code, curr_date.strftime('%Y-%m-%d'), 
                        price_info, tech, money
                    )
                    action = ai_result.get('action', 'HOLD')
                    confidence = ai_result.get('confidence', 0)
                    reason = ai_result.get('reason', '无')
                    # 👇 新增这一行 👇
                    phase = ai_result.get('market_phase', '未知')
                except: continue

                if action == "BUY":
                    # >>> 融合 3: 差异化风控 (Risk Control) <<<
                    risk_pass = True
                    risk_msg = ""
                    
                    if market == 'a_stock':
                        if price_info['change_pct'] > 9.5: risk_pass = False; risk_msg = "涨停风险"
                        if ma20_slope < -0.05: risk_pass = False; risk_msg = "下跌趋势"
                    elif market == 'hk_stock':
                        if curr_close < 1.0: risk_pass = False; risk_msg = "仙股风险"
                        if vol_ratio < 0.5: risk_pass = False; risk_msg = "流动性差"
                    elif market == 'us_stock':
                        if rsi > 85: risk_pass = False; risk_msg = "极度超买"

                    if risk_pass:
                        # 执行买入
                        buy_price = curr_close # 假设以收盘价买入
                        position['holding'] = True
                        position['buy_price'] = buy_price
                        position['buy_date'] = curr_date
                        position['highest_price'] = buy_price
                        position['hold_days'] = 0
                        
                        result['ai_approved'] += 1
                        safe_print(f"   🛒 [买入] {curr_date.strftime('%m-%d')} |  阶段: {phase} | 价格: {buy_price:.2f} | 理由: {ai_result.get('reason','无')[:20]}")
                    else:
                        safe_print(f"   🛑 风控拦截: {risk_msg}")

        # 循环结束强平
        if position['holding']:
            last_close = df_temp.iloc[-1]['close']
            profit = (last_close - position['buy_price']) / position['buy_price'] * 100
            result['signals'] += 1
            result['total_return'] += profit
            if profit > 0: result['winning'] += 1
            safe_print(f"   🔚 [清仓] 回测结束 | 收益: {profit:+.2f}%")

    except Exception as e:
        safe_print(f"❌ 线程错误 {stock_code}: {e}")

    return result


# ==========================================
# 🚀 主控系统 (修改为并行模式)
# ==========================================
class AutoSystem:
    def __init__(self):
        safe_print("DEBUG: 初始化系统组件...")
        self.scanner = GlobalMarketScanner()
        # 注意：这里不需要 self.analyzer 了，因为移到了线程内部

    def run_market_cycle(self, market='hk_stock', limit=20, days=20):
        safe_print("\n" + "="*60)
        safe_print(f"🌍 启动市场流程: {market.upper()} | 目标筛选: Top {limit} | 回测周期: 近{days}天")
        safe_print("="*60)

        # 1. 扫描选股
        stock_list = []
        if market == 'hk_stock':
            stock_list = self.scanner.get_hk_candidates(top_n=limit)
        elif market == 'us_stock':
            stock_list = self.scanner.get_us_candidates(top_n=limit)
        elif market == 'a_stock':
            stock_list = self.scanner.get_a_candidates(top_n=limit)
        
        if not stock_list:
            logger.warning(f"⚠️ {market} 未扫描到有效股票")
            return

        safe_print(f"📋 扫描结果: {stock_list}")
        
        # 2. 批量回测 (改为并行)
        self.perform_parallel_backtest(stock_list, backtest_days=days)

    def perform_parallel_backtest(self, stock_list, backtest_days=20):
        """
        使用线程池进行并行回测
        """
        # 设置线程数：建议 4-8 个，太高容易被封IP或触发API限制
        MAX_WORKERS = 5 
        
        safe_print(f"\n🚀 [并行加速模式] 启动 {MAX_WORKERS} 个线程处理 {len(stock_list)} 只股票...")
        
        # 统计汇总
        stats = {
            'total_signals': 0,
            'ai_approved': 0,
            'winning_signals': 0,
            'total_return': 0.0,
            'market_stats': {'a_stock': 0, 'hk_stock': 0, 'us_stock': 0}
        }
        
        start_time = time.time()

        # 启动线程池
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_stock = {
                executor.submit(process_single_stock_task, stock, backtest_days): stock 
                for stock in stock_list
            }
            
            # 等待完成并收集结果
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    res = future.result()
                    
                    # 汇总数据
                    if res['market'] != 'UNKNOWN':
                        stats['market_stats'][res['market']] = stats['market_stats'].get(res['market'], 0) + 1
                    
                    stats['total_signals'] += res['signals']
                    stats['ai_approved'] += res['ai_approved']
                    stats['winning_signals'] += res['winning']
                    stats['total_return'] += res['total_return']
                    
                except Exception as exc:
                    safe_print(f"❌ 股票 {stock} 线程异常: {exc}")

        duration = time.time() - start_time

        # 输出回测报告
        safe_print("\n" + "="*60)
        safe_print(f"📊 AI 并行回测报告 | 耗时: {duration:.1f}秒")
        safe_print(f"🌍 市场分布: A股{stats['market_stats'].get('a_stock',0)} | 港股{stats['market_stats'].get('hk_stock',0)} | 美股{stats['market_stats'].get('us_stock',0)}")
        
        if stats['total_signals'] > 0:
            win_rate = (stats['winning_signals'] / stats['ai_approved']) * 100 if stats['ai_approved'] > 0 else 0
            avg_return = (stats['total_return'] / stats['ai_approved']) if stats['ai_approved'] > 0 else 0
            
            safe_print(f"🤖 AI 建议买入: {stats['ai_approved']} 次")
            safe_print(f"🏆 胜率: {win_rate:.1f}%")
            safe_print(f"💰 平均收益: {avg_return:.2f}%")
            safe_print(f"📈 总收益: {stats['total_return']:.2f}%")
        else:
            safe_print("💤 全程无符合条件的交易信号")
        safe_print("="*60 + "\n")

def main():
    safe_print("DEBUG: 进入主函数...")
    try:
        system = AutoSystem()
        
        # 1. 跑港股 (如需开启，取消注释)
        system.run_market_cycle(market='hk_stock', limit=20, days=20)
        
        # 2. 跑美股 (如需开启，取消注释)
        system.run_market_cycle(market='us_stock', limit=20, days=20)
        
        # 3. 跑A股 (并行加速版)
        # system.run_market_cycle(market='a_stock', limit=20, days=20)
        
    except KeyboardInterrupt:
        safe_print("\n用户中断")
    except Exception as e:
        safe_print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TimeTravelAnalyzer(EnhancedWebStockAnalyzer):
    """
    时间穿越分析器：继承原版分析器，但具备'欺骗时间'的能力
    """
    def __init__(self):
        super().__init__()
        self.simulation_date = None  # 当前模拟的日期
        self.full_price_history = {} # 缓存全量历史数据，避免重复下载

    def set_simulation_date(self, date_str):
        """设置'假装'的今天"""
        self.simulation_date = pd.to_datetime(date_str)

    def get_stock_data(self, stock_code, period='1y'):
        """
        重写获取数据方法：
        只返回 simulation_date 之前的数据，模拟当时的市场环境
        """
        stock_code, market = self.normalize_stock_code(stock_code)
        
        # 1. 如果没有缓存全量数据，先下载一次
        if stock_code not in self.full_price_history:
            # 调用父类方法获取最新的全量数据
            # 注意：这里我们临时把 simulation_date 设为 None 以便通过父类下载最新数据
            temp_date = self.simulation_date
            self.simulation_date = None 
            try:
                # 强制重新下载最新数据（忽略缓存）以便回测
                df = super().get_stock_data(stock_code) 
                self.full_price_history[stock_code] = df
            except Exception as e:
                logger.error(f"无法获取历史数据: {e}")
                return pd.DataFrame()
            finally:
                self.simulation_date = temp_date
        
        # 2. 从缓存中切片
        full_df = self.full_price_history.get(stock_code)
        if full_df is None or full_df.empty:
            return pd.DataFrame()

        if self.simulation_date is None:
            return full_df

        # 3. 执行“时间切割”
        # 只保留 simulation_date 当天及之前的数据
        mask = full_df.index <= self.simulation_date
        sliced_df = full_df.loc[mask].copy()
        
        return sliced_df

def run_backtest(stock_list, backtest_days=20):
    """
    运行回测主程序 (稳健版：基于真实数据日期)
    """
    print("="*60)
    print(f"🚀 开始策略回测 | 股票数: {len(stock_list)} | 目标回测: 近{backtest_days}个交易日")
    print("="*60)
    
    analyzer = TimeTravelAnalyzer()
    
    # 统计结果容器
    results = []
    failed_cases = [] 
    
    for stock_code in stock_list:
        # print(f"\n📊 正在准备: {stock_code} ...")
        
        # 1. 获取该股票的全量数据
        # 注意：先调用一次 get_stock_data 确保数据已下载并缓存
        analyzer.get_stock_data(stock_code)
        # 从 analyzer 的缓存中直接拿原始 DataFrame
        full_data = analyzer.full_price_history.get(analyzer.normalize_stock_code(stock_code)[0])
        
        if full_data is None or full_data.empty:
            print(f"❌ {stock_code}: 无数据，跳过")
            continue
            
        # 2. 提取真实存在的交易日期
        # 排序并只取 datetime 类型的索引
        available_dates = full_data.index.sort_values()
        
        # 检查数据量是否足够
        # 我们至少需要: 60天(算指标) + backtest_days(回测) + 1天(验证次日结果)
        min_required = 60 + backtest_days + 1
        if len(available_dates) < min_required:
            print(f"⚠️ {stock_code}: 数据行数不足 (现有{len(available_dates)}行, 需要{min_required}行)，跳过")
            continue
            
        # 3. 选定要回测的日期范围
        # 取倒数 (backtest_days + 1) 个日期，因为我们要验证"下一天"的涨跌
        test_dates = available_dates[-(backtest_days+1):]
        
        print(f"📊 {stock_code}: 开始回测 {test_dates[0].strftime('%Y-%m-%d')} 至 {test_dates[-2].strftime('%Y-%m-%d')}")

        # 4. 遍历日期进行回测
        for i in range(len(test_dates) - 1):
            curr_date = test_dates[i]      # 假装今天是这一天
            next_date = test_dates[i+1]    # 明天（用于验证结果）
            
            # --- 核心：时间穿越 ---
            analyzer.set_simulation_date(curr_date)
            
            # 获取切片数据
            df_slice = analyzer.get_stock_data(stock_code)
            
            # 双重检查
            if df_slice.empty or len(df_slice) < 60: 
                continue

            # --- 运行策略 ---
            try:
                # 计算指标
                tech_indicators = analyzer.calculate_technical_indicators(df_slice)
                tech_score = analyzer.calculate_technical_score(tech_indicators)
                
                # 计算资金流
                money_flow = analyzer.analyze_smart_money_flow(df_slice)
                
                # 计算风控
                trade_levels = analyzer.calculate_trade_levels(df_slice)
                
               # ====================================================
                # 👇👇👇 核心修改：顺势而为策略 (拒绝下跌趋势的反弹) 👇👇👇
                # ====================================================
                
                signal = "观望"
                
                # 1. 提取基础数据
                close_price = float(df_slice.iloc[-1]['close'])
                open_price = float(df_slice.iloc[-1]['open'])
                volume = float(df_slice.iloc[-1]['volume'])
                
                # 安全获取昨天收盘价
                if len(df_slice) >= 2:
                    prev_close = float(df_slice.iloc[-2]['close'])
                else:
                    prev_close = open_price # 兜底

                # 获取均线
                ma5 = tech_indicators.get('ma5', 0)
                ma20 = tech_indicators.get('ma20', 0)
                
                # 安全计算昨天的 MA20 (用于判断趋势方向)
                try:
                    ma20_prev = df_slice['close'].iloc[:-1].rolling(20).mean().iloc[-1]
                    if pd.isna(ma20_prev): ma20_prev = 0
                except:
                    ma20_prev = 0
                
                rsi = tech_indicators.get('rsi', 50)
                money_score = money_flow.get('money_flow_score', 0)
                vol_ma20 = df_slice['volume'].rolling(20).mean().iloc[-1]
                if pd.isna(vol_ma20) or vol_ma20 == 0: vol_ma20 = 1.0

                # 2. 计算乖离率 (Bias)
                if ma20 > 0:
                    bias_20 = (close_price - ma20) / ma20
                else:
                    bias_20 = 0

                # 3. 定义过滤条件
                
                # [条件A] 生命线拐头向上 (核心救命稻草)
                # 只有 MA20 正在上涨，才说明处于上升通道。
                # 如果 MA20 在下跌，哪怕股价站上去了，也大概率是诱多。
                trend_is_up = (ma20 > ma20_prev) and (ma20 > 0)
                
                # [条件B] 股价位于均线之上
                price_above_ma20 = close_price > ma20
                
                # [条件C] 拒绝高位接盘 (收紧乖离率)
                # 之前是 10%，现在收紧到 8%。只买刚启动的，不买涨飞的。
                bias_safe = 0 < bias_20 < 0.08
                
                # [条件D] 形态确认 (收阳线)
                # 今天必须收红，且收盘价 > 昨天收盘价 (实打实的上涨)
                is_solid_up = (close_price > open_price) and (close_price > prev_close)
                
                # [条件E] 资金门槛
                money_ok = money_score >= 60
                
                # [条件F] 必须放量
                # 缩量上涨不可信
                volume_ok = volume > (vol_ma20 * 0.9)

                # 4. 综合决策
                # 逻辑：趋势向上 + 股价站稳 + 位置不高 + 资金进场 + 放量阳线
                if trend_is_up and price_above_ma20 and bias_safe and is_solid_up and money_ok and volume_ok:
                    signal = "买入"
                
                # 卖出逻辑
                elif (ma20 > 0 and close_price < ma20) or tech_score < 40:
                    signal = "卖出"
                
                # ====================================================
                # 👆👆👆 修改结束 👆👆👆
                # ====================================================
                    
                # --- 验证结果 ---
                # 获取次日真实数据
                next_day_data = full_data.loc[next_date]
                next_close = float(next_day_data['close'])
                curr_close = float(df_slice.iloc[-1]['close'])
                
                # 计算次日收益
                actual_return = (next_close - curr_close) / curr_close * 100
                
                # 记录
                record = {
                    'date': curr_date.strftime('%Y-%m-%d'),
                    'stock': stock_code,
                    'score': tech_score,
                    'money_score': money_flow.get('money_flow_score', 0),
                    'signal': signal,
                    'actual_return': actual_return,
                    'stop_loss': trade_levels.get('stop_loss', 0),
                    'low_price': float(next_day_data['low'])
                }
                results.append(record)
                
                # 收集失败案例
                if signal == "买入":
                    # 情况1: 大跌
                    if actual_return < -3:
                        failed_cases.append({**record, 'reason': '📉 暴跌: 买入后次日跌幅>3%'})
                    # 情况2: 盘中触及止损
                    elif record['low_price'] < record['stop_loss']:
                        failed_cases.append({**record, 'reason': '🛡️ 止损: 盘中触及ATR止损线'})

            except Exception as e:
                print(f"Error on {curr_date}: {e}")
                continue

    # === 生成最终报告 ===
    print("\n" + "="*60)
    print("📈 回测结果分析报告")
    print("="*60)
    
    if not results:
        print("❌ 无有效回测记录，请检查数据完整性。")
        return

    df_res = pd.DataFrame(results)
    buy_signals = df_res[df_res['signal'] == '买入']
    
    if len(buy_signals) > 0:
        # 1. 基础胜率
        win_trades = buy_signals[buy_signals['actual_return'] > 0]
        loss_trades = buy_signals[buy_signals['actual_return'] <= 0]
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        win_rate = (win_count / len(buy_signals)) * 100
        
        # 2. 收益统计
        avg_return = buy_signals['actual_return'].mean()
        max_win = buy_signals['actual_return'].max()
        max_loss = buy_signals['actual_return'].min()
        total_return = buy_signals['actual_return'].sum() # 简单单利总和
        
        # 3. 盈亏比 (Profit/Loss Ratio) - 核心指标
        avg_win_amt = win_trades['actual_return'].mean() if not win_trades.empty else 0
        avg_loss_amt = abs(loss_trades['actual_return'].mean()) if not loss_trades.empty else 1e-9 # 防除零
        pl_ratio = avg_win_amt / avg_loss_amt
        
        # 4. 最大回撤 (Max Drawdown) - 模拟资金曲线
        # 假设每次投入10000元，模拟资金曲线
        buy_signals['equity_curve'] = (1 + buy_signals['actual_return']/100).cumprod()
        peak = buy_signals['equity_curve'].expanding(min_periods=1).max()
        drawdown = (buy_signals['equity_curve'] - peak) / peak
        max_drawdown = drawdown.min() * 100 # 百分比

        print(f"🟢 交易次数: {len(buy_signals)} 次")
        print(f"🏆 胜    率: {win_rate:.1f}%  (赢{win_count} / 输{loss_count})")
        print(f"⚖️ 盈 亏 比: {pl_ratio:.2f}  (平均赚{avg_win_amt:.2f}% / 平均亏{avg_loss_amt:.2f}%)")
        print(f"💰 平均收益: {avg_return:.2f}%")
        print(f"🌊 最大回撤: {max_drawdown:.2f}% (资金曲线峰值回落)")
        print(f"🚀 最佳单笔: +{max_win:.2f}%")
        print(f"💣 最差单笔: {max_loss:.2f}%")
        
        # 打印最近交易
        print("\n📋 最近 5 次 AI 买入信号:")
        cols = ['date', 'stock', 'score', 'actual_return', 'money_score']
        # 检查列是否存在，防止报错
        existing_cols = [c for c in cols if c in buy_signals.columns]
        print(buy_signals[existing_cols].tail(5).to_string(index=False))

    else:
        print("⚠️ 策略太保守，未触发任何买入信号")

    if failed_cases:
        print(f"\n🧐 失败/风控拦截案例 (共{len(failed_cases)}次):")
        # 只打印前3个，避免刷屏
        for i, case in enumerate(failed_cases[:3]):
            print(f"[{i+1}] {case['date']} {case['stock']} | 收益: {case['actual_return']:.2f}% | 原因: {case.get('reason', '未知')}")
    else:
        print("\n🎉 完美！风控系统未记录到重大失败案例。")

# ==========================================
# 在这里输入你要批量测试的股票列表
# ==========================================
if __name__ == "__main__":
    # 你可以把之前的 30 只股票粘贴到这里
    test_stocks = [
        "300274",
"601899",
"002594",
"601888",
"601600",
"300750",
"603993",
"600498",
"002407",
"000630",
"002460",
"600362",
"601696",
"002466",
"002709",
"000878",
"300059",
"300568",
"300475",
"002326",
"688110",
"688158",
"300118",
"000792",
"000737",
"601168",
"600219",
"300390",
"002497",
"600089"
        # ... 可以继续加
    ]
    
    # 运行回测 (测过去 20 天)
    run_backtest(test_stocks, backtest_days=20)
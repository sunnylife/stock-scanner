import time
import logging
from datetime import datetime

# 导入你的模块
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
from market_scanner import MarketScanner
from trade_executor import TradeExecutor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"trading_log_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("AutoSystem")

def main():
    print("="*60)
    print("🚀 全自动量化交易系统启动 (Selection -> Analysis -> Execution)")
    print("="*60)

    # 1. 初始化各个模块
    analyzer = EnhancedWebStockAnalyzer()   # 你的核心分析器
    scanner = MarketScanner(market='a_stock') # A股扫描器
    trader = TradeExecutor(mode='sim')      # 交易执行器 (先用模拟模式)

    # 2. 运行选股 (Scanner)
    logger.info("Step 1: 启动全市场扫描...")
    candidate_codes = scanner.run_coarse_filter(top_n=5) # 为了测试，先只取前5只
    
    if not candidate_codes:
        logger.warning("⚠️ 未筛选出符合条件的股票，今日休息。")
        return

    logger.info(f"🎯 初选股票池 ({len(candidate_codes)}只): {candidate_codes}")

    # 3. 循环深度分析 (Analysis)
    for i, stock_code in enumerate(candidate_codes):
        try:
            logger.info(f"\n[{i+1}/{len(candidate_codes)}] 正在深度分析: {stock_code} ...")
            
            # 调用你现有的 analyze_stock 方法
            # 注意：enable_streaming=False 因为这是后台自动运行
            report = analyzer.analyze_stock(stock_code, enable_streaming=False)
            
            # 提取关键决策信息
            scores = report.get('scores', {})
            ai_decision = report.get('ai_trade_decision', {})
            
            logger.info(f"📊 分析完成: 综合分 {scores.get('comprehensive', 0):.1f} | AI建议: {ai_decision.get('action')}")
            
            # 4. 执行交易 (Execution)
            # 只有当 AI 明确建议 BUY 且 综合分够高时才交易
            if ai_decision.get('action') == "BUY":
                logger.info("🔥 触发买入信号！转交交易执行器...")
                trader.execute_signal(report)
            else:
                logger.info("💤 信号未达标，观望。")

            # 避免 API 频率限制，稍微休息一下
            time.sleep(2)

        except Exception as e:
            logger.error(f"❌ 处理 {stock_code} 时出错: {e}")
            continue

    logger.info("\n✅ 所有候选股分析完毕。")

    # 5. 持仓监控 (可选)
    # 这里可以添加逻辑：遍历当前持仓，调用 analyzer 分析是否需要 SELL

if __name__ == "__main__":
    main()
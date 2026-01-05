import akshare as ak
import pandas as pd
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketScanner:
    """
    市场扫描器：负责从全市场中筛选出符合初步条件的股票池
    """
    def __init__(self, market='a_stock'):
        self.market = market

    def get_all_tickers(self):
        """获取全市场实时行情"""
        logger.info("正在获取全市场实时数据...")
        try:
            if self.market == 'a_stock':
                # 获取 A 股实时行情 (速度较快)
                df = ak.stock_zh_a_spot_em()
                # 重命名方便处理
                df = df.rename(columns={
                    '代码': 'symbol', '名称': 'name', '最新价': 'price', 
                    '涨跌幅': 'pct_change', '成交量': 'volume', 
                    '成交额': 'amount', '换手率': 'turnover', '量比': 'volume_ratio'
                })
                return df
            # 如果需要港美股，可以在这里扩展
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取全市场数据失败: {e}")
            return pd.DataFrame()

    def run_coarse_filter(self, top_n=50):
        """
        粗过滤逻辑：筛选出值得进行 AI 深度分析的股票
        """
        df = self.get_all_tickers()
        if df.empty:
            return []

        # === 1. 基础清洗 ===
        # 剔除停牌、退市、ST
        df = df[df['price'] > 0]
        df = df[~df['name'].str.contains('ST|退')]
        
        # 剔除北交所 (视情况而定，这里假设只做主板/创业/科创)
        df = df[~df['symbol'].str.startswith(('8', '4', '92'))] 

        # === 2. 硬指标过滤 (根据你的策略调整) ===
        # 逻辑：找活跃股，比如 涨幅 > 2% 且 量比 > 1.5 (放量上涨)
        # 或者：找超跌股，比如 RSI 低 (需要计算，这里用涨跌幅暂代)
        
        # 示例策略：【首板/强势股挖掘】
        # 1. 涨幅在 3% 到 9.5% 之间 (未涨停，但强势)
        # 2. 量比 > 1.2 (有量)
        # 3. 换手率 > 3% (活跃)
        # 4. 流通市值 < 500亿 (剔除大盘股，波动大)
        
        condition = (
            (df['pct_change'] > 3.0) & 
            (df['pct_change'] < 9.5) &
            (df['volume_ratio'] > 1.2) &
            (df['turnover'] > 3.0) &
            (df['amount'] > 100000000) # 成交额大于1亿
        )
        
        candidates = df[condition].copy()
        
        # === 3. 排序截断 ===
        # 按量比排序，取前 N 名
        candidates = candidates.sort_values(by='volume_ratio', ascending=False).head(top_n)
        
        logger.info(f"粗选完成，筛选出 {len(candidates)} 只股票进入精选池。")
        
        # 返回股票代码列表
        return candidates['symbol'].tolist()

if __name__ == "__main__":
    scanner = MarketScanner()
    stocks = scanner.run_coarse_filter()
    print("初选股票池:", stocks)
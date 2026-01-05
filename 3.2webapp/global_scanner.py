# global_scanner.py
import akshare as ak
import pandas as pd
import logging
import time
import requests

# ==========================================
# 🛠️ 核心黑科技：全局强制增加超时时间 (解决网络慢的问题)
# ==========================================
_original_get = requests.get
_original_post = requests.post

def patched_get(*args, **kwargs):
    # 强制让所有请求至少等待 60 秒
    if 'timeout' not in kwargs or kwargs['timeout'] < 60:
        kwargs['timeout'] = 60
    return _original_get(*args, **kwargs)

def patched_post(*args, **kwargs):
    if 'timeout' not in kwargs or kwargs['timeout'] < 60:
        kwargs['timeout'] = 60
    return _original_post(*args, **kwargs)

requests.get = patched_get
requests.post = patched_post
# ==========================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class GlobalMarketScanner:
    def __init__(self):
        pass

    def get_hk_candidates(self, top_n=30):
        """获取港股候选池"""
        logger.info("📡 正在扫描港股市场 (HK) [已启用60秒超时补丁]...")
        
        # 兜底名单：如果网络全挂了，就测这些
        fallback_list = ["00700", "09988", "03690", "01211", "01810", "00981", "00388", "02020", "00992", "00005"]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0: logger.info(f"🔄 第 {attempt+1} 次重试...")
                
                df = ak.stock_hk_spot_em()
                df = df[df['最新价'] > 1.0] # 过滤仙股
                df = df[df['成交额'] > 50000000] # 过滤冷门股
                
                candidates = df.sort_values(by='成交额', ascending=False).head(top_n)
                stock_list = candidates['代码'].tolist()
                
                logger.info(f"✅ 港股扫描成功，入选 {len(stock_list)} 只")
                return stock_list

            except Exception as e:
                logger.warning(f"⚠️ 扫描失败: {e}")
                time.sleep(3)

        logger.error("❌ 接口超时，启动【蓝筹股兜底模式】")
        return fallback_list[:top_n]

    def get_us_candidates(self, top_n=30):
        """获取美股候选池"""
        logger.info("📡 正在扫描美股市场 (US)...")
        fallback_list = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL", "BABA", "PDD"]
        
        for attempt in range(3):
            try:
                df = ak.stock_us_spot_em()
                df = df[df['最新价'] > 10.0]
                # 简单提取代码逻辑
                df['symbol'] = df['代码'].apply(lambda x: str(x).split('.')[-1])
                df = df[df['symbol'].str.match(r'^[A-Z]+$')] # 只留纯字母
                
                candidates = df.sort_values(by='成交额', ascending=False).head(top_n)
                return candidates['symbol'].tolist()
            except Exception:
                time.sleep(3)

        logger.error("❌ 接口超时，启动【美股兜底模式】")
        return fallback_list[:top_n]
    
    def get_a_candidates(self, top_n=30):
        """
        获取A股候选池 (基于量价活跃度筛选)
        """
        logger.info("📡 正在扫描A股市场 (CN)...")
        
        # 兜底名单：茅台、宁德、平安、招行、东方财富、中信、紫金、立讯、迈瑞、比亚迪
        fallback_list = ["600519", "300750", "601318", "600036", "300059", "600030", "601899", "002475", "300760", "002594"]

        try:
            # 1. 获取全市场实时行情
            df = ak.stock_zh_a_spot_em()
            
            # 2. 基础过滤：剔除停牌、ST、退市、空值
            df = df[df['最新价'] > 2.0]  # 剔除2元以下低价股
            df = df[~df['名称'].str.contains('ST|退')]
            
            # 3. 剔除北交所 (代码以 4, 8, 92 开头)
            df = df[~df['代码'].str.startswith(('4', '8', '92'))]
            
            # 4. 核心策略筛选：【首板/趋势中继】
            # 逻辑：涨幅适中(2-9.5%) + 放量(量比>1.2) + 活跃(换手>3%) + 流动性好(成交>1亿)
            condition = (
                (df['涨跌幅'] > 2.0) & 
                (df['涨跌幅'] < 9.5) &
                (df['量比'] > 1.2) &
                (df['换手率'] > 3.0) &
                (df['成交额'] > 100000000) 
            )
            
            candidates = df[condition].copy()
            
            # 5. 排序截断
            # 优先按【量比】排序，寻找主力资金急切抢筹的品种
            if not candidates.empty:
                candidates = candidates.sort_values(by='量比', ascending=False).head(top_n)
                stock_list = candidates['代码'].tolist()
                logger.info(f"✅ A股扫描成功，入选 {len(stock_list)} 只 (策略: 放量活跃)")
                return stock_list
            else:
                logger.warning("⚠️ 严选策略未匹配到股票，尝试放宽条件（仅按涨幅和流动性）...")
                # 备用逻辑：只看流动性和涨幅
                backup_cond = (df['成交额'] > 300000000) & (df['涨跌幅'] > 3.0) & (df['涨跌幅'] < 9.8)
                candidates = df[backup_cond].sort_values(by='涨跌幅', ascending=False).head(top_n)
                return candidates['代码'].tolist()

        except Exception as e:
            logger.error(f"❌ A股扫描失败: {e}")
            return fallback_list[:top_n]
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
        """
        [优化版] 获取港股候选池
        策略：流动性 + 中低价股 (适合小资金) + 剔除仙股/衍生品
        """
        logger.info("📡 正在扫描港股市场 (HK)...")
        fallback_list = ["00700", "01810", "00992", "00285", "03800"] # 腾讯,小米,联想,比亚迪电子,协鑫

        for attempt in range(3):
            try:
                df = ak.stock_hk_spot_em()
                
                # 1. 基础过滤：只保留 5 位数字代码
                # 剔除窝轮牛熊证 (通常是 5 位数但名字很怪，或者代码非0开头)
                # 港股正股代码通常是 0 开头，如 00700
                df = df[df['代码'].str.match(r'^0\d{4}$')]
                
                # 2. 价格过滤 (关键！)
                # 剔除 < 2元的仙股 (风险太大)
                # 剔除 > 100元的高价股 (一手买不起)
                df = df[(df['最新价'] > 2.0) & (df['最新价'] < 100.0)]
                
                # 3. 流动性过滤
                df = df[df['成交额'] > 20000000] # 2000万港币以上
                
                # 4. 趋势过滤 (当日上涨)
                if '涨跌幅' in df.columns:
                    df = df[(df['涨跌幅'] > 1.0) & (df['涨跌幅'] < 15.0)]
                
                # 5. 排序：按量比或成交额
                # 量比反映了当日资金的活跃度
                if '量比' in df.columns:
                    df = df.sort_values(by='量比', ascending=False)
                else:
                    df = df.sort_values(by='成交额', ascending=False)
                
                candidates = df.head(top_n)
                return candidates['代码'].tolist()

            except Exception as e:
                logger.warning(f"⚠️ 港股扫描重试: {e}")
                time.sleep(3)

        return fallback_list[:top_n]

    def get_us_candidates(self, top_n=30):
        """
        [增强版] 获取美股候选池
        策略：流动性 + 动量 + 盘口强势度
        """
        logger.info("📡 正在扫描美股市场 (US) [增强策略]...")
        
        # 兜底名单：科技七巨头 + 热门股
        fallback_list = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL", "BABA", "PLTR"]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0: logger.info(f"🔄 美股扫描重试第 {attempt+1} 次...")
                
                # 1. 获取全市场实时数据
                df = ak.stock_us_spot_em()
                
                # 2. 数据清洗：提取纯字母代码 (剔除窝轮、基金等)
                # 代码格式通常为 "105.NVDA" 或 "NVDA"
                df['symbol'] = df['代码'].apply(lambda x: str(x).split('.')[-1])
                df = df[df['symbol'].str.match(r'^[A-Z]+$')]
                
                # 3. 硬性门槛过滤
                # (1) 价格过滤: 剔除 < 5美元的仙股/毛票
                df = df[df['最新价'] > 5.0]
                
                # (2) 流动性过滤: 成交额 > 5000万美元 (确保买得进卖得出)
                # 注意：部分接口返回单位可能不同，需确保是数值型
                df = df[df['成交额'] > 50000000]
                
                # (3) 趋势过滤: 
                # - 涨跌幅 > 1% (有上涨动能)
                # - 涨跌幅 < 15% (避免已经暴涨过头的妖股)
                df = df[(df['涨跌幅'] > 1.0) & (df['涨跌幅'] < 15.0)]
                
                # (4) 盘口强势度: 最新价 > 开盘价 (即今日收红/阳线)
                # 这一步非常关键，过滤掉高开低走的套人股票
                if '开盘价' in df.columns:
                    df = df[df['最新价'] > df['开盘价']]

                # 4. 综合打分排序 (核心策略)
                # 逻辑：我们需要找成交活跃且涨势不错的股票
                # 归一化处理，防止成交额数量级过大主导分数
                max_amount = df['成交额'].max()
                max_chg = df['涨跌幅'].max()
                
                # 评分公式：成交额权重 0.4 + 涨幅权重 0.4 + 换手率权重 0.2
                # (如果没有换手率数据，则忽略该项)
                if '换手率' in df.columns:
                    max_turnover = df['换手率'].max()
                    df['score'] = (
                        (df['成交额'] / max_amount) * 40 + 
                        (df['涨跌幅'] / max_chg) * 40 +
                        (df['换手率'] / max_turnover) * 20
                    )
                else:
                    df['score'] = (df['成交额'] / max_amount) * 50 + (df['涨跌幅'] / max_chg) * 50

                # 5. 取 Top N
                candidates = df.sort_values(by='score', ascending=False).head(top_n)
                
                stock_list = candidates['symbol'].tolist()
                logger.info(f"✅ 美股扫描成功，基于[量价综合评分]入选 {len(stock_list)} 只")
                
                # 打印前3名看看效果
                if not candidates.empty:
                    top3_info = candidates[['symbol', '最新价', '涨跌幅', '成交额']].head(3).to_dict('records')
                    logger.info(f"🔥 热门前三: {top3_info}")

                return stock_list

            except Exception as e:
                logger.warning(f"⚠️ 美股扫描异常: {e}")
                time.sleep(3)

        logger.error("❌ 美股接口超时或失败，启动【兜底模式】")
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
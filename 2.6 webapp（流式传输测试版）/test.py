"""
test.py
"""
import akshare as ak
import pandas as pd
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置国内网络请求适配器，提升稳定性
def get_china_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # 适配国内网络，添加超时设置
    session.timeout = 15
    return session

session = get_china_session()

def get_stock_history_kline(
    symbol: str, 
    market: str = "a_share",  # 可选：a_share（A股）、hk（港股）、us（美股）
    period: str = "daily",    # 可选：daily（日线）（A股/港股/美股均仅支持日线，避免接口不存在报错）
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    国内环境专属：全市场股票历史K线获取工具（纯AKShare，无Tushare依赖，稳定无外网依赖）
    :param symbol: 股票代码（A股：600519；港股：00700；美股：AAPL）
    :param market: 市场类型（a_share/hk/us）
    :param period: K线周期（仅支持daily日线，避免AKShare接口不存在报错）
    :param start_date: 开始日期（格式：YYYYMMDD，默认近90天）
    :param end_date: 结束日期（格式：YYYYMMDD，默认当前日期）
    :return: 格式化后的K线DataFrame
    """
    # 处理默认日期
    if not end_date:
        end_dt = datetime.now()
        end_date_ak = end_dt.strftime("%Y%m%d")
    else:
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        end_date_ak = end_date
    
    if not start_date:
        start_dt = datetime.now() - timedelta(days=90)
        start_date_ak = start_dt.strftime("%Y%m%d")
    else:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        start_date_ak = start_date

    try:
        df = pd.DataFrame()
        # 1. A股市场（纯AKShare日线接口，稳定无依赖）
        if market == "a_share":
            # 强制使用日线，避免周线/月线接口不存在报错
            if period != "daily":
                print("⚠️  A股暂仅支持日线数据，自动切换为日线查询")
                period = "daily"
            symbol_full = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
            df = ak.stock_zh_a_daily(
                symbol=symbol_full,
                start_date=start_date_ak,
                end_date=end_date_ak
            )

        # 2. 港股市场（AKShare国内接口，稳定无依赖）
        elif market == "hk":
            if period != "daily":
                raise ValueError("港股暂仅支持日线数据")
            df = ak.stock_hk_hist(
                symbol=symbol,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d")
            )

        # 3. 美股市场（AKShare国内接口，无需外网访问）
        elif market == "us":
            if period != "daily":
                raise ValueError("美股暂仅支持日线数据")
            df = ak.stock_us_daily(symbol=symbol)

        # 数据为空判断
        if df.empty:
            print(f"⚠️  未获取到{market}({symbol})原始数据")
            return pd.DataFrame()

        # 统一日期列处理
        date_candidates = ["date", "Date", "日期", "trade_date", "交易日期"]
        date_col = None
        for col in date_candidates:
            if col in df.columns:
                date_col = col
                break
        if not date_col:
            date_col = df.columns[0]
            print(f"⚠️  未识别到标准日期列，使用第一列'{date_col}'作为日期列")
        
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        # 日期筛选
        df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

        # 统一列名映射（兼容国内接口所有列名格式）
        column_mapping = {
            date_col: "date",
            "开盘": "open", "open": "open", "Open": "open", "开盘价": "open",
            "最高": "high", "high": "high", "High": "high", "最高价": "high",
            "最低": "low", "low": "low", "Low": "low", "最低价": "low",
            "收盘": "close", "close": "close", "Close": "close", "收盘价": "close",
            "成交量": "volume", "volume": "volume", "Volume": "volume", "vol": "volume"
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # 补充核心列
        core_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in core_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # 数据清洗与格式化
        df = df[core_cols].dropna(subset=["date", "close"]).sort_values(by="date").reset_index(drop=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 过滤无效数值
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] > 0)]
        return df

    except Exception as e:
        print(f"❌ 获取{market}数据失败：{str(e)}")
        return pd.DataFrame()

def print_kline_data(df: pd.DataFrame):
    """格式化打印K线数据，适配国内环境输出"""
    if df.empty:
        print("⚠️  无有效K线数据")
        return
    
    print(f"\n📊 共获取 {len(df)} 条{df.iloc[0]['date']}至{df.iloc[-1]['date']}的K线数据")
    print("-" * 88)
    for idx, row in df.head(5).iterrows():
        print(
            f"日期：{row['date']} | 开：{row['open']:.2f} | "
            f"高：{row['high']:.2f} | 低：{row['low']:.2f} | "
            f"收：{row['close']:.2f} | 量：{int(row['volume'])}"
        )
    if len(df) > 5:
        print("...")
        last_row = df.iloc[-1]
        print(
            f"日期：{last_row['date']} | 开：{last_row['open']:.2f} | "
            f"高：{last_row['high']:.2f} | 低：{last_row['low']:.2f} | "
            f"收：{last_row['close']:.2f} | 量：{int(last_row['volume'])}"
        )
    print("-" * 88)

if __name__ == "__main__":
    # 忽略无关警告（国内环境常见警告，包括pkg_resources废弃警告）
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 88)
    print("🎯 国内环境专属：全市场股票历史K线获取工具（纯AKShare，无Tushare依赖）")
    print("=" * 88)

    # 示例1：A股贵州茅台（600519）近90天日线
    print("\n===== 示例1：A股贵州茅台（600519）近90天日线 =====")
    a_share_df = get_stock_history_kline(
        symbol="600519",
        market="a_share",
        period="daily",
        start_date=(datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
    )
    print_kline_data(a_share_df)

    # 示例2：港股腾讯控股（00700）近30天日线
    print("\n===== 示例2：港股小米集团（01810）近30天日线 =====")
    hk_df = get_stock_history_kline(
        symbol="01810",
        market="hk",
        period="daily",
        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    )
    print_kline_data(hk_df)

    # 示例3：美股苹果（AAPL）近60天日线
    print("\n===== 示例3：美股苹果（AAPL）近60天日线 =====")
    us_df = get_stock_history_kline(
        symbol="AAPL",
        market="us",
        period="daily",
        start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
    )
    print_kline_data(us_df)

    # 示例4：A股宁德时代（300750）近60天日线（原周线切换为日线，避免接口不存在报错）
    print("\n===== 示例4：A股宁德时代（300750）近60天日线 =====")
    a_share_daily_df = get_stock_history_kline(
        symbol="300750",
        market="a_share",
        period="daily",
        start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
    )
    print_kline_data(a_share_daily_df)

    print("\n🎉 所有市场K线查询完成！")
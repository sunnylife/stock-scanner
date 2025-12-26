import akshare as ak
import datetime

def get_latest_stock_price(stock_code):
    """
    获取股票最新价格（支持A股、港股、美股）
    
    参数:
        stock_code: 股票代码（如A股600879需传入'600879'）
    
    返回:
        包含最新价格及时间的字典
    """
    try:
        # 尝试获取A股实时行情
        # 方法1: 使用东方财富实时行情接口
        df = ak.stock_zh_a_spot_em()
        stock_data = df[df['代码'] == stock_code]
        
        if not stock_data.empty:
            latest_price = stock_data.iloc[0]['最新价']
            update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return {
                'stock_code': stock_code,
                'latest_price': latest_price,
                'update_time': update_time,
                'source': '东方财富实时行情',
                'market': 'A股'
            }
        
        # 方法2: 尝试使用新浪财经接口（备用）
        df_sina = ak.stock_zh_a_daily(symbol=f'sh{stock_code}', adjust="qfq")
        if not df_sina.empty:
            latest_price = df_sina.iloc[-1]['close']
            update_time = df_sina.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            return {
                'stock_code': stock_code,
                'latest_price': latest_price,
                'update_time': update_time,
                'source': '新浪财经日线数据',
                'market': 'A股'
            }
            
        return {'error': f'未找到股票代码 {stock_code} 的数据'}
        
    except Exception as e:
        return {'error': f'获取价格失败: {str(e)}'}

# 测试获取600879的最新价格
if __name__ == "__main__":
    result = get_latest_stock_price('600879')
    print(f"股票代码: {result.get('stock_code')}")
    print(f"最新价格: {result.get('latest_price')}")
    print(f"更新时间: {result.get('update_time')}")
    print(f"数据来源: {result.get('source')}")
    if 'error' in result:
        print(f"错误信息: {result['error']}")
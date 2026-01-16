# check_data.py
import pandas as pd
import numpy as np
from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer

analyzer = EnhancedWebStockAnalyzer()
stock_code = "02800"  # 港股盈富基金

# 获取数据
df = analyzer.get_stock_data(stock_code)
print(f"数据行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print(f"\n最近5行数据:")
print(df.tail())

# 计算技术指标
tech = analyzer.calculate_technical_indicators(df)
print(f"\n技术指标:")
for key, value in tech.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")

# 检查价格和MA20
if 'close' in df.columns and 'ma20' in tech:
    latest_price = df['close'].iloc[-1]
    ma20_value = tech.get('ma20', 0)
    if ma20_value > 0:
        deviation = (latest_price - ma20_value) / ma20_value * 100
        print(f"\n🔍 数据质量检查:")
        print(f"  最新收盘价: {latest_price:.4f}")
        print(f"  MA20值: {ma20_value:.4f}")
        print(f"  偏离度: {deviation:.2f}%")
        
        if abs(deviation) > 100:
            print(f"  ❌ 严重数据异常！偏离超过100%")
            # 检查历史数据
            print(f"\n  检查历史MA20计算:")
            df['ma20_calc'] = df['close'].rolling(20).mean()
            print(df[['close', 'ma20_calc']].tail(10))
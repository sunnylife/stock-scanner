import json
import requests
from datetime import datetime

def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('api_keys', {}).get('longport', {})
    except:
        return None

def diagnostic_test():
    cfg = load_config()
    if not cfg: return
    
    headers = {
        "x-api-key": cfg.get('app_key'),
        "x-access-token": cfg.get('access_token'),
        "Accept": "application/json"
    }

    # 大陆接入点
    host = "https://openapi.longportapp.cn"
    
    # 测试三个不同维度的基础接口
    tests = [
        {
            "name": "实时报价 (Real-time Quote)",
            "url": f"{host}/v1/quote/realtime",
            "params": {"symbol": "1810.HK"}
        },
        {
            "name": "证券基础信息 (Static Info)",
            "url": f"{host}/v1/asset/quote/stock/list",
            "params": {"symbol": "1810.HK"}
        },
        {
            "name": "账户资产 (Account Balance - 交易类)",
            "url": f"{host}/v1/asset/account/balance",
            "params": {}
        }
    ]

    print(f"🔍 正在诊断长桥网关: {host}\n")

    for t in tests:
        print(f"--- 正在测试: {t['name']} ---")
        try:
            resp = requests.get(t['url'], headers=headers, params=t['params'], timeout=10)
            print(f"HTTP 状态码: {resp.status_code}")
            print(f"响应内容: {resp.text}")
        except Exception as e:
            print(f"网络异常: {e}")
        print("\n")

if __name__ == "__main__":
    diagnostic_test()
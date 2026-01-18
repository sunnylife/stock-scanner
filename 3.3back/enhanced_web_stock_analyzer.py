"""
Web版增强分析系统 - 支持AI流式输出 + gm分析
基于最新 stock_analyzer.py 修正版本，新增AI流式返回功能和gm支持
支持市场：A、g、m
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import json
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import time
import yfinance as yf
import random
import requests
import pandas_datareader.data as web
import pandas as pd
import openai
# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志 - 只输出到命令行
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 只保留命令行输出
    ]
)

class EnhancedWebStockAnalyzer:
    """增强版Web分析器（支持A/g/m + AI流式输出）"""
    
    def __init__(self, config_file='config_back.json'):
        """初始化分析器"""
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.last_api_request_time = 0
        
        # 加载配置文件
        self.config = self._load_config()
        
        # ✅ 新增：Tavily API Key
        self.tavily_api_key = self.config.get('api_keys', {}).get('tavily', '')  # 从配置文件读取
        if not self.tavily_api_key:
            self.logger.warning("⚠️ 未配置 Tavily API Key，新闻搜索功能将不可用")
        
        # 缓存配置
        cache_config = self.config.get('cache', {})
        self.cache_duration = timedelta(hours=cache_config.get('price_hours', 1))
        self.fundamental_cache_duration = timedelta(hours=cache_config.get('fundamental_hours', 6))
        self.news_cache_duration = timedelta(hours=cache_config.get('news_hours', 2))
        
        self.price_cache = {}
        self.fundamental_cache = {}
        self.news_cache = {}
        
        # 分析权重配置
        weights = self.config.get('analysis_weights', {})
        self.analysis_weights = {
            'technical': weights.get('technical', 0.4),
            'fundamental': weights.get('fundamental', 0.4),
            'sentiment': weights.get('sentiment', 0.2)
        }
        
        # 流式推理配置
        streaming = self.config.get('streaming', {})
        self.streaming_config = {
            'enabled': streaming.get('enabled', True),
            'show_thinking': streaming.get('show_thinking', True),
            'delay': streaming.get('delay', 0.1)
        }
        
        # AI配置
        ai_config = self.config.get('ai', {})
        self.ai_config = {
            'max_tokens': ai_config.get('max_tokens', 4000),
            'temperature': ai_config.get('temperature', 0.7),
            'model_preference': ai_config.get('model_preference', 'openai')
        }
        
        # 分析参数配置
        params = self.config.get('analysis_params', {})
        self.analysis_params = {
            'max_news_count': params.get('max_news_count', 100),
            'technical_period_days': params.get('technical_period_days', 180),
            'financial_indicators_count': params.get('financial_indicators_count', 25)
        }
        
        # 市场配置
        markets = self.config.get('markets', {})
        self.market_config = {
            'a_stock': markets.get('a_stock', {'enabled': True, 'currency': 'CNY', 'timezone': 'Asia/Shanghai'}),
            'hk_stock': markets.get('hk_stock', {'enabled': True, 'currency': 'HKD', 'timezone': 'Asia/Hong_Kong'}),
            'us_stock': markets.get('us_stock', {'enabled': True, 'currency': 'USD', 'timezone': 'America/New_York'})
        }
        
        # API密钥配置
        self.api_keys = self.config.get('api_keys', {})
        
        self.logger.info("增强版Web分析器初始化完成（支持A/g/m + AI流式输出）")
        self._log_config_status()

        # === 新增：初始化本地存储目录 ===
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir =  os.path.join(script_dir, "data_cache")
        self.history_dir = os.path.join(script_dir, "analysis_history")
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            
        self.logger.info(f"📁 本地缓存目录已就绪: {self.cache_dir}, {self.history_dir}")

    def _load_config(self):
        """加载JSON配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"✅ 成功加载配置文件: {self.config_file}")
                return config
            else:
                self.logger.warning(f"⚠️ 配置文件 {self.config_file} 不存在，使用默认配置")
                default_config = self._get_default_config()
                self._save_config(default_config)
                return default_config
                
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ 配置文件格式错误: {e}")
            self.logger.info("使用默认配置并备份错误文件")
            
            if os.path.exists(self.config_file):
                backup_name = f"{self.config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.config_file, backup_name)
                self.logger.info(f"错误配置文件已备份为: {backup_name}")
            
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
            
        except Exception as e:
            self.logger.error(f"❌ 加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self):
        """获取增强版默认配置（支持gm）"""
        return {
            "api_keys": {
                "openai": "",
                "anthropic": "",
                "zhipu": "",
                "notes": "请填入您的API密钥"
            },
            "ai": {
                "model_preference": "openai",
                "models": {
                    "openai": "gpt-4o-mini",
                    "anthropic": "claude-3-haiku-20240307",
                    "zhipu": "chatglm_turbo"
                },
                "max_tokens": 4000,
                "temperature": 0.7,
                "api_base_urls": {
                    "openai": "https://api.openai.com/v1",
                    "notes": "如使用中转API，修改上述URL"
                }
            },
            "analysis_weights": {
                "technical": 0.4,
                "fundamental": 0.4,
                "sentiment": 0.2,
                "notes": "权重总和应为1.0"
            },
            "cache": {
                "price_hours": 1,
                "fundamental_hours": 6,
                "news_hours": 2
            },
            "streaming": {
                "enabled": True,
                "show_thinking": False,
                "delay": 0.05
            },
            "analysis_params": {
                "max_news_count": 100,
                "technical_period_days": 180,
                "financial_indicators_count": 25
            },
            "markets": {
                "a_stock": {
                    "enabled": True,
                    "currency": "CNY",
                    "timezone": "Asia/Shanghai",
                    "trading_hours": "09:30-15:00",
                    "notes": "中国A市场"
                },
                "hk_stock": {
                    "enabled": True,
                    "currency": "HKD", 
                    "timezone": "Asia/Hong_Kong",
                    "trading_hours": "09:30-16:00",
                    "notes": "香g市场"
                },
                "us_stock": {
                    "enabled": True,
                    "currency": "USD",
                    "timezone": "America/New_York", 
                    "trading_hours": "09:30-16:00",
                    "notes": "m国市场"
                }
            },
            "web_auth": {
                "enabled": False,
                "password": "",
                "session_timeout": 3600,
                "notes": "Web界面密码鉴权配置"
            },
            "_metadata": {
                "version": "3.1.0-multi-market-streaming",
                "created": datetime.now().isoformat(),
                "description": "增强版AI分析系统配置文件（支持A/g/m + AI流式输出）"
            }
        }

    def _save_config(self, config):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ 配置文件已保存: {self.config_file}")
        except Exception as e:
            self.logger.error(f"❌ 保存配置文件失败: {e}")

    def _log_config_status(self):
        """记录配置状态"""
        self.logger.info("=== 增强版系统配置状态（支持A/g/m + AI流式输出）===")
        
        # 检查API密钥状态
        available_apis = []
        for api_name, api_key in self.api_keys.items():
            if api_name != 'notes' and api_key and api_key.strip():
                available_apis.append(api_name)
        
        if available_apis:
            self.logger.info(f"🤖 可用AI API: {', '.join(available_apis)}")
            primary = self.config.get('ai', {}).get('model_preference', 'openai')
            self.logger.info(f"🎯 主要API: {primary}")
            self.logger.info(f"🌊 AI流式输出: 支持")
            
            # 显示自定义配置
            api_base = self.config.get('ai', {}).get('api_base_urls', {}).get('openai')
            if api_base and api_base != 'https://api.openai.com/v1':
                self.logger.info(f"🔗 自定义API地址: {api_base}")
        else:
            self.logger.warning("⚠️ 未配置任何AI API密钥")
        
        # 检查市场支持
        enabled_markets = []
        for market, config in self.market_config.items():
            if config.get('enabled', True):
                enabled_markets.append(market.upper().replace('_', ''))
        
        self.logger.info(f"🌍 支持市场: {', '.join(enabled_markets)}")
        
        self.logger.info(f"📊 财务指标数量: {self.analysis_params['financial_indicators_count']}")
        self.logger.info(f"📰 最大新闻数量: {self.analysis_params['max_news_count']}")
        self.logger.info(f"📈 技术分析周期: {self.analysis_params['technical_period_days']} 天")
        
        # 检查Web鉴权配置
        web_auth = self.config.get('web_auth', {})
        if web_auth.get('enabled', False):
            self.logger.info(f"🔐 Web鉴权: 已启用")
        else:
            self.logger.info(f"🔓 Web鉴权: 未启用")
        
        self.logger.info("=" * 50)

    def search_market_news(self, query):
        """
        使用 Tavily 搜索实时新闻
        """
        if not self.tavily_api_key:
            return "未配置搜索API，跳过新闻分析"
            
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",  # basic 速度快，advanced 更深
                "include_answer": True,   # 让 Tavily 直接生成答案摘要
                "max_results": 3
            }
            # 简单的 requests 调用，不依赖额外库
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # 优先返回 Tavily 生成的直接答案，如果没有则拼接摘要
                return data.get("answer") or " ".join([r['content'][:200] for r in data.get('results', [])])
            else:
                return f"搜索失败: {response.status_code}"
        except Exception as e:
            self.logger.error(f"Tavily 搜索异常: {e}")
            return "搜索服务不可用"

    def get_llm_trade_decision(self, stock_code, date_str, price_info, tech_indicators, money_flow):
        """
        专门为回测设计的轻量级决策函数
        只输入纯数据，要求返回 JSON 格式的买卖指令
        """
        api_key = self.config.get('api_keys', {}).get('openai')
        api_base = self.config.get('ai', {}).get('api_base_urls', {}).get('openai')
        try:
            client = openai.OpenAI(api_key=api_key)
            if api_base:
                client.base_url = api_base

            model_name = self.config.get('ai', {}).get('models', {}).get('openai', 'gpt-4o-mini')
            max_tokens = self.config.get('ai', {}).get('max_tokens', 6000)

            # 关键修复：确保 tech_indicators 包含 ma20_slope
            # 如果传入的 tech_indicators 没有，尝试计算
            if 'ma20_slope' not in tech_indicators:
                # 这里需要传入原始的 df_slice 来计算，你需要调整函数参数
                # 最简单的方式：在调用此函数前，确保计算并传入了 ma20_slope
                # 提示：你需要在 auto_backtest_runner.py 中计算并传入该值
                pass  # 此处先留空，我们会在下面说明如何修改调用方
            
            # 确保所有用于格式化的值都是标量
            def get_scalar_value(value, default=0):
                """将可能的Series或其他类型转换为标量"""
                if hasattr(value, 'iloc'):  # 如果是Series
                    return value.iloc[-1] if len(value) > 0 else default
                return value if value is not None else default
            
            # 1. 安全提取标量数据 (防止报错的核心步骤)
            def get_val(data, key, default=0):
                val = data.get(key, default)
                if hasattr(val, 'iloc'): return val.iloc[-1]
                return val
            # 提取标量值
            # --- 基础数据 ---
            close = get_val(price_info, 'close')
            change_pct = get_val(price_info, 'change_pct')
            vol_ratio = get_val(price_info, 'vol_ratio', 1.0)
            money_flow_score = get_val(money_flow, 'money_flow_score', 50)
            
            # ✅ 提取传入的新闻摘要
            news_summary = price_info.get('news_summary', '暂无新闻')
            
            # --- 原有指标 ---
            ma20 = get_val(tech_indicators, 'ma20')
            ma20_slope = get_val(tech_indicators, 'ma20_slope', 0)
            rsi = get_val(tech_indicators, 'rsi', 50)
            
            # --- [新增] 增强指标 (Step 1 计算出来的) ---
            k_value = get_val(tech_indicators, 'kdj_k', 50)
            d_value = get_val(tech_indicators, 'kdj_d', 50)
            kdj_signal = tech_indicators.get('kdj_signal', '中性')
            
            wr = get_val(tech_indicators, 'wr', 50)
            atr = get_val(tech_indicators, 'atr', 0)
            atr_percent = get_val(tech_indicators, 'atr_percent', 0)
            
            # 👇👇👇 [新增] 提取区间位置 👇👇👇
            pp_20d = get_val(tech_indicators, 'price_position_20d', 50)

            resonance = tech_indicators.get('resonance_signals', [])
            resonance_str = " + ".join(resonance) if resonance else "无明显共振"

            #提取传入的新闻摘要
            news_summary = price_info.get('news_summary', '暂无新闻')
            # ====================================================
            # 2. 构建 Prompt (这是"换脑"部分：让AI知道新指标)
            # ====================================================
            # ====================================================
        # 2. 构建自主决策型 Prompt
        # ====================================================
            prompt = f"""
            你是一位专业的**证券市场分析师**，擅长结合**市场舆情**与**技术指标**进行趋势量化分析。
            请基于以下数据，进行**模拟盘**的趋势研判。注意：这是单纯的数据分析任务，不涉及真实资金操作。

            [核心数据档案]
            - 标的: {stock_code} (日期: {date_str})
            - 📰 **舆情摘要**: {news_summary}  <-- (请评估此信息对短期情绪的影响)
            
            [技术面盘口]
            - 价格形态: 现价 {close} (涨跌: {change_pct:.2f}%) | 波动率(ATR): {atr:.3f}
            - 趋势状态: 20日线斜率 {ma20_slope:.4f} ({'向上' if ma20_slope > 0.001 else '走平' if ma20_slope > -0.001 else '向下'})
            - 资金位置: 相对位置 {pp_20d:.1f}% (0=底/100=顶) | 资金强度 {money_flow_score:.1f}/100 | 量比 {vol_ratio:.2f}

            [量化指标参考]
            - RSI(6): {rsi:.1f} (注：>85超买，<20超卖)
            - KDJ(9,3,3): K={k_value:.1f}, D={d_value:.1f}
            - 威廉WR(14): {wr:.1f}
            - ★信号共振★: {resonance_str}

            [分析任务]
            1. **舆情评估**: 
               - 是否有重大黑天鹅？(如造假、退市风险) -> 也就是 "SELL" 信号。
               - 是否利好兑现？
            
            2. **趋势辨析**: 
               - 相对位置 > 90% 时，是主升浪还是诱多？
               - RSI > 85 时，警惕回调风险。

            [最终研判]
            请综合上述信息，给出模拟交易建议。
            - **BUY**: 胜率高，盈亏比合适（如：底部启动、强势突破）。
            - **HOLD**: 趋势不明朗，或已有持仓建议继续持有。
            - **SELL**: 趋势走坏，或见顶风险大，或基本面恶化。

            [输出要求]
            请仅返回一个标准的 JSON 对象（不要Markdown格式，不要 ```json 包裹）：
            {{
                "market_phase": "当前状态(如：底部反转/主升浪/高位震荡/阴跌)",
                "action": "BUY", "HOLD", 或 "SELL",
                "confidence": 0到100的整数,
                "reason": "简要分析逻辑（如：虽然有利好，但RSI超买，建议观望）",
                "risk_warning": "主要风险点"
            }}
            """

            print(f"   🤖 调用AI模型: {model_name}, API: {api_base or '官方'}")
            # 关键修改：捕获原始响应
            raw_response = None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1, # 零温度，保证结果稳定
                    response_format={"type": "json_object"}
                )
                raw_response = response.choices[0].message.content
                print(f"   📥 AI原始响应: {raw_response[:100]}...")  # 打印前100字符
                # 尝试解析JSON
                try:
                    # 关键修复：移除Markdown代码块标记
                    cleaned_response = raw_response.strip()
                    
                    # 如果响应以 ```json 开头，去除这个标记
                    if cleaned_response.startswith('```json'):cleaned_response = cleaned_response[7:]  # 移除 ```json
                    if cleaned_response.startswith('```'):cleaned_response = cleaned_response[3:]  # 移除 ```
                    # 如果响应以 ``` 结尾，去除这个标记
                    if cleaned_response.endswith('```'):cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()

                    # result = json.loads(cleaned_response)
                    
                    # 2. 尝试解析 (这是你要查错的关键位置)
                    try:
                        result = json.loads(cleaned_response, strict=False)
                        # print(f"   ✅ AI决策: {result.get('action')} (置信度: {result.get('confidence')})")
                        # 验证必要的字段
                        if 'action' not in result:
                            result['action'] = 'HOLD'
                        if 'confidence' not in result:
                            result['confidence'] = 50
                        if 'reason' not in result:
                            result['reason'] = '默认'
                        return result
                    
                    except json.JSONDecodeError as json_err:
                        # ====================================================
                        # 🕵️‍♂️ 侦探模式：这里会把导致错误的罪魁祸首打印出来
                        # ====================================================
                        print("\n" + "!"*60)
                        print(f"❌ [JSON 解析崩溃现场调查]")
                        print(f"📉 股票代码: {stock_code} | 日期: {date_str}")
                        print(f"🐛 错误信息: {json_err}")
                        print("-" * 30)
                        print("📄 1. 原始返回 (Raw Response):")
                        # 使用 repr() 可以把换行符 \n 显示出来，方便看哪里断行了
                        print(repr(raw_response)) 
                        print("-" * 30)
                        print("🧹 2. 清理后文本 (Cleaned Content):")
                        print(repr(cleaned_response))
                        print("!"*60 + "\n")
                        
                        # 可以在这里暂时返回 HOLD，防止程序中断，让你能继续看后面的日志
                        return {"action": "HOLD", "confidence": 0, "reason": "JSON_DEBUG_ERROR"}

                             
                    print(f"   ✅ AI决策: {result.get('action')} (置信度: {result.get('confidence')})")
                    return result
                    
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON解析失败！清理后响应: {cleaned_response[:200]}")
                    print(f"   错误详情: {e}")
                    # 尝试更激进的清理
                    try:
                        # 尝试找到第一个 { 和最后一个 }
                        start_idx = raw_response.find('{')
                        end_idx = raw_response.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = raw_response[start_idx:end_idx]
                            result = json.loads(json_str)
                            print(f"   🔧 二次解析成功: {result.get('action')}")
                            return result
                    except:
                        pass
                    return {"action": "HOLD", "confidence": 0, "reason": "JSON解析失败"}
                    
            except openai.APIConnectionError as e:
                print(f"   ❌ API连接错误: {e}")
                return {"action": "HOLD", "confidence": 0, "reason": "API连接失败"}
            except openai.APIError as e:
                print(f"   ❌ API错误: {e}")
                return {"action": "HOLD", "confidence": 0, "reason": "API错误"}
            except Exception as e:
                print(f"   ❌ 未知AI调用错误: {e}")
                return {"action": "HOLD", "confidence": 0, "reason": "AI调用异常"}

        except Exception as e:
            print(f"❌ get_llm_trade_decision 整体错误: {e}")
            return {"action": "HOLD", "confidence": 0, "reason": "函数执行异常"}

    def detect_market(self, stock_code):
        """检测所属市场"""
        stock_code = stock_code.strip().upper()
        
        # A检测（6位数字）
        if re.match(r'^\d{6}$', stock_code):
            return 'a_stock'
        
        # g检测（5位数字，通常以0开头）
        elif re.match(r'^\d{5}$', stock_code):
            return 'hk_stock'
        
        # g检测（带HK前缀）
        elif re.match(r'^HK\d{5}$', stock_code):
            return 'hk_stock'
        
        # m检测（字母代码）
        elif re.match(r'^[A-Z]{1,5}$', stock_code):
            return 'us_stock'
        
        # 默认返回A
        else:
            self.logger.warning(f"⚠️ 无法识别代码格式: {stock_code}，默认为A")
            return 'a_stock'

    def normalize_stock_code(self, stock_code, market=None):
        """标准化代码"""
        stock_code = stock_code.strip().upper()
        
        if market is None:
            market = self.detect_market(stock_code)
        
        if market == 'hk_stock':
            # 移除HK前缀（如果有）
            if stock_code.startswith('HK'):
                stock_code = stock_code[2:]
            # g代码补零到5位
            if len(stock_code) < 5:
                stock_code = stock_code.zfill(5)
        
        return stock_code, market

    def _wait_for_rate_limit(self, min_interval=2.0):
        """强制网络请求间隔（秒）"""
        elapsed = time.time() - self.last_api_request_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            # self.logger.info(f"⏳ 触发频控，等待 {sleep_time:.2f} 秒...")
            time.sleep(sleep_time)
        self.last_api_request_time = time.time()

    def _get_yf_session(self):
        """获取带伪装头的 Session，防止 Yahoo 403/429"""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        return session

    def _safe_yf_download(self, ticker, period="1y", start=None):
        """
        安全的 Yahoo 下载包装器 (适配新版 yfinance)
        去除手动 session，保留限流和重试机制
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. 依然保留随机限流，这是防止封IP最有效的手段
                self._wait_for_rate_limit(min_interval=random.uniform(2.0, 4.0)) 
                
                # 2. 调用 yfinance (注意：不再传递 session 参数！)
                # 新版 yfinance 会自动处理 curl_cffi 伪装
                if start:
                    # start如果是datetime对象，yfinance能直接识别
                    df = yf.download(ticker, start=start, progress=False, timeout=20)
                else:
                    df = yf.download(ticker, period=period, progress=False, timeout=20)
                
                # 3. 数据有效性检查
                if not df.empty:
                    return df
                    
            except Exception as e:
                # 捕获错误并等待
                wait = (attempt + 1) * 3
                # 只有当不是最后一次重试时才打印警告，避免刷屏
                if attempt < max_retries - 1:
                    self.logger.warning(f"Yahoo下载重试({ticker}): {str(e)[:50]}... 等待{wait}秒")
                time.sleep(wait)
        
        return pd.DataFrame()

    def get_stock_data(self, stock_code, period='1y'):
        """
        获取股票数据 [终极稳定版：集成 Baostock]
        优先级: Akshare -> Baostock -> Yahoo
        """
        # 1. 标准化代码
        stock_code, market = self.normalize_stock_code(stock_code)
        cache_key = f"{market}_{stock_code}"
        # --- 第一层：本地文件缓存 ---
        today_str = datetime.now().strftime('%Y%m%d')
        cache_filename = f"{market}_{stock_code}_{today_str}.csv"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        

        # 检查缓存 (1小时内有效)
        if os.path.exists(cache_path):
            try:
                if (os.path.getsize(cache_path) > 100) and \
                   (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path)) < timedelta(hours=1)):
                    df = pd.read_csv(cache_path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    df = self._standardize_price_data_columns(df, market)
                    if not df.empty and 'close' in df.columns:
                        self.logger.info(f"📦 命中本地缓存: {cache_filename}")
                        return df
            except:
                pass # 缓存读取失败直接跳过
        
        # 3. === 核心修复：定义时间范围 ===
        end_date_dt = datetime.now()
        
        # 解析 period (例如 '1y', '6mo', '5d')
        days = 365 # 默认
        try:
            if isinstance(period, str):
                if period.endswith('y'):
                    days = int(period[:-1]) * 365
                elif period.endswith('mo'):
                    days = int(period[:-2]) * 30
                elif period.endswith('d'):
                    days = int(period[:-1])
        except:
            days = 365
            
        # 关键修复：在这里明确定义 start_date_dt
        start_date_dt = end_date_dt - timedelta(days=days)
        
        start_date_str = start_date_dt.strftime('%Y-%m-%d')
        end_date_str = end_date_dt.strftime('%Y-%m-%d')
        # =================================
        # --- 第二层：网络请求 ---
        self.logger.info(f"🌐 正在下载 {market.upper()} {stock_code}...")
        
        import akshare as ak
        import baostock as bs  # 引入新援
        import yfinance as yf
        
        stock_data = pd.DataFrame()
        end_date = datetime.now().strftime('%Y%m%d')
        days = self.analysis_params.get('technical_period_days', 365)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

        try:
            # ================= A股逻辑 =================
            if market == 'a_stock':
                
                # --- 方案 A: Akshare (前复权) ---
                try:
                    # 快速尝试一下，不行立刻跳过，不纠结
                    stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                except:
                    pass

                # --- 方案 B: Baostock (核心救火队员) ---
                if stock_data.empty:
                    try:
                        # 格式化代码: 600036 -> sh.600036
                        if stock_code.startswith('6'):
                            bs_code = f"sh.{stock_code}"
                        elif stock_code.startswith('0') or stock_code.startswith('3'):
                            bs_code = f"sz.{stock_code}"
                        elif stock_code.startswith('8') or stock_code.startswith('4'):
                            bs_code = f"bj.{stock_code}"
                        else:
                            bs_code = f"sh.{stock_code}" # 默认

                        self.logger.info(f"🛡️ 启用 Baostock 下载: {bs_code}")
                        
                        # 1. 登录
                        bs.login() # 不打印登录信息
                        
                        # 2. 下载 (日线, 前复权=2)
                        rs = bs.query_history_k_data_plus(
                            bs_code,
                            "date,open,high,low,close,volume,amount,pctChg",
                            start_date=start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:], # YYYY-MM-DD
                            end_date=end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:],
                            frequency="d", adjustflag="2"
                        )
                        
                        # 3. 转 DataFrame
                        data_list = []
                        while (rs.error_code == '0') & rs.next():
                            data_list.append(rs.get_row_data())
                        
                        if data_list:
                            stock_data = pd.DataFrame(data_list, columns=rs.fields)
                            # Baostock 返回全是字符串，需要转换
                            stock_data['date'] = pd.to_datetime(stock_data['date'])
                            cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
                            for col in cols:
                                if col in stock_data.columns:
                                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                            
                            # 映射列名以适配系统 (pctChg -> change_pct)
                            stock_data = stock_data.rename(columns={'pctChg': 'change_pct'})
                            
                        # 4. 登出
                        bs.logout()
                        
                    except Exception as e:
                        self.logger.warning(f"Baostock 下载异常: {e}")
                        try: bs.logout() 
                        except: pass

                # --- 方案 C: Yahoo (最后的挣扎，带错误捕获) ---
                if stock_data.empty:
                    try:
                        self.logger.info("⚠️ 尝试 Yahoo Finance (可能被限流)...")
                        yf_code = f"{stock_code}.SS" if stock_code.startswith('6') else f"{stock_code}.SZ"
                        # 强制等待一下
                        time.sleep(random.uniform(1.0, 3.0))
                        # 不带 Session，让它自己处理
                        df = yf.download(yf_code, start=datetime.now()-timedelta(days=days), progress=False, timeout=10)
                        if not df.empty:
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            df.columns = df.columns.str.lower()
                            stock_data = df.reset_index()
                    except Exception as e:
                        self.logger.error(f"Yahoo 彻底失败: {str(e)[:50]}")

                # --- 方案 D: 只有实时数据 (最后的兜底) ---
                if stock_data.empty:
                    try:
                        spot_df = ak.stock_zh_a_spot_em()
                        row = spot_df[spot_df['代码'] == stock_code]
                        if not row.empty:
                            self.logger.warning("⚠️ 仅获取到今日实时数据，历史数据缺失")
                            # 构造单行数据
                            stock_data = pd.DataFrame({
                                'date': [datetime.now().strftime('%Y-%m-%d')],
                                'open': [float(row.iloc[0]['今开'])],
                                'close': [float(row.iloc[0]['最新价'])],
                                'high': [float(row.iloc[0]['最高'])],
                                'low': [float(row.iloc[0]['最低'])],
                                'volume': [float(row.iloc[0]['成交量'])],
                                'change_pct': [float(row.iloc[0]['涨跌幅'])]
                            })
                    except:
                        pass

            # ================= 港股逻辑 =================
            elif market == 'hk_stock':
                # 1. Akshare
                try:
                    self._wait_for_rate_limit(1.0)
                    stock_data = ak.stock_hk_hist(symbol=stock_code, period="daily", start_date=start_date_str, end_date=end_date, adjust="qfq")
                except:
                    pass
                
                # 2. Yahoo (安全模式)
                if stock_data.empty:
                    yf_code = f"{int(stock_code):04d}.HK"
                    stock_data = self._safe_yf_download(yf_code, start=start_date_dt)
                    if not stock_data.empty:
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            stock_data.columns = stock_data.columns.get_level_values(0)
                        stock_data.columns = stock_data.columns.str.lower()

            # ================= 美股逻辑 =================
            # === 美股 (Stooq源) ===
            elif market == 'us_stock':
                try:
                    start_dt = datetime.now() - timedelta(days=days)
                    df = web.DataReader(stock_code, 'stooq', start=start_dt, end=datetime.now())
                    
                    if df is not None and not df.empty:
                        # ⚠️ 关键：Stooq 默认是倒序(新->旧)，必须转为正序(旧->新)
                        df = df.sort_index(ascending=True).reset_index()
                        
                        # 重命名列以匹配系统标准
                        df = df.rename(columns={
                            "Date": "date", 
                            "Open": "open", 
                            "High": "high", 
                            "Low": "low", 
                            "Close": "close", 
                            "Volume": "volume"
                        })
                        
                        # 设置日期索引
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # 补充 change_pct
                        df['change_pct'] = df['close'].pct_change() * 100
                        
                        stock_data = df
                        print("   ✅ Stooq 源下载成功")
                    else:
                        print("   ❌ Stooq 返回空数据")
                except Exception as e:
                    self.logger.error(f"Stooq 获取失败: {e}")

            # --- 最终处理 ---
            if stock_data is None or stock_data.empty:
                raise ValueError(f"所有数据源均无法获取 {stock_code}")

            stock_data = self._standardize_price_data_columns(stock_data, market)

            if 'date' in stock_data.columns and stock_data.index.name != 'date':
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                stock_data.set_index('date', inplace=True)

            # 保存缓存
            self.price_cache[cache_key] = (datetime.now(), stock_data)
            try:
                stock_data.to_csv(cache_path)
            except:
                pass

            # === [新增] 数据质量清洗 ===
            if not stock_data.empty:
                # 1. 填充空值 (用前一天的数据填)
                stock_data = stock_data.fillna(method='ffill')
                
                # 2. 异常值处理 (如成交量为0但价格在动的诡异数据)
                if 'volume' in stock_data.columns:
                    stock_data['volume'] = stock_data['volume'].replace(0, np.nan).fillna(method='ffill')
                
                # 3. 确保数值类型
                cols = ['open', 'close', 'high', 'low', 'volume']
                for c in cols:
                    if c in stock_data.columns:
                        stock_data[c] = pd.to_numeric(stock_data[c], errors='coerce')
            # ==========================
            return stock_data

        except Exception as e:
            self.logger.error(f"❌ 获取数据失败 {stock_code}: {e}")
            return pd.DataFrame()

    def _standardize_price_data_columns(self, stock_data, market):
        """标准化价格数据列名 (基于列名映射，稳健版)"""
        try:
            # 记录原始列名以供调试
            self.logger.info(f"处理前列名: {list(stock_data.columns)}")

            # 通用中文列名映射（覆盖A股、港股中文列，避免重复代码）
            chinese_common_map = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate'
            }

            # === A股处理逻辑 ===
            if market == 'a_stock':
                # 直接使用通用中文映射，精准替换
                stock_data = stock_data.rename(columns=chinese_common_map)

            # === 港股处理逻辑 (修复重点) ===
            elif market == 'hk_stock':
                # 1. 修正映射关系：根据日志分析，extra_1 才是涨跌幅
                hk_rename_map = {
                    **chinese_common_map,
                    'extra_0': 'turnover',   # 成交额
                    'extra_1': 'change_pct', # 修正：这里是涨跌幅 (-0.69)
                    'extra_2': 'change_amount', # 修正：这里是涨跌额 (-0.18)
                    'extra_3': 'amplitude'    # 推测
                }
                
                # 2. 执行重命名
                stock_data = stock_data.rename(columns=hk_rename_map)

            # === 美股处理逻辑 ===
            elif market == 'us_stock':
                stock_data.columns = stock_data.columns.str.lower()
                stock_data = stock_data.rename(columns=chinese_common_map)
                # 兜底映射
                if 'close' not in stock_data.columns and len(stock_data.columns) >= 6:
                    cols = ['date', 'open', 'close', 'high', 'low', 'volume']
                    remain = [f'extra_{i}' for i in range(len(stock_data.columns)-6)]
                    stock_data.columns = cols + remain

            # === 通用数据清洗（必做，三市场共用） ===
            # 1. 处理时间索引（确保index为datetime类型）
            if 'date' in stock_data.columns:
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                stock_data = stock_data.set_index('date')
            elif stock_data.index.name != 'date':
                try:
                    stock_data.index = pd.to_datetime(stock_data.index)
                    stock_data.index.name = 'date'
                except:
                    self.logger.warning("⚠️ 时间索引转换失败，可能影响回测")

            # 2. 强制转换核心数值列（避免字符串干扰计算）
            core_numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'change_pct', 'turnover', 'amplitude']
            for col in core_numeric_cols:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').fillna(0)

            # 3. 确保change_pct字段必存在（兜底逻辑，防止缺失）
            if 'change_pct' not in stock_data.columns:
                if 'close' in stock_data.columns and len(stock_data) >= 2:
                    stock_data['change_pct'] = stock_data['close'].pct_change() * 100
                else:
                    stock_data['change_pct'] = 0

            self.logger.info(f"处理后列名: {list(stock_data.columns)}")
            return stock_data
                
        except Exception as e:
            self.logger.error(f"列名标准化失败: {e}")
            return stock_data

    def get_comprehensive_fundamental_data(self, stock_code):
        """获取综合财务指标数据（支持多市场）"""
        stock_code, market = self.normalize_stock_code(stock_code)

        # === 1. 生成缓存文件名 ===
        # 按月缓存基本面 (因为财报更新慢，没必要每天下)
        month_str = datetime.now().strftime('%Y%m') 
        cache_filename = f"fund_{market}_{stock_code}_{month_str}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        # === 2. 检查本地文件 ===
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"📦 命中基本面文件缓存: {cache_filename}")
                return data
            except Exception as e:
                self.logger.warning(f"读取基本面缓存失败: {e}")


        cache_key = f"{market}_{stock_code}"
        
        if cache_key in self.fundamental_cache:
            cache_time, data = self.fundamental_cache[cache_key]
            if datetime.now() - cache_time < self.fundamental_cache_duration:
                self.logger.info(f"使用缓存的基本面数据: {cache_key}")
                return data
        
        try:
            import akshare as ak
            
            fundamental_data = {}
            self.logger.info(f"开始获取 {market.upper()} {stock_code} 的综合财务指标...")
            
            if market == 'a_stock':
                fundamental_data = self._get_a_stock_fundamental_data(stock_code)
            elif market == 'hk_stock':
                fundamental_data = self._get_hk_stock_fundamental_data(stock_code)
            elif market == 'us_stock':
                fundamental_data = self._get_us_stock_fundamental_data(stock_code)
            # === 4. 保存到硬盘 ===
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(fundamental_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"💾 基本面数据已缓存至: {cache_path}")
            except Exception as e:
                self.logger.error(f"写入基本面缓存失败: {e}")
            # 缓存数据
            self.fundamental_cache[cache_key] = (datetime.now(), fundamental_data)
            self.logger.info(f"✓ {market.upper()} {stock_code} 综合基本面数据获取完成并已缓存")
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"获取综合基本面数据失败: {str(e)}")
            return {
                'basic_info': {},
                'financial_indicators': {},
                'valuation': {},
                'performance_forecast': [],
                'dividend_info': [],
                'industry_analysis': {}
            }

    def _get_a_stock_fundamental_data(self, stock_code):
        """获取A股基本面数据 - 修复版"""
        import akshare as ak
        
        fundamental_data = {}
        
        # 1. 基本信息
        try:
            self.logger.info("正在获取A股基本信息...")
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            # 这里的 stock_info 返回的是 DataFrame，需要转 dict
            # DataFrame 结构通常是 item, value 两列
            info_dict = dict(zip(stock_info['item'], stock_info['value']))
            fundamental_data['basic_info'] = info_dict
            self.logger.info("✓ A股基本信息获取成功")
        except Exception as e:
            self.logger.warning(f"获取A股基本信息失败: {e}")
            fundamental_data['basic_info'] = {}
        
        # 2. 财务指标
        try:
            self.logger.info("正在获取A股财务指标...")
            financial_indicators = self._get_a_stock_financial_indicators(stock_code)
            fundamental_data['financial_indicators'] = financial_indicators
        except Exception as e:
            self.logger.warning(f"获取A股财务指标失败: {e}")
            fundamental_data['financial_indicators'] = {}
        
        # 3. 估值指标 (修复点：替换失效接口 stock_a_indicator_lg)
        try:
            # 使用百度接口获取个股估值，包含PE, PB, 市值等
            # 优先检查接口是否存在（防止版本过低报错）
            if hasattr(ak, 'stock_zh_a_valuation_baidu'):
                valuation_data = ak.stock_zh_a_valuation_baidu(symbol=stock_code)
                if not valuation_data.empty:
                    latest_valuation = valuation_data.iloc[-1].to_dict()
                    fundamental_data['valuation'] = self._clean_financial_data({
                        '市盈率(TTM)': latest_valuation.get('pe_ttm'),
                        '市净率': latest_valuation.get('pb'),
                        '股息率': latest_valuation.get('dividend_yield'),
                        '总市值': latest_valuation.get('total_market_cap')
                    })
                    self.logger.info("✓ A股估值指标获取成功")
                else:
                    fundamental_data['valuation'] = {}
            else:
                # 备用方案：如果新接口不存在，尝试从基本信息里找（旧版本兼容）
                self.logger.warning("AkShare版本较低，使用备用估值获取方式")
                if 'basic_info' in fundamental_data:
                    info = fundamental_data['basic_info']
                    fundamental_data['valuation'] = {
                        '市盈率(TTM)': info.get('市盈率-动态'), # 只有部分接口有
                        '市净率': info.get('市净率'),
                        '总市值': info.get('总市值')
                    }
        except Exception as e:
            # 降级为 DEBUG 级别日志，避免刷屏吓人，因为这不是致命错误
            self.logger.debug(f"A股估值指标获取受限: {e} (已跳过)")
            fundamental_data['valuation'] = {}
        
        # 4. 业绩预告
        try:
            performance_forecast = ak.stock_yjbb_em(symbol=stock_code)
            if not performance_forecast.empty:
                fundamental_data['performance_forecast'] = performance_forecast.head(10).to_dict('records')
        except Exception as e:
            fundamental_data['performance_forecast'] = []
        
        # 5. 分红信息
        try:
            dividend_info = ak.stock_fhpg_em(symbol=stock_code)
            if not dividend_info.empty:
                fundamental_data['dividend_info'] = dividend_info.head(10).to_dict('records')
        except Exception as e:
            fundamental_data['dividend_info'] = []
        
        # 6. 行业分析
        fundamental_data['industry_analysis'] = self._get_industry_analysis(stock_code, 'a_stock')
        
        return fundamental_data

    def _get_hk_stock_fundamental_data(self, stock_code):
        """获取g基本面数据"""
        import akshare as ak
        
        fundamental_data = {}
        
        # 1. 基本信息
        try:
            self.logger.info("正在获取g基本信息...")
            # g基本信息
            hk_info = ak.stock_hk_spot_em()
            stock_info = hk_info[hk_info['代码'] == stock_code]
            if not stock_info.empty:
                fundamental_data['basic_info'] = stock_info.iloc[0].to_dict()
            else:
                fundamental_data['basic_info'] = {'代码': stock_code, '市场': 'g'}
            self.logger.info("✓ g基本信息获取成功")
        except Exception as e:
            self.logger.warning(f"获取g基本信息失败: {e}")
            fundamental_data['basic_info'] = {'代码': stock_code, '市场': 'g'}
        
        # 2. 财务指标（g财务数据较少）
        try:
            financial_indicators = {}
            
            # 尝试获取g财务数据
            try:
                hk_financial = ak.stock_hk_valuation_baidu(symbol=stock_code)
                if not hk_financial.empty:
                    latest_data = hk_financial.iloc[-1].to_dict()
                    financial_indicators.update(self._clean_financial_data(latest_data))
            except:
                pass
            
            # 计算基本财务指标
            if financial_indicators:
                core_indicators = self._calculate_hk_financial_indicators(financial_indicators)
                fundamental_data['financial_indicators'] = core_indicators
            else:
                fundamental_data['financial_indicators'] = self._get_default_financial_indicators('g')
                
        except Exception as e:
            self.logger.warning(f"获取g财务指标失败: {e}")
            fundamental_data['financial_indicators'] = self._get_default_financial_indicators('g')
        
        # 3. 估值指标
        fundamental_data['valuation'] = {}
        
        # 4. 业绩预告
        fundamental_data['performance_forecast'] = []
        
        # 5. 分红信息
        fundamental_data['dividend_info'] = []
        
        # 6. 行业分析
        fundamental_data['industry_analysis'] = self._get_industry_analysis(stock_code, 'hk_stock')
        
        return fundamental_data

    def _get_us_stock_fundamental_data(self, stock_code):
        """获取m基本面数据"""
        import akshare as ak
        
        fundamental_data = {}
        
        # 1. 基本信息
        try:
            self.logger.info("正在获取m基本信息...")
            # m基本信息
            us_info = ak.stock_us_spot_em()
            stock_info = us_info[us_info['代码'] == stock_code.upper()]
            if not stock_info.empty:
                fundamental_data['basic_info'] = stock_info.iloc[0].to_dict()
            else:
                fundamental_data['basic_info'] = {'代码': stock_code.upper(), '市场': 'm'}
            self.logger.info("✓ m基本信息获取成功")
        except Exception as e:
            self.logger.warning(f"获取m基本信息失败: {e}")
            fundamental_data['basic_info'] = {'代码': stock_code.upper(), '市场': 'm'}
        
        # 2. 财务指标
        try:
            financial_indicators = {}
            
            # 尝试获取m财务数据
            try:
                us_financial = ak.stock_us_fundamental(symbol=stock_code.upper())
                if not us_financial.empty:
                    latest_data = us_financial.iloc[-1].to_dict()
                    financial_indicators.update(self._clean_financial_data(latest_data))
            except:
                pass
            
            if financial_indicators:
                core_indicators = self._calculate_us_financial_indicators(financial_indicators)
                fundamental_data['financial_indicators'] = core_indicators
            else:
                fundamental_data['financial_indicators'] = self._get_default_financial_indicators('m')
                
        except Exception as e:
            self.logger.warning(f"获取m财务指标失败: {e}")
            fundamental_data['financial_indicators'] = self._get_default_financial_indicators('m')
        
        # 3. 估值指标
        fundamental_data['valuation'] = {}
        
        # 4. 业绩预告
        fundamental_data['performance_forecast'] = []
        
        # 5. 分红信息
        fundamental_data['dividend_info'] = []
        
        # 6. 行业分析
        fundamental_data['industry_analysis'] = self._get_industry_analysis(stock_code, 'us_stock')
        
        return fundamental_data

    def _get_a_stock_financial_indicators(self, stock_code):
        """获取A股详细财务指标 - 增强健壮性版"""
        import akshare as ak
        
        financial_indicators = {}
        
        try:
            # 利润表数据
            income_statement = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
            if income_statement is not None and not income_statement.empty:
                latest_income = income_statement.iloc[0].to_dict()
                financial_indicators.update(latest_income)
        except Exception as e:
            self.logger.warning(f"获取利润表数据失败: {e}")
        
        try:
            # 财务分析指标
            balance_sheet = ak.stock_financial_analysis_indicator(symbol=stock_code)
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[-1].to_dict()
                financial_indicators.update(latest_balance)
        except Exception as e:
            self.logger.warning(f"获取财务分析指标失败: {e}")
        
        try:
            # 现金流量表 (修复点：增加 None 判断)
            cash_flow = ak.stock_cash_flow_sheet_by_report_em(symbol=stock_code)
            if cash_flow is not None and not cash_flow.empty:
                latest_cash = cash_flow.iloc[-1].to_dict()
                financial_indicators.update(latest_cash)
            else:
                self.logger.warning("现金流量表数据为空")
        except Exception as e:
            self.logger.warning(f"获取现金流量表失败: {e}")
        
        # 计算25项核心财务指标
        core_indicators = self._calculate_core_financial_indicators(financial_indicators)
        return core_indicators

    def _calculate_hk_financial_indicators(self, raw_data):
        """计算g财务指标"""
        indicators = {}
        
        def safe_get(key, default=0):
            value = raw_data.get(key, default)
            try:
                if value is None or value == '' or str(value).lower() in ['nan', 'none', '--']:
                    return default
                num_value = float(value)
                if math.isnan(num_value) or math.isinf(num_value):
                    return default
                return num_value
            except (ValueError, TypeError):
                return default
        
        # g基本指标
        indicators['市盈率'] = safe_get('市盈率')
        indicators['市净率'] = safe_get('市净率')
        indicators['息收益率'] = safe_get('息收益率')
        indicators['市值'] = safe_get('市值')
        indicators['流通市值'] = safe_get('流通市值')
        
        # 添加其他默认指标
        for i in range(20):
            key = f'g指标_{i+1}'
            indicators[key] = safe_get(key, 0)
        
        return indicators

    def _calculate_us_financial_indicators(self, raw_data):
        """计算m财务指标"""
        indicators = {}
        
        def safe_get(key, default=0):
            value = raw_data.get(key, default)
            try:
                if value is None or value == '' or str(value).lower() in ['nan', 'none', '--']:
                    return default
                num_value = float(value)
                if math.isnan(num_value) or math.isinf(num_value):
                    return default
                return num_value
            except (ValueError, TypeError):
                return default
        
        # m基本指标
        indicators['PE_Ratio'] = safe_get('PE_Ratio')
        indicators['PB_Ratio'] = safe_get('PB_Ratio')
        indicators['Dividend_Yield'] = safe_get('Dividend_Yield')
        indicators['Market_Cap'] = safe_get('Market_Cap')
        indicators['Revenue'] = safe_get('Revenue')
        indicators['Net_Income'] = safe_get('Net_Income')
        indicators['EPS'] = safe_get('EPS')
        indicators['ROE'] = safe_get('ROE')
        
        # 添加其他默认指标
        for i in range(17):
            key = f'US_Metric_{i+1}'
            indicators[key] = safe_get(key, 0)
        
        return indicators

    def _get_default_financial_indicators(self, market):
        """获取默认财务指标"""
        if market == 'g':
            return {
                '市盈率': 0,
                '市净率': 0,
                '息收益率': 0,
                '市值': 0,
                '数据完整度': '有限'
            }
        elif market == 'm':
            return {
                'PE_Ratio': 0,
                'PB_Ratio': 0,
                'Dividend_Yield': 0,
                'Market_Cap': 0,
                'Data_Completeness': 'Limited'
            }
        else:
            return {}

    def _calculate_core_financial_indicators(self, raw_data):
        """计算25项核心财务指标（A）"""
        try:
            indicators = {}
            
            def safe_get(key, default=0):
                value = raw_data.get(key, default)
                try:
                    if value is None or value == '' or str(value).lower() in ['nan', 'none', '--']:
                        return default
                    num_value = float(value)
                    if math.isnan(num_value) or math.isinf(num_value):
                        return default
                    return num_value
                except (ValueError, TypeError):
                    return default
            
            # 1-5: 盈利能力指标
            indicators['净利润率'] = safe_get('净利润率')
            indicators['净资产收益率'] = safe_get('净资产收益率')
            indicators['总资产收益率'] = safe_get('总资产收益率')
            indicators['毛利率'] = safe_get('毛利率')
            indicators['营业利润率'] = safe_get('营业利润率')
            
            # 6-10: 偿债能力指标
            indicators['流动比率'] = safe_get('流动比率')
            indicators['速动比率'] = safe_get('速动比率')
            indicators['资产负债率'] = safe_get('资产负债率')
            indicators['产权比率'] = safe_get('产权比率')
            indicators['利息保障倍数'] = safe_get('利息保障倍数')
            
            # 11-15: 营运能力指标
            indicators['总资产周转率'] = safe_get('总资产周转率')
            indicators['存货周转率'] = safe_get('存货周转率')
            indicators['应收账款周转率'] = safe_get('应收账款周转率')
            indicators['流动资产周转率'] = safe_get('流动资产周转率')
            indicators['固定资产周转率'] = safe_get('固定资产周转率')
            
            # 16-20: 发展能力指标
            indicators['营收同比增长率'] = safe_get('营收同比增长率')
            indicators['净利润同比增长率'] = safe_get('净利润同比增长率')
            indicators['总资产增长率'] = safe_get('总资产增长率')
            indicators['净资产增长率'] = safe_get('净资产增长率')
            indicators['经营现金流增长率'] = safe_get('经营现金流增长率')
            
            # 21-25: 市场表现指标
            indicators['市盈率'] = safe_get('市盈率')
            indicators['市净率'] = safe_get('市净率')
            indicators['市销率'] = safe_get('市销率')
            indicators['PEG比率'] = safe_get('PEG比率')
            indicators['息收益率'] = safe_get('息收益率')
            
            # 过滤掉无效的指标
            valid_indicators = {k: v for k, v in indicators.items() if v not in [0, None, 'nan']}
            
            self.logger.info(f"✓ 成功计算 {len(valid_indicators)} 项有效财务指标")
            return valid_indicators
            
        except Exception as e:
            self.logger.error(f"计算核心财务指标失败: {e}")
            return {}

    def _clean_financial_data(self, data_dict):
        """清理财务数据中的NaN值"""
        cleaned_data = {}
        for key, value in data_dict.items():
            if pd.isna(value) or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                cleaned_data[key] = None
            else:
                cleaned_data[key] = value
        return cleaned_data

    def _get_industry_analysis(self, stock_code, market):
        """获取行业分析数据（多市场）"""
        try:
            import akshare as ak
            
            industry_data = {}
            
            if market == 'a_stock':
                # A行业分析
                try:
                    industry_info = ak.stock_board_industry_name_em()
                    stock_industry = industry_info[industry_info.iloc[:, 0].astype(str).str.contains(stock_code, na=False)]
                    if not stock_industry.empty:
                        industry_data['industry_info'] = stock_industry.iloc[0].to_dict()
                except Exception as e:
                    self.logger.warning(f"获取A行业信息失败: {e}")
            
            elif market == 'hk_stock':
                # g行业分析
                industry_data['market'] = 'g'
                industry_data['currency'] = 'HKD'
                
            elif market == 'us_stock':
                # m行业分析
                industry_data['market'] = 'm'
                industry_data['currency'] = 'USD'
            
            return industry_data
            
        except Exception as e:
            self.logger.warning(f"行业分析失败: {e}")
            return {'market': market.replace('_', '').upper()}

    def get_comprehensive_news_data(self, stock_code, days=15):
        """获取综合新闻数据（支持多市场）"""
        stock_code, market = self.normalize_stock_code(stock_code)
       # === 1. 生成缓存文件名 (核心修改点) ===
        utc_now = datetime.utcnow()
        beijing_now = utc_now + timedelta(hours=8)
        date_str = beijing_now.strftime('%Y%m%d')
        current_time_str = beijing_now.strftime('%H%M')

        if current_time_str < "1000":
            # 00:00 - 09:59 -> 使用盘前缓存
            period_suffix = "PRE"
        elif current_time_str < "1330":
            # 10:00 - 13:29 -> 使用10点更新的缓存
            period_suffix = "1000"
        else:
            # 13:30 - 23:59 -> 使用13点半更新的缓存
            period_suffix = "1330"
            
        # 文件名示例: news_us_stock_AAPL_20251225_1000.json
        cache_filename = f"news_{market}_{stock_code}_{date_str}_{period_suffix}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # # 判断当前是 上午(AM) 还是 下午(PM)
        # # 0-11点为 AM，12-23点为 PM
        # period_str = "AM" if now.hour < 12 else "PM"
        
        # # 文件名示例: news_us_stock_AAPL_20251225_AM.json
        # # 这样每天 00:00 和 12:00 会各更新一次
        # cache_filename = f"news_{market}_{stock_code}_{date_str}_{period_str}.json"
        # cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # === 2. 检查本地文件 ===
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"📦 命中新闻文件缓存: {cache_filename}")
                return data
            except Exception as e:
                self.logger.warning(f"读取新闻缓存失败: {e}")
        self.logger.info(f"🌐 正在下载 {market.upper()} {stock_code} 的新闻数据...")

        # cache_key = f"{market}_{stock_code}_{days}"
        
        # if cache_key in self.news_cache:
        #     cache_time, data = self.news_cache[cache_key]
        #     if datetime.now() - cache_time < self.news_cache_duration:
        #         self.logger.info(f"使用缓存的新闻数据: {cache_key}")
        #         return data
        
        # self.logger.info(f"开始获取 {market.upper()} {stock_code} 的综合新闻数据（最近{days}天）...")
        
        try:
            import akshare as ak
            
            all_news_data = {
                'company_news': [],
                'announcements': [],
                'research_reports': [],
                'industry_news': [],
                'market_sentiment': {},
                'news_summary': {}
            }
            all_news_data = {}
            if market == 'a_stock':
                all_news_data = self._get_a_stock_news_data(stock_code, days)
            elif market == 'hk_stock':
                all_news_data = self._get_hk_stock_news_data(stock_code, days)
            elif market == 'us_stock':
                all_news_data = self._get_us_stock_news_data(stock_code, days)
            
            # === 3. 保存到硬盘 ===
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(all_news_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"💾 新闻数据已缓存至: {cache_path}")
            except Exception as e:
                self.logger.error(f"写入新闻缓存失败: {e}")

            # 内存缓存
            cache_key = f"{market}_{stock_code}_{days}"
            # 缓存数据
            self.news_cache[cache_key] = (datetime.now(), all_news_data)
            
            self.logger.info(f"✓ {market.upper()} {stock_code} 综合新闻数据获取完成，总计 {all_news_data['news_summary'].get('total_news_count', 0)} 条")
            return all_news_data
            
        except Exception as e:
            self.logger.error(f"获取综合新闻数据失败: {str(e)}")
            return {
                'company_news': [],
                'announcements': [],
                'research_reports': [],
                'industry_news': [],
                'market_sentiment': {},
                'news_summary': {'total_news_count': 0}
            }

    def _get_a_stock_news_data(self, stock_code, days):
        """获取A股新闻数据 - 修复版"""
        import akshare as ak
        
        all_news_data = {
            'company_news': [],
            'announcements': [],
            'research_reports': [],
            'industry_news': [],
            'market_sentiment': {},
            'news_summary': {}
        }
        
        # 1. 公司新闻 (修复点：增加异常处理)
        try:
            # 尝试使用东财个股资讯
            company_news = ak.stock_news_em(symbol=stock_code)
            if company_news is not None and not company_news.empty:
                processed_news = []
                for _, row in company_news.head(20).iterrows():
                    # 东财返回列名通常为: 关键词, 标题, 内容, 发布时间, 文章来源, 网址
                    news_item = {
                        'title': str(row.get('新闻标题') or row.get('title') or row.iloc[1]),
                        'content': str(row.get('新闻内容') or row.get('content') or row.iloc[2]),
                        'date': str(row.get('发布时间') or row.get('date') or row.iloc[3]),
                        'source': 'eastmoney',
                        'relevance_score': 1.0
                    }
                    processed_news.append(news_item)
                all_news_data['company_news'] = processed_news
        except Exception as e:
            self.logger.warning(f"获取A股公司新闻失败: {e}")
        
        # 2. 公司公告 (修复点：替换失效接口 stock_zh_a_alerts_cls)
        try:
            # 使用东财公告接口替代财联社
            announcements = ak.stock_notice_report(symbol=stock_code)
            if announcements is not None and not announcements.empty:
                processed_announcements = []
                for _, row in announcements.head(20).iterrows():
                    announcement = {
                        'title': str(row.get('公告标题')),
                        'content': str(row.get('公告类型')), # 公告通常只有标题和类型
                        'date': str(row.get('公告日期')),
                        'type': '公告',
                        'relevance_score': 1.0
                    }
                    processed_announcements.append(announcement)
                
                all_news_data['announcements'] = processed_announcements
        except Exception as e:
            self.logger.warning(f"获取A股公司公告失败: {e}")
        
        # 3. 研究报告
        try:
            research_reports = ak.stock_research_report_em(symbol=stock_code)
            if research_reports is not None and not research_reports.empty:
                processed_reports = []
                for _, row in research_reports.head(20).iterrows():
                    report = {
                        'title': str(row.get('报告名称') or row.iloc[0]),
                        'institution': str(row.get('机构名称') or row.iloc[1]),
                        'rating': str(row.get('评级') or row.iloc[2]),
                        'date': str(row.get('发布日期') or row.iloc[4]),
                        'relevance_score': 0.9
                    }
                    processed_reports.append(report)
                
                all_news_data['research_reports'] = processed_reports
        except Exception as e:
            self.logger.warning(f"获取A股研究报告失败: {e}")
        
        # 统计新闻数量
        total_news = (len(all_news_data['company_news']) + 
                     len(all_news_data['announcements']) + 
                     len(all_news_data['research_reports']))
        
        all_news_data['news_summary'] = {
            'total_news_count': total_news,
            'company_news_count': len(all_news_data['company_news']),
            'announcements_count': len(all_news_data['announcements']),
            'research_reports_count': len(all_news_data['research_reports']),
            'industry_news_count': 0,
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market': 'A股'
        }
        
        return all_news_data

    def _get_hk_stock_news_data(self, stock_code, days):
        """获取g新闻数据"""
        # g新闻数据相对有限，返回基本结构
        return {
            'company_news': [],
            'announcements': [],
            'research_reports': [],
            'industry_news': [],
            'market_sentiment': {},
            'news_summary': {
                'total_news_count': 0,
                'company_news_count': 0,
                'announcements_count': 0,
                'research_reports_count': 0,
                'industry_news_count': 0,
                'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market': 'g',
                'note': 'g新闻数据来源有限'
            }
        }

    def _get_us_stock_news_data(self, stock_code, days):
        """获取m新闻数据"""
        # m新闻数据相对有限，返回基本结构
        return {
            'company_news': [],
            'announcements': [],
            'research_reports': [],
            'industry_news': [],
            'market_sentiment': {},
            'news_summary': {
                'total_news_count': 0,
                'company_news_count': 0,
                'announcements_count': 0,
                'research_reports_count': 0,
                'industry_news_count': 0,
                'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market': 'm',
                'note': 'm新闻数据来源有限'
            }
        }

    def calculate_advanced_sentiment_analysis(self, comprehensive_news_data):
        """计算高级情绪分析（支持多市场）"""
        self.logger.info("开始高级情绪分析...")
        
        try:
            # 准备所有新闻文本
            all_texts = []
            
            # 收集所有新闻文本
            for news in comprehensive_news_data.get('company_news', []):
                text = f"{news.get('title', '')} {news.get('content', '')}"
                all_texts.append({'text': text, 'type': 'company_news', 'weight': 1.0})
            
            for announcement in comprehensive_news_data.get('announcements', []):
                text = f"{announcement.get('title', '')} {announcement.get('content', '')}"
                all_texts.append({'text': text, 'type': 'announcement', 'weight': 1.2})
            
            for report in comprehensive_news_data.get('research_reports', []):
                text = f"{report.get('title', '')} {report.get('rating', '')}"
                all_texts.append({'text': text, 'type': 'research_report', 'weight': 0.9})
            
            if not all_texts:
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_by_type': {},
                    'sentiment_trend': '中性',
                    'confidence_score': 0.0,
                    'total_analyzed': 0
                }
            
            # 多语言情绪词典
            positive_words = {
                # 中文
                '上涨', '涨停', '利好', '突破', '增长', '盈利', '收益', '回升', '强势', '看好',
                '买入', '推荐', '优秀', '领先', '创新', '发展', '机会', '潜力', '稳定', '改善',
                '提升', '超预期', '积极', '乐观', '向好', '受益', '龙头', '热点', '爆发', '翻倍',
                # 英文
                'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'bull', 'positive', 
                'upgrade', 'outperform', 'beat', 'exceed', 'surge', 'rally', 'boom'
            }
            
            negative_words = {
                # 中文
                '下跌', '跌停', '利空', '破位', '下滑', '亏损', '风险', '回调', '弱势', '看空',
                '卖出', '减持', '较差', '落后', '滞后', '困难', '危机', '担忧', '悲观', '恶化',
                '下降', '低于预期', '消极', '压力', '套牢', '被套', '暴跌', '崩盘', '踩雷', '退市',
                # 英文
                'sell', 'weak', 'decline', 'loss', 'bear', 'negative', 'downgrade', 
                'underperform', 'miss', 'fall', 'drop', 'crash', 'plunge', 'slump'
            }
            
            # 分析每类新闻的情绪
            sentiment_by_type = {}
            overall_scores = []
            
            for text_data in all_texts:
                try:
                    text = text_data['text'].lower()  # 转换为小写以匹配英文词汇
                    text_type = text_data['type']
                    weight = text_data['weight']
                    
                    if not text.strip():
                        continue
                    
                    positive_count = sum(1 for word in positive_words if word in text)
                    negative_count = sum(1 for word in negative_words if word in text)
                    
                    # 计算情绪得分
                    total_sentiment_words = positive_count + negative_count
                    if total_sentiment_words > 0:
                        sentiment_score = (positive_count - negative_count) / total_sentiment_words
                    else:
                        sentiment_score = 0.0
                    
                    # 应用权重
                    weighted_score = sentiment_score * weight
                    overall_scores.append(weighted_score)
                    
                    # 按类型统计
                    if text_type not in sentiment_by_type:
                        sentiment_by_type[text_type] = []
                    sentiment_by_type[text_type].append(weighted_score)
                    
                except Exception as e:
                    continue
            
            # 计算总体情绪
            overall_sentiment = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            # 计算各类型平均情绪
            avg_sentiment_by_type = {}
            for text_type, scores in sentiment_by_type.items():
                avg_sentiment_by_type[text_type] = sum(scores) / len(scores) if scores else 0.0
            
            # 判断情绪趋势
            if overall_sentiment > 0.3:
                sentiment_trend = '非常积极'
            elif overall_sentiment > 0.1:
                sentiment_trend = '偏向积极'
            elif overall_sentiment > -0.1:
                sentiment_trend = '相对中性'
            elif overall_sentiment > -0.3:
                sentiment_trend = '偏向消极'
            else:
                sentiment_trend = '非常消极'
            
            # 计算置信度
            confidence_score = min(len(all_texts) / 50, 1.0)
            
            result = {
                'overall_sentiment': overall_sentiment,
                'sentiment_by_type': avg_sentiment_by_type,
                'sentiment_trend': sentiment_trend,
                'confidence_score': confidence_score,
                'total_analyzed': len(all_texts),
                'type_distribution': {k: len(v) for k, v in sentiment_by_type.items()},
                'positive_ratio': len([s for s in overall_scores if s > 0]) / len(overall_scores) if overall_scores else 0,
                'negative_ratio': len([s for s in overall_scores if s < 0]) / len(overall_scores) if overall_scores else 0
            }
            
            self.logger.info(f"✓ 高级情绪分析完成: {sentiment_trend} (得分: {overall_sentiment:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"高级情绪分析失败: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_by_type': {},
                'sentiment_trend': '分析失败',
                'confidence_score': 0.0,
                'total_analyzed': 0
            }

    def calculate_technical_indicators(self, price_data):
        """计算技术指标（增加 MA200 趋势线）"""
        try:
            if price_data.empty:
                return self._get_default_technical_analysis()
            
            technical_analysis = {}
            
            # 安全的数值处理函数
            def safe_float(value, default=50.0):
                try:
                    if pd.isna(value):
                        return default
                    num_value = float(value)
                    if math.isnan(num_value) or math.isinf(num_value):
                        return default
                    return num_value
                except (ValueError, TypeError):
                    return default
            
            # 移动平均线
            # 1. 基础移动平均线 (原有逻辑保持)
            try:
                price_data['ma5'] = price_data['close'].rolling(window=5, min_periods=1).mean()
                price_data['ma10'] = price_data['close'].rolling(window=10, min_periods=1).mean()
                price_data['ma20'] = price_data['close'].rolling(window=20, min_periods=1).mean()
                price_data['ma60'] = price_data['close'].rolling(window=60, min_periods=1).mean()
                price_data['ma200'] = price_data['close'].rolling(window=200, min_periods=1).mean()
                
                latest_price = safe_float(price_data['close'].iloc[-1])
                ma5 = safe_float(price_data['ma5'].iloc[-1], latest_price)
                ma10 = safe_float(price_data['ma10'].iloc[-1], latest_price)
                ma20 = safe_float(price_data['ma20'].iloc[-1], latest_price)
                ma60 = safe_float(price_data['ma60'].iloc[-1], latest_price)
                ma200 = safe_float(price_data['ma200'].iloc[-1], latest_price)

                technical_analysis['ma5'] = ma5
                technical_analysis['ma10'] = ma10
                technical_analysis['ma20'] = ma20
                technical_analysis['ma60'] = ma60
                technical_analysis['ma200'] = ma200
                technical_analysis['price_above_ma200'] = latest_price > ma200
                
                if latest_price > ma5 > ma10 > ma20:
                    technical_analysis['ma_trend'] = '多头排列'
                elif latest_price < ma5 < ma10 < ma20:
                    technical_analysis['ma_trend'] = '空头排列'
                else:
                    technical_analysis['ma_trend'] = '震荡整理'
            except Exception:
                technical_analysis['ma_trend'] = '计算失败'
            
            # ================= [新增] 2. KDJ 指标 =================
            try:
                low_9 = price_data['low'].rolling(window=9, min_periods=1).min()
                high_9 = price_data['high'].rolling(window=9, min_periods=1).max()
                rsv = 100 * (price_data['close'] - low_9) / (high_9 - low_9 + 1e-10)
                
                price_data['k'] = rsv.ewm(com=2).mean()
                price_data['d'] = price_data['k'].ewm(com=2).mean()
                price_data['j'] = 3 * price_data['k'] - 2 * price_data['d']
                
                current_k = safe_float(price_data['k'].iloc[-1], 50)
                current_d = safe_float(price_data['d'].iloc[-1], 50)
                current_j = safe_float(price_data['j'].iloc[-1], 50)
                
                technical_analysis['kdj_k'] = current_k
                technical_analysis['kdj_d'] = current_d
                technical_analysis['kdj_j'] = current_j
                
                # KDJ信号判断
                if len(price_data) >= 2:
                    k_prev = safe_float(price_data['k'].iloc[-2], current_k)
                    d_prev = safe_float(price_data['d'].iloc[-2], current_d)
                    if current_k > current_d and k_prev <= d_prev:
                        technical_analysis['kdj_signal'] = '金叉'
                    elif current_k < current_d and k_prev >= d_prev:
                        technical_analysis['kdj_signal'] = '死叉'
                    else:
                        technical_analysis['kdj_signal'] = '中性'
                else:
                    technical_analysis['kdj_signal'] = '数据不足'
                
                # KDJ状态
                if current_k > 80: technical_analysis['kdj_status'] = '超买'
                elif current_k < 20: technical_analysis['kdj_status'] = '超卖'
                else: technical_analysis['kdj_status'] = '正常'
            except Exception as e:
                technical_analysis['kdj_signal'] = '计算失败'
                technical_analysis['kdj_status'] = '未知'

            # ================= [新增] 3. 威廉指标 (WR) =================
            try:
                n = 14
                high_n = price_data['high'].rolling(window=n, min_periods=1).max()
                low_n = price_data['low'].rolling(window=n, min_periods=1).min()
                wr = 100 * (high_n - price_data['close']) / (high_n - low_n + 1e-10)
                current_wr = safe_float(wr.iloc[-1], 50)
                technical_analysis['wr'] = current_wr
                technical_analysis['wr_signal'] = '超卖' if current_wr > 80 else '超买' if current_wr < 20 else '正常'
            except Exception:
                technical_analysis['wr'] = 50.0
                technical_analysis['wr_signal'] = '计算失败'

            # ================= [新增] 4. ATR 波动率 =================
            try:
                high_low = price_data['high'] - price_data['low']
                high_close = np.abs(price_data['high'] - price_data['close'].shift())
                low_close = np.abs(price_data['low'] - price_data['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14, min_periods=1).mean()
                current_atr = safe_float(atr.iloc[-1], 0)
                technical_analysis['atr'] = current_atr
                # ATR百分比 (波动幅度)
                technical_analysis['atr_percent'] = safe_float(current_atr / price_data['close'].iloc[-1] * 100, 0)
            except Exception:
                technical_analysis['atr'] = 0
                technical_analysis['atr_percent'] = 0

            # ================= [新增] 10. 20日区间位置 (Price Position) =================
            # 作用：判断价格在过去20天波动范围内的位置。
            # 0 = 最低点，100 = 最高点，>90 暗示可能突破，<10 暗示支撑
            try:
                window = 20
                # 计算过去20天的最低价和最高价
                period_low = price_data['low'].rolling(window=window, min_periods=1).min()
                period_high = price_data['high'].rolling(window=window, min_periods=1).max()
                current_close = price_data['close']
                
                # 计算相对位置 (0-100)
                # 加 1e-9 防止最高价等于最低价时除以零
                pp_20d = (current_close - period_low) / (period_high - period_low + 1e-9) * 100
                
                # 存入字典
                technical_analysis['price_position_20d'] = safe_float(pp_20d.iloc[-1], 50.0)
                
            except Exception as e:
                # self.logger.debug(f"PP20d计算失败: {e}") # 可选打印
                technical_analysis['price_position_20d'] = 50.0 # 默认给中间值

            # 5. RSI指标 (原有逻辑)
            try:
                def calculate_rsi(prices, window=14):
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                rsi_series = calculate_rsi(price_data['close'])
                technical_analysis['rsi'] = safe_float(rsi_series.iloc[-1], 50.0)
            except Exception:
                technical_analysis['rsi'] = 50.0
            
            # 6. MACD指标 (原有逻辑)
            try:
                ema12 = price_data['close'].ewm(span=12, min_periods=1).mean()
                ema26 = price_data['close'].ewm(span=26, min_periods=1).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, min_periods=1).mean()
                histogram = macd_line - signal_line
                
                if len(histogram) >= 2:
                    current_hist = safe_float(histogram.iloc[-1])
                    prev_hist = safe_float(histogram.iloc[-2])
                    if current_hist > prev_hist and current_hist > 0:
                        technical_analysis['macd_signal'] = '金叉向上'
                    elif current_hist < prev_hist and current_hist < 0:
                        technical_analysis['macd_signal'] = '死叉向下'
                    else:
                        technical_analysis['macd_signal'] = '横盘整理'
                else:
                    technical_analysis['macd_signal'] = '数据不足'
            except Exception:
                technical_analysis['macd_signal'] = '计算失败'
            
            # 7. 布林带 (原有逻辑)
            try:
                bb_window = min(20, len(price_data))
                bb_middle = price_data['close'].rolling(window=bb_window, min_periods=1).mean()
                bb_std = price_data['close'].rolling(window=bb_window, min_periods=1).std()
                bb_upper = bb_middle + 2 * bb_std
                bb_lower = bb_middle - 2 * bb_std
                
                latest_close = safe_float(price_data['close'].iloc[-1])
                bb_upper_val = safe_float(bb_upper.iloc[-1])
                bb_lower_val = safe_float(bb_lower.iloc[-1])
                
                if bb_upper_val != bb_lower_val and bb_upper_val > bb_lower_val:
                    bb_position = (latest_close - bb_lower_val) / (bb_upper_val - bb_lower_val)
                    technical_analysis['bb_position'] = safe_float(bb_position, 0.5)
                else:
                    technical_analysis['bb_position'] = 0.5
            except Exception:
                technical_analysis['bb_position'] = 0.5
            
            # ================= [增强] 8. 成交量分析 (增强版) =================
            try:
                volume_window = min(20, len(price_data))
                avg_volume_5 = price_data['volume'].rolling(window=5, min_periods=1).mean().iloc[-1]
                avg_volume_20 = price_data['volume'].rolling(window=volume_window, min_periods=1).mean().iloc[-1]
                recent_volume = safe_float(price_data['volume'].iloc[-1])
                
                if 'change_pct' in price_data.columns:
                    price_change = safe_float(price_data['change_pct'].iloc[-1])
                else:
                    price_change = 0
                
                avg_volume_5 = safe_float(avg_volume_5, recent_volume)
                avg_volume_20 = safe_float(avg_volume_20, recent_volume)
                
                # 计算量比
                vol_ratio_5 = recent_volume / avg_volume_5 if avg_volume_5 > 0 else 1.0
                vol_ratio_20 = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
                
                technical_analysis['vol_ratio_5'] = safe_float(vol_ratio_5, 1.0)
                technical_analysis['vol_ratio_20'] = safe_float(vol_ratio_20, 1.0)
                
                # 状态判断
                if recent_volume > avg_volume_20 * 2.0:
                    technical_analysis['volume_status'] = '巨量' + ('上涨' if price_change > 0 else '下跌')
                elif recent_volume > avg_volume_20 * 1.5:
                    technical_analysis['volume_status'] = '放量' + ('上涨' if price_change > 0 else '下跌')
                elif recent_volume < avg_volume_20 * 0.6:
                    technical_analysis['volume_status'] = '极度缩量'
                else:
                    technical_analysis['volume_status'] = '量能温和'
            except Exception:
                technical_analysis['volume_status'] = '数据不足'
                technical_analysis['vol_ratio_5'] = 1.0
                technical_analysis['vol_ratio_20'] = 1.0

            # ================= [新增] 9. 共振分析 (Resonance) =================
            try:
                resonance_signals = []
                resonance_score = 0
                
                # 均线共振
                if technical_analysis.get('ma_trend') == '多头排列': 
                    resonance_signals.append('均线多头')
                    resonance_score += 1
                
                # 动量共振 (RSI + MACD + KDJ)
                bull_signals = 0
                if 45 < technical_analysis.get('rsi', 50) < 75: bull_signals += 1
                if '金叉' in technical_analysis.get('macd_signal', ''): bull_signals += 1
                if technical_analysis.get('kdj_signal') == '金叉': bull_signals += 1
                
                if bull_signals >= 2:
                    resonance_signals.append(f'动量共振({bull_signals}/3)')
                    resonance_score += 1.5

                # 抄底共振 (RSI低位 + WR低位)
                oversold_cnt = 0
                if technical_analysis.get('rsi', 50) < 30: oversold_cnt += 1
                if technical_analysis.get('wr', 50) > 80: oversold_cnt += 1 # WR>80是超卖
                if technical_analysis.get('kdj_status') == '超卖': oversold_cnt += 1
                
                if oversold_cnt >= 2:
                    resonance_signals.append('超卖共振(反弹预期)')
                    resonance_score += 1
                
                technical_analysis['resonance_signals'] = resonance_signals
                technical_analysis['resonance_rating'] = '强力' if resonance_score >= 2.5 else '一般'
            except Exception:
                technical_analysis['resonance_rating'] = '无'

            return technical_analysis
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败: {str(e)}")
            return self._get_default_technical_analysis()

    def analyze_smart_money_flow(self, df):
        """
        主力资金流向分析 (Smart Money Flow)
        检测机构建仓、洗盘和出货信号
        """
        try:
            if df.empty or len(df) < 30:
                return {}
            
            analysis = {}
            
            # 1. 计算 OBV (能量潮) - 核心资金指标
            # 逻辑：收阳线成交量加，收阴线成交量减。
            # 机构建仓特征：股价不涨，OBV 持续上涨
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # 2. 计算主力吸筹信号 (Stealth Accumulation)
            # 定义：当日量比 > 1.5 且 涨幅在 -1% 到 2% 之间 (放量不涨，多为主力吸筹)
            vol_ma20 = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / vol_ma20
            
            # 标记吸筹日
            accumulation_days = df[
                (df['vol_ratio'] > 1.5) & 
                (df['change_pct'] > -1.0) & 
                (df['change_pct'] < 2.5) &
                (df['close'] > df['open']) # 阳线吸筹更可信
            ]
            
            # 统计最近 30 天有多少个吸筹日
            recent_acc_days = len(accumulation_days[accumulation_days.index > df.index[-30]])
            
            # 3. 资金流向打分
            flow_score = 50
            status = "资金观望"
            
            # OBV 趋势判断
            obv_ma10 = df['obv'].rolling(10).mean().iloc[-1]
            obv_ma30 = df['obv'].rolling(30).mean().iloc[-1]
            current_obv = df['obv'].iloc[-1]
            
            if current_obv > obv_ma10 > obv_ma30:
                flow_score += 20
                status = "资金持续流入"
            elif current_obv < obv_ma10 < obv_ma30:
                flow_score -= 20
                status = "资金持续流出"
                
            # 吸筹力度加分
            if recent_acc_days >= 3:
                flow_score += 15
                status = "机构隐蔽建仓"
            elif recent_acc_days >= 5:
                flow_score += 25
                status = "机构强势抢筹"
                
            # 4. 筹码稳定性 (波动率收缩 VCP)
            # 建仓末期通常波动率极低
            volatility_5d = df['change_pct'].tail(5).std()
            if volatility_5d < 1.5 and flow_score > 60:
                status += " (即将爆发)"
                
            analysis = {
                'money_flow_score': flow_score,
                'flow_status': status,
                'accumulation_days': recent_acc_days, # 最近30天吸筹天数
                'obv_trend': '向上' if current_obv > obv_ma30 else '向下',
                'volatility_status': '极低' if volatility_5d < 1.5 else '正常'
            }
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"主力资金分析失败: {e}")
            return {'money_flow_score': 50, 'flow_status': '数据不足'}

    # === 新增方法：计算 ATR 止损位和支撑阻力 ===
    def calculate_trade_levels(self, df, total_capital=100000, risk_per_trade=0.02):
        """
        计算交易点位及仓位管理 (ATR风控 + 2%资金风险模型)
        """
        try:
            if df.empty or len(df) < 20:
                return {}

            # 数据转换
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            close = pd.to_numeric(df['close'], errors='coerce')
            current_price = close.iloc[-1]
            
            # 1. 计算 ATR (波动率)
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # 2. 确定止损位 (ATR吊灯止损)
            # 止损距离 = 2倍 ATR (给波动留出呼吸空间)
            stop_loss_distance = 2.0 * atr
            stop_loss_price = current_price - stop_loss_distance
            
            # 3. 仓位管理 (核心治愈代码)
            # 这是一个铁律：每笔交易最多只允许亏掉总资金的 2%
            # 比如 10万本金，最多亏 2000块。
            max_risk_amount = total_capital * risk_per_trade
            
            # 计算买入股数 = 允许亏损金额 / 单股亏损金额
            # 比如：允许亏2000 / (现价100 - 止损90) = 买200股
            if stop_loss_distance > 0:
                suggested_shares = int(max_risk_amount / stop_loss_distance)
                # 针对A股/港股调整为 100 的倍数 (手)
                suggested_shares = (suggested_shares // 100) * 100
            else:
                suggested_shares = 0
                
            # 计算建议投入本金
            suggested_position_value = suggested_shares * current_price
            position_percent = (suggested_position_value / total_capital) * 100

            return {
                "atr": round(atr, 2),
                "stop_loss": round(stop_loss_price, 2),
                "take_profit": round(current_price + (3.0 * atr), 2), # 1:3 盈亏比
                "support_20d": round(low.tail(20).min(), 2),
                "resistance_20d": round(high.tail(20).max(), 2),
                # === 新增：仓位建议 ===
                "suggested_shares": suggested_shares,
                "suggested_position_value": round(suggested_position_value, 2),
                "position_percent": round(position_percent, 1),
                "max_risk_money": max_risk_amount
            }
        except Exception as e:
            self.logger.warning(f"风控计算失败: {e}")
            return {}

    def _get_default_technical_analysis(self):
        """获取默认技术分析结果"""
        return {
            'ma_trend': '数据不足',
            'rsi': 50.0,
            'macd_signal': '数据不足',
            'bb_position': 0.5,
            'volume_status': '数据不足'
        }

    def calculate_technical_score(self, technical_analysis):
        """计算技术分析得分"""
        try:
            score = 50
            
            # 1. 趋势得分 (权重最高)
            # 如果股价在 200 日均线之上，说明处于长期牛市，基础分直接给高
            if technical_analysis.get('price_above_ma200', False):
                score += 10
            else:
                score -= 10

            # 2. 均线形态
            ma_trend = technical_analysis.get('ma_trend', '数据不足')
            if ma_trend == '多头排列':
                score += 15
            elif ma_trend == '空头排列':
                score -= 15
            
            # 3. RSI (核心修改：结合趋势判断)
            rsi = technical_analysis.get('rsi', 50)
            is_bull_market = technical_analysis.get('price_above_ma200', False)
            
            if is_bull_market:
                # 牛市里，RSI低位是买点 (回调)
                if rsi < 40: score += 15  # 黄金坑
                elif 40 <= rsi <= 70: score += 5
                elif rsi > 80: score -= 5 # 只有极度超买才减分
            else:
                # 熊市里，RSI低位可能是陷阱 (阴跌)，不加分
                if rsi < 30: score += 0   # 甚至可以不加分
                elif rsi > 60: score -= 10 # 熊市反弹一波就要跑
            
            # 4. MACD
            macd_signal = technical_analysis.get('macd_signal', '横盘整理')
            if macd_signal == '金叉向上':
                score += 15
            elif macd_signal == '死叉向下':
                score -= 15
            
            bb_position = technical_analysis.get('bb_position', 0.5)
            if 0.2 <= bb_position <= 0.8:
                score += 5
            elif bb_position < 0.2:
                score += 10
            elif bb_position > 0.8:
                score -= 5
            
            volume_status = technical_analysis.get('volume_status', '数据不足')
            if '放量上涨' in volume_status:
                score += 10
            elif '放量下跌' in volume_status:
                score -= 10
            
            score = max(0, min(100, score))
            return score
            
        except Exception as e:
            self.logger.error(f"技术分析评分失败: {str(e)}")
            return 50

    def calculate_fundamental_score(self, fundamental_data):
        """计算基本面得分（引入 PEG 和 动态估值）"""
        try:
            score = 50
            financials = fundamental_data.get('financial_indicators', {})
            
            # 数据提取 (兼容中英文key)
            def get_val(keys, default=0):
                for k in keys:
                    if k in financials and financials[k] is not None:
                        try:
                            return float(financials[k])
                        except: pass
                return default

            # 关键指标获取
            pe = get_val(['市盈率', 'PE_Ratio', '市盈率(TTM)'])
            roe = get_val(['净资产收益率', 'ROE'])
            growth = get_val(['净利润同比增长率', 'Net_Income_Growth', '营业收入同比增长率'])
            
            # === 1. 盈利能力 (ROE) ===
            # ROE 是公司的底色，依然重要
            if roe > 20: score += 15
            elif roe > 15: score += 10
            elif roe > 10: score += 5
            elif roe < 5: score -= 10

            # === 2. 核心修改：PEG 估值法 (取代死板的 PE<20) ===
            # PEG = 市盈率 / (净利润增长率 * 100)
            # 彼得·林奇法则：PEG < 1 低估，PEG > 1 合理，PEG > 2 高估
            
            if pe > 0 and growth > 0:
                peg = pe / growth
                if peg < 0.8: score += 20     # 极度低估 (成长快且便宜)
                elif 0.8 <= peg <= 1.2: score += 10 # 合理估值
                elif 1.2 < peg <= 2.0: score += 0   # 略贵但可接受
                elif peg > 2.0: score -= 15         # 泡沫严重
            else:
                # 如果没有增长数据，回退到动态 PE 逻辑
                # 逻辑：如果增速快(>20%)，允许高PE；否则必须低PE
                if growth > 20:
                    if pe < 40: score += 10
                elif growth > 10:
                    if pe < 25: score += 10
                else:
                    # 龟速增长股，PE必须低
                    if 0 < pe < 15: score += 10
                    elif pe > 25: score -= 10

            # === 3. 避雷指标 (一票否决项) ===
            # 扣分项：如果这些指标很烂，哪怕 PEG 很好也要扣分
            
            # 负债率过高 > 70%
            debt_ratio = get_val(['资产负债率', 'Debt_Ratio'], 50)
            if debt_ratio > 80: score -= 15
            elif debt_ratio > 70: score -= 5
            
            # 现金流为负 (赚假钱)
            # 假设有一个指标叫 '经营现金流占比' 或者简单判断现金流是否为负
            # 这里简单判断：如果 ROE < 0 (亏损)，直接扣分
            if roe < 0: score -= 10

            score = max(0, min(100, score))
            return score
            
        except Exception as e:
            self.logger.error(f"基本面评分失败: {e}")
            return 50

    def calculate_sentiment_score(self, sentiment_analysis):
        """计算情绪分析得分"""
        try:
            overall_sentiment = sentiment_analysis.get('overall_sentiment', 0.0)
            confidence_score = sentiment_analysis.get('confidence_score', 0.0)
            total_analyzed = sentiment_analysis.get('total_analyzed', 0)
            
            # 基础得分：将情绪得分从[-1,1]映射到[0,100]
            base_score = (overall_sentiment + 1) * 50
            
            # 置信度调整
            confidence_adjustment = confidence_score * 10
            
            # 新闻数量调整
            news_adjustment = min(total_analyzed / 100, 1.0) * 10
            
            final_score = base_score + confidence_adjustment + news_adjustment
            final_score = max(0, min(100, final_score))
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"情绪得分计算失败: {e}")
            return 50

    def calculate_comprehensive_score(self, scores):
        """计算综合得分"""
        try:
            technical_score = scores.get('technical', 50)
            fundamental_score = scores.get('fundamental', 50)
            sentiment_score = scores.get('sentiment', 50)
            
            comprehensive_score = (
                technical_score * self.analysis_weights['technical'] +
                fundamental_score * self.analysis_weights['fundamental'] +
                sentiment_score * self.analysis_weights['sentiment']
            )
            
            comprehensive_score = max(0, min(100, comprehensive_score))
            return comprehensive_score
            
        except Exception as e:
            self.logger.error(f"计算综合得分失败: {e}")
            return 50

    def get_stock_name(self, stock_code):
        """获取名称（支持多市场）"""
        try:
            stock_code, market = self.normalize_stock_code(stock_code)
            
            import akshare as ak
            
            if market == 'a_stock':
                try:
                    stock_info = ak.stock_individual_info_em(symbol=stock_code)
                    if not stock_info.empty:
                        info_dict = dict(zip(stock_info['item'], stock_info['value']))
                        stock_name = info_dict.get('简称', stock_code)
                        if stock_name and stock_name != stock_code:
                            return stock_name
                except Exception as e:
                    self.logger.warning(f"获取A名称失败: {e}")
            
            elif market == 'hk_stock':
                try:
                    hk_info = ak.stock_hk_spot_em()
                    stock_info = hk_info[hk_info['代码'] == stock_code]
                    if not stock_info.empty:
                        return stock_info['名称'].iloc[0]
                except Exception as e:
                    self.logger.warning(f"获取g名称失败: {e}")
            
            elif market == 'us_stock':
                try:
                    us_info = ak.stock_us_spot_em()
                    stock_info = us_info[us_info['代码'] == stock_code.upper()]
                    if not stock_info.empty:
                        return stock_info['名称'].iloc[0]
                except Exception as e:
                    self.logger.warning(f"获取m名称失败: {e}")
            
            return f"{market.upper()}_{stock_code}"
            
        except Exception as e:
            self.logger.warning(f"获取名称时出错: {e}")
            return stock_code

    def get_price_info(self, price_data):
        """从价格数据中提取关键信息（支持多市场）"""
        try:
            if price_data.empty or 'close' not in price_data.columns:
                self.logger.warning("价格数据为空或缺少收盘价列")
                return {
                    'current_price': 0.0,
                    'price_change': 0.0,
                    'volume_ratio': 1.0,
                    'volatility': 0.0
                }
            
            # 获取最新数据
            latest = price_data.iloc[-1]
            
            # 确保使用收盘价作为当前价格
            current_price = float(latest['close'])
            self.logger.info(f"✓ 当前价格(收盘价): {current_price}")
            
            # 安全的数值处理函数
            def safe_float(value, default=0.0):
                try:
                    if pd.isna(value):
                        return default
                    num_value = float(value)
                    if math.isnan(num_value) or math.isinf(num_value):
                        return default
                    return num_value
                except (ValueError, TypeError):
                    return default
            
            # 计算价格变化
            price_change = 0.0
            try:
                if 'change_pct' in price_data.columns and not pd.isna(latest['change_pct']):
                    price_change = safe_float(latest['change_pct'])
                elif len(price_data) > 1:
                    prev = price_data.iloc[-2]
                    prev_price = safe_float(prev['close'])
                    if prev_price > 0:
                        price_change = safe_float(((current_price - prev_price) / prev_price * 100))
            except Exception as e:
                self.logger.warning(f"计算价格变化失败: {e}")
                price_change = 0.0
            
            # 计算成交量比率
            volume_ratio = 1.0
            try:
                if 'volume' in price_data.columns:
                    volume_data = price_data['volume'].dropna()
                    if len(volume_data) >= 5:
                        recent_volume = volume_data.tail(5).mean()
                        avg_volume = volume_data.mean()
                        if avg_volume > 0:
                            volume_ratio = safe_float(recent_volume / avg_volume, 1.0)
            except Exception as e:
                self.logger.warning(f"计算成交量比率失败: {e}")
                volume_ratio = 1.0
            
            # 计算波动率
            volatility = 0.0
            try:
                close_prices = price_data['close'].dropna()
                if len(close_prices) >= 20:
                    returns = close_prices.pct_change().dropna()
                    if len(returns) >= 20:
                        volatility = safe_float(returns.tail(20).std() * 100)
            except Exception as e:
                self.logger.warning(f"计算波动率失败: {e}")
                volatility = 0.0
            
            result = {
                'current_price': safe_float(current_price),
                'price_change': safe_float(price_change),
                'volume_ratio': safe_float(volume_ratio, 1.0),
                'volatility': safe_float(volatility)
            }
            
            self.logger.info(f"✓ 价格信息提取完成: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"获取价格信息失败: {e}")
            return {
                'current_price': 0.0,
                'price_change': 0.0,
                'volume_ratio': 1.0,
                'volatility': 0.0
            }

    def generate_recommendation(self, scores, market=None):
        """根据得分生成投资建议（支持多市场）"""
        try:
            comprehensive_score = scores.get('comprehensive', 50)
            technical_score = scores.get('technical', 50)
            fundamental_score = scores.get('fundamental', 50)
            sentiment_score = scores.get('sentiment', 50)
            
            # 基础建议逻辑
            if comprehensive_score >= 80:
                if technical_score >= 75 and fundamental_score >= 75:
                    base_recommendation = "强烈推荐买入"
                else:
                    base_recommendation = "推荐买入"
            elif comprehensive_score >= 65:
                if sentiment_score >= 60:
                    base_recommendation = "建议买入"
                else:
                    base_recommendation = "谨慎买入"
            elif comprehensive_score >= 45:
                base_recommendation = "持有观望"
            elif comprehensive_score >= 30:
                base_recommendation = "建议减仓"
            else:
                base_recommendation = "建议卖出"
            
            # 根据市场特点调整建议
            if market == 'hk_stock':
                base_recommendation += " (g)"
            elif market == 'us_stock':
                base_recommendation += " (m)"
            elif market == 'a_stock':
                base_recommendation += " (A)"
                
            return base_recommendation
                
        except Exception as e:
            self.logger.warning(f"生成投资建议失败: {e}")
            return "数据不足，建议谨慎"

    def _build_enhanced_ai_analysis_prompt(self, stock_code, stock_name, scores, technical_analysis, 
                                        fundamental_data, sentiment_analysis, price_info, market=None,trade_levels=None,money_flow=None,ai_trade_decision=None):
        """构建增强版AI分析提示词（支持多市场）"""
        
        market_info = ""
        if market:
            market_config = self.market_config.get(market, {})
            currency = market_config.get('currency', 'CNY')
            timezone = market_config.get('timezone', 'Asia/Shanghai')
            market_info = f"""
**市场信息：**
- 交易市场：{market.upper().replace('_', '')}
- 计价货币：{currency}
- 时区：{timezone}
"""
        
        # 提取财务指标
        financial_indicators = fundamental_data.get('financial_indicators', {})
        financial_text = ""
        if financial_indicators:
            financial_text = "**财务指标详情：**\n"
            for i, (key, value) in enumerate(financial_indicators.items(), 1):
                if isinstance(value, (int, float)) and value != 0:
                    financial_text += f"{i}. {key}: {value}\n"
        
        trade_levels_text = ""
        if trade_levels:
            trade_levels_text = f"""
**量化风控与仓位建议 (基于2%本金风险模型)**：
- 波动率(ATR)：{trade_levels.get('atr', 'N/A')}
- 🛑 刚性止损位：{trade_levels.get('stop_loss', 'N/A')} (触及必须无条件离场)
- 💰 建议仓位：{trade_levels.get('suggested_shares', 0)} 股 (约占本金 {trade_levels.get('position_percent', 0)}%)
- ⚠️ 最大风险敞口：-{trade_levels.get('max_risk_money', 0)} 元 (即使止损离场，也只损失本金的2%)
"""

        money_flow_text = ""
        if money_flow:
            money_flow_text = f"""
**主力资金监控 (Smart Money)**：
- 资金状态：{money_flow.get('flow_status', '未知')} (得分: {money_flow.get('money_flow_score', 50)})
- 隐蔽吸筹：最近30天出现 {money_flow.get('accumulation_days', 0)} 次主力吸筹信号
- OBV趋势：{money_flow.get('obv_trend', '未知')}
- 爆发潜力：{'高 (波动率收缩+资金流入)' if money_flow.get('volatility_status') == '极低' and money_flow.get('money_flow_score', 0) > 70 else '一般'}
"""

        # === 👇 新增：构建量化信号文本 👇 ===
        quant_signal_text = ""
        if ai_trade_decision:
            action = ai_trade_decision.get('action', 'HOLD')
            conf = ai_trade_decision.get('confidence', 0)
            reason = ai_trade_decision.get('reason', '无')
            
            quant_signal_text = f"""
**核心量化信号 (最高优先级参考)**：
- 🤖 策略引擎建议：{action} (置信度 {conf}%)
- 🎯 信号逻辑：{reason}
- ⚠️ 写作要求：你的“实战操作建议”必须与此量化信号保持一致！如果量化模型建议 BUY，你必须解释为何买入；如果建议 HOLD，你必须解释为何观望。
"""
        # =================================

        # 构建完整的提示词
        prompt = f"""
# Role
你是一位拥有20年实战经验的**资深全球量化交易员**。你的风格是**结论先行、数据驱动、拒绝废话**。
你不需要向我解释什么是ETF或，也不需要科普监管环境。你需要基于我提供的详细数据，像写**交易日志**一样，给出直击要害的分析和操作计划。

# Input Data (市场情报)
**基础信息**：
- 代码：{stock_code} ({stock_name})
- 现价：{price_info.get('current_price', 0):.2f} (涨跌: {price_info.get('price_change', 0):.2f}%)
- 波动率：{price_info.get('volatility', 0):.2f}% | 量比：{price_info.get('volume_ratio', 1):.2f}

{trade_levels_text}  

{money_flow_text}

{quant_signal_text}

**技术信号**：
- 趋势：{technical_analysis.get('ma_trend', '未知')}
- 指标：RSI={technical_analysis.get('rsi', 50):.1f} | MACD={technical_analysis.get('macd_signal', '未知')}
- 布林带位置：{technical_analysis.get('bb_position', 0.5):.2f} (0=下轨, 1=上轨)
- 成交量状态：{technical_analysis.get('volume_status', '未知')}

**情绪与评分**：
- 市场情绪：{sentiment_analysis.get('sentiment_trend', '中性')} (得分: {sentiment_analysis.get('overall_sentiment', 0):.3f})
- 综合评分：{scores.get('comprehensive', 50):.1f}/100 (技术:{scores.get('technical', 50):.1f} | 基本面:{scores.get('fundamental', 50):.1f})

**补充情报**：
{market_info}
{financial_text}
---

# Output Requirement (输出要求)
请严格模仿“专业投研报告”的格式，按照以下结构输出：

## {stock_name} ({stock_code}) 深度交易策略报告

### 核心观点 (Core Thesis)
(用一句话定性：看多/看空/震荡。结合综合评分 {scores.get('comprehensive', 0):.1f} 和情绪，给出明确的方向性判断。)

### 1. 基本面驱动逻辑 (Fundamental Drivers)
* **核心逻辑**：(基于 `{financial_text}`，简述营收、利润或宏观驱动力。如果是m/ETF，重点分析宏观利率、汇率影响或成分表现。)
* **估值与资金**：(分析当前价格是否合理，是否有大资金流入流出迹象。)

### 2. 技术面狙击 (Technical Sniper)
* **形态与趋势**：(结合 `{technical_analysis.get('ma_trend')}` 和 涨跌幅，描述K线形态，如“突破箱体”、“缩量回调”等。)
* **量价配合**：(基于量比 {price_info.get('volume_ratio'):.2f} 和成交量状态，分析主力意图。)
* **指标共振**：
    * MACD: {technical_analysis.get('macd_signal')} (解读其含义，如“多头趋势确认”或“顶背离警示”)
    * RSI ({technical_analysis.get('rsi'):.1f}): (解读是否超买/超卖，结合布林带位置 {technical_analysis.get('bb_position'):.2f} 判断反弹或回调压力。)
* **关键点位预测**：
    * 🔴 **强阻力位**：参考量化模型提供的 {trade_levels.get('resistance_20d') if trade_levels else '阻力位'}，结合布林带分析。
    * 🟢 **强支撑位**：参考量化模型提供的 {trade_levels.get('support_20d') if trade_levels else '支撑位'}，结合均线分析。

### 3. 多空博弈与风险 (Risk & Opportunity)
* **多头逻辑**：(上涨的催化剂是什么？)
* **空头风险**：(下跌的风险点，包括地缘政治、汇率风险或技术破位风险。)

### 4. 实战操作建议 (Action Plan)
(综合以上分析，给出具体的操盘逻辑。是左侧低吸？还是右侧追涨？还是空仓观望？)

### AI 交易决策 (AI Signal)
(声明：仅供参考，不构成投资建议)
| 操作方向 | 建议价格区间 | 建议仓位 | 期望收益率(EV) | 策略置信度 |
| :--- | :--- | :--- | :--- | :--- |
| [买入/卖出/观望] | [具体数值] | [如: 30%] | [如: +15%] | [0.0-1.0] |

---
**注意**：
1. 语言风格要**犀利、专业**，像是在给基金经理写汇报。
2. 必须给出**具体的数字**（支撑位、阻力位），不要给模糊的范围。
3. 结合“全球视角”，如果是跨境投资，简要提及汇率或m联储政策的影响，但不要展开写科普文。
"""
        return prompt

    def generate_ai_analysis(self, analysis_data, enable_streaming=False, stream_callback=None):
        """生成AI增强分析（支持多市场）"""
        try:
            self.logger.info("🤖 开始AI深度分析（支持多市场）...")
            
            stock_code = analysis_data.get('stock_code', '')
            stock_name = analysis_data.get('stock_name', stock_code)
            scores = analysis_data.get('scores', {})
            technical_analysis = analysis_data.get('technical_analysis', {})
            fundamental_data = analysis_data.get('fundamental_data', {})
            sentiment_analysis = analysis_data.get('sentiment_analysis', {})
            price_info = analysis_data.get('price_info', {})
            
            # 检测市场
            _, market = self.normalize_stock_code(stock_code)
            
            trade_levels = analysis_data.get('trade_levels', {})
            money_flow = analysis_data.get('money_flow', {})
            ai_trade_decision = analysis_data.get('ai_trade_decision', {})

            # 构建增强版AI分析提示词
            prompt = self._build_enhanced_ai_analysis_prompt(
                stock_code, stock_name, scores, technical_analysis, 
                fundamental_data, sentiment_analysis, price_info, market,
                trade_levels,money_flow, ai_trade_decision
            )
            
            # 调用AI API（支持流式）
            ai_response = self._call_ai_api(prompt, enable_streaming, stream_callback)

            if ai_response:
                self.logger.info("✅ AI深度分析完成（多市场）")
                # 👉 修改点：返回元组 (ai_response, prompt)
                return ai_response, prompt
            else:
                self.logger.warning("⚠️ AI API不可用，使用高级分析模式")
                fallback = self._advanced_rule_based_analysis(analysis_data, market)
                # 👉 修改点：返回元组 (fallback, 说明文字)
                return fallback, "（API不可用，使用规则引擎分析）"
                
        except Exception as e:
            self.logger.error(f"AI分析失败: {e}")
            fallback = self._advanced_rule_based_analysis(analysis_data, self.detect_market(stock_code))
            # 👉 修改点：返回元组
            return fallback, f"（分析出错: {e}，使用规则引擎）"

    def _call_ai_api(self, prompt, enable_streaming=False, stream_callback=None):
        """调用AI API - 支持流式输出（多市场通用）"""
        try:
            model_preference = self.config.get('ai', {}).get('model_preference', 'openai')
            
            if model_preference == 'openai' and self.api_keys.get('openai'):
                result = self._call_openai_api(prompt, enable_streaming, stream_callback)
                if result:
                    return result
            
            elif model_preference == 'anthropic' and self.api_keys.get('anthropic'):
                result = self._call_claude_api(prompt, enable_streaming, stream_callback)
                if result:
                    return result
                    
            elif model_preference == 'zhipu' and self.api_keys.get('zhipu'):
                result = self._call_zhipu_api(prompt, enable_streaming, stream_callback)
                if result:
                    return result
            
            # 尝试其他可用的服务
            if self.api_keys.get('openai') and model_preference != 'openai':
                self.logger.info("尝试备用OpenAI API...")
                result = self._call_openai_api(prompt, enable_streaming, stream_callback)
                if result:
                    return result
                    
            if self.api_keys.get('anthropic') and model_preference != 'anthropic':
                self.logger.info("尝试备用Claude API...")
                result = self._call_claude_api(prompt, enable_streaming, stream_callback)
                if result:
                    return result
                    
            if self.api_keys.get('zhipu') and model_preference != 'zhipu':
                self.logger.info("尝试备用智谱AI API...")
                result = self._call_zhipu_api(prompt, enable_streaming, stream_callback)
                if result:
                    return result
            
            return None
                
        except Exception as e:
            self.logger.error(f"AI API调用失败: {e}")
            return None

    def _call_openai_api(self, prompt, enable_streaming=False, stream_callback=None):
        """调用OpenAI API"""
        try:
          
            api_key = self.api_keys.get('openai')
            if not api_key:
                return None
            
            openai.api_key = api_key
            
            api_base = self.config.get('ai', {}).get('api_base_urls', {}).get('openai')
            if api_base:
                openai.api_base = api_base
            
            model = self.config.get('ai', {}).get('models', {}).get('openai', 'gpt-4o-mini')
            max_tokens = self.config.get('ai', {}).get('max_tokens', 6000)
            temperature = self.config.get('ai', {}).get('temperature', 0.7)
            
            # messages = [
            #     {"role": "system", "content": "你是一位资深的全球分析师，具有丰富的多市场投资经验。请提供专业、客观、有深度的分析。"},
            #     {"role": "user", "content": prompt}
            # ]
            messages = [
                {
                    "role": "system", 
                    # 关键修改：把“分析师”改成“交易员”，并强调“拒绝废话”
                    "content": "你是一位资深全球量化交易员。请严格根据用户提供的数据，以实战、犀利的风格输出交易策略报告，拒绝模棱两可的废话。" 
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            # 检测OpenAI库版本
            try:
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI(api_key=api_key)
                    if api_base:
                        client.base_url = api_base
                    
                    if enable_streaming and stream_callback:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True
                        )
                        
                        full_response = ""
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                if stream_callback:
                                    stream_callback(content)
                        
                        return full_response
                    else:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                
                else:
                    if enable_streaming and stream_callback:
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True
                        )
                        
                        full_response = ""
                        for chunk in response:
                            if chunk.choices[0].delta.get('content'):
                                content = chunk.choices[0].delta.content
                                full_response += content
                                if stream_callback:
                                    stream_callback(content)
                        
                        return full_response
                    else:
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                    
            except Exception as api_error:
                self.logger.error(f"OpenAI API调用错误: {api_error}")
                return None
                
        except ImportError:
            self.logger.error("OpenAI库未安装")
            return None
        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {e}")
            return None

    def _call_claude_api(self, prompt, enable_streaming=False, stream_callback=None):
        """调用Claude API"""
        try:
            import anthropic
            
            api_key = self.api_keys.get('anthropic')
            if not api_key:
                return None
            
            client = anthropic.Anthropic(api_key=api_key)
            
            model = self.config.get('ai', {}).get('models', {}).get('anthropic', 'claude-3-haiku-20240307')
            max_tokens = self.config.get('ai', {}).get('max_tokens', 6000)
            
            if enable_streaming and stream_callback:
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ) as stream:
                    full_response = ""
                    for text in stream.text_stream:
                        full_response += text
                        if stream_callback:
                            stream_callback(text)
                
                return full_response
            else:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Claude API调用失败: {e}")
            return None

    def _call_zhipu_api(self, prompt, enable_streaming=False, stream_callback=None):
        """调用智谱AI API"""
        try:
            api_key = self.api_keys.get('zhipu')
            if not api_key:
                return None
            
            model = self.config.get('ai', {}).get('models', {}).get('zhipu', 'chatglm_turbo')
            max_tokens = self.config.get('ai', {}).get('max_tokens', 6000)
            temperature = self.config.get('ai', {}).get('temperature', 0.7)
            
            try:
                import zhipuai
                zhipuai.api_key = api_key
                
                if hasattr(zhipuai, 'ZhipuAI'):
                    client = zhipuai.ZhipuAI(api_key=api_key)
                    
                    if enable_streaming and stream_callback:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True
                        )
                        
                        full_response = ""
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                if stream_callback:
                                    stream_callback(content)
                        
                        return full_response
                    else:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        return response.choices[0].message.content
                
                else:
                    response = zhipuai.model_api.invoke(
                        model=model,
                        prompt=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    if isinstance(response, dict):
                        if 'data' in response and 'choices' in response['data']:
                            return response['data']['choices'][0]['content']
                        elif 'choices' in response:
                            return response['choices'][0]['content']
                        elif 'data' in response:
                            return response['data']
                    
                    return str(response)
                    
            except ImportError:
                self.logger.error("智谱AI库未安装")
                return None
            except Exception as api_error:
                self.logger.error(f"智谱AI API调用错误: {api_error}")
                return None
            
        except Exception as e:
            self.logger.error(f"智谱AI API调用失败: {e}")
            return None

    def _advanced_rule_based_analysis(self, analysis_data, market=None):
        """高级规则分析（支持多市场）"""
        try:
            self.logger.info(f"🧠 使用高级规则引擎进行分析（{market or 'Unknown'}市场）...")
            
            stock_code = analysis_data.get('stock_code', '')
            stock_name = analysis_data.get('stock_name', stock_code)
            scores = analysis_data.get('scores', {})
            technical_analysis = analysis_data.get('technical_analysis', {})
            fundamental_data = analysis_data.get('fundamental_data', {})
            sentiment_analysis = analysis_data.get('sentiment_analysis', {})
            price_info = analysis_data.get('price_info', {})
            
            analysis_sections = []
            
            # 1. 市场特征分析
            market_info = ""
            if market:
                market_config = self.market_config.get(market, {})
                currency = market_config.get('currency', 'CNY')
                
                if market == 'a_stock':
                    market_info = "**A市场特征：** 中国内地主板市场，以人民币计价，T+1交易制度，涨跌停限制±10%。"
                elif market == 'hk_stock':
                    market_info = "**g市场特征：** 香g联合交易所，g币计价，T+0交易制度，无涨跌停限制，国际化程度高。"
                elif market == 'us_stock':
                    market_info = "**m市场特征：** 纳斯达克/纽交所，m元计价，T+0交易制度，盘前盘后交易，全球影响力最大。"
            
            # 2. 综合评估
            comprehensive_score = scores.get('comprehensive', 50)
            analysis_sections.append(f"""## 📊 多市场综合评估

{market_info}

基于技术面、基本面和市场情绪的综合分析，{stock_name}({stock_code})的综合得分为{comprehensive_score:.1f}分。

- 技术面得分：{scores.get('technical', 50):.1f}/100
- 基本面得分：{scores.get('fundamental', 50):.1f}/100  
- 情绪面得分：{scores.get('sentiment', 50):.1f}/100""")
            
            # 3. 财务分析
            financial_indicators = fundamental_data.get('financial_indicators', {})
            if financial_indicators:
                key_metrics = []
                for key, value in list(financial_indicators.items())[:10]:
                    if isinstance(value, (int, float)) and value != 0:
                        key_metrics.append(f"- {key}: {value}")
                
                financial_text = f"""## 💰 财务健康度分析

获取到{len(financial_indicators)}项财务指标，主要指标如下：

{chr(10).join(key_metrics[:8])}

财务健康度评估：{'优秀' if scores.get('fundamental', 50) >= 70 else '良好' if scores.get('fundamental', 50) >= 50 else '需关注'}"""
                analysis_sections.append(financial_text)
            
            # 4. 技术面分析
            tech_analysis = f"""## 📈 技术面分析

当前技术指标显示：
- 均线趋势：{technical_analysis.get('ma_trend', '未知')}
- RSI指标：{technical_analysis.get('rsi', 50):.1f}
- MACD信号：{technical_analysis.get('macd_signal', '未知')}
- 成交量状态：{technical_analysis.get('volume_status', '未知')}

技术面评估：{'强势' if scores.get('technical', 50) >= 70 else '中性' if scores.get('technical', 50) >= 50 else '偏弱'}"""
            analysis_sections.append(tech_analysis)
            
            # 5. 市场情绪
            sentiment_desc = f"""## 📰 市场情绪分析

基于{sentiment_analysis.get('total_analyzed', 0)}条新闻的分析：
- 整体情绪：{sentiment_analysis.get('sentiment_trend', '中性')}
- 情绪得分：{sentiment_analysis.get('overall_sentiment', 0):.3f}
- 置信度：{sentiment_analysis.get('confidence_score', 0):.2%}

新闻分布：
- 公司新闻：{len(sentiment_analysis.get('company_news', []))}条
- 公司公告：{len(sentiment_analysis.get('announcements', []))}条  
- 研究报告：{len(sentiment_analysis.get('research_reports', []))}条"""
            analysis_sections.append(sentiment_desc)
            
            # 6. 投资建议
            recommendation = self.generate_recommendation(scores, market)
            market_specific_advice = ""
            
            if market == 'hk_stock':
                market_specific_advice = """
**g投资注意事项：**
- 考虑汇率风险（g币对人民币）
- 关注南下资金流向
- 注意g通交易限制
- 考虑香g政治经济环境影响"""
            elif market == 'us_stock':
                market_specific_advice = """
**m投资注意事项：**
- 考虑汇率风险（m元对人民币）
- 关注m联储政策影响
- 注意ADR与正价差
- 考虑税务和资金成本"""
            elif market == 'a_stock':
                market_specific_advice = """
**A投资注意事项：**
- 关注政策导向和监管变化
- 注意涨跌停限制
- 考虑T+1交易制度
- 关注机构资金流向"""
            
            strategy = f"""## 🎯 投资策略建议

**投资建议：{recommendation}**

{'**积极配置**：各项指标表现优异，可适当加大仓位。' if comprehensive_score >= 80 else 
 '**谨慎买入**：整体表现良好，但需要关注风险点。' if comprehensive_score >= 60 else
 '**观望为主**：当前风险收益比一般，建议等待更好时机。' if comprehensive_score >= 40 else
 '**规避风险**：多项指标显示风险较大，建议减仓或观望。'}

**操作建议：**
- 买入时机：技术面突破关键位置时
- 止损位置：跌破重要技术支撑
- 持有周期：中长期为主

{market_specific_advice}"""
            analysis_sections.append(strategy)
            
            return "\n\n".join(analysis_sections)
            
        except Exception as e:
            self.logger.error(f"高级规则分析失败: {e}")
            return "分析系统暂时不可用，请稍后重试。"

    def set_streaming_config(self, enabled=True, show_thinking=True):
        """设置流式推理配置"""
        self.streaming_config.update({
            'enabled': enabled,
            'show_thinking': show_thinking
        })

    def analyze_stock(self, stock_code, enable_streaming=None, stream_callback=None):
        """分析的主方法（支持多市场 + AI流式输出）"""
        if enable_streaming is None:
            enable_streaming = self.streaming_config.get('enabled', False)
        
        try:
            # 标准化代码并检测市场
            normalized_code, market = self.normalize_stock_code(stock_code)
            
            self.logger.info(f"开始增强版分析: {normalized_code} ({market.upper()})")
            
            # 检查市场是否启用
            if not self.market_config.get(market, {}).get('enabled', True):
                raise ValueError(f"市场 {market.upper()} 未启用")
            
            # 获取名称
            stock_name = self.get_stock_name(normalized_code)
            
            # 1. 获取价格数据和技术分析
            self.logger.info(f"正在进行 {market.upper()} 技术分析...")
            price_data = self.get_stock_data(normalized_code)
            if price_data.empty:
                raise ValueError(f"无法获取 {market.upper()} {normalized_code} 的价格数据")
            
            price_info = self.get_price_info(price_data)
            technical_analysis = self.calculate_technical_indicators(price_data)
            technical_score = self.calculate_technical_score(technical_analysis)

            # === 新增：主力资金分析 ===
            money_flow = self.analyze_smart_money_flow(price_data)
            self.logger.info(f"资金分析完成: {money_flow.get('flow_status')}")

            # ai_trade_decision = analysis_data.get('ai_trade_decision', {})

            # === 新增：计算量化交易点位 ===
            trade_levels = self.calculate_trade_levels(price_data)
            self.logger.info(f"量化点位计算完成: 止损 {trade_levels.get('stop_loss')}")
            
            # 2. 获取财务指标和基本面分析
            self.logger.info(f"正在进行 {market.upper()} 财务指标分析...")
            fundamental_data = self.get_comprehensive_fundamental_data(normalized_code)
            fundamental_score = self.calculate_fundamental_score(fundamental_data)
            
            # 3. 获取新闻数据和情绪分析
            self.logger.info(f"正在进行 {market.upper()} 新闻和情绪分析...")
            comprehensive_news_data = self.get_comprehensive_news_data(normalized_code, days=30)
            sentiment_analysis = self.calculate_advanced_sentiment_analysis(comprehensive_news_data)
            sentiment_score = self.calculate_sentiment_score(sentiment_analysis)
            
            # 合并新闻数据到情绪分析结果中
            sentiment_analysis.update(comprehensive_news_data)
            
            # 4. 计算综合得分
            scores = {
                'technical': technical_score,
                'fundamental': fundamental_score,
                'sentiment': sentiment_score,
                'comprehensive': self.calculate_comprehensive_score({
                    'technical': technical_score,
                    'fundamental': fundamental_score,
                    'sentiment': sentiment_score
                })
            }
            # ============================================================
            # 👇👇👇 [新增] 插入 AI 策略决策逻辑 👇👇👇
            # ============================================================
            ai_decision = {"action": "HOLD", "confidence": 0, "reason": "初始化"}
            
            if not price_data.empty and len(price_data) > 30:
                # A. 计算策略专用数据
                df_strategy = self._calculate_strategy_features(price_data)
                
                # B. 运行硬规则风控
                is_valid, reject_reason = self._check_v5_rules(df_strategy)
                
                if not is_valid:
                    ai_decision = {
                        "action": "HOLD",
                        "confidence": 0,
                        "reason": f"风控拦截: {reject_reason}"
                    }
                else:
                    # C. 调用 AI
                    # 如果开启流式，发送通知
                    if enable_streaming and stream_callback:
                        stream_callback(f"\n🤖 [策略] 风控通过，正在进行交易决策...\n")
                    
                    prompt = self._build_strategy_prompt(df_strategy)
                    ai_res_text = self._call_strategy_ai(prompt) 
                    
                    try:
                        match = re.search(r"(\{.*\})", ai_res_text, re.DOTALL)
                        if match:
                            ai_decision = json.loads(match.group(1))
                        else:
                            ai_decision = json.loads(ai_res_text)
                    except:
                        ai_decision = {"action": "HOLD", "confidence": 0, "reason": "AI解析失败"}
            # ============================================================
            # 5. 生成投资建议
            recommendation = self.generate_recommendation(scores, market)
            
            # 6. AI增强分析（支持多市场 + 流式输出）
            ai_analysis, used_prompt = self.generate_ai_analysis({
                'stock_code': normalized_code,
                'stock_name': stock_name,
                'price_info': price_info,
                'technical_analysis': technical_analysis,
                'fundamental_data': fundamental_data,
                'sentiment_analysis': sentiment_analysis,
                'scores': scores,
                'market': market,
                'money_flow': money_flow,
                'trade_levels': trade_levels,
                'ai_trade_decision': ai_decision
            }, enable_streaming, stream_callback)
            
            # ==========================================
            # 👉 【插入在这里】 保存历史记录 👈
            # ==========================================
            if ai_analysis:
                try:
                    saved_path = self.save_analysis_history(
                        stock_code=stock_code,
                        prompt=used_prompt,  # 👉 这里传入真实的 prompt 变量
                        ai_response=ai_analysis,
                        scores=scores
                    )
                    self.logger.info(f"📝 历史记录已保存: {saved_path}")
                except Exception as e:
                    self.logger.warning(f"保存历史记录失败: {e}")
        
            # 7. 生成最终报告
            report = {
                'stock_code': normalized_code,
                'original_code': stock_code,
                'stock_name': stock_name,
                'market': market,
                'market_info': self.market_config.get(market, {}),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price_info': price_info,
                'technical_analysis': technical_analysis,
                'fundamental_data': fundamental_data,
                'comprehensive_news_data': comprehensive_news_data,
                'sentiment_analysis': sentiment_analysis,
                'scores': scores,
                'analysis_weights': self.analysis_weights,
                'recommendation': recommendation,
                'ai_analysis': ai_analysis,
                'data_quality': {
                    'financial_indicators_count': len(fundamental_data.get('financial_indicators', {})),
                    'total_news_count': sentiment_analysis.get('total_analyzed', 0),
                    'analysis_completeness': '完整' if len(fundamental_data.get('financial_indicators', {})) >= 10 else '部分',
                    'market_coverage': market.upper()
                },
                'ai_trade_decision': ai_decision
            }
            
            self.logger.info(f"✓ 增强版分析完成: {normalized_code} ({market.upper()})")
            self.logger.info(f"  - 市场类型: {market.upper()}")
            self.logger.info(f"  - 财务指标: {len(fundamental_data.get('financial_indicators', {}))} 项")
            self.logger.info(f"  - 新闻数据: {sentiment_analysis.get('total_analyzed', 0)} 条")
            self.logger.info(f"  - 综合得分: {scores['comprehensive']:.1f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"增强版分析失败 {stock_code}: {str(e)}")
            raise

    def save_analysis_history(self, stock_code, prompt, ai_response, scores):
        """保存分析问答历史到本地 Markdown"""
        
        stock_code, market = self.normalize_stock_code(stock_code)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 文件名: 20251225_120000_QQQ_us_stock.md
        filename = f"{timestamp}_{stock_code}_{market}.md"
        filepath = os.path.join(self.history_dir, filename)
        
        # 构建 Markdown 内容
        content = f"""# 📈 股票分析报告: {stock_code} ({market})

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**综合评分**: {scores.get('comprehensive', 0):.1f} 分

---

## 🙋‍♂️ 你的问题 (Prompt Context)
> 这是一个基于自动数据的分析请求。
*(为节省空间，此处通常不保存完整的庞大 Prompt，只保存关键输入)*

## 🤖 AI 的深度分析
{ai_response}

---
*Generated by EnhancedWebStockAnalyzer*
"""
        
        # 写入文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"📝 分析报告已归档: {filename}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存历史失败: {e}")
            return None


    def analyze_stock_with_streaming(self, stock_code, streamer):
        """带流式回调的分析方法"""
        def stream_callback(content):
            """AI流式内容回调"""
            if streamer:
                streamer.send_ai_stream(content)
        
        return self.analyze_stock(stock_code, enable_streaming=True, stream_callback=stream_callback)

    def get_supported_markets(self):
        """获取支持的市场列表"""
        supported_markets = []
        for market, config in self.market_config.items():
            if config.get('enabled', True):
                market_info = {
                    'market': market,
                    'name': market.upper().replace('_', ''),
                    'currency': config.get('currency', 'CNY'),
                    'timezone': config.get('timezone', 'Asia/Shanghai'),
                    'trading_hours': config.get('trading_hours', '09:30-15:00')
                }
                supported_markets.append(market_info)
        
        return supported_markets

    def validate_stock_code(self, stock_code):
        """验证代码格式"""
        try:
            normalized_code, market = self.normalize_stock_code(stock_code)
            
            # 检查市场是否启用
            if not self.market_config.get(market, {}).get('enabled', True):
                return False, f"市场 {market.upper()} 未启用"
            
            # 基本格式验证
            if market == 'a_stock' and not re.match(r'^\d{6}$', normalized_code):
                return False, "A代码应为6位数字"
            elif market == 'hk_stock' and not re.match(r'^\d{5}$', normalized_code):
                return False, "g代码应为5位数字"
            elif market == 'us_stock' and not re.match(r'^[A-Z]{1,5}$', normalized_code):
                return False, "m代码应为1-5位字母"
            
            return True, f"有效的{market.upper()}代码"
            
        except Exception as e:
            return False, f"代码验证失败: {str(e)}"

    # 兼容旧版本的方法名
    def get_fundamental_data(self, stock_code):
        """兼容方法：获取基本面数据"""
        return self.get_comprehensive_fundamental_data(stock_code)
    
    def get_news_data(self, stock_code, days=30):
        """兼容方法：获取新闻数据"""
        return self.get_comprehensive_news_data(stock_code, days)
    
    def calculate_news_sentiment(self, news_data):
        """兼容方法：计算新闻情绪"""
        return self.calculate_advanced_sentiment_analysis(news_data)
    
    def get_sentiment_analysis(self, stock_code):
        """兼容方法：获取情绪分析"""
        news_data = self.get_comprehensive_news_data(stock_code)
        return self.calculate_advanced_sentiment_analysis(news_data)

    # ============================================================
    # 👇👇👇 [新增] V5.3 策略组件 (已适配您的 config.json) 👇👇👇
    # ============================================================

    def _calculate_strategy_features(self, df):
        """策略专用指标计算"""
        try:
            df = df.copy()
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA20_slope'] = df['MA20'].diff()
            
            std = df['close'].rolling(20).std()
            mid = df['MA20']
            upper = mid + 2 * std
            lower = mid - 2 * std
            range_bb = upper - lower
            
            df['bb_pos'] = 0.5
            mask = range_bb > 0
            df.loc[mask, 'bb_pos'] = (df.loc[mask, 'close'] - lower[mask]) / range_bb[mask]
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = exp12 - exp26
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD_Bar'] = 2 * (df['DIF'] - df['DEA'])
            
            vol_ma20 = df['volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['volume'] / vol_ma20
            
            return df
        except Exception as e:
            self.logger.error(f"策略指标计算错误: {e}")
            return df

    def _check_v5_rules(self, df_slice):
        """V5.3 哑铃策略风控"""
        if df_slice.empty: return False, "数据不足"
        curr = df_slice.iloc[-1]
        prev = df_slice.iloc[-2]
        
        rsi = curr.get('RSI', 50)
        close = curr['close']
        ma5 = curr.get('MA5', 0)
        ma20 = curr.get('MA20', 0)
        ma20_slope = curr.get('MA20_slope', 0)
        bb_pos = curr.get('bb_pos', 0.5)
        
        if rsi > 70: return False, f"RSI过热({rsi:.1f}>70)"

        is_uptrend = close > ma20 
        if is_uptrend:
            if ma20_slope < -0.01: return False, f"MA20趋势向下({ma20_slope:.3f})"
            bias = (close - ma20) / ma20 * 100
            if bias > 8: return False, f"乖离率过大({bias:.1f}%)"
            if close < ma5 and curr.get('change_pct', 0) < -3: return False, "破位大跌"
        else:
            is_oversold = rsi < 35 
            is_boll_low = bb_pos < 0.15
            is_shrink = curr.get('Vol_Ratio', 1.0) < 0.8
            if not ((is_oversold or is_boll_low) and is_shrink):
                return False, "左侧条件不足(需超跌+缩量)"

        try:
            macd_val = curr.get('MACD_Bar', 0)
            prev_macd = prev.get('MACD_Bar', 0)
            if macd_val < -0.2 and macd_val < prev_macd: return False, "MACD加速下跌"
        except: pass

        return True, "符合策略"

    def _build_strategy_prompt(self, df_enriched):
        """构建策略 Prompt"""
        curr = df_enriched.iloc[-1]
        recent = df_enriched.tail(10)
        table = "| 日期 | 收盘 | 涨跌% | MA20 | MACD | 量比 |\n|---|---|---|---|---|---|\n"
        for d, r in recent.iterrows():
            d_str = d.strftime('%m-%d')
            mac_icon = "🔴" if r['MACD_Bar'] > 0 else "🟢"
            table += f"| {d_str} | {r['close']:.2f} | {r.get('change_pct',0):.2f} | {r['MA20']:.2f} | {mac_icon} | {r.get('Vol_Ratio',0):.1f} |\n"

        return f"""
你是一名资深量化交易员，擅长【哑铃策略】。
【近期数据】
{table}
【当前指标】
- 价格: {curr['close']} (MA20: {curr.get('MA20',0):.2f})
- 布林位置: {curr.get('bb_pos',0.5):.2f}
- RSI: {curr.get('RSI',50):.1f}
【任务】
判断当前是**右侧顺势**还是**左侧震荡**，并给出操作建议。
输出JSON: action (BUY/HOLD/SELL), confidence (0-100), reason (简短理由)。
"""

    def _call_strategy_ai(self, prompt):
        """
        [关键修改] 策略专用 AI 调用
        兼容您的 config.json 格式 (简单的字符串 key)
        """
        try:
            import openai
            
            # === 直接获取字符串格式的 Key ===
            api_key = self.api_keys.get('openai')
            if not api_key:
                return '{"action": "HOLD", "confidence": 0, "reason": "No API Key"}'

            # 设置 Key 和 Base URL
            openai.api_key = api_key
            api_base = self.config.get('ai', {}).get('api_base_urls', {}).get('openai')
            if api_base:
                openai.api_base = api_base
            
            # 获取模型配置
            model = self.config.get('ai', {}).get('models', {}).get('openai', 'gpt-4o-mini')

            # 调用
            if hasattr(openai, 'OpenAI'): # 新版 SDK
                client = openai.OpenAI(api_key=api_key, base_url=api_base)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional trader. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            else: # 旧版 SDK 兼容
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional trader. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"策略AI调用出错: {e}")
            return '{"action": "HOLD", "confidence": 0, "reason": "API Error"}'

# 为了保持向后兼容，创建一个别名
WebStockAnalyzer = EnhancedWebStockAnalyzer


def main():
    """主函数"""
    analyzer = EnhancedWebStockAnalyzer()
    
    # 显示支持的市场
    markets = analyzer.get_supported_markets()
    print(f"支持的市场: {', '.join([m['name'] for m in markets])}")
    
    # 测试分析 - 包含多个市场的
    test_stocks = [
        '000001',  # A：平安银行
        '00700',   # g：腾讯
        'AAPL',    # m：苹果
        '600036',  # A：招商银行
        '00388',   # g：香g交易所
        'TSLA'     # m：特斯拉
    ]
    
    for stock_code in test_stocks:
        try:
            print(f"\n=== 开始多市场增强版分析 {stock_code} ===")
            
            # 验证代码
            is_valid, message = analyzer.validate_stock_code(stock_code)
            print(f"代码验证: {message}")
            
            if not is_valid:
                continue
            
            # 定义流式回调函数
            def print_stream(content):
                print(content, end='', flush=True)
            
            report = analyzer.analyze_stock(stock_code, enable_streaming=True, stream_callback=print_stream)
            
            print(f"\n代码: {report['stock_code']} (原始: {report['original_code']})")
            print(f"名称: {report['stock_name']}")
            print(f"交易市场: {report['market'].upper()}")
            print(f"计价货币: {report['market_info'].get('currency', 'Unknown')}")
            print(f"当前价格: {report['price_info']['current_price']:.2f}")
            print(f"涨跌幅: {report['price_info']['price_change']:.2f}%")
            print(f"财务指标数量: {report['data_quality']['financial_indicators_count']}")
            print(f"新闻数据量: {report['data_quality']['total_news_count']}")
            print(f"综合得分: {report['scores']['comprehensive']:.1f}")
            print(f"投资建议: {report['recommendation']}")
            print("=" * 60)
            
        except Exception as e:
            print(f"分析 {stock_code} 失败: {e}")


if __name__ == "__main__":
    main()

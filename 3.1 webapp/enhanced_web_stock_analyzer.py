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
    
    def __init__(self, config_file='config.json'):
        """初始化分析器"""
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        
        # 加载配置文件
        self.config = self._load_config()
        
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

    def get_stock_data(self, stock_code, period='1y'):
        """获取股票数据（带本地缓存 + 内存缓存 + 网络请求）"""
        # 1. 标准化代码
        stock_code, market = self.normalize_stock_code(stock_code)
        
        # --- 第一层：本地文件缓存检查 ---
        # 缓存文件名: us_stock_QQQ_20251225.csv
        today_str = datetime.now().strftime('%Y%m%d')
        cache_filename = f"{market}_{stock_code}_{today_str}.csv"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # 如果本地有今天的文件，直接读
        if os.path.exists(cache_path):
            self.logger.info(f"📦 命中本地文件缓存: {cache_filename}")
            try:
                df = pd.read_csv(cache_path)
                # 尝试恢复日期索引
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                # 兼容不同CSV格式，如果第一列是日期但叫 'Unnamed: 0'
                elif df.index.name != 'date' and 'date' not in df.columns:
                    df.index = pd.to_datetime(df.iloc[:, 0])
                    df = df.iloc[:, 1:] 
                return df
            except Exception as e:
                self.logger.warning(f"读取本地缓存失败，准备重新下载: {e}")
                # 读取失败不返回，继续往下走网络请求

        # --- 第二层：内存缓存检查 (兼容你原有逻辑) ---
        cache_key = f"{market}_{stock_code}"
        if cache_key in self.price_cache:
            cache_time, data = self.price_cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                self.logger.info(f"⚡ 使用内存缓存数据: {cache_key}")
                return data

        # --- 第三层：网络请求 (核心逻辑) ---
        self.logger.info(f"🌐 正在从网络下载 {stock_code} 数据...")
        
        try:
            import akshare as ak
            
            end_date = datetime.now().strftime('%Y%m%d')
            days = self.analysis_params.get('technical_period_days', 180)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            stock_data = None
            
            # === A股 ===
            if market == 'a_stock':
                stock_data = ak.stock_zh_a_hist(
                    symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
                )
            
            # === 港股 ===
            elif market == 'hk_stock':
                try:
                    stock_data = ak.stock_hk_hist(
                        symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
                    )
                except:
                    # 备用接口
                    stock_data = ak.stock_hk_daily(symbol=stock_code, adjust="qfq")
                    if not stock_data.empty:
                        stock_data = stock_data[stock_data.index >= start_date]

            # === 美股 (Stooq源) ===
            elif market == 'us_stock':
                try:
                    start_dt = datetime.now() - timedelta(days=days)
                    df = web.DataReader(stock_code, 'stooq', start=start_dt, end=datetime.now())
                    
                    if df is not None and not df.empty:
                        df = df.sort_index(ascending=True).reset_index()
                        df = df.rename(columns={
                            "Date": "date", "Open": "open", "High": "high", 
                            "Low": "low", "Close": "close", "Volume": "volume"
                        })
                        df['date'] = pd.to_datetime(df['date'])
                        if df['date'].dt.tz is not None:
                            df['date'] = df['date'].dt.tz_localize(None)
                        stock_data = df[df['date'] >= pd.to_datetime(start_date)]
                except Exception as e:
                    self.logger.error(f"Stooq 获取失败: {e}")

            # 检查数据有效性
            if stock_data is None or stock_data.empty:
                raise ValueError(f"无法获取 {market.upper()} {stock_code} 的数据")

            # 标准化列名
            stock_data = self._standardize_price_data_columns(stock_data, market)

            # --- 保存缓存 (内存 + 本地文件) ---
            # 1. 存内存
            self.price_cache[cache_key] = (datetime.now(), stock_data)
            
            # 2. 存本地文件
            try:
                stock_data.to_csv(cache_path)
                self.logger.info(f"💾 数据已保存至本地: {cache_path}")
            except Exception as e:
                self.logger.error(f"写入本地文件失败: {e}")

            return stock_data

        except Exception as e:
            self.logger.error(f"获取数据全流程失败: {str(e)}")
            return pd.DataFrame()

    def _standardize_price_data_columns(self, stock_data, market):
        """标准化价格数据列名"""
        try:
            actual_columns = len(stock_data.columns)
            self.logger.info(f"获取到 {actual_columns} 列数据，列名: {list(stock_data.columns)}")
            
            # 根据市场和实际列数进行映射
            if market == 'a_stock':
                # A列名映射
                if actual_columns >= 11:
                    standard_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate']
                else:
                    standard_columns = [f'col_{i}' for i in range(actual_columns)]
                    
            elif market == 'hk_stock':
                # g列名映射
                if actual_columns >= 6:
                    standard_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
                    if actual_columns > 6:
                        standard_columns.extend([f'extra_{i}' for i in range(actual_columns - 6)])
                else:
                    standard_columns = [f'col_{i}' for i in range(actual_columns)]
                    
            elif market == 'us_stock':
                # m列名映射
                if actual_columns >= 6:
                    standard_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
                    if actual_columns > 6:
                        standard_columns.extend([f'extra_{i}' for i in range(actual_columns - 6)])
                else:
                    standard_columns = [f'col_{i}' for i in range(actual_columns)]
            
            # 创建列名映射
            column_mapping = dict(zip(stock_data.columns, standard_columns))
            stock_data = stock_data.rename(columns=column_mapping)
            
            # 确保必要的列存在
            required_columns = ['close', 'open', 'high', 'low', 'volume']
            for col in required_columns:
                if col not in stock_data.columns:
                    similar_cols = [c for c in stock_data.columns if col in c.lower() or c.lower() in col]
                    if similar_cols:
                        stock_data[col] = stock_data[similar_cols[0]]
                        self.logger.info(f"✓ 映射列 {similar_cols[0]} -> {col}")
            
            # 处理日期列
            try:
                if 'date' in stock_data.columns:
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    stock_data = stock_data.set_index('date')
                else:
                    stock_data.index = pd.to_datetime(stock_data.index)
            except Exception as e:
                self.logger.warning(f"日期处理失败: {e}")
            
            # 确保数值列为数值类型
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            for col in numeric_columns:
                if col in stock_data.columns:
                    try:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                    except:
                        pass
            
            return stock_data
            
        except Exception as e:
            self.logger.warning(f"列名标准化失败: {e}")
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
        """计算技术指标（通用于多市场）"""
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
            try:
                price_data['ma5'] = price_data['close'].rolling(window=5, min_periods=1).mean()
                price_data['ma10'] = price_data['close'].rolling(window=10, min_periods=1).mean()
                price_data['ma20'] = price_data['close'].rolling(window=20, min_periods=1).mean()
                price_data['ma60'] = price_data['close'].rolling(window=60, min_periods=1).mean()
                
                latest_price = safe_float(price_data['close'].iloc[-1])
                ma5 = safe_float(price_data['ma5'].iloc[-1], latest_price)
                ma10 = safe_float(price_data['ma10'].iloc[-1], latest_price)
                ma20 = safe_float(price_data['ma20'].iloc[-1], latest_price)
                
                if latest_price > ma5 > ma10 > ma20:
                    technical_analysis['ma_trend'] = '多头排列'
                elif latest_price < ma5 < ma10 < ma20:
                    technical_analysis['ma_trend'] = '空头排列'
                else:
                    technical_analysis['ma_trend'] = '震荡整理'
                
            except Exception as e:
                technical_analysis['ma_trend'] = '计算失败'
            
            # RSI指标
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
                
            except Exception as e:
                technical_analysis['rsi'] = 50.0
            
            # MACD指标
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
                
            except Exception as e:
                technical_analysis['macd_signal'] = '计算失败'
            
            # 布林带
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
                
            except Exception as e:
                technical_analysis['bb_position'] = 0.5
            
            # 成交量分析
            try:
                volume_window = min(20, len(price_data))
                avg_volume = price_data['volume'].rolling(window=volume_window, min_periods=1).mean().iloc[-1]
                recent_volume = safe_float(price_data['volume'].iloc[-1])
                
                if 'change_pct' in price_data.columns:
                    price_change = safe_float(price_data['change_pct'].iloc[-1])
                elif len(price_data) >= 2:
                    current_price = safe_float(price_data['close'].iloc[-1])
                    prev_price = safe_float(price_data['close'].iloc[-2])
                    if prev_price > 0:
                        price_change = ((current_price - prev_price) / prev_price) * 100
                    else:
                        price_change = 0
                else:
                    price_change = 0
                
                avg_volume = safe_float(avg_volume, recent_volume)
                if recent_volume > avg_volume * 1.5:
                    technical_analysis['volume_status'] = '放量上涨' if price_change > 0 else '放量下跌'
                elif recent_volume < avg_volume * 0.5:
                    technical_analysis['volume_status'] = '缩量调整'
                else:
                    technical_analysis['volume_status'] = '温和放量'
                
            except Exception as e:
                technical_analysis['volume_status'] = '数据不足'
            
            return technical_analysis
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败: {str(e)}")
            return self._get_default_technical_analysis()

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
            
            ma_trend = technical_analysis.get('ma_trend', '数据不足')
            if ma_trend == '多头排列':
                score += 20
            elif ma_trend == '空头排列':
                score -= 20
            
            rsi = technical_analysis.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 10
            elif rsi < 30:
                score += 5
            elif rsi > 70:
                score -= 5
            
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
        """计算基本面得分（支持多市场）"""
        try:
            score = 50
            
            # 财务指标评分
            financial_indicators = fundamental_data.get('financial_indicators', {})
            if len(financial_indicators) >= 10:  # 调整阈值以适应不同市场
                score += 15
                
                # 通用盈利能力评分（适应不同市场的指标名称）
                roe = (financial_indicators.get('净资产收益率', 0) or 
                      financial_indicators.get('ROE', 0) or 
                      financial_indicators.get('roe', 0))
                if roe > 15:
                    score += 10
                elif roe > 10:
                    score += 5
                elif roe < 5:
                    score -= 5
                
                # 通用估值指标
                pe_ratio = (financial_indicators.get('市盈率', 0) or 
                           financial_indicators.get('PE_Ratio', 0) or 
                           financial_indicators.get('pe_ratio', 0))
                if 0 < pe_ratio < 20:
                    score += 10
                elif pe_ratio > 50:
                    score -= 5
                
                # 债务水平评估
                debt_ratio = (financial_indicators.get('资产负债率', 50) or 
                             financial_indicators.get('debt_ratio', 50))
                if debt_ratio < 30:
                    score += 5
                elif debt_ratio > 70:
                    score -= 10
            
            # 估值评分
            valuation = fundamental_data.get('valuation', {})
            if valuation:
                score += 10
            
            # 业绩预告评分
            performance_forecast = fundamental_data.get('performance_forecast', [])
            if performance_forecast:
                score += 10
            
            score = max(0, min(100, score))
            return score
            
        except Exception as e:
            self.logger.error(f"基本面评分失败: {str(e)}")
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
                                        fundamental_data, sentiment_analysis, price_info, market=None):
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
    * 🔴 **强阻力位**：[基于波动率和布林带估算价格]
    * 🟢 **强支撑位**：[基于波动率和布林带估算价格]

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
            
            # 构建增强版AI分析提示词
            prompt = self._build_enhanced_ai_analysis_prompt(
                stock_code, stock_name, scores, technical_analysis, 
                fundamental_data, sentiment_analysis, price_info, market
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
            import openai
            
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
                'market': market
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
            # ==========================================

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
                }
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

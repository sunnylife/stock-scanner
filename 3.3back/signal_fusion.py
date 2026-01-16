# signal_fusion.py
import numpy as np

class SignalFusionEngine:
    """信号融合引擎"""
    
    def __init__(self):
        # 信号权重配置（可根据市场调整）
        self.weights = {
            'technical': {
                'ma_trend': 0.15,
                'rsi': 0.10,
                'macd': 0.10,
                'kdj': 0.08,
                'volume': 0.07
            },
            'fundamental': 0.20,
            'sentiment': 0.10,
            'money_flow': 0.10,
            'risk': 0.10
        }
    
    def fuse_signals(self, signals_dict):
        """融合多个信号源"""
        fused_score = 50  # 基准分
        
        # 1. 技术信号融合
        tech_score = 0
        tech_weight_total = 0
        
        for indicator, weight in self.weights['technical'].items():
            if indicator in signals_dict.get('technical', {}):
                value = signals_dict['technical'][indicator]
                tech_score += self._normalize_tech_value(indicator, value) * weight
                tech_weight_total += weight
        
        if tech_weight_total > 0:
            tech_score = tech_score / tech_weight_total * 100
            fused_score += (tech_score - 50) * 0.3
        
        # 2. 基本面信号
        if 'fundamental' in signals_dict:
            fund_score = signals_dict['fundamental']
            fused_score += (fund_score - 50) * 0.2
        
        # 3. 情绪信号
        if 'sentiment' in signals_dict:
            sent_score = signals_dict['sentiment']
            fused_score += (sent_score - 50) * 0.1
        
        # 4. 资金流信号
        if 'money_flow' in signals_dict:
            flow_score = signals_dict['money_flow']
            fused_score += (flow_score - 50) * 0.1
        
        # 5. 风险信号（风险越高，分数越低）
        if 'risk' in signals_dict:
            risk_score = 100 - signals_dict['risk']  # 风险转为负向指标
            fused_score += (risk_score - 50) * 0.1
        
        # 确保分数在0-100之间
        fused_score = max(0, min(100, fused_score))
        
        # 生成最终信号
        if fused_score >= 75:
            action = "BUY"
        elif fused_score >= 60:
            action = "HOLD"
        elif fused_score >= 40:
            action = "WATCH"
        else:
            action = "SELL"
        
        return {
            'action': action,
            'confidence': fused_score,
            'fused_score': round(fused_score, 1),
            'details': {
                'technical': round(tech_score, 1) if tech_weight_total > 0 else None,
                'fundamental': signals_dict.get('fundamental'),
                'sentiment': signals_dict.get('sentiment'),
                'money_flow': signals_dict.get('money_flow'),
                'risk': signals_dict.get('risk')
            }
        }
    
    def _normalize_tech_value(self, indicator, value):
        """标准化技术指标值"""
        if indicator == 'ma_trend':
            return 80 if value == '多头排列' else 20 if value == '空头排列' else 50
        elif indicator == 'rsi':
            if value < 30:
                return 80  # 超卖，机会
            elif value < 50:
                return 60
            elif value < 70:
                return 40
            else:
                return 20  # 超买，风险
        elif indicator == 'macd':
            if '金叉' in str(value):
                return 80
            elif '死叉' in str(value):
                return 20
            else:
                return 50
        elif indicator == 'kdj':
            if '金叉' in str(value):
                return 70
            elif '死叉' in str(value):
                return 30
            else:
                return 50
        elif indicator == 'volume':
            if value > 1.5:
                return 70  # 放量
            elif value > 0.8:
                return 50  # 正常
            else:
                return 30  # 缩量
        
        return 50  # 默认值
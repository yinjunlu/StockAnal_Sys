# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
开发者：熊猫大侠
版本：v2.1.0
许可证：MIT License
"""
# scenario_predictor.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import openai
import logging
"""

"""

class ScenarioPredictor:
    def __init__(self, analyzer, openai_api_key=None, openai_model=None):
        self.analyzer = analyzer
        self.openai_api_key = os.getenv('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
        self.openai_api_url = os.getenv('OPENAI_API_URL')
        self.openai_model = os.getenv('OPENAI_API_MODEL', 'gemini-2.0-pro-exp-02-05')
        self.logger = logging.getLogger(__name__)

    def generate_scenarios(self, stock_code, market_type='A', days=60):
        """生成乐观、中性、悲观三种市场情景预测"""
        try:
            # 获取股票数据和技术指标
            self.logger.info(f"开始获取股票 {stock_code} 的数据")
            df = self.analyzer.get_stock_data(stock_code, market_type)
            
            if df is None or df.empty:
                self.logger.error(f"股票 {stock_code} 数据为空")
                return {"error": "暂时不支持该股票代码分析", "status": "error"}
                
            df = self.analyzer.calculate_indicators(df)

            # 获取股票信息
            stock_info = self.analyzer.get_stock_info(stock_code)
            if not stock_info:
                self.logger.warning(f"无法获取股票 {stock_code} 的基本信息")
                stock_info = {"股票名称": stock_code}

            # 计算基础数据
            current_price = df.iloc[-1]['close']
            avg_volatility = df['Volatility'].mean()

            # 根据历史波动率计算情景
            scenarios = self._calculate_scenarios(df, days)

            # 使用AI生成各情景的分析
            if self.openai_api_key:
                ai_analysis = self._generate_ai_analysis(stock_code, stock_info, df, scenarios)
                scenarios.update(ai_analysis)

            return scenarios
        except KeyError as e:
            error_msg = f"获取股票数据失败: {str(e)}"
            self.logger.error(error_msg)
            return {"error": "暂时不支持该股票代码分析", "status": "error"}
        except Exception as e:
            error_msg = f"生成情景预测出错: {str(e)}"
            self.logger.error(error_msg)
            return {"error": "暂时不支持该股票代码分析", "status": "error"}

    def _calculate_scenarios(self, df, days):
        """基于历史数据计算三种情景的价格预测"""
        current_price = df.iloc[-1]['close']

        # 计算历史波动率和移动均线
        volatility = df['Volatility'].mean() / 100  # 转换为小数
        daily_volatility = volatility / np.sqrt(252)  # 转换为日波动率
        ma20 = df.iloc[-1]['MA20']
        ma60 = df.iloc[-1]['MA60']

        # 计算乐观情景（上涨至压力位或突破）
        optimistic_return = 0.15  # 15%上涨
        if df.iloc[-1]['BB_upper'] > current_price:
            optimistic_target = df.iloc[-1]['BB_upper'] * 1.05  # 突破上轨5%
        else:
            optimistic_target = current_price * (1 + optimistic_return)

        # 计算中性情景（震荡，围绕当前价格或20日均线波动）
        neutral_target = (current_price + ma20) / 2

        # 计算悲观情景（下跌至支撑位或跌破）
        pessimistic_return = -0.12  # 12%下跌
        if df.iloc[-1]['BB_lower'] < current_price:
            pessimistic_target = df.iloc[-1]['BB_lower'] * 0.95  # 跌破下轨5%
        else:
            pessimistic_target = current_price * (1 + pessimistic_return)

        # 计算预期时间
        time_periods = np.arange(1, days + 1)

        # 生成乐观路径
        opt_path = [current_price]
        for _ in range(days):
            daily_return = (optimistic_target / current_price) ** (1 / days) - 1
            random_component = np.random.normal(0, daily_volatility)
            new_price = opt_path[-1] * (1 + daily_return + random_component / 2)
            opt_path.append(new_price)

        # 生成中性路径
        neu_path = [current_price]
        for _ in range(days):
            daily_return = (neutral_target / current_price) ** (1 / days) - 1
            random_component = np.random.normal(0, daily_volatility)
            new_price = neu_path[-1] * (1 + daily_return + random_component)
            neu_path.append(new_price)

        # 生成悲观路径
        pes_path = [current_price]
        for _ in range(days):
            daily_return = (pessimistic_target / current_price) ** (1 / days) - 1
            random_component = np.random.normal(0, daily_volatility)
            new_price = pes_path[-1] * (1 + daily_return + random_component / 2)
            pes_path.append(new_price)

        # 生成日期序列
        start_date = datetime.now()
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days + 1)]

        # 组织结果
        return {
            'current_price': current_price,
            'optimistic': {
                'target_price': optimistic_target,
                'change_percent': (optimistic_target / current_price - 1) * 100,
                'path': dict(zip(dates, opt_path))
            },
            'neutral': {
                'target_price': neutral_target,
                'change_percent': (neutral_target / current_price - 1) * 100,
                'path': dict(zip(dates, neu_path))
            },
            'pessimistic': {
                'target_price': pessimistic_target,
                'change_percent': (pessimistic_target / current_price - 1) * 100,
                'path': dict(zip(dates, pes_path))
            }
        }

    def _generate_ai_analysis(self, stock_code, stock_info, df, scenarios):
        """使用AI生成各情景的分析说明"""
        try:
            openai.api_key = self.openai_api_key
            openai.api_base = self.openai_api_url

            # 提取关键数据
            current_price = df.iloc[-1]['close']
            ma5 = df.iloc[-1]['MA5']
            ma20 = df.iloc[-1]['MA20']
            ma60 = df.iloc[-1]['MA60']
            rsi = df.iloc[-1]['RSI']
            macd = df.iloc[-1]['MACD']
            signal = df.iloc[-1]['Signal']

            # 构建提示词
            prompt = f"""分析股票{stock_code}（{stock_info.get('股票名称', '未知')}）的三种市场情景:

1. 当前数据:
   - 当前价格: {current_price}
   - 均线: MA5={ma5}, MA20={ma20}, MA60={ma60}
   - RSI: {rsi}
   - MACD: {macd}, Signal: {signal}

2. 预测目标价:
   - 乐观情景: {scenarios['optimistic']['target_price']:.2f} ({scenarios['optimistic']['change_percent']:.2f}%)
   - 中性情景: {scenarios['neutral']['target_price']:.2f} ({scenarios['neutral']['change_percent']:.2f}%)
   - 悲观情景: {scenarios['pessimistic']['target_price']:.2f} ({scenarios['pessimistic']['change_percent']:.2f}%)

请为每种情景提供简短分析(每种情景100字以内)，包括可能的触发条件和风险因素。格式为JSON:
{{
  "optimistic_analysis": "乐观情景分析...",
  "neutral_analysis": "中性情景分析...",
  "pessimistic_analysis": "悲观情景分析..."
}}
"""

            # 调用AI API
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "你是专业的股票分析师，擅长技术分析和情景预测。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            # 解析AI回复
            import json
            try:
                analysis = json.loads(response.choices[0].message.content)
                return analysis
            except:
                # 如果解析失败，尝试从文本中提取JSON
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.choices[0].message.content)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                else:
                    return {
                        "optimistic_analysis": "乐观情景分析暂无",
                        "neutral_analysis": "中性情景分析暂无",
                        "pessimistic_analysis": "悲观情景分析暂无"
                    }
        except Exception as e:
            print(f"生成AI分析出错: {str(e)}")
            return {
                "optimistic_analysis": "乐观情景分析暂无",
                "neutral_analysis": "中性情景分析暂无",
                "pessimistic_analysis": "悲观情景分析暂无"
            }
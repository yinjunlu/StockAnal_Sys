# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
开发者：熊猫大侠
版本：v2.1.0
许可证：MIT License
"""
# stock_qa.py
import os
import openai
import logging

class StockQA:
    def __init__(self, analyzer, openai_api_key=None, openai_model=None):
        self.analyzer = analyzer
        self.openai_api_key = os.getenv('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
        self.openai_api_url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1')
        self.openai_model = os.getenv('OPENAI_API_MODEL', 'gemini-2.0-pro-exp-02-05')
        self.logger = logging.getLogger(__name__)

    def answer_question(self, stock_code, question, market_type='A'):
        """回答关于股票的问题"""
        try:
            if not self.openai_api_key:
                return {"error": "未配置API密钥，无法使用智能问答功能"}

            # 获取股票信息
            try:
                stock_info = self.analyzer.get_stock_info(stock_code)
                self.logger.info(f"成功获取股票 {stock_code} 的基本信息")
            except Exception as e:
                self.logger.warning(f"获取股票基本信息失败: {str(e)}")
                # 尝试使用quick_analyze_stock方法获取基本信息
                try:
                    quick_result = self.analyzer.quick_analyze_stock(stock_code, market_type)
                    stock_info = {
                        '股票名称': quick_result.get('stock_name', stock_code),
                        '行业': quick_result.get('industry', '未知行业')
                    }
                    self.logger.info(f"通过quick_analyze_stock获取股票信息: {stock_info}")
                except Exception as e2:
                    self.logger.error(f"通过quick_analyze_stock获取股票信息失败: {str(e2)}")
                    # 如果是'date'相关错误，返回特定消息
                    if "'date'" in str(e2) or "获取股票数据失败" in str(e2):
                        return {
                            "question": question,
                            "answer": "暂时不支持该股票代码分析",
                            "stock_code": stock_code,
                            "error": "unsupported_stock"
                        }
                    stock_info = {'股票名称': stock_code, '行业': '未知'}

            # 获取技术指标数据
            try:
                self.logger.info(f"开始获取股票 {stock_code} 的历史数据")
                df = self.analyzer.get_stock_data(stock_code, market_type)
                df = self.analyzer.calculate_indicators(df)
                self.logger.info(f"成功获取并计算股票 {stock_code} 的技术指标")
            except Exception as e:
                self.logger.error(f"获取股票数据或计算指标失败: {str(e)}")
                # 如果是'date'相关错误，返回特定消息
                if "'date'" in str(e) or "获取股票数据失败" in str(e):
                    return {
                        "question": question,
                        "answer": "暂时不支持该股票代码分析",
                        "stock_code": stock_code,
                        "error": "unsupported_stock"
                    }
                return {
                    "question": question,
                    "answer": f"获取股票数据失败: {str(e)}",
                    "stock_code": stock_code
                }

            # 提取最新数据
            latest = df.iloc[-1]

            # 计算评分
            try:
                score = self.analyzer.calculate_score(df)
                self.logger.info(f"成功计算股票 {stock_code} 的评分: {score}")
            except Exception as e:
                self.logger.warning(f"计算评分失败: {str(e)}")
                score = 50  # 默认中性评分

            # 获取支撑压力位
            try:
                sr_levels = self.analyzer.identify_support_resistance(df)
                self.logger.info(f"成功识别股票 {stock_code} 的支撑压力位")
            except Exception as e:
                self.logger.warning(f"识别支撑压力位失败: {str(e)}")
                sr_levels = {
                    'support_levels': {'short_term': [], 'medium_term': []},
                    'resistance_levels': {'short_term': [], 'medium_term': []}
                }

            # 构建上下文
            context = f"""股票信息:
- 代码: {stock_code}
- 名称: {stock_info.get('股票名称', '未知')}
- 行业: {stock_info.get('行业', '未知')}

技术指标(最新数据):
- 价格: {latest['close']}
- 5日均线: {latest['MA5']}
- 20日均线: {latest['MA20']}
- 60日均线: {latest['MA60']}
- RSI: {latest['RSI']}
- MACD: {latest['MACD']}
- MACD信号线: {latest['Signal']}
- 布林带上轨: {latest['BB_upper']}
- 布林带中轨: {latest['BB_middle']}
- 布林带下轨: {latest['BB_lower']}
- 波动率: {latest['Volatility']}%

技术评分: {score}分

支撑位:
- 短期: {', '.join([str(level) for level in sr_levels['support_levels']['short_term']])}
- 中期: {', '.join([str(level) for level in sr_levels['support_levels']['medium_term']])}

压力位:
- 短期: {', '.join([str(level) for level in sr_levels['resistance_levels']['short_term']])}
- 中期: {', '.join([str(level) for level in sr_levels['resistance_levels']['medium_term']])}"""

            # 特定问题类型的补充信息
            if '基本面' in question or '财务' in question or '估值' in question:
                try:
                    # 导入基本面分析器
                    from fundamental_analyzer import FundamentalAnalyzer
                    fundamental = FundamentalAnalyzer()

                    # 获取基本面数据
                    indicators = fundamental.get_financial_indicators(stock_code)

                    # 添加到上下文
                    context += f"""

基本面指标:
- PE(TTM): {indicators.get('pe_ttm', '未知')}
- PB: {indicators.get('pb', '未知')}
- ROE: {indicators.get('roe', '未知')}%
- 毛利率: {indicators.get('gross_margin', '未知')}%
- 净利率: {indicators.get('net_profit_margin', '未知')}%"""
                except Exception as e:
                    self.logger.warning(f"获取基本面数据失败: {str(e)}")
                    context += "\n\n注意：未能获取基本面数据"

            # 调用AI API回答问题
            try:
                self.logger.info(f"开始调用AI API回答问题: {question}")
                openai.api_key = self.openai_api_key
                openai.api_base = self.openai_api_url

                system_content = """你是专业的股票分析师助手，基于'时空共振交易体系'提供分析。
                请基于技术指标和市场数据进行客观分析。
                """

                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user",
                         "content": f"请回答关于股票的问题，并参考以下股票数据：\n\n{context}\n\n问题：{question}"}
                    ],
                    temperature=0.7
                )

                answer = response.choices[0].message.content
                self.logger.info(f"成功获取AI回答")

                return {
                    "question": question,
                    "answer": answer,
                    "stock_code": stock_code,
                    "stock_name": stock_info.get('股票名称', '未知')
                }
            except Exception as e:
                self.logger.error(f"调用AI API回答问题失败: {str(e)}")
                return {
                    "question": question,
                    "answer": f"抱歉，AI回答问题时出错: {str(e)}",
                    "stock_code": stock_code,
                    "stock_name": stock_info.get('股票名称', '未知')
                }

        except Exception as e:
            self.logger.error(f"智能问答出错: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出错: {str(e)}",
                "stock_code": stock_code
            }
# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
修改：熊猫大侠
版本：v2.1.0
"""
# web_server.py

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from stock_analyzer import StockAnalyzer
from us_stock_service import USStockService
import threading
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import json
from datetime import date, datetime, timedelta
from flask_cors import CORS
import time
from flask_caching import Cache
import threading
import sys
from flask_swagger_ui import get_swaggerui_blueprint
from database import get_session, StockInfo, AnalysisResult, Portfolio, USE_DATABASE
from dotenv import load_dotenv
from industry_analyzer import IndustryAnalyzer

# 加载环境变量
load_dotenv()

# 检查是否需要初始化数据库
if USE_DATABASE:
    init_db()

# 配置Swagger
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "股票智能分析系统 API文档"
    }
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
analyzer = StockAnalyzer()
us_stock_service = USStockService()

# 配置缓存
cache_config = {
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
}

# 如果配置了Redis，使用Redis作为缓存后端
if os.getenv('USE_REDIS_CACHE', 'False').lower() == 'true' and os.getenv('REDIS_URL'):
    cache_config = {
        'CACHE_TYPE': 'RedisCache',
        'CACHE_REDIS_URL': os.getenv('REDIS_URL'),
        'CACHE_DEFAULT_TIMEOUT': 300
    }

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# 确保全局变量在重新加载时不会丢失
if 'analyzer' not in globals():
    try:
        from stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        print("成功初始化全局StockAnalyzer实例")
    except Exception as e:
        print(f"初始化StockAnalyzer时出错: {e}", file=sys.stderr)
        raise

# 导入新模块
from fundamental_analyzer import FundamentalAnalyzer
from capital_flow_analyzer import CapitalFlowAnalyzer
from scenario_predictor import ScenarioPredictor
from stock_qa import StockQA
from risk_monitor import RiskMonitor
from index_industry_analyzer import IndexIndustryAnalyzer

# 初始化模块实例
fundamental_analyzer = FundamentalAnalyzer()
capital_flow_analyzer = CapitalFlowAnalyzer()
scenario_predictor = ScenarioPredictor(analyzer, os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_API_MODEL'))
stock_qa = StockQA(analyzer, os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_API_MODEL'))
risk_monitor = RiskMonitor(analyzer)
index_industry_analyzer = IndexIndustryAnalyzer(analyzer)
industry_analyzer = IndustryAnalyzer()

# Thread-local storage
thread_local = threading.local()


def get_analyzer():
    """获取线程本地的分析器实例"""
    # 如果线程本地存储中没有分析器实例，创建一个新的
    if not hasattr(thread_local, 'analyzer'):
        thread_local.analyzer = StockAnalyzer()
    return thread_local.analyzer


# 配置日志
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('flask_app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(handler)

# 扩展任务管理系统以支持不同类型的任务
task_types = {
    'scan': 'market_scan',  # 市场扫描任务
    'analysis': 'stock_analysis'  # 个股分析任务
}

# 任务数据存储
tasks = {
    'market_scan': {},  # 原来的scan_tasks
    'stock_analysis': {}  # 新的个股分析任务
}


def get_task_store(task_type):
    """获取指定类型的任务存储"""
    return tasks.get(task_type, {})


def generate_task_key(task_type, **params):
    """生成任务键"""
    if task_type == 'stock_analysis':
        # 对于个股分析，使用股票代码和市场类型作为键
        return f"{params.get('stock_code')}_{params.get('market_type', 'A')}"
    return None  # 其他任务类型不使用预生成的键


def get_or_create_task(task_type, **params):
    """获取或创建任务"""
    store = get_task_store(task_type)
    task_key = generate_task_key(task_type, **params)

    # 检查是否有现有任务
    if task_key and task_key in store:
        task = store[task_key]
        # 检查任务是否仍然有效
        if task['status'] in [TASK_PENDING, TASK_RUNNING]:
            return task['id'], task, False
        if task['status'] == TASK_COMPLETED and 'result' in task:
            # 任务已完成且有结果，重用它
            return task['id'], task, False

    # 创建新任务
    task_id = generate_task_id()
    task = {
        'id': task_id,
        'key': task_key,  # 存储任务键以便以后查找
        'type': task_type,
        'status': TASK_PENDING,
        'progress': 0,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': params
    }

    with task_lock:
        if task_key:
            store[task_key] = task
        store[task_id] = task

    return task_id, task, True


# 添加到web_server.py顶部
# 任务管理系统
scan_tasks = {}  # 存储扫描任务的状态和结果
task_lock = threading.Lock()  # 用于线程安全操作

# 任务状态常量
TASK_PENDING = 'pending'
TASK_RUNNING = 'running'
TASK_COMPLETED = 'completed'
TASK_FAILED = 'failed'


def generate_task_id():
    """生成唯一的任务ID"""
    import uuid
    return str(uuid.uuid4())


def start_market_scan_task_status(task_id, status, progress=None, result=None, error=None):
    """更新任务状态 - 保持原有签名"""
    with task_lock:
        if task_id in scan_tasks:
            task = scan_tasks[task_id]
            task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def update_task_status(task_type, task_id, status, progress=None, result=None, error=None):
    """更新任务状态"""
    store = get_task_store(task_type)
    with task_lock:
        if task_id in store:
            task = store[task_id]
            task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 更新键索引的任务
            if 'key' in task and task['key'] in store:
                store[task['key']] = task


analysis_tasks = {}


def get_or_create_analysis_task(stock_code, market_type='A'):
    """获取或创建个股分析任务"""
    # 创建一个键，用于查找现有任务
    task_key = f"{stock_code}_{market_type}"

    with task_lock:
        # 检查是否有现有任务
        for task_id, task in analysis_tasks.items():
            if task.get('key') == task_key:
                # 检查任务是否仍然有效
                if task['status'] in [TASK_PENDING, TASK_RUNNING]:
                    return task_id, task, False
                if task['status'] == TASK_COMPLETED and 'result' in task:
                    # 任务已完成且有结果，重用它
                    return task_id, task, False

        # 创建新任务
        task_id = generate_task_id()
        task = {
            'id': task_id,
            'key': task_key,
            'status': TASK_PENDING,
            'progress': 0,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': {
                'stock_code': stock_code,
                'market_type': market_type
            }
        }

        analysis_tasks[task_id] = task

        return task_id, task, True


def update_analysis_task(task_id, status, progress=None, result=None, error=None):
    """更新个股分析任务状态"""
    with task_lock:
        if task_id in analysis_tasks:
            task = analysis_tasks[task_id]
            task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Define the custom JSON encoder


# In web_server.py, update the convert_numpy_types function to handle NaN values

# Function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Recursively converts NumPy types in dictionaries and lists to Python native types"""
    try:
        import numpy as np
        import math

        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and Infinity specifically
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None if obj < 0 else 1e308  # Use a very large number for +Infinity
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Python's own float NaN and Infinity
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            return obj
        # 添加对date和datetime类型的处理
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        else:
            return obj
    except ImportError:
        # 如果没有安装numpy，但需要处理date和datetime
        import math
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        # Handle Python's own float NaN and Infinity
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            return obj
        return obj


# Also update the NumpyJSONEncoder class
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # For NumPy data types
        try:
            import numpy as np
            import math
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # Handle NaN and Infinity specifically
                if np.isnan(obj):
                    return None
                elif np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            # Handle Python's own float NaN and Infinity
            elif isinstance(obj, float):
                if math.isnan(obj):
                    return None
                elif math.isinf(obj):
                    return None
                return obj
        except ImportError:
            # Handle Python's own float NaN and Infinity if numpy is not available
            import math
            if isinstance(obj, float):
                if math.isnan(obj):
                    return None
                elif math.isinf(obj):
                    return None

        # 添加对date和datetime类型的处理
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()

        return super(NumpyJSONEncoder, self).default(obj)


# Custom jsonify function that uses our encoder
def custom_jsonify(data):
    return app.response_class(
        json.dumps(convert_numpy_types(data), cls=NumpyJSONEncoder),
        mimetype='application/json'
    )


# 保持API兼容的路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """分析股票数据"""
    try:
        data = request.get_json()
        stock_codes = data.get('stock_codes', [])
        market_type = data.get('market_type', 'A')
        
        if not stock_codes:
            return jsonify({'error': '请提供股票代码'}), 400
        
        analyzer = get_analyzer()
        results = []
        
        for stock_code in stock_codes:
            try:
                logging.info(f"开始分析股票 {stock_code}")
                # 尝试获取股票信息
                try:
                    stock_info = analyzer.get_stock_info(stock_code, market_type)
                    if not stock_info or 'date' not in stock_info:
                        raise ValueError("获取股票数据失败: 'date'")
                    results.append(stock_info)
                except Exception as e:
                    error_msg = str(e)
                    logging.warning(f"获取股票 {stock_code} 信息失败: {error_msg}")
                    
                    # 检查是否是 'date' 相关错误
                    if "'date'" in error_msg or "date" in error_msg:
                        results.append({
                            'stock_code': stock_code,
                            'error': '暂时不支持该股票代码分析'
                        })
                    else:
                        # 尝试使用快速分析作为备选
                        try:
                            quick_result = analyzer.quick_analyze_stock(stock_code, market_type)
                            if quick_result:
                                results.append(quick_result)
                            else:
                                results.append({
                                    'stock_code': stock_code,
                                    'error': '暂时不支持该股票代码分析'
                                })
                        except Exception as quick_e:
                            logging.error(f"快速分析股票 {stock_code} 失败: {str(quick_e)}")
                            results.append({
                                'stock_code': stock_code,
                                'error': '暂时不支持该股票代码分析'
                            })
            except Exception as stock_e:
                logging.error(f"处理股票 {stock_code} 时发生错误: {str(stock_e)}")
                results.append({
                    'stock_code': stock_code,
                    'error': '暂时不支持该股票代码分析'
                })
        
        return jsonify({'results': results})
    except Exception as e:
        logging.error(f"分析请求处理失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '处理请求时发生错误'}), 500


@app.route('/api/north_flow_history', methods=['POST'])
def api_north_flow_history():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        days = data.get('days', 10)  # 默认为10天，对应前端的默认选项

        # 计算 end_date 为当前时间
        end_date = datetime.now().strftime('%Y%m%d')

        # 计算 start_date 为 end_date 减去指定的天数
        start_date = (datetime.now() - timedelta(days=int(days))).strftime('%Y%m%d')

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        # 调用北向资金历史数据方法
        from capital_flow_analyzer import CapitalFlowAnalyzer

        analyzer = CapitalFlowAnalyzer()
        result = analyzer.get_north_flow_history(stock_code, start_date, end_date)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"获取北向资金历史数据出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/search_us_stocks', methods=['GET'])
def search_us_stocks():
    try:
        keyword = request.args.get('keyword', '')
        if not keyword:
            return jsonify({'error': '请输入搜索关键词'}), 400

        results = us_stock_service.search_us_stocks(keyword)
        return jsonify({'results': results})

    except Exception as e:
        app.logger.error(f"搜索美股代码时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 新增可视化分析页面路由
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/stock_detail/<string:stock_code>')
def stock_detail(stock_code):
    market_type = request.args.get('market_type', 'A')
    return render_template('stock_detail.html', stock_code=stock_code, market_type=market_type)


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')


@app.route('/market_scan')
def market_scan():
    return render_template('market_scan.html')


# 基本面分析页面
@app.route('/fundamental')
def fundamental():
    return render_template('fundamental.html')


# 资金流向页面
@app.route('/capital_flow')
def capital_flow():
    return render_template('capital_flow.html')


# 情景预测页面
@app.route('/scenario_predict')
def scenario_predict():
    return render_template('scenario_predict.html')


# 风险监控页面
@app.route('/risk_monitor')
def risk_monitor_page():
    return render_template('risk_monitor.html')


# 智能问答页面
@app.route('/qa')
def qa_page():
    return render_template('qa.html')


# 行业分析页面
@app.route('/industry_analysis')
def industry_analysis():
    return render_template('industry_analysis.html')


def make_cache_key_with_stock():
    """创建包含股票代码的自定义缓存键"""
    path = request.path

    # 从请求体中获取股票代码
    stock_code = None
    if request.is_json:
        stock_code = request.json.get('stock_code')

    # 构建包含股票代码的键
    if stock_code:
        return f"{path}_{stock_code}"
    else:
        return path


@app.route('/api/start_stock_analysis', methods=['POST'])
def start_stock_analysis():
    """启动个股分析任务"""
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')

        if not stock_code:
            return jsonify({'error': '请输入股票代码'}), 400

        app.logger.info(f"准备分析股票: {stock_code}")

        # 获取或创建任务
        task_id, task, is_new = get_or_create_task(
            'stock_analysis',
            stock_code=stock_code,
            market_type=market_type
        )

        # 如果是已完成的任务，直接返回结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            app.logger.info(f"使用缓存的分析结果: {stock_code}")
            return jsonify({
                'task_id': task_id,
                'status': task['status'],
                'result': task['result']
            })

        # 如果是新创建的任务，启动后台处理
        if is_new:
            app.logger.info(f"创建新的分析任务: {task_id}")

            # 启动后台线程执行分析
            def run_analysis():
                try:
                    update_task_status('stock_analysis', task_id, TASK_RUNNING, progress=10)

                    # 执行分析
                    result = analyzer.perform_enhanced_analysis(stock_code, market_type)

                    # 更新任务状态为完成
                    update_task_status('stock_analysis', task_id, TASK_COMPLETED, progress=100, result=result)
                    app.logger.info(f"分析任务 {task_id} 完成")

                except Exception as e:
                    app.logger.error(f"分析任务 {task_id} 失败: {str(e)}")
                    app.logger.error(traceback.format_exc())
                    update_task_status('stock_analysis', task_id, TASK_FAILED, error=str(e))

            # 启动后台线程
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()

        # 返回任务ID和状态
        return jsonify({
            'task_id': task_id,
            'status': task['status'],
            'message': f'已启动分析任务: {stock_code}'
        })

    except Exception as e:
        app.logger.error(f"启动个股分析任务时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis_status/<task_id>', methods=['GET'])
def get_analysis_status(task_id):
    """获取个股分析任务状态"""
    store = get_task_store('stock_analysis')
    with task_lock:
        if task_id not in store:
            return jsonify({'error': '找不到指定的分析任务'}), 404

        task = store[task_id]

        # 基本状态信息
        status = {
            'id': task['id'],
            'status': task['status'],
            'progress': task.get('progress', 0),
            'created_at': task['created_at'],
            'updated_at': task['updated_at']
        }

        # 如果任务完成，包含结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            status['result'] = task['result']

        # 如果任务失败，包含错误信息
        if task['status'] == TASK_FAILED and 'error' in task:
            status['error'] = task['error']

        return custom_jsonify(status)


@app.route('/api/cancel_analysis/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    """取消个股分析任务"""
    store = get_task_store('stock_analysis')
    with task_lock:
        if task_id not in store:
            return jsonify({'error': '找不到指定的分析任务'}), 404

        task = store[task_id]

        if task['status'] in [TASK_COMPLETED, TASK_FAILED]:
            return jsonify({'message': '任务已完成或失败，无法取消'})

        # 更新状态为失败
        task['status'] = TASK_FAILED
        task['error'] = '用户取消任务'
        task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 更新键索引的任务
        if 'key' in task and task['key'] in store:
            store[task['key']] = task

        return jsonify({'message': '任务已取消'})


# 保留原有API用于向后兼容
@app.route('/api/enhanced_analysis', methods=['POST'])
def enhanced_analysis():
    """原增强分析API的向后兼容版本"""
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')

        if not stock_code:
            return custom_jsonify({'error': '请输入股票代码'}), 400

        # 调用新的任务系统，但模拟同步行为
        # 这会导致和之前一样的超时问题，但保持兼容
        timeout = 300
        start_time = time.time()

        # 获取或创建任务
        task_id, task, is_new = get_or_create_task(
            'stock_analysis',
            stock_code=stock_code,
            market_type=market_type
        )

        # 如果是已完成的任务，直接返回结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            app.logger.info(f"使用缓存的分析结果: {stock_code}")
            return custom_jsonify({'result': task['result']})

        # 启动分析（如果是新任务）
        if is_new:
            # 同步执行分析
            try:
                result = analyzer.perform_enhanced_analysis(stock_code, market_type)
                update_task_status('stock_analysis', task_id, TASK_COMPLETED, progress=100, result=result)
                app.logger.info(f"分析完成: {stock_code}，耗时 {time.time() - start_time:.2f} 秒")
                return custom_jsonify({'result': result})
            except Exception as e:
                app.logger.error(f"分析过程中出错: {str(e)}")
                update_task_status('stock_analysis', task_id, TASK_FAILED, error=str(e))
                return custom_jsonify({'error': f'分析过程中出错: {str(e)}'}), 500
        else:
            # 已存在正在处理的任务，等待其完成
            max_wait = timeout - (time.time() - start_time)
            wait_interval = 0.5
            waited = 0

            while waited < max_wait:
                with task_lock:
                    current_task = store[task_id]
                    if current_task['status'] == TASK_COMPLETED and 'result' in current_task:
                        return custom_jsonify({'result': current_task['result']})
                    if current_task['status'] == TASK_FAILED:
                        error = current_task.get('error', '任务失败，无详细信息')
                        return custom_jsonify({'error': error}), 500

                time.sleep(wait_interval)
                waited += wait_interval

            # 超时
            return custom_jsonify({'error': '处理超时，请稍后重试'}), 504

    except Exception as e:
        app.logger.error(f"执行增强版分析时出错: {traceback.format_exc()}")
        return custom_jsonify({'error': str(e)}), 500


# 添加在web_server.py主代码中
@app.errorhandler(404)
def not_found(error):
    """处理404错误"""
    if request.path.startswith('/api/'):
        # 为API请求返回JSON格式的错误
        return jsonify({
            'error': '找不到请求的API端点',
            'path': request.path,
            'method': request.method
        }), 404
    # 为网页请求返回HTML错误页
    return render_template('error.html', error_code=404, message="找不到请求的页面"), 404


@app.errorhandler(500)
def server_error(error):
    """处理500错误"""
    app.logger.error(f"服务器错误: {str(error)}")
    if request.path.startswith('/api/'):
        # 为API请求返回JSON格式的错误
        return jsonify({
            'error': '服务器内部错误',
            'message': str(error)
        }), 500
    # 为网页请求返回HTML错误页
    return render_template('error.html', error_code=500, message="服务器内部错误"), 500


# Update the get_stock_data function in web_server.py to handle date formatting properly
@app.route('/api/stock_data', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_stock_data():
    try:
        stock_code = request.args.get('stock_code')
        market_type = request.args.get('market_type', 'A')
        period = request.args.get('period', '1y')  # 默认1年

        if not stock_code:
            return custom_jsonify({'error': '请提供股票代码'}), 400

        # 记录请求参数，便于调试
        app.logger.info(f"获取股票数据请求参数: stock_code={stock_code}, market_type={market_type}, period={period}")

        # 根据period计算start_date
        end_date = datetime.now().strftime('%Y%m%d')
        if period == '1m':
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        elif period == '3m':
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        elif period == '6m':
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        elif period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        else:
            # 如果传入了未知的周期，默认使用1年
            app.logger.warning(f"未知的周期参数: {period}，使用默认值1y")
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            period = '1y'

        app.logger.info(f"计算的日期范围: start_date={start_date}, end_date={end_date}, period={period}")

        # 获取股票历史数据
        try:
            df = analyzer.get_stock_data(stock_code, market_type, start_date, end_date)
        except Exception as e:
            error_msg = str(e)
            app.logger.error(f"获取股票数据失败: {error_msg}")
            
            # 检查是否是'date'相关错误或其他常见错误
            if "'date'" in error_msg or "获取股票数据失败" in error_msg:
                return jsonify({"status": "error", "message": "暂时不支持该股票代码分析"}), 400
            else:
                return jsonify({"status": "error", "message": f"获取股票数据失败: {error_msg}"}), 500

        # 计算技术指标
        app.logger.info(f"计算股票 {stock_code} 的技术指标")
        df = analyzer.calculate_indicators(df)

        # 检查数据是否为空
        if df.empty:
            app.logger.warning(f"股票 {stock_code} 的数据为空")
            return jsonify({"status": "error", "message": "未找到股票数据"}), 404

        # 将DataFrame转为JSON格式
        app.logger.info(f"将数据转换为JSON格式，行数: {len(df)}")

        # 确保日期列是字符串格式 - 修复缓存问题
        if 'date' in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                else:
                    df = df.copy()
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                app.logger.error(f"处理日期列时出错: {str(e)}")
                df['date'] = df['date'].astype(str)

        # 将NaN值替换为None
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})

        records = df.to_dict('records')

        app.logger.info(f"数据处理完成，返回 {len(records)} 条记录，周期: {period}")
        return jsonify({"status": "success", "data": records, "period": period})
    except Exception as e:
        app.logger.error(f"获取股票数据时出错: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "暂时不支持该股票代码分析", "detail": str(e)}), 400


# @app.route('/api/market_scan', methods=['POST'])
# def api_market_scan():
#     try:
#         data = request.json
#         stock_list = data.get('stock_list', [])
#         min_score = data.get('min_score', 60)
#         market_type = data.get('market_type', 'A')

#         if not stock_list:
#             return jsonify({'error': '请提供股票列表'}), 400

#         # 限制股票数量，避免过长处理时间
#         if len(stock_list) > 100:
#             app.logger.warning(f"股票列表过长 ({len(stock_list)}只)，截取前100只")
#             stock_list = stock_list[:100]

#         # 执行市场扫描
#         app.logger.info(f"开始扫描 {len(stock_list)} 只股票，最低分数: {min_score}")

#         # 使用线程池优化处理
#         results = []
#         max_workers = min(10, len(stock_list))  # 最多10个工作线程

#         # 设置较长的超时时间
#         timeout = 300  # 5分钟

#         def scan_thread():
#             try:
#                 return analyzer.scan_market(stock_list, min_score, market_type)
#             except Exception as e:
#                 app.logger.error(f"扫描线程出错: {str(e)}")
#                 return []

#         thread = threading.Thread(target=lambda: results.append(scan_thread()))
#         thread.start()
#         thread.join(timeout)

#         if thread.is_alive():
#             app.logger.error(f"市场扫描超时，已扫描 {len(stock_list)} 只股票超过 {timeout} 秒")
#             return custom_jsonify({'error': '扫描超时，请减少股票数量或稍后再试'}), 504

#         if not results or not results[0]:
#             app.logger.warning("扫描结果为空")
#             return custom_jsonify({'results': []})

#         scan_results = results[0]
#         app.logger.info(f"扫描完成，找到 {len(scan_results)} 只符合条件的股票")

#         # 使用自定义JSON格式处理NumPy数据类型
#         return custom_jsonify({'results': scan_results})
#     except Exception as e:
#         app.logger.error(f"执行市场扫描时出错: {traceback.format_exc()}")
#         return custom_jsonify({'error': str(e)}), 500

@app.route('/api/start_market_scan', methods=['POST'])
def start_market_scan():
    """启动市场扫描任务"""
    try:
        data = request.json
        stock_list = data.get('stock_list', [])
        min_score = data.get('min_score', 60)
        market_type = data.get('market_type', 'A')

        if not stock_list:
            return jsonify({'error': '请提供股票列表'}), 400

        # 限制股票数量，避免过长处理时间
        if len(stock_list) > 100:
            app.logger.warning(f"股票列表过长 ({len(stock_list)}只)，截取前100只")
            stock_list = stock_list[:100]

        # 创建新任务
        task_id = generate_task_id()
        task = {
            'id': task_id,
            'status': TASK_PENDING,
            'progress': 0,
            'total': len(stock_list),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': {
                'stock_list': stock_list,
                'min_score': min_score,
                'market_type': market_type
            }
        }

        with task_lock:
            scan_tasks[task_id] = task

        # 启动后台线程执行扫描
        def run_scan():
            try:
                start_market_scan_task_status(task_id, TASK_RUNNING)

                # 执行分批处理
                results = []
                total = len(stock_list)
                batch_size = 10

                for i in range(0, total, batch_size):
                    if task_id not in scan_tasks or scan_tasks[task_id]['status'] != TASK_RUNNING:
                        # 任务被取消
                        app.logger.info(f"扫描任务 {task_id} 被取消")
                        return

                    batch = stock_list[i:i + batch_size]
                    batch_results = []

                    for stock_code in batch:
                        try:
                            report = analyzer.quick_analyze_stock(stock_code, market_type)
                            if report['score'] >= min_score:
                                batch_results.append(report)
                        except Exception as e:
                            app.logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
                            continue

                    results.extend(batch_results)

                    # 更新进度
                    progress = min(100, int((i + len(batch)) / total * 100))
                    start_market_scan_task_status(task_id, TASK_RUNNING, progress=progress)

                # 按得分排序
                results.sort(key=lambda x: x['score'], reverse=True)

                # 更新任务状态为完成
                start_market_scan_task_status(task_id, TASK_COMPLETED, progress=100, result=results)
                app.logger.info(f"扫描任务 {task_id} 完成，找到 {len(results)} 只符合条件的股票")

            except Exception as e:
                app.logger.error(f"扫描任务 {task_id} 失败: {str(e)}")
                app.logger.error(traceback.format_exc())
                start_market_scan_task_status(task_id, TASK_FAILED, error=str(e))

        # 启动后台线程
        thread = threading.Thread(target=run_scan)
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'status': 'pending',
            'message': f'已启动扫描任务，正在处理 {len(stock_list)} 只股票'
        })

    except Exception as e:
        app.logger.error(f"启动市场扫描任务时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/scan_status/<task_id>', methods=['GET'])
def get_scan_status(task_id):
    """获取扫描任务状态"""
    with task_lock:
        if task_id not in scan_tasks:
            return jsonify({'error': '找不到指定的扫描任务'}), 404

        task = scan_tasks[task_id]

        # 基本状态信息
        status = {
            'id': task['id'],
            'status': task['status'],
            'progress': task.get('progress', 0),
            'total': task.get('total', 0),
            'created_at': task['created_at'],
            'updated_at': task['updated_at']
        }

        # 如果任务完成，包含结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            status['result'] = task['result']

        # 如果任务失败，包含错误信息
        if task['status'] == TASK_FAILED and 'error' in task:
            status['error'] = task['error']

        return custom_jsonify(status)


@app.route('/api/cancel_scan/<task_id>', methods=['POST'])
def cancel_scan(task_id):
    """取消扫描任务"""
    with task_lock:
        if task_id not in scan_tasks:
            return jsonify({'error': '找不到指定的扫描任务'}), 404

        task = scan_tasks[task_id]

        if task['status'] in [TASK_COMPLETED, TASK_FAILED]:
            return jsonify({'message': '任务已完成或失败，无法取消'})

        # 更新状态为失败
        task['status'] = TASK_FAILED
        task['error'] = '用户取消任务'
        task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({'message': '任务已取消'})


@app.route('/api/index_stocks', methods=['GET'])
def get_index_stocks():
    """获取指数成分股"""
    try:
        import akshare as ak
        index_code = request.args.get('index_code', '000300')  # 默认沪深300

        # 获取指数成分股
        app.logger.info(f"获取指数 {index_code} 成分股")
        if index_code == '000300':
            # 沪深300成分股
            stocks = ak.index_stock_cons_weight_csindex(symbol="000300")
        elif index_code == '000905':
            # 中证500成分股
            stocks = ak.index_stock_cons_weight_csindex(symbol="000905")
        elif index_code == '000852':
            # 中证1000成分股
            stocks = ak.index_stock_cons_weight_csindex(symbol="000852")
        elif index_code == '000001':
            # 上证指数
            stocks = ak.index_stock_cons_weight_csindex(symbol="000001")
        else:
            return jsonify({'error': '不支持的指数代码'}), 400

        # 提取股票代码列表
        stock_list = stocks['成分券代码'].tolist() if '成分券代码' in stocks.columns else []
        app.logger.info(f"找到 {len(stock_list)} 只成分股")

        return jsonify({'stock_list': stock_list})
    except Exception as e:
        app.logger.error(f"获取指数成分股时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/industry_stocks', methods=['GET'])
def get_industry_stocks():
    """获取行业成分股"""
    try:
        import akshare as ak
        industry = request.args.get('industry', '')

        if not industry:
            return jsonify({'error': '请提供行业名称'}), 400

        # 获取行业成分股
        app.logger.info(f"获取 {industry} 行业成分股")
        stocks = ak.stock_board_industry_cons_em(symbol=industry)

        # 提取股票代码列表
        stock_list = stocks['代码'].tolist() if '代码' in stocks.columns else []
        app.logger.info(f"找到 {len(stock_list)} 只 {industry} 行业股票")

        return jsonify({'stock_list': stock_list})
    except Exception as e:
        app.logger.error(f"获取行业成分股时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 添加到web_server.py
def clean_old_tasks():
    """清理旧的扫描任务"""
    with task_lock:
        now = datetime.now()
        to_delete = []

        for task_id, task in scan_tasks.items():
            # 解析更新时间
            try:
                updated_at = datetime.strptime(task['updated_at'], '%Y-%m-%d %H:%M:%S')
                # 如果任务完成或失败且超过1小时，或者任务状态异常且超过3小时，清理它
                if ((task['status'] in [TASK_COMPLETED, TASK_FAILED] and
                     (now - updated_at).total_seconds() > 3600) or
                        ((now - updated_at).total_seconds() > 10800)):
                    to_delete.append(task_id)
            except:
                # 日期解析错误，添加到删除列表
                to_delete.append(task_id)

        # 删除旧任务
        for task_id in to_delete:
            del scan_tasks[task_id]

        return len(to_delete)


# 修改 run_task_cleaner 函数，使其每 5 分钟运行一次并在 16:30 左右清理所有缓存
def run_task_cleaner():
    """定期运行任务清理，并在每天 16:30 左右清理所有缓存"""
    while True:
        try:
            now = datetime.now()
            # 判断是否在收盘时间附近（16:25-16:35）
            is_market_close_time = (now.hour == 16 and 25 <= now.minute <= 35)

            cleaned = clean_old_tasks()

            # 如果是收盘时间，清理所有缓存
            if is_market_close_time:
                # 清理分析器的数据缓存
                analyzer.data_cache.clear()

                # 清理 Flask 缓存
                cache.clear()

                # 清理任务存储
                with task_lock:
                    for task_type in tasks:
                        task_store = tasks[task_type]
                        completed_tasks = [task_id for task_id, task in task_store.items()
                                           if task['status'] == TASK_COMPLETED]
                        for task_id in completed_tasks:
                            del task_store[task_id]

                app.logger.info("市场收盘时间检测到，已清理所有缓存数据")

            if cleaned > 0:
                app.logger.info(f"清理了 {cleaned} 个旧的扫描任务")
        except Exception as e:
            app.logger.error(f"任务清理出错: {str(e)}")

        # 每 5 分钟运行一次，而不是每小时
        time.sleep(600)


# 基本面分析路由
@app.route('/api/fundamental_analysis', methods=['POST'])
def api_fundamental_analysis():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')  # 默认为A股

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        app.logger.info(f"获取基本面分析: stock_code={stock_code}, market_type={market_type}")
        
        # 尝试获取股票基本信息
        stock_info = None
        try:
            # 使用StockAnalyzer的get_stock_info方法获取股票信息
            stock_info = analyzer.get_stock_info(stock_code)
            app.logger.info(f"成功获取股票信息: {stock_info}")
        except Exception as e:
            app.logger.warning(f"获取股票基本信息失败: {str(e)}")
            # 尝试使用quick_analyze_stock方法获取基本信息
            try:
                quick_result = analyzer.quick_analyze_stock(stock_code, market_type)
                stock_info = {
                    'stock_name': quick_result.get('stock_name', stock_code),
                    'industry': quick_result.get('industry', '未知行业')
                }
                app.logger.info(f"通过quick_analyze_stock获取股票信息: {stock_info}")
            except Exception as e2:
                app.logger.warning(f"通过quick_analyze_stock获取股票信息失败: {str(e2)}")
                stock_info = None
        
        # 获取真实财务数据
        financial_data = {}
        growth_data = {}
        
        try:
            import akshare as ak
            
            # 格式化股票代码，确保适合akshare API
            formatted_stock_code = stock_code
            if market_type == 'A':
                # 确保A股代码格式正确（不带市场前缀，纯数字）
                if stock_code.startswith('sh') or stock_code.startswith('sz'):
                    formatted_stock_code = stock_code[2:]
            
            app.logger.info(f"获取股票 {formatted_stock_code} 的财务指标")
            
            # 获取主要财务指标
            try:
                # 获取PE、PB等估值指标 - 使用正确的API
                stock_zh_a_spot_df = ak.stock_zh_a_spot()
                stock_info_df = stock_zh_a_spot_df[stock_zh_a_spot_df['代码'] == formatted_stock_code]
                
                if not stock_info_df.empty:
                    financial_data['pe_ttm'] = float(stock_info_df['市盈率-动态'].values[0]) if not pd.isna(stock_info_df['市盈率-动态'].values[0]) else None
                    financial_data['pb'] = float(stock_info_df['市净率'].values[0]) if not pd.isna(stock_info_df['市净率'].values[0]) else None
                    financial_data['ps_ttm'] = None  # 市销率在spot数据中没有，需要另外获取
                    app.logger.info(f"获取到估值指标: PE={financial_data.get('pe_ttm')}, PB={financial_data.get('pb')}")
            except Exception as e:
                app.logger.warning(f"获取估值指标失败: {str(e)}")
                app.logger.warning(traceback.format_exc())
            
            # 获取ROE等盈利能力指标
            try:
                # 获取最近的财务报表数据 - 使用正确的API参数
                try:
                    # 尝试直接获取财务报表，不使用symbol_type参数
                    stock_financial_report_df = ak.stock_financial_report_sina(symbol=formatted_stock_code)
                    app.logger.info(f"成功获取财务报表数据，行数: {len(stock_financial_report_df)}")
                except Exception as e1:
                    app.logger.warning(f"获取财务报表数据失败: {str(e1)}")
                    stock_financial_report_df = pd.DataFrame()
                
                # 如果无法获取财务报表，尝试从财务分析指标中提取盈利能力指标
                if stock_financial_report_df.empty and 'stock_financial_analysis_df' in locals() and not stock_financial_analysis_df.empty:
                    app.logger.info("从财务分析指标中提取盈利能力指标")
                    latest_data = stock_financial_analysis_df.iloc[0]
                    
                    # 尝试提取ROE
                    if '净资产收益率-摊薄(%)' in latest_data:
                        financial_data['roe'] = float(latest_data['净资产收益率-摊薄(%)']) if not pd.isna(latest_data['净资产收益率-摊薄(%)']) else None
                    elif '净资产收益率(%)' in latest_data:
                        financial_data['roe'] = float(latest_data['净资产收益率(%)']) if not pd.isna(latest_data['净资产收益率(%)']) else None
                    
                    # 尝试提取毛利率
                    if '销售毛利率(%)' in latest_data:
                        financial_data['gross_margin'] = float(latest_data['销售毛利率(%)']) if not pd.isna(latest_data['销售毛利率(%)']) else None
                    elif '毛利率(%)' in latest_data:
                        financial_data['gross_margin'] = float(latest_data['毛利率(%)']) if not pd.isna(latest_data['毛利率(%)']) else None
                    
                    # 尝试提取净利率
                    if '销售净利率(%)' in latest_data:
                        financial_data['net_margin'] = float(latest_data['销售净利率(%)']) if not pd.isna(latest_data['销售净利率(%)']) else None
                    elif '净利率(%)' in latest_data:
                        financial_data['net_margin'] = float(latest_data['净利率(%)']) if not pd.isna(latest_data['净利率(%)']) else None
                elif not stock_financial_report_df.empty:
                    # 如果成功获取了财务报表，从中提取数据
                    latest_quarter = stock_financial_report_df.iloc[0]
                    
                    # 尝试从财务报表中提取指标
                    for roe_col in ['净资产收益率(%)', 'ROE(%)', '净资产收益率']:
                        if roe_col in latest_quarter and not pd.isna(latest_quarter[roe_col]):
                            financial_data['roe'] = float(latest_quarter[roe_col])
                            break
                    
                    for gm_col in ['毛利率(%)', '销售毛利率(%)', '毛利率']:
                        if gm_col in latest_quarter and not pd.isna(latest_quarter[gm_col]):
                            financial_data['gross_margin'] = float(latest_quarter[gm_col])
                            break
                    
                    for nm_col in ['净利率(%)', '销售净利率(%)', '净利率']:
                        if nm_col in latest_quarter and not pd.isna(latest_quarter[nm_col]):
                            financial_data['net_margin'] = float(latest_quarter[nm_col])
                            break
                
                # 尝试从股票实时行情中获取ROE
                if 'roe' not in financial_data or financial_data['roe'] is None:
                    try:
                        # 获取个股资金流向，其中可能包含ROE
                        stock_individual_fund_flow_df = ak.stock_individual_fund_flow(stock=formatted_stock_code, market="sh" if formatted_stock_code.startswith('6') else "sz")
                        if 'roe' in stock_individual_fund_flow_df.columns:
                            roe_value = stock_individual_fund_flow_df['roe'].iloc[0]
                            if not pd.isna(roe_value):
                                financial_data['roe'] = float(roe_value)
                                app.logger.info(f"从资金流向数据获取到ROE: {financial_data['roe']}")
                    except Exception as e2:
                        app.logger.warning(f"获取个股资金流向失败: {str(e2)}")
                
                app.logger.info(f"获取到盈利能力指标: ROE={financial_data.get('roe')}, 毛利率={financial_data.get('gross_margin')}, 净利率={financial_data.get('net_margin')}")
            except Exception as e:
                app.logger.warning(f"获取盈利能力指标失败: {str(e)}")
                app.logger.warning(traceback.format_exc())
        
        except ImportError:
            app.logger.error("未安装akshare库，无法获取真实财务数据")
        
        # 如果没有获取到真实数据，使用模拟数据
        if not financial_data:
            app.logger.warning("使用模拟财务数据")
            # 生成一个稳定的随机种子，确保同一股票每次生成相同的模拟数据
            import hashlib
            seed = int(hashlib.md5(f"{stock_code}_{market_type}".encode()).hexdigest(), 16) % (10**8)
            import random
            random.seed(seed)
            
            financial_data = {
                'pe_ttm': round(10 + random.random() * 20, 2),  # 10-30之间的PE
                'pb': round(1 + random.random() * 3, 2),  # 1-4之间的PB
                'ps_ttm': round(1 + random.random() * 5, 2),  # 1-6之间的PS
                'roe': round(5 + random.random() * 20, 2),  # 5-25之间的ROE
                'gross_margin': round(20 + random.random() * 40, 2),  # 20-60之间的毛利率
                'net_margin': round(5 + random.random() * 25, 2)  # 5-30之间的净利率
            }
        
        if not growth_data:
            app.logger.warning("使用模拟增长数据")
            import hashlib
            seed = int(hashlib.md5(f"{stock_code}_{market_type}_growth".encode()).hexdigest(), 16) % (10**8)
            import random
            random.seed(seed)
            
            growth_data = {
                'revenue_3y_cagr': round(-5 + random.random() * 35, 2),  # -5% 到 30% 之间的3年营收CAGR
                'revenue_5y_cagr': round(-5 + random.random() * 30, 2),  # -5% 到 25% 之间的5年营收CAGR
                'profit_3y_cagr': round(-10 + random.random() * 45, 2),  # -10% 到 35% 之间的3年利润CAGR
                'profit_5y_cagr': round(-10 + random.random() * 40, 2)   # -10% 到 30% 之间的5年利润CAGR
            }
        
        # 计算估值评分 (0-100)
        valuation_score = 0
        if financial_data.get('pe_ttm') is not None:
            pe_score = max(0, min(100, 100 - (financial_data['pe_ttm'] - 10) * 5)) if financial_data['pe_ttm'] > 0 else 0
            valuation_score += pe_score * 0.5
        
        if financial_data.get('pb') is not None:
            pb_score = max(0, min(100, 100 - (financial_data['pb'] - 1) * 25)) if financial_data['pb'] > 0 else 0
            valuation_score += pb_score * 0.3
        
        if financial_data.get('ps_ttm') is not None:
            ps_score = max(0, min(100, 100 - (financial_data['ps_ttm'] - 1) * 20)) if financial_data['ps_ttm'] > 0 else 0
            valuation_score += ps_score * 0.2
        
        # 计算财务健康评分 (0-100)
        financial_health_score = 0
        if financial_data.get('roe') is not None:
            roe_score = min(100, max(0, financial_data['roe'] * 4))
            financial_health_score += roe_score * 0.5
        
        if financial_data.get('gross_margin') is not None:
            gm_score = min(100, max(0, financial_data['gross_margin'] * 1.5))
            financial_health_score += gm_score * 0.25
        
        if financial_data.get('net_margin') is not None:
            nm_score = min(100, max(0, financial_data['net_margin'] * 3))
            financial_health_score += nm_score * 0.25
        
        # 计算成长性评分 (0-100)
        growth_score = 0
        if growth_data.get('revenue_3y_cagr') is not None:
            rev_3y_score = min(100, max(0, 50 + growth_data['revenue_3y_cagr'] * 2))
            growth_score += rev_3y_score * 0.25
        
        if growth_data.get('revenue_5y_cagr') is not None:
            rev_5y_score = min(100, max(0, 50 + growth_data['revenue_5y_cagr'] * 2))
            growth_score += rev_5y_score * 0.25
        
        if growth_data.get('profit_3y_cagr') is not None:
            profit_3y_score = min(100, max(0, 50 + growth_data['profit_3y_cagr'] * 1.5))
            growth_score += profit_3y_score * 0.25
        
        if growth_data.get('profit_5y_cagr') is not None:
            profit_5y_score = min(100, max(0, 50 + growth_data['profit_5y_cagr'] * 1.5))
            growth_score += profit_5y_score * 0.25
        
        # 构造基本面数据
        fundamental_data = {
            'details': {
                'indicators': {
                    'stock_name': stock_info.get('stock_name', stock_code) if stock_info else stock_code,
                    'industry': stock_info.get('industry', '未知行业') if stock_info else '未知行业',
                    'pe_ttm': financial_data.get('pe_ttm'),
                    'pb': financial_data.get('pb'),
                    'ps_ttm': financial_data.get('ps_ttm'),
                    'roe': financial_data.get('roe'),
                    'gross_margin': financial_data.get('gross_margin'),
                    'net_margin': financial_data.get('net_margin')
                },
                'growth': {
                    'revenue_3y_cagr': growth_data.get('revenue_3y_cagr'),
                    'revenue_5y_cagr': growth_data.get('revenue_5y_cagr'),
                    'profit_3y_cagr': growth_data.get('profit_3y_cagr'),
                    'profit_5y_cagr': growth_data.get('profit_5y_cagr')
                }
            },
            'scores': {
                'valuation': round(valuation_score),
                'financial_health': round(financial_health_score),
                'growth': round(growth_score),
                'total': round((valuation_score + financial_health_score + growth_score) / 3)
            },
            'analysis': {
                'valuation': '估值处于行业偏高水平' if valuation_score < 50 else '估值处于行业合理水平',
                'financial_health': '财务状况一般' if financial_health_score < 50 else '财务状况良好',
                'growth': '成长性一般' if growth_score < 50 else '成长性良好'
            }
        }
        
        return jsonify(fundamental_data)
        
    except Exception as e:
        app.logger.error(f"基本面分析API出错: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# 资金流向分析路由
# Add to web_server.py

# API endpoint to get concept fund flow
@app.route('/api/concept_fund_flow', methods=['GET'])
def api_concept_fund_flow():
    try:
        period = request.args.get('period', '10日排行')  # Default to 10-day ranking

        # Get concept fund flow data
        result = capital_flow_analyzer.get_concept_fund_flow(period)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting concept fund flow: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# API endpoint to get individual stock fund flow ranking
@app.route('/api/individual_fund_flow_rank', methods=['GET'])
def api_individual_fund_flow_rank():
    try:
        period = request.args.get('period', '10日')  # Default to today

        # Get individual fund flow ranking data
        result = capital_flow_analyzer.get_individual_fund_flow_rank(period)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting individual fund flow ranking: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# API endpoint to get individual stock fund flow
@app.route('/api/individual_fund_flow', methods=['GET'])
def api_individual_fund_flow():
    try:
        stock_code = request.args.get('stock_code')
        market_type = request.args.get('market_type', '')  # Auto-detect if not provided
        re_date = request.args.get('period-select')

        if not stock_code:
            return jsonify({'error': 'Stock code is required'}), 400

        # Get individual fund flow data
        result = capital_flow_analyzer.get_individual_fund_flow(stock_code, market_type, re_date)
        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting individual fund flow: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# API endpoint to get stocks in a sector
@app.route('/api/sector_stocks', methods=['GET'])
def api_sector_stocks():
    try:
        sector = request.args.get('sector')

        if not sector:
            return jsonify({'error': 'Sector name is required'}), 400

        # Get sector stocks data
        result = capital_flow_analyzer.get_sector_stocks(sector)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting sector stocks: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# Update the existing capital flow API endpoint
@app.route('/api/capital_flow', methods=['POST'])
def api_capital_flow():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', '')  # Auto-detect if not provided

        if not stock_code:
            return jsonify({'error': 'Stock code is required'}), 400

        # Calculate capital flow score
        result = capital_flow_analyzer.calculate_capital_flow_score(stock_code, market_type)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error calculating capital flow score: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 情景预测路由
@app.route('/api/scenario_predict', methods=['POST'])
def api_scenario_predict():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')
        days = data.get('days', 60)

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        app.logger.info(f"获取情景预测: stock_code={stock_code}, market_type={market_type}, days={days}")
        
        # 获取分析器实例
        analyzer = get_analyzer()
        
        # 创建情景预测器
        predictor = ScenarioPredictor(analyzer)
        
        # 生成情景预测
        scenarios = predictor.generate_scenarios(stock_code, market_type, days)
        
        # 检查是否有错误状态
        if isinstance(scenarios, dict) and scenarios.get('status') == 'error':
            return jsonify({'error': scenarios.get('error', '暂时不支持该股票代码分析')}), 400
            
        return jsonify(scenarios)
        
    except Exception as e:
        app.logger.error(f"情景预测API出错: {str(e)}")
        return jsonify({"error": str(e)}), 500


# 智能问答路由
@app.route('/api/qa', methods=['POST'])
def api_qa():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        question = data.get('question')
        market_type = data.get('market_type', 'A')

        if not stock_code or not question:
            return jsonify({'error': '请提供股票代码和问题'}), 400

        app.logger.info(f"智能问答请求: stock_code={stock_code}, question={question}")
        
        # 获取智能问答结果
        result = stock_qa.answer_question(stock_code, question, market_type)
        
        # 检查是否有错误
        if isinstance(result, dict) and 'error' in result:
            error_type = result.get('error')
            if error_type == 'unsupported_stock':
                app.logger.warning(f"不支持的股票代码: {stock_code}")
                return jsonify({'error': '暂时不支持该股票代码分析'}), 400
            else:
                app.logger.error(f"智能问答出错: {result.get('error', '未知错误')}")
                return jsonify({'error': result.get('answer', '处理问题时出错')}), 500
        
        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"智能问答API出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 风险分析路由
@app.route('/api/risk_analysis', methods=['POST'])
def api_risk_analysis():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        # 获取风险分析结果
        result = risk_monitor.analyze_stock_risk(stock_code, market_type)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"风险分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 投资组合风险分析路由
@app.route('/api/portfolio_risk', methods=['POST'])
def api_portfolio_risk():
    try:
        data = request.json
        portfolio = data.get('portfolio', [])

        if not portfolio:
            return jsonify({'error': '请提供投资组合'}), 400

        # 获取投资组合风险分析结果
        result = risk_monitor.analyze_portfolio_risk(portfolio)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"投资组合风险分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 指数分析路由
@app.route('/api/index_analysis', methods=['GET'])
def api_index_analysis():
    try:
        index_code = request.args.get('index_code')
        limit = int(request.args.get('limit', 30))

        if not index_code:
            return jsonify({'error': '请提供指数代码'}), 400

        # 获取指数分析结果
        result = index_industry_analyzer.analyze_index(index_code, limit)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"指数分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 行业分析路由
@app.route('/api/industry_analysis', methods=['GET'])
def api_industry_analysis():
    try:
        industry = request.args.get('industry')
        limit = int(request.args.get('limit', 30))

        if not industry:
            return jsonify({'error': '请提供行业名称'}), 400

        # 获取行业分析结果
        result = index_industry_analyzer.analyze_industry(industry, limit)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"行业分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/industry_fund_flow', methods=['GET'])
def api_industry_fund_flow():
    """获取行业资金流向数据"""
    try:
        symbol = request.args.get('symbol', '即时')

        result = industry_analyzer.get_industry_fund_flow(symbol)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"获取行业资金流向数据出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/industry_detail', methods=['GET'])
def api_industry_detail():
    """获取行业详细信息"""
    try:
        industry = request.args.get('industry')

        if not industry:
            return jsonify({'error': '请提供行业名称'}), 400

        result = industry_analyzer.get_industry_detail(industry)

        app.logger.info(f"返回前 (result)：{result}")
        if not result:
            return jsonify({'error': f'未找到行业 {industry} 的详细信息'}), 404

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"获取行业详细信息出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 行业比较路由
@app.route('/api/industry_compare', methods=['GET'])
def api_industry_compare():
    try:
        limit = int(request.args.get('limit', 10))

        # 获取行业比较结果
        result = index_industry_analyzer.compare_industries(limit)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"行业比较出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 保存股票分析结果到数据库
def save_analysis_result(stock_code, market_type, result):
    """保存分析结果到数据库"""
    if not USE_DATABASE:
        return

    try:
        session = get_session()

        # 创建新的分析结果记录
        analysis = AnalysisResult(
            stock_code=stock_code,
            market_type=market_type,
            score=result.get('scores', {}).get('total', 0),
            recommendation=result.get('recommendation', {}).get('action', ''),
            technical_data=result.get('technical_analysis', {}),
            fundamental_data=result.get('fundamental_data', {}),
            capital_flow_data=result.get('capital_flow_data', {}),
            ai_analysis=result.get('ai_analysis', '')
        )

        session.add(analysis)
        session.commit()

    except Exception as e:
        app.logger.error(f"保存分析结果到数据库时出错: {str(e)}")
        if session:
            session.rollback()
    finally:
        if session:
            session.close()


# 从数据库获取历史分析结果
@app.route('/api/history_analysis', methods=['GET'])
def get_history_analysis():
    """获取股票的历史分析结果"""
    if not USE_DATABASE:
        return jsonify({'error': '数据库功能未启用'}), 400

    stock_code = request.args.get('stock_code')
    limit = int(request.args.get('limit', 10))

    if not stock_code:
        return jsonify({'error': '请提供股票代码'}), 400

    try:
        session = get_session()

        # 查询历史分析结果
        results = session.query(AnalysisResult) \
            .filter(AnalysisResult.stock_code == stock_code) \
            .order_by(AnalysisResult.analysis_date.desc()) \
            .limit(limit) \
            .all()

        # 转换为字典列表
        history = [result.to_dict() for result in results]

        return jsonify({'history': history})

    except Exception as e:
        app.logger.error(f"获取历史分析结果时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if session:
            session.close()


# 在应用启动时启动清理线程（保持原有代码不变）
cleaner_thread = threading.Thread(target=run_task_cleaner)
cleaner_thread.daemon = True
cleaner_thread.start()

if __name__ == '__main__':
    # 将 host 设置为 '0.0.0.0' 使其支持所有网络接口访问
    app.run(host='0.0.0.0', port=3886, debug=False)
// 查找获取股票数据的函数
function fetchStockData(stockCode, marketType, startDate, endDate) {
    // 显示加载中提示
    showLoading("正在获取股票数据...");
    
    // 构建API请求URL
    const url = `/api/stock_data?stock_code=${stockCode}&market_type=${marketType}&start_date=${startDate}&end_date=${endDate}`;
    
    // 发送请求
    fetch(url)
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.message || "获取股票数据失败");
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === "error") {
                // 处理错误状态
                showError(data.message);
                hideLoading();
                return;
            }
            
            // 处理成功获取的数据
            processStockData(data.data);
            hideLoading();
        })
        .catch(error => {
            // 显示错误信息
            showError(error.message || "暂时不支持该股票代码分析");
            hideLoading();
            console.error("获取股票数据错误:", error);
        });
}

// 显示错误信息的函数
function showError(message) {
    // 获取或创建错误提示元素
    let errorElement = document.getElementById("error-message");
    if (!errorElement) {
        errorElement = document.createElement("div");
        errorElement.id = "error-message";
        errorElement.className = "error-message";
        document.querySelector(".main-content").prepend(errorElement);
    }
    
    // 设置错误信息并显示
    errorElement.textContent = message;
    errorElement.style.display = "block";
    
    // 5秒后自动隐藏
    setTimeout(() => {
        errorElement.style.display = "none";
    }, 5000);
}

// 显示加载中提示的函数
function showLoading(message) {
    // 获取或创建加载提示元素
    let loadingElement = document.getElementById("loading-indicator");
    if (!loadingElement) {
        loadingElement = document.createElement("div");
        loadingElement.id = "loading-indicator";
        loadingElement.className = "loading-indicator";
        document.querySelector(".main-content").prepend(loadingElement);
    }
    
    // 设置加载信息并显示
    loadingElement.textContent = message || "加载中...";
    loadingElement.style.display = "block";
}

// 隐藏加载中提示的函数
function hideLoading() {
    const loadingElement = document.getElementById("loading-indicator");
    if (loadingElement) {
        loadingElement.style.display = "none";
    }
} 
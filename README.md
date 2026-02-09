# QIS Dashboard

量化投资策略分析仪表板应用

## 项目说明

这是一个用于量化投资策略分析的Dashboard应用，提供数据可视化和分析功能。

## 功能特性

- 📊 实时数据分析与可视化
- 📈 QIS子账簿分析
- 🔍 股票代码映射与查询
- 🤖 集成Copilot SDK支持

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/jasper-wutong/QIS_Dashboard.git
cd QIS_Dashboard
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

### Flask Web应用

```bash
python app.py
```

应用将在 http://localhost:5000 启动

### 命令行工具

```bash
python research_cli.py
```

## 项目结构

```
QIS_Dashboard/
├── app.py                          # Flask主应用
├── research_cli.py                 # 命令行研究工具
├── ticker_mapping.py               # 股票代码映射
├── QIS_SubBook_Analysis.ipynb     # Jupyter分析笔记本
├── templates/
│   └── dashboard.html             # Dashboard模板
├── copilot-sdk/                   # Copilot SDK
├── requirements.txt               # Python依赖
└── README.md                      # 项目说明
```

## 数据文件

项目包含示例数据文件：
- `EDSLib Realtime Result as of 2026-02-06.xlsx`

## 依赖包

主要依赖项（详见 requirements.txt）：
- Flask - Web框架
- pandas - 数据分析
- openpyxl - Excel文件处理
- requests - HTTP请求

## 开发

### Jupyter Notebook

要运行分析笔记本：

```bash
jupyter notebook QIS_SubBook_Analysis.ipynb
```

## 注意事项

- 确保Python版本 >= 3.8
- 首次运行前请安装所有依赖
- 数据文件较大，请耐心等待加载

## License

MIT License

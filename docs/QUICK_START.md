# 快速开始指南

本指南帮助你运行 CHAMPS 项目并查看模型预测结果。

## 📋 前置要求

### 1. 下载 Kaggle CHAMPS 竞赛数据

访问 Kaggle 竞赛页面：
https://www.kaggle.com/c/champs-scalar-coupling/data

下载以下文件并放入 `data/` 目录：
- `train.csv`
- `test.csv`
- `structures.csv`

### 2. 安装依赖包

```powershell
# 安装基础依赖
pip install -r requirements.txt

# 安装 RDKit（通过 conda）
conda install -c rdkit rdkit
```

## 🚀 运行步骤

### 步骤 1：下载预训练模型

```powershell
# 方法 A：使用 Python 脚本（推荐）
python download_models.py

# 方法 B：使用原始 bash 脚本（需要 Git Bash）
bash get_saved_models.sh
```

### 步骤 2：数据预处理

```powershell
cd src
python pipeline_pre.py 1   # 耗时约 1-2 小时
python pipeline_pre.py 2   # 较快
```

### 步骤 3：运行预测

```powershell
# 使用预训练模型进行预测
python predictor.py
```

预测结果将保存在 `submissions/` 目录中。

## 📊 预期输出

- 每个模型的预测结果：`submissions/[model_name].csv.bz2`
- 最终集成结果：`submissions/submission.csv`

## ⚠️ 注意事项

1. **硬件要求**：
   - 需要 NVIDIA GPU（推荐）
   - 至少 16GB 内存
   - 足够的磁盘空间（约 50GB）

2. **CUDA 要求**：
   - CUDA 10.1 或更高
   - 相应的 PyTorch 版本

3. **数据预处理时间**：
   - `pipeline_pre.py 1` 可能需要 1-2 小时
   - 请耐心等待

## 🐛 常见问题

### Q: CUDA out of memory 错误
A: 参考 README.md 中的"Notes on Saving Memory"部分

### Q: RDKit 安装失败
A: 必须使用 conda 安装：`conda install -c rdkit rdkit`

### Q: 缺少数据文件
A: 从 Kaggle 下载竞赛数据并放入 `data/` 目录

## 📖 更多信息

详细信息请参考 `README.md`

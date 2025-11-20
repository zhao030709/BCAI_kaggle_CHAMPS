# ⚠️ 缺少竞赛数据文件

## 需要的文件

请从 Kaggle CHAMPS 竞赛下载以下文件并放入 `data/` 目录：

### 下载地址
🔗 https://www.kaggle.com/c/champs-scalar-coupling/data

### 需要下载的文件：
1. ✅ `train.csv` - 训练数据（约 4.7M 行）
2. ✅ `test.csv` - 测试数据（约 2.5M 行）
3. ✅ `structures.csv` - 分子结构数据

## 下载步骤

### 方法 1：使用 Kaggle API（推荐）

```powershell
# 1. 安装 Kaggle CLI
pip install kaggle

# 2. 配置 API Token
# 访问 https://www.kaggle.com/[你的用户名]/account
# 点击 "Create New API Token"，下载 kaggle.json
# 将 kaggle.json 放到 C:\Users\LENOVO\.kaggle\

# 3. 下载数据
cd data
kaggle competitions download -c champs-scalar-coupling
unzip champs-scalar-coupling.zip
```

### 方法 2：手动下载

1. 访问竞赛页面：https://www.kaggle.com/c/champs-scalar-coupling/data
2. 点击 "Download All"
3. 解压下载的文件
4. 将 `train.csv`, `test.csv`, `structures.csv` 复制到本项目的 `data/` 目录

## 文件大小参考

- `train.csv`: ~320 MB
- `test.csv`: ~165 MB  
- `structures.csv`: ~11 MB

## 验证文件

下载完成后，运行以下命令验证：

```powershell
cd data
Get-ChildItem
```

应该看到：
```
README
structures.csv
test.csv
train.csv
```

## 下一步

文件准备好后，继续执行：
```powershell
cd ../src
python pipeline_pre.py 1
```

# SMILES to J-Coupling Prediction with Set Transformer

一个基于Set Transformer和Hungarian Loss的端到端SMILES到NMR J-coupling预测系统。

## 项目简介

本项目实现了从SMILES分子表示直接预测NMR自旋-自旋耦合常数(J-coupling)的完整流程。核心创新点包括:

- **Set Transformer架构**: 6层Transformer编码器,处理不定数量的J-coupling预测
- **Hungarian Loss**: 解决预测与真实J值的匹配问题,实现端到端训练
- **NMR谱图解析**: 自动解析实验NMR谱图,提取J-coupling数值
- **峰匹配与分配**: 基于分子结构和实验谱图的J-coupling分配算法

### 主要结果

- **测试集性能**: J-MAE = 0.946 Hz (1692个匹配耦合对, 覆盖666个分子)
- **按耦合类型**:
  - 2JHC: MAE = 0.638 Hz
  - 3JHC: MAE = 0.794 Hz  
  - 3JHH: MAE = 1.134 Hz

## 快速开始

### 环境安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

**GPU环境** (推荐):
```bash
conda env create -f environment_gpu.yml
conda activate set_transformer
```

### 数据准备

1. 下载Kaggle CHAMPS数据集 (详见 `data/DOWNLOAD_INSTRUCTIONS.md`)
2. 数据文件放置在 `data/` 目录:
   - `train.csv`: 训练集
   - `test.csv`: 测试集  
   - `structures.csv`: 分子结构

### 训练模型

```bash
python src/train_set_transformer.py \
    --config config/set_transformer_config.json \
    --data pseudo_labeled_dataset_1000.csv \
    --val_split 0.2 \
    --epochs 60
```

### 评估模型

```bash
python src/evaluate_set_transformer.py \
    --config config/set_transformer_config.json \
    --checkpoint checkpoints/set_transformer/checkpoint_epoch35_best.pt \
    --data pseudo_labeled_dataset_1000.csv
```

### 预测

```bash
python src/predict_set_transformer.py \
    --config config/set_transformer_config.json \
    --checkpoint checkpoints/set_transformer/checkpoint_epoch35_best.pt \
    --input filtered_test_dataset_1000.csv \
    --output outputs/predictions/test_predictions.csv
```

## 项目结构

```
BCAI_kaggle_CHAMPS/
├── README.md                          # 项目说明
├── requirements.txt                   # Python依赖 (CPU版本)
├── environment_gpu.yml                # Conda环境 (GPU版本)
├── LICENSE                            # 许可证
│
├── config/                            # 配置文件
│   ├── set_transformer_config.json   # 模型超参数配置
│   └── manual_bond_order_fix.json    # 键级修正规则
│
├── data/                              # 数据集
│   ├── train.csv                     # Kaggle训练集
│   ├── test.csv                      # Kaggle测试集
│   ├── structures.csv                # 分子结构
│   ├── DOWNLOAD_INSTRUCTIONS.md      # 数据下载说明
│   └── README                        # 数据集说明
│
├── src/                               # 源代码
│   ├── train_set_transformer.py      # 训练脚本
│   ├── evaluate_set_transformer.py   # 评估脚本
│   ├── predict_set_transformer.py    # 预测脚本
│   ├── set_transformer_model.py      # Set Transformer模型定义
│   ├── set_transformer_data.py       # 数据加载器
│   ├── set_transformer_loss.py       # Hungarian Loss实现
│   ├── parse_nmr_data.py             # NMR谱图解析
│   ├── assign_peaks.py               # 峰分配算法
│   ├── match_peaks.py                # 峰匹配算法
│   ├── xyz2mol.py                    # 3D坐标转分子图
│   └── modules/                      # 工具模块
│
├── checkpoints/                       # 模型检查点
│   └── set_transformer/
│       ├── checkpoint_epoch35_best.pt # Stage 2最佳模型
│       └── checkpoint_epoch48_best.pt # Stage 2最终模型
│
├── outputs/                           # 输出结果
│   ├── predictions/                  # 预测结果
│   └── evaluation/                   # 评估结果
│
├── cache_features/                    # 3D特征缓存
├── archived/                          # 归档数据
│   └── matched_pairs_1000_original.csv # 测试集ground truth
│
└── docs/                              # 详细文档
    ├── PROJECT_SUMMARY.md            # 完整技术报告
    └── PSEUDO_LABELING_FINAL_REPORT.md # 伪标注报告
```

## 模型架构

### Set Transformer
- **编码器**: 6层, d_model=256, nhead=8, dim_feedforward=512
- **输入**: SMILES分子特征 + 3D构象信息
- **输出**: 不定数量的J-coupling预测 (变长集合)

### Hungarian Loss
- 解决预测集合与真实集合的最优匹配问题
- 结合Chamfer Loss平衡召回率
- 按耦合类型加权: 2JHC(1.0), 3JHC(1.0), 3JHH(1.5)

### 伪标注策略
- **阶段1**: 38分子验证集 (92个J-coupling) 训练初始模型
- **阶段2**: 使用初始模型预测1000分子测试集,生成905个高置信度伪标签
- **阶段3**: 合并验证集+伪标签数据,扩展训练至55 epochs

## 性能指标

| 数据集 | 分子数 | 匹配对数 | J-MAE (Hz) | RMSE (Hz) |
|--------|--------|----------|------------|-----------|
| 验证集 | 38     | 92       | -          | -         |
| 测试集 | 1000   | 1692     | 0.946      | 1.115     |

**按耦合类型分析**:
| 耦合类型 | 数量 | MAE (Hz) | RMSE (Hz) |
|----------|------|----------|-----------|
| 2JHC     | 485  | 0.638    | 0.737     |
| 3JHC     | 690  | 0.794    | 0.945     |
| 3JHH     | 517  | 1.134    | 1.484     |

**误差分布**:
- |Error| < 0.5 Hz: 42.7%
- |Error| < 1.0 Hz: 69.6%
- |Error| < 2.0 Hz: 92.0%

## 使用示例

### 从SMILES预测J-coupling

```python
import torch
from src.set_transformer_model import SetTransformer
from src.set_transformer_data import prepare_smiles_features

# 加载模型
config = {...}  # 从config/set_transformer_config.json加载
model = SetTransformer(config)
model.load_state_dict(torch.load('checkpoints/set_transformer/checkpoint_epoch35_best.pt'))
model.eval()

# 准备输入
smiles = "CCO"  # 乙醇
features = prepare_smiles_features(smiles)

# 预测
with torch.no_grad():
    predictions = model(features)
    j_couplings = predictions['j_values']  # 预测的J-coupling值
    coupling_types = predictions['types']   # 耦合类型
```

## 技术细节

详细的技术文档请参考:
- **完整报告**: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)
- **伪标注流程**: [docs/PSEUDO_LABELING_FINAL_REPORT.md](docs/PSEUDO_LABELING_FINAL_REPORT.md)

## 系统要求

- **Python**: 3.8+
- **PyTorch**: 1.10+ (推荐使用GPU版本)
- **内存**: 16GB+ (训练时)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (可选,但强烈推荐)

## 依赖包

主要依赖:
- `torch >= 1.10.0`
- `rdkit >= 2022.03.1`
- `scipy >= 1.7.0`
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`

完整依赖列表见 `requirements.txt` 或 `environment_gpu.yml`

## 引用

如果本项目对您的研究有帮助,请引用:

```bibtex
@misc{smiles_to_j_coupling,
  title={SMILES to J-Coupling Prediction with Set Transformer},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/BCAI_kaggle_CHAMPS}}
}
```

## 致谢

- 数据集来源: [Kaggle CHAMPS Scalar Coupling Competition](https://www.kaggle.com/c/champs-scalar-coupling)
- Set Transformer: Lee et al., "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks", ICML 2019
- Hungarian Algorithm: Kuhn-Munkres algorithm for optimal assignment

## 许可证

本项目使用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub Issues: 提交bug报告或功能请求
- Email: your.email@example.com

---

**Note**: 本项目基于Kaggle CHAMPS竞赛数据集开发,主要用于学术研究和教育目的。

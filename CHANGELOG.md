# CHANGELOG

项目变更记录

## [1.0.0] - 2025-11-20

### 项目重构 - 从BCAI_kaggle_CHAMPS到SMILES-to-J独立项目

#### 清理与优化

**删除的内容**:
- ✅ 原BCAI项目的README和各种shell/bat脚本 (14+个文件)
- ✅ BCAI_kaggle_CHAMPS/子目录 (原项目完整副本)
- ✅ archived/中的临时分析文件和冗余matched_pairs (保留matched_pairs_1000_original.csv)
- ✅ src/中的原BCAI训练脚本和临时测试脚本 (21个文件)
- ✅ models/目录中的model_A到model_M (13个原BCAI预训练模型文件夹)
- ✅ checkpoints/中的中间epoch检查点 (保留epoch35_best.pt和epoch48_best.pt)
- ✅ data/champs-scalar-coupling/子目录 (冗余数据)
- ✅ config/models.json (原BCAI模型配置)
- ✅ outputs/中的冗余评估和预测文件
- ✅ runs/目录 (TensorBoard日志)
- ✅ docs/中的临时文档 (entry_points.md, GPU_FULL_RUN.md等)

**保留的核心内容**:

*数据文件*:
- `pseudo_labeled_dataset.csv` (38分子验证集, 92耦合)
- `pseudo_labeled_dataset_1000.csv` (905耦合训练集)
- `filtered_test_dataset.csv` (2505分子完整测试集)
- `filtered_test_dataset_1000.csv` (1000分子测试子集)
- `archived/matched_pairs_1000_original.csv` (1692个ground truth耦合对)
- `cache_features/` (1000个分子的3D特征缓存, 5.61 MB)

*源代码* (src/目录):
- `train_set_transformer.py` - 训练脚本
- `evaluate_set_transformer.py` - 评估脚本
- `predict_set_transformer.py` - 预测脚本
- `set_transformer_model.py` - Set Transformer模型定义
- `set_transformer_data.py` - 数据加载器
- `set_transformer_loss.py` - Hungarian Loss实现
- `parse_nmr_data.py` - NMR谱图解析
- `assign_peaks.py` - 峰分配算法
- `match_peaks.py` - 峰匹配算法
- `xyz2mol.py` - 3D坐标转分子图
- `evaluate_matched_pairs.py` - 评估脚本
- `modules/` - 工具模块 (embeddings, optimizations等)
- `utils/` - 辅助工具

*模型检查点*:
- `checkpoints/set_transformer/checkpoint_epoch35_best.pt` (Stage 2最佳, Val MAE: 0.946 Hz)
- `checkpoints/set_transformer/checkpoint_epoch48_best.pt` (Stage 2最终)

*配置文件*:
- `config/set_transformer_config.json` - 模型超参数
- `config/manual_bond_order_fix.json` - 键级修正规则

*文档*:
- `docs/PROJECT_SUMMARY.md` - 完整技术报告
- `docs/PSEUDO_LABELING_FINAL_REPORT.md` - 伪标注详细流程
- `docs/QUICK_START.md` - 快速开始指南

#### 新增内容

**文档**:
- ✅ `README.md` - 全新的GitHub项目首页,简洁专业
- ✅ `CHANGELOG.md` - 项目变更记录 (本文件)

**配置**:
- ✅ `.gitignore` - 更新以适配新项目结构,保留关键文件

**输出结果**:
- `outputs/evaluation/` - 保留evaluation_results_1000mol.json/csv, matched_pairs_1000_evaluation.json, prediction_statistics_1000.json
- `outputs/predictions/` - 保留test_predictions_1000_filtered.csv

#### 项目统计

**代码减少**:
- Python文件: 32 → 11 (src/目录)
- Shell/Bat脚本: 14+ → 0
- 文档文件: 7 → 3 (docs/目录)
- 配置文件: 3 → 2 (config/目录)

**磁盘空间优化**:
- 删除的检查点文件: ~60个中间epoch检查点
- 删除的模型文件夹: 13个BCAI预训练模型
- 删除的子目录: BCAI_kaggle_CHAMPS/完整副本
- cache_features/: 保留 (5.61 MB, 优化用户体验)

**最终项目结构**:
```
BCAI_kaggle_CHAMPS/
├── README.md                  # 新建
├── CHANGELOG.md               # 新建
├── requirements.txt
├── environment_gpu.yml
├── LICENSE
├── .gitignore                 # 更新
│
├── config/                    # 2个配置文件
├── data/                      # Kaggle数据集 (用户自行下载)
├── src/                       # 11个核心Python文件
├── checkpoints/               # 2个最佳检查点
├── outputs/                   # 评估和预测结果
├── cache_features/            # 3D特征缓存 (5.61 MB)
├── archived/                  # 1个ground truth文件
├── docs/                      # 3个文档文件
├── processed/                 # 空目录(供用户使用)
├── submissions/               # 空目录(供用户使用)
└── validation_subset/         # 验证子集数据
```

### 模型性能 (已验证)

**测试集评估** (1000分子):
- **J-MAE**: 0.946 Hz
- **RMSE**: 1.115 Hz
- **匹配对数**: 1692 (覆盖666个分子)

**按耦合类型**:
| 类型  | 数量 | MAE (Hz) | RMSE (Hz) |
|-------|------|----------|-----------|
| 2JHC  | 485  | 0.638    | 0.737     |
| 3JHC  | 690  | 0.794    | 0.945     |
| 3JHH  | 517  | 1.134    | 1.484     |

**误差分布**:
- |Error| < 0.5 Hz: 42.7%
- |Error| < 1.0 Hz: 69.6%
- |Error| < 2.0 Hz: 92.0%

### 训练配置 (已修正)

**Stage 1**: 38分子验证集
- Epochs: 20
- Validation MAE: ~2.0 Hz
- 用途: 初始模型训练

**Stage 2**: 扩展训练 (905耦合对)
- **实际Epochs**: 55 (之前文档错误记录为65)
- **最佳Epoch**: 35 (Val MAE: 0.946 Hz)
- **最终Epoch**: 48 (保存用于对比)

### 技术栈

- **模型**: Set Transformer (6层, d_model=256, 4.97M参数)
- **损失函数**: Hungarian Loss + Chamfer Loss
- **框架**: PyTorch 1.10+
- **化学工具**: RDKit 2022.03+
- **优化算法**: scipy.optimize.linear_sum_assignment

### 未来计划

- [ ] 添加模型API文档
- [ ] 创建Web演示界面
- [ ] 扩展到更多NMR参数 (化学位移预测)
- [ ] 优化3D构象生成速度
- [ ] 支持更大规模数据集训练

### 致谢

- 数据集: Kaggle CHAMPS Scalar Coupling Competition
- 原项目: BCAI_kaggle_CHAMPS (竞赛代码基础)
- 核心创新: Set Transformer + Hungarian Loss for SMILES→J prediction

---

## 版本说明

- **v1.0.0**: 首个独立发布版本,完成从BCAI竞赛代码到SMILES-to-J预测项目的完整转型

## 贡献者

- 主要开发: [Your Name]
- 项目重构: 2025年11月

## 许可证

MIT License - 详见LICENSE文件

<div align="center">

<img src="assets/banner.svg" alt="PrismLoop-RL：面向视觉智能体的工具感知强化学习" width="100%" />

# PrismLoop-RL

**面向图像与视频智能体的工具感知强化学习实验项目**

让模型在“观察 → 推理 → 执行 Python → 读取反馈”的循环中学习。

[English](README.md) · [方法设计](docs/METHOD.md) · [实验说明](docs/EXPERIMENTS.md) · [复现指南](docs/REPRODUCTION.md) · [研究路线](docs/ROADMAP.md)

</div>

## 项目定位

PrismLoop-RL 基于 [PyVision-RL](https://github.com/agents-x-project/PyVision-RL)，围绕多模态 Agentic RL 的奖励处理、训练样本选择、工具调用约束与训练可靠性进行复现和扩展。

**Prism** 对应图像与视频等视觉输入；**Loop** 对应模型与 Python 环境交互，以及策略持续更新的过程。项目名称表达研究方向，不代表已经提出或验证全新的强化学习算法。

<img src="assets/architecture.svg" alt="视觉输入、工具交互、奖励与样本选择、GRPO 更新和评测审计流程" width="100%" />

## 已完成的工作

| 方向 | 设计出发点 | 当前实现 |
|---|---|---|
| 最终答案解析 | 推理和代码中的数字可能干扰奖励计算 | 提取最终数值答案，再沿用原 MRA 评分公式 |
| 数据源配额 | 不同任务的奖励方差可能影响采样构成 | 按来源分配配额，在来源内部排序问题组 |
| 工具质量约束 | 重复执行或空输出不应反复获得奖励 | 检查成功状态、输出非空、代码与反馈去重，限制累计奖励 |
| 执行稳定性 | 工具阻塞影响 rollout 完成和资源释放 | spawn 工作进程与有界清理 |
| 消融与对照 | 继续训练的收益需要与方法收益分开 | 显式组件开关、固定起点、原方法同期对照队列 |
| 实验证据 | 有 checkpoint 不等于权重确实更新 | 参数与优化器审计、导出验证、评测记录与协议核验 |

这些是项目对上游工作的扩展。工具质量约束属于启发式规则，不能证明一次工具调用对解题具有因果贡献。

## 当前实测结果

每个实验新增 **40 个 rollout global step**。Image 从第 20 步到第 60 步；Video 从第 120 步到第 160 步。下面使用服务器统一评测条件与本机 32B judge，数值为百分比；VStar 为 avg@32，VSI 为 8 类任务宏平均。

| 模型 / 实验 | VStar ↑ | HRBench-4K ↑ | VSI 宏平均 ↑ |
|---|---:|---:|---:|
| 官方 SFT 权重 | 68.80 | 63.38 | 35.71 |
| 官方发布的 RL 权重 | 83.97 | 72.38 | 43.18 |
| 服务器复现原方法的起点 checkpoint | 70.14 | 66.00 | 35.98 |
| 核心扩展，新增 40 步 | 71.53 | 66.50 | 36.98 |
| 核心扩展 + PPO epochs=2，新增 40 步 | **71.65** | **68.75** | **37.52** |

粗体只表示当前扩展实验中最高的结果。Image 与 Video 对应不同模型。

<img src="assets/short-run-gains.png" alt="新增 40 步后相对训练起点的观测提升，不代表同预算下方法的净收益" width="100%" />

**结果边界：** 相对起点的提升同时包含继续训练的收益；本发布快照尚未纳入同期原方法对照及完整消融结果。PPO epochs=2 的参数更新次数是 epochs=1 的两倍。当前是单随机种子短程实验，尚不能宣称统计显著、达到 SOTA，或全面优于原方法。官方发布 RL 权重的训练预算没有与本实验对齐。

[全部结果](results/benchmark-results.csv) · [评测协议与异常说明](docs/EXPERIMENTS.md)

## 如何使用

先阅读 [方法设计](docs/METHOD.md) 与 [复现指南](docs/REPRODUCTION.md)。当前发布的是研究源码快照，历史脚本中的路径、GPU 分配和端口需要根据机器调整。模型权重、训练数据和评测数据需要另行获取；评测框架作为外部依赖列出。

训练代码、数据准备、评测编排与证据核验脚本均已发布。尚未实现的空间一致性、工具修复信用分配和自适应 rollout 分配，明确保留在 [后续研究计划](docs/ROADMAP.md) 中。

## 来源与署名

本项目基于 **PyVision-RL 及其包含的 verl/RAGEN 衍生组件**，保留上游许可证、版权和文档。详见 [上游说明](docs/UPSTREAM_README.md)、[第三方声明](THIRD_PARTY_NOTICES.txt) 与 [源码版本记录](docs/source-provenance.json)。使用时请同时署名、引用相关上游工作。

维护者：[carlyer](https://github.com/carlyer) · [MIT License](LICENSE)

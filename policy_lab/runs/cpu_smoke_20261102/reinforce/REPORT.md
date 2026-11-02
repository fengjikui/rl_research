# 视觉策略梯度实验 REINFORCE

环境：CarRacing-v3；状态：completed。
所有得分为环境原始奖励之和；相同固定种子和相同评估策略用于阶段比较。
达到评估步数上限的记录明确标为 capped，不能称完整回合最终成绩。

| 阶段 | 采集决策 | 均分 | 标准差 | 有截断 | 录像 |
|---|---:|---:|---:|---|---|
| before_s000000000 | 0 | 5.21 | 0.00 | True | [MP4](before_s000000000/evaluation/play.mp4) |
| final_s000000429 | 429 | 8.42 | 0.00 | True | [MP4](final_s000000429/evaluation/play.mp4) |

训练前后 model.pt 均为实际保存的权重；source/、config.json、逐步原始轨迹和 update 日志可用于复核。
latest.pt 用于继续学习；模拟器和未更新的 on-policy 轨迹不恢复，续训从新一局重新采集。
本程序尚不能由代码本身证明游戏成绩提升；以本目录实际评估和录像为准。

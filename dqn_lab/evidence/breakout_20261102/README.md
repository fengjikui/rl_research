# 本机打砖块实验的公开证据

两套模型分别从零训练，均使用本机 MPS、真实 ALE Breakout 和相同游戏规则。本目录只精选可分享的证据；全部逐步训练日志、优化器和磁盘回放池仍在本机原运行目录。

- [全部已保存阶段的分数与配置](stage_scores.json)：包含中间回落；不是挑选最佳结果后的平均。
- [DQN 最佳阶段完整首局录像](dqn/best/play.mp4)：318 分、1830 个决策，原始奖励逐步记录见 [JSONL](dqn/best/episode_00.jsonl)，[模型权重](dqn/best/model.pt)用于推理，不用于恢复训练。
- [DQN 训练前完整首局录像](dqn/before/play.mp4)：0 分。训练约五分钟的9分阶段保留了[原始评分与动作](dqn/early/evaluation.json)，该阶段旧录像已按留存规则精简，不能提供其原视频。
- [Rainbow 最佳阶段首局录像](rainbow/best/play.mp4)与[原始逐步奖励](rainbow/best/episode_00.jsonl)：准确分数见 evaluation.json，不代表稳定达到该水平。
- [DQN 曲线](dqn_curve.png)、[Rainbow 曲线](rainbow_curve.png)。

固定种子贪心评估中的三局常是同一轨迹，不能当作三份独立泛化证据。DQN 318 分阶段另做过十局 epsilon=0.05 的复测，均分为113.3，[逐局分数与原始动作奖励](dqn/best/heldout/evaluation.json)均保留；不能把318称为稳定平均能力。Rainbow的最新独立复测另见本目录JSON（若存在）。

两次训练均已按用户要求停止，自动续训开关已关闭。代码保留训练入口，但不会因为打开文档而启动任务。运行前需自行选择新目录并检查资源。

时间比较：按累计训练窗口（包含阶段评估，排除段间停顿），约30分钟 DQN22分、Rainbow3分；约60分钟 DQN31分、Rainbow17分。同数据量约49万步时分别22、23分。单个训练种子的早期实验不等于算法的一般排名。

完整回放文件可以直接从仓库下载；GitHub网页不一定内嵌播放MP4。其他历史文档中的 runs/ 链接可能只指本机完整实验目录，公开读者优先使用本证据目录。

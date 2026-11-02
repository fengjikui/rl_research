"""The complete DQN regression update, shared by Catch and real Atari."""
import json
import numpy as np
import torch
from torch.nn import functional as F


def update(online, target_net, optimizer, transitions, gamma, audit_dir=None,
           action_names=None):
    # ===== 形状约定：这一份更新函数同时用于接球与 Atari =====
    # B = 本批经验数量，不是同时运行的环境数量。
    # A = 每个状态的动作数量；接球 A=3，本文 Breakout A=4。
    # S = 一条观察的形状：接球 (2,6,5)，Atari (4,84,84)。
    # 一批观察写成 [B,*S]；*S 表示展开观察本身的各维度。
    # 实测接球 B=64 -> [64,2,6,5]；Atari 短实验 B=16 -> [16,4,84,84]。
    # transitions 是长度为 B 的 Python 列表，每项都是
    # (state, action, reward, next_state, terminated)。不是一个整局轨迹。

    # zip(*transitions)：把 B 行、每行 5 个字段，转为 5 组、每组 B 项。
    # np.asarray 分别将每组堆成 NumPy 数组：
    # states / next_states: [B,*S]；actions / rewards / terminated: [B]。
    # 状态数组：接球 float32，Atari uint8 像素；标量字段随后明确指定 dtype。
    states, actions, rewards, next_states, terminated = map(
        np.asarray, zip(*transitions))
    # NumPy -> PyTorch；as_tensor 不增减维度，并可能与 NumPy 共享底层存储。
    # 数据移到在线网络所在设备：CPU / MPS / CUDA，形状和更新公式不变。
    device = next(online.parameters()).device
    states = torch.as_tensor(states, device=device)
    next_states = torch.as_tensor(next_states, device=device)
    # [B] int64：这是动作索引，供 gather 使用，不是回归标签，也不是 one-hot。
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    # [B] float32：每条经验对应一个实际获得的单步奖励。
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    # [B] float32：False/True -> 0.0/1.0，便于计算 (1-terminated)。
    # 这里表示游戏真正终止；单纯采集时间上限不自动等于环境终止。
    terminated = torch.as_tensor(terminated, dtype=torch.float32, device=device)

    # [batch, actions] -> [batch]: select only the action actually taken.
    # 按顺序拆开这一行：
    # 1) online(states): [B,*S] -> [B,A]，每行有当前状态的全部动作 Q 值。
    # 2) actions[:,None]: [B] -> [B,1]；增加一维以匹配 gather 的索引维数。
    # 3) gather(1,...): 沿第 1 维（动作列）逐行选实际动作 -> [B,1]。
    #    例如 actions[i]=2，就取 online(states)[i,2]，不是取最大的动作。
    # 4) squeeze(1): 仅移除那个长度为 1 的动作列维 -> [B]。
    #    显式写 squeeze(1)，即使 B=1，也保留批次维而不变成标量。
    # prediction: [B] float32，保留计算图，可对在线网络参数反向传播。
    prediction = online(states).gather(1, actions[:, None]).squeeze(1)
    # 标签分支不建反向传播图；目标网络参数也在训练初始化时被冻结。
    with torch.no_grad():
        # target_net(next_states): [B,*S] -> [B,A]。
        # max(dim=1) 沿动作列取最大值，返回 values 和 indices，均为 [B]。
        # 这里只用 .values，也就是每条经验的下一状态最佳价值估计。
        next_value = target_net(next_states).max(dim=1).values
        # This is the regression label. It is not the correct action ID.
        # rewards、terminated、next_value 均为 [B]，下面逐元素运算：
        # (1-terminated): [B]；gamma 是 Python 标量，乘法不会增加维度。
        # terminated[i]=1 -> label[i]=rewards[i]；否则加折扣后的 next_value[i]。
        # label: [B] float32、requires_grad=False；它才是本次回归的标签。
        label = rewards + gamma * (1-terminated) * next_value
    # prediction 和 label 一一对应，都是 [B]；不会对未执行动作另外构造标签。
    # reduction='none' 保留每条经验的 Huber/Smooth L1 损失 -> [B]。
    per_item_loss = F.smooth_l1_loss(prediction, label, reduction='none')
    # 对 B 项取平均 -> 标量张量，shape 为 []（不是 [1]），保留梯度图。
    loss = per_item_loss.mean()

    if audit_dir is not None:
        # 以下只是保存真实批次供检查，不参与构造 loss 或梯度。
        audit_dir.mkdir(parents=True, exist_ok=True)
        # 原始观察/下一观察仍为 [B,*S]；其余数组为 [B]。
        # 日志先复制回 CPU；不改变训练张量的设备。
        np.savez_compressed(audit_dir/'batch_inputs.npz',
            states=states.cpu().numpy(), actions=actions.cpu().numpy(), rewards=rewards.cpu().numpy(),
            next_states=next_states.cpu().numpy(), terminated=terminated.cpu().numpy())
        # 对 [B] 的每个数组取 [i] -> 标量张量，转成 Python 数值写 JSON。
        # 预测/损失先 detach，仅断开日志数值的计算图，不改变原来的 loss。
        rows = [dict(index=i, action=int(actions[i]),
            action_name=action_names[int(actions[i])] if action_names else str(int(actions[i])),
            reward=float(rewards[i]), terminated=bool(terminated[i]),
            next_max_q=float(next_value[i]), gamma=gamma,
            regression_label=float(label[i]), prediction_before_update=float(prediction[i].detach()),
            smooth_l1_loss=float(per_item_loss[i].detach())) for i in range(len(actions))]
        record = dict(input_shape=list(states.shape), input_dtype=str(states.dtype),
            mean_loss=float(loss.detach()), rows=rows,
            label_formula='r + gamma * (1 - terminated) * max_a Q_target(next_state, a)',
            label_requires_grad=label.requires_grad)
        (audit_dir/'loss_batch.json').write_text(json.dumps(record, indent=2)+'\n')

    # 清理上一批参数梯度；不会清零预测值、标签或网络权重。
    optimizer.zero_grad()
    # 从标量 loss 反向传播。每个参数的 grad 与该参数本身形状相同；
    # 梯度经 prediction 回到在线网络，不沿 label 流入目标网络。
    loss.backward()
    # 对在线网络全部参数的梯度做整体范数裁剪；不改变各梯度的形状。
    torch.nn.utils.clip_grad_norm_(online.parameters(), 5.)
    # Adam 使用这些梯度更新在线网络参数；参数形状保持原样。
    optimizer.step()
    # 报告用 Python float，不再是带计算图的张量。
    return float(loss.detach())

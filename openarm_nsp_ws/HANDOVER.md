# OpenArm 控制栈 — 工作交接文档

> 更新：2026-08-27。本文档是全部工作的**总入口**：系统全貌、文档地图、
> 当前实机状态、已验证结论、未解决问题与恢复指引。接手者从这份文件开始。

---

## 0. 一页全貌

为 OpenArm v1.0（7-DoF 双臂，达妙电机，CAN-FD）构建的完整控制栈，三个层次：

```
NSP 逆运动学（零空间投影，主动远离奇异/限位）        ← openarm_pinocchio_nsp
  ↓ (times, q_path)
离线轨迹规划：直线 / SE(3)弧线 / 关节回放 / B样条    ← cartesian_planner / bspline_planner
  ↓ (执行)
250Hz 实时 web 面板：状态机 + 重力补偿 G(q) +        ← openarm_dashboard
   笛卡尔 6D 阻抗控制（当前实机功能边界）
```

**当前实机可用**（v8，2026-08-27 回退后验证）：使能/零力矩/抱住/归零、
示教-直线-弧线-关节回放- B样条执行、重力补偿、定点阻抗控制全套
（推-回弹 / 拖动模式 / 方向感知奇异防护 / 软挡块 / 内推屈肘）、
自碰撞检测、全程调试日志。

**已开发但已回退**：轨迹录制回放 + 运动中阻抗（IMP_TRACK）——因实机
曲线冻结 bug 回退到 v8（见 §5.1），代码保留在 git 历史。

---

## 1. 目录与仓库

| 路径 | 内容 |
|---|---|
| `/ros2_ws/openarm_nsp_ws/` | **主工作区**（NSP + dashboard + 全部文档）|
| `/ros2_ws/openarm_pinocchio_ik/` | 早期 IK 工作区（FK/IK 验证、moveit_web 面板、备份清单）|
| `/ros2_ws/opeanrm_pinocchio_ik_nsp/` | **git 发布仓库**（以上两个工作区的源码快照，push 到 GitHub `leiying107/opeanrm_pinocchio_ik_nsp`）|

**Git 状态（发布仓库）**：本地 `fe71aa3` 领先远端多个提交，**待 push**
（clash 代理关闭时无法推送；开启后用 memory `github-repo-leiying107` 里的
隧道命令）。关键提交：

```
fe71aa3  回退 dashboard 到 v8（曲线冻结）★ 当前实机版本
3802802  编码器防护误触发修复（使能闪断）
f2b2d1f  CAN 通道交换（物理换线：left=ch1 / right=ch0）
99e4db3  编码器重索引防护
e845090  碰撞感知规划 + 双臂规划器
cfd7149  轨迹回放关节空间化（Kr=0）      ┓
485938a  回放腕部爆炸修复               ┃ 已回退的
54b2e0a  轨迹卡片 UI 配色               ┃ 轨迹功能
ac95197  IMP_TRACK 压测 + 文档           ┃ （保留在历史）
38647fb  轨迹录制回放 + IMP_TRACK       ┛
1576519  阻抗 v8 快照（回退目标）★
d0afcc9  笛卡尔 6D 阻抗控制
943b872  web_panel + 重力补偿
33d564a  初始上传（NSP-IK + 规划器 + dashboard）
```

---

## 2. 文档地图（按功能）

| 文档 | 内容 | 何时读 |
|---|---|---|
| **USAGE.md**（760 行，18 节）| 总手册：环境/编译/CAN、CLI 工具、dashboard 全用法、示教-复现、弧线静止启停、离线验证、API 速查、故障排查；§13-15 web_panel/重力补偿/误差显示、§17 阻抗 | 日常操作查这份 |
| **IMPEDANCE_THEORY.md**（294 行）| 阻抗控制原理：控制律逐项、奇异防护 v5（W 矩阵方向淡化 + 速率型屈肘）、饱和链、线程模型、实时性保障、压测体系 §8.1 | 想懂原理/改控制律 |
| **IMPEDANCE_SAFETY.md**（295 行）| 阻抗安全手册：最保守参数（照抄）、每个参数含义、启动流程、中止条件、四次实机事故完整复盘（§8-10：超时/回弹/15Hz振荡/编码器）| **上机前必读**；出事后对照 §7 调试日志 |
| **TRAJECTORY_TEST.md**（204 行）| 轨迹录制回放的仿真验证（R1-R5 + TL1-TL6 压测 157 场景）+ 实机测试规程 | **恢复轨迹功能时**读（当前功能已回退）|
| **COLLISION_AVOIDANCE.md**（146 行）| 自碰撞检测（163 对）+ 让位 IK + 三策略避障 + 14 维双臂 B 样条 | 碰撞/双臂规划 |
| `openarm_pinocchio_nsp/README.md` | NSP-IK 工具箱：纯规划（不碰硬件）、plan_trajectory 用法、规划/执行分离 | 用 IK/规划 API |
| `openarm_pinocchio_ik/BACKUP_MANIFEST.md` | 早期 IK 工作区清单 | 考古 |
| **本文档 HANDOVER.md** | 总入口/交接 | 接手第一天 |

---

## 3. 各功能模块现状

### 3.1 逆运动学（NSP-IK）— ✅ 稳定
- 两阶段：DLS 收敛 + 零空间多目标爬升（可操控度 + 关节限位居中）
- 规避了 KDL 在归零奇异位的失效；`ik_nsp`/`ik_multi` 热启动
- 位置：`openarm_pinocchio_nsp/kinematics.py`；验证：`validation/`

### 3.2 正运动学 — ✅ 稳定
- `PinocchioModel.fk(q)` → (xyz, xyzw)；`_pose_and_jac` 一次 FK+Jacobian
（阻抗 250Hz 用，比 fk+jacobian6 省一半计算）

### 3.3 轨迹规划 — ✅ 稳定
- `plan_cartesian`（直线/路点）、`fit_arc`（SE(3) 圆弧拟合）、
  `pose_replay_traj`（关节回放重定时）、`BSplineOptimizer`（SLSQP+自碰撞）
- 全部输出 `(times, q_path)`；五次缓入缓出 `ease_in_out_retime`
- **规划与执行分离**：规划器零 CAN 代码

### 3.4 重力补偿 — ✅ 实机验证
- `gravity.py`：pinocchio `computeGeneralizedGravity` on v1_simple.urdf，
  **无电机偏置**（与 MuJoCo/KDL 三方一致验证）；`MITParam(0,0,0,0,+G)` 纯力矩
- UI：checkbox + scale 滑条；与零力矩/阻抗共用路径

### 3.5 笛卡尔 6D 阻抗控制（v8）— ✅ 实机验证（当前功能边界）
- **控制律**：`τ = Jᵀ(K·W·Δ − D·W·ẋ) + 关节弹簧 + G(q)`，纯力矩 MIT 路径
- **四档预设**：极软(100)/软(300)/中(800)/硬(1500) N/m；ζ 滑条 0.5-1.5（**硬上限 1.5**）
- **方向感知奇异防护**：W=U·diag(σ²/(σ²+0.05²))·Uᵀ 只淡化失控方向；
  速率型屈肘控制沿 Vt[5]（软挡块 + 内推屈肘，实机验证）
- **回位**：kx 比例姿态弹簧 + 未投影配置回位弹簧（肘部深位移回位 83-94%）
- **安全链**：入口 σ 门禁 → σ 持续硬退出(0.5s) → 超速衰减计数 → 关节限位 → 看门狗
- 实机事故四次全部根因修复并归档在 IMPEDANCE_SAFETY.md §8-10

### 3.6 自碰撞检测 — ✅ 实机验证
- 163 对（双臂/基座），20mm 边距，实时监护 + 让位 IK

### 3.7 轨迹录制回放 + 运动中阻抗（IMP_TRACK）— ⚠️ 已回退
- 功能已完整实现并通过 157 场景压测，但实机出现**曲线冻结 bug**
（250Hz 线程静默停摆）→ 2026-08-27 回退到 v8，实机恢复正常
- **代码在 git** `38647fb..3802802`；恢复步骤见 §5.2

### 3.8 调试日志体系 — ✅ 全功能内建
- `log/panel_<时间戳>/`：panel.log + commands.jsonl + 250Hz ctrl CSV
  （49 列：q/dq/τ感/τ令/温度/Δx/F_est/σ/屈肘）+ can_stats.jsonl +
  异常事件自动转储（前 5 秒）+ ring_tail
- **32 个历史会话**可考古；分析方法见 IMPEDANCE_SAFETY.md §7

---

## 4. 环境速查（细节在 USAGE.md §3）

```bash
# 编译（venv python 驱动 colcon，--symlink-install 改码即生效）
/ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/python -m colcon build \
    --packages-select openarm_pinocchio_nsp openarm_dashboard --symlink-install

# 起面板
source /opt/ros/humble/setup.bash
source /ros2_ws/openarm_ros2/install/setup.bash
source /ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/activate
source /ros2_ws/openarm_nsp_ws/install/setup.bash
ros2 run openarm_dashboard web_panel        # :8050
```

- **venv 是硬要求**（系统 numpy 2.x 与 pinocchio 二进制不兼容，导入即段错误）
- **CAN 映射（2026-08-27 物理换线后）**：LEFT=`can_slot1_ch1`、
  RIGHT=`can_slot1_ch0`（代码里两处 CAN_MAP 已同步；换回线材要改回）
- 本机还跑着默认域的实机栈：**测试用 ROS_DOMAIN_ID=42 隔离，绝不要 kill**
- 回归测试：`scripts/test_impedance_sim.py`（10 项，改阻抗代码后必跑）

---

## 5. 未解决问题（按优先级）

### 5.1 轨迹功能冻结 bug（恢复轨迹功能的前置）
- **现象**：实机面板启动 ~1.2s 后曲线冻结——250Hz 线程静默停摆
  （120 行数据后无输出、无错误日志、进程存活）
- **已确认**：软件问题（回退到 v8 后实机正常，排除线缆）
- **嫌疑**：编码器防护的 `_dump_event` 半初始化死锁，或轨迹命令路径死锁
- **定位方法**：在 v8 基础上逐个 cherry-pick 轨迹提交，每次实机验证曲线

### 5.2 右臂 CAN 物理故障（硬件，未修复）
- **症状**（换线前 ch1）：每会话 tx_error +12.6万~44.2万、tx_dropped
  4k-14k、292 次 bus-off；错误风暴导致**编码器重索引**（全部关节单 tick
  跳变 0.3-3.1 rad，重力前馈瞬间反号 → 肩部整圈旋转撞机身）
- **换线提供了判别实验**：看故障跟臂走还是跟通道走——新会话里
  ch0（现接右臂）是否还报错。**尚未执行**
- 检查项：终端电阻、连接器、走线与动力线捆扎

### 5.3 待 push
8 个 commit 在本地 `opeanrm_pinocchio_ik_nsp`。clash 开启后推送
（命令在 memory `github-repo-leiying107`，需走 clash 隧道）。

---

## 6. 恢复轨迹功能步骤（未来）

1. 先解决 §5.1（二分定位冻结 bug）
2. 从 git 恢复：
   ```bash
   cd /ros2_ws/opeanrm_pinocchio_ik_nsp
   git checkout cfd7149 -- openarm_nsp_ws/src/openarm_dashboard/...
   # （cfd7149 = 关节空间回放最终版；恢复后同步到 openarm_nsp_ws）
   ```
3. 同步恢复 `traj_rec.py`、`/traj` 端点、index.html 轨迹卡片、测试脚本
4. 跑 `test_traj_sim.py`（5 项）+ `test_traj_stress.py`（157 场景）
5. 按 TRAJECTORY_TEST.md §2 实机阶梯验证
6. **注意**：恢复时保留 CAN 交换（left=ch1）与 v8 之后的阻抗修复

---

## 7. 关键教训索引（防回归，详见各文档）

| 教训 | 出处 |
|---|---|
| 腕部 ZOH 阻尼上限 0.2·I/dt 是承重墙，永不可封弹簧 | IMPEDANCE_THEORY §2.2 |
| IMP_KD 实机必须为 0（电机侧 kd × 量化速度 = 抖动）| IMPEDANCE_SAFETY §0 |
| BLAS 必须单线程 + GC 关闭（实时性）| IMPEDANCE_SAFETY §8-9 |
| 屈肘控制必须速率型（位置/力形式全部不稳定）| IMPEDANCE_THEORY §4 |
| ζ 硬上限 1.5（方向阻尼 >1.5 确定性失稳）| memory |
| 压测 harness 必须每场景重置控制器滤波状态 | memory |
| ease 时钟必须有下限（bump 在 τ=0 为零 → 死锁）| memory |
| 编码器防护需要使能后稳定窗（否则使能闪断）| §5.1 嫌疑链 |

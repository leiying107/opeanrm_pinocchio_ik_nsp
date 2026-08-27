# OpenArm 控制栈 — 工作交接文档

> 本文档是全部工作的总入口：系统全貌、文档地图、
> 功能实现说明、环境速查、已解决问题记录。接手者从这份文件开始。

---

## 0. 一页全貌

为 OpenArm v1.0（7-DoF 双臂，达妙电机，CAN-FD）构建的完整控制栈，三个层次：

```
NSP 逆运动学（零空间投影，主动远离奇异/限位）        ← openarm_pinocchio_nsp
  ↓ (times, q_path)
离线轨迹规划：直线 / SE(3)弧线 / 关节回放 / B样条    ← cartesian_planner / bspline_planner
  ↓ (执行)
250Hz 实时 web 面板：状态机 + 重力补偿 G(q) +        ← openarm_dashboard
   笛卡尔 6D 阻抗控制
```

实机功能：使能/零力矩/抱住/归零、示教-直线-弧线-关节回放-B样条执行、
重力补偿、定点阻抗控制全套（推-回弹 / 拖动模式 / 方向感知奇异防护 /
软挡块 / 内推屈肘）、自碰撞检测、全程调试日志。

---

## 1. 目录

| 路径 | 内容 |
|---|---|
| `/ros2_ws/openarm_nsp_ws/` | 主工作区（NSP + dashboard + 全部文档）|
| `/ros2_ws/openarm_pinocchio_ik/` | IK 工作区（FK/IK 验证、moveit_web 面板）|

---

## 2. 文档地图（按功能）

| 文档 | 内容 |
|---|---|
| **USAGE.md** | 总手册：环境/编译/CAN、CLI 工具、dashboard 全用法、示教-复现、弧线静止启停、离线验证、API 速查、故障排查；§13-15 web_panel/重力补偿/误差显示、§17 阻抗 |
| **IMPEDANCE_THEORY.md**（294 行）| 阻抗控制原理：控制律逐项、奇异防护（W 矩阵方向淡化 + 速率型屈肘）、饱和链、线程模型、实时性保障、压测体系 §8.1 |
| **IMPEDANCE_SAFETY.md**（295 行）| 阻抗安全手册：最保守参数、每个参数含义、启动流程、中止条件、四次实机事故完整复盘（§8-10）与调试日志分析方法（§7）|
| **COLLISION_AVOIDANCE.md**（146 行）| 自碰撞检测（163 对）+ 让位 IK + 三策略避障 + 14 维双臂 B 样条 |
| `openarm_pinocchio_nsp/README.md` | NSP-IK 工具箱：纯规划（不碰硬件）、plan_trajectory 用法、规划/执行分离 |
| **本文档 HANDOVER.md** | 总入口/交接 |

---

## 3. 功能实现

### 3.1 逆运动学（NSP-IK）
- 两阶段求解：阻尼最小二乘（DLS）收敛 + 零空间多目标爬升
  （可操控度最大化 + 关节限位居中），经 `P = I − J⁺J` 投影
- 规避了 KDL 在归零奇异位的失效问题；`ik_nsp`/`ik_multi` 支持热启动种子
- 实现：`openarm_pinocchio_nsp/kinematics.py`、`ik_nsp.py`；
  离线验证：`openarm_pinocchio_nsp/validation/`

### 3.2 正运动学
- `PinocchioModel.fk(q)` → (xyz, xyzw)
- `_pose_and_jac(q)` 一次调用同时返回位姿和雅可比
  （供阻抗 250Hz 循环使用，比分别调 fk + jacobian6 省一半计算）

### 3.3 轨迹规划
- `plan_cartesian`（直线/多路点，逐点 IK + 时间参数化）
- `fit_arc`（SE(3) 圆弧拟合：位置圆 + 姿态 SLERP）
- `pose_replay_traj`（示教点 → 关节空间弧长均匀重定时）
- `BSplineOptimizer`（SLSQP 优化 + 自碰撞约束）
- 统一输出 `(times, q_path)`；`ease_in_out_retime` 五次缓入缓出
- 规划与执行分离：规划器零 CAN 代码，执行器任意（ros2_control / dashboard / 自定义）

### 3.4 重力补偿
- `gravity.py`：pinocchio `computeGeneralizedGravity` on v1_simple.urdf，
  无电机偏置（与 MuJoCo、KDL 三方数值一致验证）
- 发送：`MITParam(0, 0, 0, 0, +G)` 纯力矩（与零力矩同一已验证路径）
- UI：checkbox 开关 + scale 滑条（0-1.5 可调，1.0 = 全补偿）

### 3.5 笛卡尔 6D 阻抗控制
- **控制律**（每 250Hz tick）：
  `τ = Jᵀ(K·W·Δx − D·W·ẋ) + 关节弹簧(姿态+回位) − 阻尼 + G(q)`，
  以纯力矩 MIT 帧发送
- **四档预设**：极软(100)/软(300)/中(800)/硬(1500) N/m；
  ζ 滑条 0.5-1.5（硬上限 1.5）；漏速滑条 0-2/s（0=回弹，>0=柔顺拖动）
- **方向感知奇异防护**：
  - W = U·diag(σ²/(σ²+0.05²))·Uᵀ 只淡化失控（径向）方向，
    其余 5 个方向满刚度——伸直位侧向依然可推
  - 速率型屈肘控制沿折叠方向 Vt[5]：向外拉遇渐进"软挡块"；
    沿臂轴向内推触发屈肘让位（含 τ 残差推力检测）
- **回位**：姿态弹簧随 Kx 缩放 + 未投影配置回位弹簧
  （解决零空间投影吞掉肘部回位力的问题，深位移回位 83-94%）
- **安全链**：入口 σ≥0.05 门禁 → σ<0.02 持续 0.5s 硬退出 →
  超速衰减计数器（防振荡混叠）→ 关节限位余量 0.10rad → 计算看门狗（3 连超时）
- 2 秒刚度渐入；预设切换时 Kx 滑条自动吸附

### 3.6 自碰撞检测
- 163 对链接对（双臂间 + 臂-基座），20mm 边距
- 实时监护（面板警告）+ 让位 IK（零空间推斥）+ 规划期约束

### 3.7 调试日志体系
- `log/panel_<时间戳>/` 自动落盘：
  - `panel.log`（事件日志）、`commands.jsonl`（全部 HTTP 命令）
  - `ctrl_<side>.csv`（250Hz × 49 列：q/dq/τ感/τ令/电机温度/Δx/F_est/σ/屈肘）
  - `can_stats.jsonl`（CAN 错误/丢帧计数）
  - 异常退出时自动转储前 5 秒事件 CSV + 关停时 ring_tail
- 写入在独立线程，控制循环只入队不落盘

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
  RIGHT=`can_slot1_ch0`（代码里两处 CAN_MAP 已同步）
- 本机跑着实机栈：**测试用 ROS_DOMAIN_ID=42 隔离，绝不要 kill**
- 回归测试：`scripts/test_impedance_sim.py`（10 项，改阻抗代码后必跑）

---

## 5. 已发现并解决的问题（含解法与出处）

### 5.1 首次开阻抗剧烈抖动
- **根因**：电机侧 MIT kd=[1,1,.8,.8,.4,...] 放大量化速度噪声到力矩
  （0.003 kg·m² 腕部）；默认中档刚度；无渐入
- **解决**：IMP_KD 实机置零（纯力矩路径）；默认极软档；ζ=1.2；
  2 秒平滑渐入；速度反馈 15Hz 低通
- **查看**：IMPEDANCE_SAFETY.md §0

### 5.2 阻抗计算超时退出（两种根因）
- **根因 A**：OpenBLAS 多线程池同步停顿（RK3588 八核负载下 p99=10ms）
  + Python GC 停顿；单次超时即退出过于敏感
- **解决**：threadpoolctl 限单线程 + gc.freeze()/disable()；
  看门狗改为连续 3 次超时；FK 换 `_pose_and_jac` 一次调用
- **根因 B**：环境变量在 numpy import 后设置无效（线程池已定型）
- **解决**：web_panel 启动时显式 `threadpool_limits(1)` 并在缺失时告警
- **查看**：IMPEDANCE_SAFETY.md §8、§9

### 5.3 推到极限大力回弹
- **根因**：91ms 看门狗窗口内臂加速到 11-20 rad/s，退出后 kp=70 的
  PD 抱住高速臂 → 硬刹车过冲（非"回到原始位置"，日志证伪）
- **解决**：退出时若 |q̇|>2 rad/s 加 0.5 秒纯力矩阻尼刹车窗
- **查看**：IMPEDANCE_SAFETY.md §9

### 5.4 15Hz 持续振荡
- **根因 A**：软逃逸窄带 [0.02,0.05] 边界猎振（位姿骑在阈值上开关切换）
- **解决**：混合带加宽 + 每 tick 斜率限制（权限随 σ 单调，无跳变沿）
- **根因 B**：连续 tick 超速计数被 15Hz 振荡混叠击败（从未连续 200ms 超速）
- **解决**：衰减计数器（超速 +1、正常 −1，累计 75 触发）
- **查看**：IMPEDANCE_SAFETY.md §10

### 5.5 回弹弱、回不到位（20-45mm 残余）
- **根因**：姿态弹簧（20 Nm/rad）压过弱任务弹簧（极软 Kx=100），
  静止平衡在离锚点 20-45mm 处
- **解决**：姿态弹簧随 Kx 缩放（kx/300 倍）+ 奇异区地板 ×(1+2·blend)；
  残余降至极软 7mm / 软 1mm
- **查看**：IMPEDANCE_THEORY.md §2.3 注记

### 5.6 腕部 ~1Hz 不收敛抖动
- **根因**：W 矩阵加权后腕部转动阻尼折算仅 ~0.2 Nm·s/rad
  （临界需 0.36），欠阻尼；指令力矩自身反转驱动腕部
- **解决**：阻尼力矩上限加下限 DAMP_CAP_FLOOR=0.20 Nm
  （制动能力 0.15→0.20 Nm ≥ 实测振荡所需 0.18 Nm，kd 隐含值距
  ZOH 失稳边缘仍有 4-7 倍余量）
- **查看**：IMPEDANCE_THEORY.md §2.2

### 5.7 推肘后肘部不回位（j4 卡死）
- **根因**（两层）：①姿态弹簧沿折叠方向的让位分量 + 屈肘阻尼反向
  抵抗回位运动；②零空间投影 Nᵀ 删掉 99% 的 j4 姿态回位力
  （1.35 Nm → 0.01 Nm），极软任务弹簧补不动 → 合计 0.4-0.7 Nm
  低于肘部带载静摩擦（实测 ≥2.18 Nm）
- **解决**：①让位仅在屈肘控制主动外拉时生效；②加未投影配置回位弹簧
  `τ_cfg = clip(K_CFG·(q_post−q), ±5 Nm)`，交互时自动让位
- **查看**：IMPEDANCE_THEORY.md §2.3

### 5.8 右臂 CAN 错误风暴 → 编码器重索引 → 肩部整圈旋转
- **根因**：右 CAN 物理层故障（每会话 tx_error 十万级 + 丢帧 +
  292 次 bus-off）；错误风暴使全部 7 电机单 tick"跳变"0.3-3.1 rad
  （多圈计数重置，速度反馈仍 ~0——物理不可能），重力前馈瞬间反号
  主动驱动肩部旋转撞机身
- **解决**：编码器跳变防护（单 tick 关节跳变 >0.1rad 且速度 <8rad/s
  → 立即失能 + 事件转储）；后按使能后 0.5s 稳定窗修复误触发
- **查看**：IMPEDANCE_SAFETY.md 事故记录；防护实现 arm_controller.py `_step`

### 5.9 实时性保障（通用）
- BLAS 单线程 + GC 关闭后实机 tick p99 = 5.5ms（47000 ticks 压测）
- **查看**：IMPEDANCE_THEORY.md §6.1

---

## 6. 关键教训（防回归）

| 教训 | 解法 | 查看处 |
|---|---|---|
| 腕部 ZOH 阻尼上限是承重墙 | 永不封弹簧项，只限阻尼 | IMPEDANCE_THEORY §2.2 |
| 电机侧 kd 实机必须为 0 | 纯力矩路径（kp=0/kd=0）| IMPEDANCE_SAFETY §0 |
| BLAS 多线程 + GC 停顿咬实时 | threadpool_limits(1) + gc 冻结关闭 | IMPEDANCE_THEORY §6.1 |
| 屈肘力矩必须速率型 | 位置/力形式均不稳定（70Hz 振颤/极限环）| IMPEDANCE_THEORY §4.2 |
| ζ 上限 1.5 | >1.5 方向阻尼确定性失稳 | impedance.py 常量注释 |
| 奇异防护只淡失控方向 | W 矩阵保 5 个好方向满刚度 | IMPEDANCE_THEORY §4.1 |
| 混合带宽与斜率 | 窄带导致边界猎振（15Hz 事故）| IMPEDANCE_SAFETY §10 |
| 振荡检测用衰减计数 | 连续计数被振荡混叠击败 | IMPEDANCE_SAFETY §10 |
| 速度噪声先低通再进阻尼 | 15Hz 仍有残余致 22Hz 力矩回路 | impedance.py DQ_LPF_HZ |
| 重力只走 _grav_tau 路径 | 阻抗模块含重力必双重计算 | gravity.py 模块注释 |
| v1_simple.urdf 无电机偏置 | 与 MuJoCo/KDL 三方一致 | USAGE §14 |

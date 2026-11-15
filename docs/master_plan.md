# VGGT + Token-Scale 3DGS 空间记忆主计划

## 当前状态与完成判据

2026-11-15：全新独立项目；旧仓库一次性基础设施盘点已关闭。没有读取旧实验报告或业务代码，没有导入旧历史、会话或方法结论，没有终止进程。基础资产只读路径登记在 `configs/runtime_paths.yaml`。只读要求由加载方式和项目内缓存路径落实，软链接不作为只读保护。

阶段顺序：1. 只读盘点/独立初始化 → 2. Builder-First AGENTS → 3. 本方案 → 4. Step 1 几何与三场景冒烟。阶段 3 的方案写入本文件，不等待额外行政确认即执行 Step 1。阶段状态以实际测试/训练结果为准，不把找到路径、成功导入、GPU 运算成功、GPU 渲染成功混为一谈。

初始提交 `5ae42a4` 已完成。初始有限盘点未找到权重；用户随后提供了只读共享盘上的完整官方 VGGT `model.safetensors` 和兼容的 miniColmap 环境，已核实文件头 1797 个张量、344 个 DINOv2 patch_embed 张量，并严格匹配加载实际使用的 aggregator 与 point_head。独立 DINOv2 文件未确认，使用完整模型内嵌的官方编码器参数。小型补充依赖仅写项目 `.deps`，不修改既有 Conda、venv 或系统。

Step 1 已实际执行：真实几何 11 项通过，其他模块/API/数据测试 9 项通过，完成 seed=17 的 1000 步三场景真实 VGGT 过拟合。Loss 前后 30 步均值 1.812 → 0.198。三个场景各 V1–V5 已生成并打开查看，原始权重/数据未复制，训练结束无本任务后台 GPU 作业。详见 `docs/step1_report.md`。当前证实可微训练链路与过拟合收敛；尚未证实稳定的 Pose 精化增益、O6 几何选择性或局部擦除因果作用，留出走廊查询仍失败。下一轮先改进这些具体问题，不直接启动 15k–20k 主训练，不做组件消融，不关闭方法家族。V6 保留到 Step 3。

## 痛点与整体通路

VGGT 全帧注意力随帧数 O(T²) 增长；滑窗丢失长期全局参考，重访会漂移并产生双层墙；按时间保留的 2D KV 冗余且无显式三维融合/遮挡机制。

将每个 14×14 patch token 对应的局部视锥抬升为粗高斯，形成 `G_m=(mu_m,Sigma_m,alpha_m,z_m)`。按三维空间融合去重，容量随实际表面积与复杂度增长；不设极小全局固定槽数，不像素级加密。建图完成后可读取的历史状态仅为空间高斯，不保存供查询读取的原图或帧级 2D KV。反传需要的张量仅服务梯度，不能通过旁路给查询头使用。

多视角建图 → Token Writer → 空间矩匹配记忆 → 初始位姿 T0 → O5 渲染 + O6 Cross-Attention → VGGT 接收交互层 → DPT + Pose 精化头 → 世界点图与 T1。冻结 DINOv2；Writer、双 Reader、接收层、DPT、Pose 可训练。骨干必须是实际官方 VGGT，轻量参考模块测试不能替代此接入。

## 坐标、尺度与输入隔离

世界系固定为首个建图相机，米制尺度由建图 RGB-D/监督确定。此 gauge 是预先定义的输入坐标变换，不能根据预测事后对齐。若以后改为纯 RGB 建图，必须显式学习/校准 metric gauge，不假定 VGGT 自带任意尺度等于米。

当前明确允许的建图设置：历史建图帧 RGB、K、metric depth、建图 c2w；所有建图 c2w 预先变换到首帧 gauge。查询输入只包含当前 RGB、K 和公共记忆，不接收查询深度/真值外参。监督对象单独传给 loss/evaluator。先验证这个设置，不能把建图真值当作预测建图结果。

OpenCV 相机轴向 x 向右、y 向下、z 向前；深度是 z 而非射线长度。VGGT extrinsics 为 w2c 3×4，gsplat viewmats 为 w2c 4×4，c2w 为其刚体逆，列向量表达 `X_c=R_wc X_w+t_wc`。gsplat quaternion 为 WXYZ；Sigma=R(q)diag(s²)R(q)^T。

VGGT/OpenCV 图像像素中心为整数；gsplat 光栅采样为索引+0.5。禁止隐式混用。原图 K_cv 转为边缘坐标 `K_edge=A(+0.5,+0.5) K_cv`。patch 网格边缘坐标 `K_edge,p=diag(1/14,1/14,1) K_edge`；若回到 OpenCV patch 整数坐标，再左乘 A(-0.5,-0.5)。patch 中心对应原图 `(j+0.5)*14-0.5`。所有 resize/crop/pad 都要把相同仿射乘进 K，不能只缩放 focal。非 14 整除图像先明确 resize/crop，不能随意取整。

## Writer 与空间融合

从历史 patch token 和该 patch 的物理深度建立中心。以局部视锥 `r_i=14*depth_i/(2*focal)` 初始化尺寸。头预测有界 Δlog s、归一化 WXYZ q、sigmoid alpha 和 z∈R^256；中心允许有界局部残差以训练 mu。明确 depth_i 与 O6 射线方向 d_i 的区别。

按自适应 voxel/马氏邻域和法向相容性形成融合组。离散成员选择可 stop-gradient；组内 mu/Sigma/alpha/z 的融合保持可微。令 w 为正置信权重，`mu=sum(w_i mu_i)/sum(w_i)`，`Sigma=sum(w_i(Sigma_i+(mu_i-mu)(mu_i-mu)^T))/sum(w_i)`。用 Cholesky/SPD 下界稳定；潜特征和 alpha 由权重及可学习门控更新。不得漏掉中心差二阶矩。正负法向视作同一无向平面。公共总数与每次活跃读取数分别记录，不通过删公共记忆控制显存。

## O5：可微光栅读取

在 H_p×W_p 渲染。对于按前后排序的高斯，`w_im=T_im*a_im`，`O_i=sum_m w_im=1-T_i`。将 per-Gaussian 属性 `[mu_world,z,depth_cam,depth_cam²]` 作为 channels 用相同权重光栅累积；深度指中心在当前相机的 z。

`Xhat=sum(w mu)/max(O,eps)`，`Zhat=sum(w z)/max(O,eps)`，`Dhat=sum(w d)/max(O,eps)`，`var_D=max(sum(w d²)/max(O,eps)-Dhat²,0)`。低覆盖位置用 mask 明确无效，不把零输出当真实世界原点。渲染中心属性的期望不等于每像素曲面射线交点，测试只在受控覆盖/混合条件下比较。

使用 `O*phi(Xhat,Dhat,var_D,Zhat)+(1-O)*e_unseen`，e_unseen 可学习。背景颜色为零且先去除背景项再归一化，防止 alpha 加权和被误认为期望值。接收层有可学习门控。O5 在初始位姿 T0 下提供遮挡优先的几何锚点；T1 下再次渲染仅作校准诊断/精化需要，不输入真值位姿。若显示 RGB，颜色只能来自明确记录的建图 RGB 聚合属性；否则标注为几何/潜特征渲染。

## O6：参数 PE 与几何跨注意力

从当前视锥及缓冲带选活跃高斯，必要时分块/预算控制单次显存，不删除公共记忆。中心使用 sin/cos 高频三维位置编码；严禁以尺寸或 Sigma 衰减中心高频相位。

对射线 `o+t*d, t>=0` 与椭球，A=Sigma^-1，v=mu-o，`t*=max(d^T A v/(d^T A d),0)`，`rho²=(t*d-v)^T A(t*d-v)`。d 采用单位方向，t* 是射线长度；与 O5 的 z 深度比较前必须乘相机坐标 d_z。将以下量送入 per-head bias/几何 PE：rho²、最佳接近 z、与 Dhat 的有符号前后差、视线与最薄主轴 n 的 |cos|、三轴 log s、alpha。法向只从几何通道进入，不以完整 Sigma 污染点基线普通特征。

K/V 来自 z 与中心 PE，Cross-Attention 注入 VGGT 交互层，残差门控零初始化。门控初期开启前内部零梯度合法；数步后必须看到 O5/O6、Writer 的参数变化及记忆 mu/Sigma/alpha/z 梯度。Sigma 特征的 eigensystem 在重根处需诊断；用不同基础三轴尺寸避免任意法向的不可微退化。

## 无查询真值位姿的两阶段自举

初始 Pose 头只接收当前浅层特征和粗记忆摘要（例如全局 mu/feature 加权统计），预测世界系 T0。T0 构造查询射线并启动 O5/O6；DPT 与精化相机头输出 P_world 和 T1。Pose 输出的旋转需使用连续稳定参数化。GT 查询 T 仅作为训练 loss/评测目标；训练、验证接口均禁止传入读取函数。单场景 ID、目标点图或查询深度也不得充当旁路输入。

## 损失与指标

有效深度 mask 下世界点 L1/EPE（m）；旋转测地误差（°）；平移误差（m）；T0/T1 的前后变化；跨视图与重访的墙面厚度、重影一致性。直接在固定世界系计算，不 Umeyama、ICP、事后刚体/尺度对齐、去均值或改相对深度。报告每场景与总体曲线。先用明确权重平衡 point/pose/稳定项，按真实梯度做正常科研调优，不设繁琐行政门禁。

## Step 1：现在执行

1. 首先 `/home/liwt/anaconda3/bin/python -B -m pytest tests/test_geometry_conventions.py`。覆盖 WXYZ、非平凡 c2w/w2c、轴向/像素中心/z 深度、14 倍 patch K 变换、世界反投影/重投影、受控 gsplat 深度/坐标/混合/透明度。缺 CUDA/gsplat 的测试明确 skip，不算渲染通过；必要时启用严格渲染开关使缺项直接失败。
2. 选三个真实 Sun3D 室内场景：教室 harvard_c5、办公室 brown_bm_1、另一教室 harvard_c6，须检查目录、数据格式、覆盖与外参。固定 seed=17；明确建图/查询帧，并固定场景不交叉的 train/val 主线划分。开发集可视化正常使用，不引入 fresh/used。
3. 800–1500 更新步（默认 1000），轮换三场景，冻结 DINOv2，训练其余指定模块。当前冒烟为 112×140 图像、8×10 patch 网格；历史 Writer 直接使用官方冻结 DINOv2 patch tokens，查询使用完整官方 24 组交替交互层与原生 DPT，在第 4 号 global block 后注入 O5/O6。训练这个接收 global block、DPT、Writer/融合、双 Reader 与两阶段 Pose；其余预训练交互块冻结参数但保留回传到接收层的计算图。图像低分辨率只用于过拟合冒烟，不能当作主线最终质量。冒烟属于三场景训练集过拟合；同场景固定 held-out query 仅作诊断，不谎称跨场景 val。每场景应有足够更新次数。

   查询原生单帧 DPT 输出在模型内部视为相机系点，并以预测 T1 转为世界点后计算直接世界 loss；此处没有真值位姿或事后拟合。`configs/splits.yaml` 固定场景划分和建图/查询索引；未来主线 val 为 harvard_c11，和三个 train 场景分离。
4. 记录实际 Loss/EPE/pose T0/T1，alpha/scale/特征的范围，mu/Sigma/alpha/z 的 autograd，门控开启后各分支 grad norm、参数更新及 finite 检查。持续断梯度/NaN 必须修复。明显下降的判据预设为前后窗口可见下降并检查曲线，不以一次波动判停。
5. 实际生成并打开 V1–V5 后写观察。缺大权重/依赖时不以合成/缩小的替身实验宣布 Step 1 通过；保存实际可完成的几何/模块检查，并列出最小解阻项。没有 V6 不阻塞此阶段。

## Step 2：主线可学习性与因果

完成 Step 1 后才开始 15k–20k 更新，固定 seed，多视图建记忆 → 未知位姿新视角世界重建。主对比 No-Memory/Local（展示局部参考、对齐和重影），Full（O5+O6），固定同权重零重训的空记忆/错配记忆/3D bbox 擦除。完整系统收敛后才补 Render-Only、PE-Only、Isotropic-Point；点基线特征层不得泄露完整 Sigma。以训练收敛、验证指标和可视化共同判断；若负结果直接诊断，不永久关闭家族。定型总结时补多 seed 稳健性。

## Step 3：流式 Read-then-Write

同一融合模块切换为先读历史记忆、输出当前世界点/位姿、再用当前预测写入。禁止给读者保留历史原图/KV；不使用 GT 查询位姿写回伪装流式自定位。长间隔室内探索—徘徊—重访，报告全局一致性、累积位姿漂移、墙面厚度、已探索区域记忆增长、新区域合理扩容。更新 V1–V5 并生成 V6。

## 可视化交付（results/<run_name>/vis/）

- **V1**：冷灰历史、亮橙查询预测，固定世界系直接叠合，无任何后对齐；3D+鸟瞰+选定墙面厚度切片，展示实际漂移与重影。
- **V2**：1.5σ 椭球三维外壳，最薄主轴 |n| 映射 RGB，alpha 控制透明度；远景大面元与近景放大框。不能仅画中心点替代椭球。
- **V3**：同行四联图：输入真实 RGB（白墙/半遮挡）；T0 几何或明确颜色来源的记忆渲染+红绿边缘错位+1-O 盲区；T1 边缘对齐；最终世界几何与同权重无记忆对照。标清渲染 channel，潜特征不能称照片。
- **V4**：查询图选半遮挡/边缘/跨尺度 patch，世界系射线与 attention Top-30 高斯椭球热力图；明确层/头/平均方式，记录活跃集数，不用随机权重。
- **V5**：同图同权重，3D bbox 精准擦除；擦除位置/前后输出/局部变化/区域外和世界系稳定性均展示。若全局退化照实分析，不预写仅局部退化结论。
- **V6**（仅 Step 3）：公共总基元数、每次活跃读取数、帧级 KV 随时间增长基线；标注探索/徘徊/重访。活跃预算平台不能称公共记忆自然饱和。

所有图片必须实际生成并打开，不交空路径，不将预期现象写成已观察事实。报告使用“改了什么/原因—Loss 与指标—实际可视化—下一步”，不做行政 JSON。

## 几何参考（官方基础设施资料）

- [VGGT 官方几何实现](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py)：OpenCV w2c、整数网格、camera-z 深度。
- [gsplat 数据约定](https://docs.gsplat.studio/main/conventions/data_conventions.html)：WXYZ 和 w2c。
- [gsplat 光栅接口](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html)：任意属性通道、累积 alpha、深度类型和相机梯度限制；内参不作为可学习参数。
- [SUN3D 官方工具箱](https://github.com/PrincetonVision/SUN3Dsfm)：`depthRead.m` 规定 uint16 右旋 3 位后除 1000；`outputCameraExtrinsics.m` 导出 c2w；`depth2XYZcamera.m` 使用 1 起始 MATLAB 坐标，本项目加载时主点减 1 再统一到 OpenCV。真实三场景数据已做跨视图 z 残差检查。

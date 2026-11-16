# Step2 v4：完整18k Full与单帧No-Memory主对比

当前执行用户授权的连续18k主实验；保留full/v2/v3所有旧产物与原`configs/step2_splits.yaml`。Full目录`results/step2_v4_full_seed17`，基线目录`results/step2_nomemory_seed17`。二者均从官方VGGT完整权重独立初始化，168×224、seed17、18000更新，无预测后对齐。

## 实现

- 逐patch乘法深度锚定：依据本patch的O5深度/opacity/相对方差/可学习置信度形成局部比例，置信度尾层bias=+1。每个patch独立双线性提升比例，不混合相邻patch的因子；因此右侧擦除不改变左侧锚定，opacity=0严格保持原生点，patch内原生深度边缘保留。没有全图gamma。
- T1残差使用T0_read处局部校准点；最终在T1.detach处重新渲染并从raw点独立锚定，不重复修正。世界变换也用T1.detach，最终点损失不代偿相机位姿。
- 对称上下文：建图帧no_grad运行冻结的全局块0–3及frame块0–4，停止于查询注入前相同位置；Writer与查询匹配投影均读[shallow,pre_inject]。公共记忆只有mu/Sigma/alpha/z，build_memory不传出图像、token或KV。
- 训练读入微扰覆盖[0,3°]/[0,6cm]；SpatialRefinement空间上下文门控sigmoid bias=−2，小初值，零残差/无覆盖严格恒等。旧架构路径保留以复现对照。
- No-Memory仅有原生VGGT/DPT和单帧位姿头，forward唯一参数QueryInput(rgb,K)。不构建高斯、不读取建图RGB-D/位姿、不注入O5/O6、不做地图锚定；DINO冻结，同样训练原生块4/11和DPT。训练无随机世界规范，G=I，目标仍由相同episode首建图相机定义；地图数据仅用于数据集构造目标和V1参考，不作为网络输入。基线重放Full训练episode ID序列，绝不加载Full参数或预测。

## 先行验证

Full启动前`REQUIRE_GSPLAT=1`全部74项通过，包括局部锚定严格擦除隔离/空记忆恒等、无历史上下文状态、门控及微扰范围、v3几何和T1梯度隔离。12步独立官方初始化profile损失/梯度有限，峰值4.62GiB、中位约0.68秒/步；正式Full重新初始化。No-Memory增加接口与深度/位姿梯度隔离单测；其12步官方全新GPU profile前全部75项REQUIRE_GSPLAT=1严格测试通过，损失/梯度有限，峰值4.53GiB、约0.33秒/步。正式基线仍待Full完成后再从官方权重独立初始化，不复用profile权重。

## 执行与评测

Full连续18000步，0/1500/3000/6000/12000/18000保存完整126查询评测和里程碑检查点；500步快速子集评估与收敛曲线。3000步不因个别长尾/局部负例停机，只检查非有限、持续断梯度或跨场景总EPE>0.35m系统崩溃。所有难帧保留。

Full完成后紧接独立官方全新No-Memory 18000步，同划分/尺寸/更新数；同一126评测episode和建图帧顺序，世界参考系首建图相机、G=I，无GT查询输入或后对齐。最终输出均值/中位数/p90/最差难例、robotics远距分桶，Full四干预与同权重单建图帧对照；实际生成并打开Full/No-Memory V1和Full V1–V5/全部曲线。

## 实际结果（运行中，未提前填写）

Full正在训练，No-Memory尚未启动。最终结论以18000步实测为准。

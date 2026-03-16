# UNAGI IPF 靶点发现：逻辑、代码与工作流完全指南

## 概述

UNAGI（**U**nsupervised i**N**-silico cellular dyn**A**mics and dru**G** d**I**scovery）是一个深度生成模型框架，专为解析复杂疾病时序单细胞数据中的细胞动态而设计，并通过计算机模拟药物干扰来识别治疗靶点。本文以特发性肺纤维化（IPF）为具体案例，系统梳理 UNAGI 发现 IPF 靶点的核心逻辑与代码工作流。

---

## 一、整体架构

UNAGI 采用 **VAE-GAN + 图神经网络（GCN）** 架构，将疾病时序 snRNA-seq 数据转换为低维细胞嵌入表示，再通过动态图和基因调控网络（GRN）推断，最终在潜变量空间执行计算机模拟扰动，输出候选靶点排名。

```
时序 snRNA-seq 数据 (h5ad)
        ↓
   [数据预处理]        ← 按疾病分期拆分、构建 KNN 细胞图
        ↓
   [VAE-GAN 训练]     ← 迭代学习疾病特异性细胞嵌入
        ↓
   [动态图构建]        ← 跨期 KL 散度连边 + iDREM 基因调控网络
        ↓
   [标志物发现]        ← 动态标志物 + 层级静态标志物
        ↓
   [通路扰动分析]      ← GSEA 通路 → ΔD 评分
        ↓
   [药物扰动分析]      ← CMAP 数据库 → ΔD 评分 → IPF 靶点排名
```

入口类是 `UNAGI.UNAGI_tool.UNAGI`，整个流程通过以下两个核心调用驱动：

```python
# UNAGI/UNAGI_tool.py
unagi = UNAGI()
unagi.run_UNAGI(idrem_dir)          # 训练阶段
unagi.analyse_UNAGI(data_path, ...)  # 分析阶段
```

---

## 二、阶段一：数据预处理与细胞图构建

### 2.1 数据加载与分期拆分

`setup_data` 方法接受包含所有时期的 h5ad 文件，将其按 `stage_key` 字段拆分为独立的分期文件：

```python
# UNAGI/UNAGI_tool.py
def setup_data(self, data_path, stage_key, total_stage,
               gcn_connectivities=False, neighbors=25, threads=20):
    """
    data_path:  h5ad 文件路径（可以是整体数据集或已分期的文件夹）
    stage_key:  标注疾病分期的 obs 列名
    total_stage: 时期总数（IPF 典型值为 4：对照组 + 3 个纤维化进展期）
    neighbors:  构建 KNN 细胞图时的邻居数，默认 25
    """
    if total_stage < 2:
        raise ValueError('The total number of stages should be larger than 1')
    ...
    if not gcn_connectivities:
        print('Cell graphs not found, calculating cell graphs for individual stages!')
        self.calculate_neighbor_graph(neighbors, threads)
```

对于 IPF 数据集，典型配置为 4 个时期（对照组 + 进展期 1/2/3），`neighbors=25` 用于构建细胞图。

### 2.2 细胞 KNN 邻接图

细胞图（GCN 图）由 `get_gcn_exp` 构建，结果存储于每个分期 h5ad 的 `adata.obsp['gcn_connectivities']`，供后续图卷积网络（GCN）编码器使用。细胞图的作用是捕捉同一分期内细胞邻域的上下文信息，使模型能感知细胞微环境。

---

## 三、阶段二：VAE-GAN 迭代训练

### 3.1 模型架构

UNAGI 使用 **图变分自编码器（VAE）+ 判别器（GAN）** 的联合架构。

**图编码器（GCN + VAE 编码器）**：

```python
# UNAGI/model/models.py
class Graph_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_dim, latent_dim):
        super(Graph_encoder, self).__init__()
        self.fc_graph = GCNLayer(input_dim, graph_dim)  # 图卷积层
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)   # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)   # 对数方差

    def forward(self, x, adj, idx=None):
        # 第一步：GCN 聚合邻居信息
        h0 = F.softplus(self.BN(self.fc_graph(x, adj)))
        # 第二步：非线性映射
        h1 = F.softplus(self.BN1(self.fc1(h0)))
        # 输出：均值和对数方差（VAE 潜变量参数）
        return self.fc21(h1), self.fc22(h1)
```

**GCN 层**通过 `torch.sparse.mm(adj, x)` 将细胞图结构融入基因表达特征，让嵌入同时包含细胞自身表达和邻域信息：

```python
class GCNLayer(nn.Module):
    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return torch.sparse.mm(adj, x)  # 图传播
```

**判别器**用于区分真实与重建的细胞表达，增强生成表示的真实性：

```python
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))
```

**分布选择**：UNAGI 支持零膨胀分布族，以更准确建模单细胞 RNA-seq 数据的过度零值问题，可选 `ziln`（零膨胀对数正态，推荐用于 IPF）、`zinb`、`zig` 等。

### 3.2 训练配置

```python
# UNAGI/UNAGI_tool.py
unagi.setup_training(
    task='IPF_fibroblast',
    dist='ziln',           # 零膨胀对数正态分布
    device='cuda:0',
    GPU=True,
    latent_dim=64,         # 潜变量空间维度
    hidden_dim=256,
    graph_dim=1024,        # GCN 图层维度
    lr=1e-4,
    lr_dis=5e-4,           # 判别器学习率（通常高于 VAE）
    beta=1,
    epoch_initial=20,      # 初始迭代 epoch 数
    epoch_iter=10,         # 后续迭代 epoch 数
    max_iter=10,           # 最大迭代轮数
    BATCHSIZE=512
)
```

模型构建逻辑：

```python
# UNAGI/UNAGI_tool.py（setup_training 内部）
if GCN:
    self.model = VAE(self.input_dim, self.hidden_dim, self.graph_dim,
                     self.latent_dim, beta=self.beta, distribution=self.dist)
else:
    self.model = Plain_VAE(...)  # 无图版本

if self.adversarial:
    self.dis_model = Discriminator(self.input_dim)
```

### 3.3 迭代训练流程

迭代训练是 UNAGI 的关键创新，每轮迭代更新基因权重（gene weights）反馈给下轮训练，逐步让模型关注疾病关键基因：

```python
# UNAGI/UNAGI_tool.py
def run_UNAGI(self, idrem_dir, CPO=True, ...):
    for iteration in range(self.max_iter):
        unagi_runner = UNAGI_runner(
            self.data_folder, self.ns, iteration,
            self.unagi_trainer, idrem_dir, ...
        )
        unagi_runner.run(CPO)  # 执行单轮迭代
```

`UNAGI_runner.run()` 方法封装了每轮迭代的完整步骤：

```python
# UNAGI/train/runner.py
def run(self, CPO):
    self.load_stage_data()                  # 1. 加载各期数据
    self.trainer.train(self.all_in_one, ...) # 2. 训练 VAE-GAN
    if CPO:
        self.run_CPO()                       # 3. 自动优化聚类参数
    self.update_cell_attributes(CPO)         # 4. 更新细胞嵌入与属性
    self.build_temporal_dynamics_graph()     # 5. 构建时序动态图
    self.run_IDREM()                         # 6. 运行 iDREM 推断调控网络
    self.update_gene_weights_table()         # 7. 更新基因权重（用于下轮训练）
    self.build_iteration_dataset()           # 8. 构建下一轮数据集
```

---

## 四、阶段三：动态图构建与基因调控网络推断

### 4.1 CPO 自动聚类参数优化

UNAGI 使用 CPO（Clustering Parameter Optimization）为每个时期自动确定最优邻居数和 Leiden 分辨率：

```python
# UNAGI/train/runner.py
def run_CPO(self):
    self.neighbor_parameters, anchor_index = get_neighbors(
        self.adata_stages, num_cells,
        anchor_neighbors=15, max_neighbors=35, min_neighbors=10
    )
    self.resolutions, _ = auto_resolution(
        self.adata_stages, anchor_index, self.neighbor_parameters,
        resolution_min=0.8, resolution_max=1.5
    )
```

聚类后的潜变量表示用于细胞 UMAP 可视化：

```python
# UNAGI/train/runner.py（annotate_stage_data 内）
z_locs, z_scales, cell_embeddings = self.trainer.get_latent_representation(adata, ...)
adata.obsm['z'] = cell_embeddings
sc.tl.leiden(adata, resolution=self.resolutions[stage])
sc.tl.umap(adata, min_dist=0.05, init_pos='paga')
```

### 4.2 跨期动态图边的构建

UNAGI 通过对各期细胞簇计算 **KL 散度** 来衡量它们之间的距离，并使用 **差异基因重叠相似度** 进行补充，只连接距离最近且 p 值显著的簇对：

```python
# UNAGI/dynamic_graphs/buildGraph.py
def nodesDistance(rep1, rep2, topgene1, topgene2):
    """
    计算两个时期间所有簇对的综合距离
    rep1/rep2: 各簇的高斯分布参数 (均值、方差)
    topgene1/topgene2: 各簇的 Top 差异基因
    """
    distance = [[] for _ in range(len(rep2))]
    for i in range(len(rep2)):
        for j in range(len(rep1)):
            gaussiankl_1 = calculateKL(rep2[i], rep1[j])
            gaussiankl_2 = calculateKL(rep1[j], rep2[i])
            gaussiankl = (gaussiankl_1 + gaussiankl_2) / 2    # 对称 KL
            similarityDE = getSimilarity(topgene2, topgene1, i, j)  # 基因重叠度
            distance[i].append([gaussiankl, similarityDE])
    ...

def connectNodes(distances, cutoff=0.05):
    """
    连接最近邻且 p 值 < cutoff 的簇对
    """
    edges = []
    for i in range(len(distances)):
        leftend = np.argmin(distances[i])  # 最近邻簇
        pval = norm.cdf(
            distances[i][leftend],
            loc=np.mean(distances),
            scale=np.std(distances)
        )
        if pval < cutoff:   # 显著性过滤
            edges.append([leftend, i])
    return edges
```

全局边构建逻辑：

```python
# UNAGI/dynamic_graphs/buildGraph.py
def getandUpadateEdges(total_stage, midpath, iteration, cutoff=0.05):
    edges = []
    for i in range(total_stage - 1):
        edges.append(buildEdges(i, i+1, midpath, iteration, cutoff=cutoff))
    edges = updateEdges(edges, midpath, iteration)
    return edges
```

边结果存储于 `adata.uns['edges']`，格式为 `{stage_idx: [[prev_cluster, next_cluster], ...]}`。

### 4.3 iDREM 基因调控网络推断

UNAGI 调用 iDREM（改进的 DREM）根据细胞轨迹推断转录因子（TF）调控网络，识别驱动每条轨迹的关键调控因子：

```python
# UNAGI/train/runner.py
def run_IDREM(self):
    averageValues = np.load(...)
    paths = getClusterPaths(self.edges, self.total_stage)   # 提取完整轨迹路径
    idrem = getClusterIdrem(paths, averageValues, self.total_stage)  # 整合平均表达
    runIdrem(paths, self.data_path, idrem, self.genenames,
             self.iteration, self.idrem_dir, species=self.species)
```

iDREM 输出每条轨迹对应的 DREM.json 文件，后续动态标志物发现从中提取每个基因沿轨迹的平均表达值及变化趋势。

### 4.4 基因权重更新

iDREM 结果用于识别每条轨迹的关键转录因子及其靶基因，并将 TF-靶基因关系与表达折叠变化相匹配，更新模型的基因权重表，供下一轮迭代使用：

```python
# UNAGI/train/runner.py
def update_gene_weights_table(self, topN=100):
    TFs = getTFs(idrem_results_path, total_stage=self.total_stage)
    scope = getTargetGenes(idrem_results_path, topN)
    if self.species == 'Human':
        p = matchTFandTGWithFoldChange(
            TFs, scope, self.averageValues,
            get_data_file_path('human_encode.txt'),  # ENCODE TF 结合数据库
            self.genenames, self.total_stage
        )
    updateLoss = updataGeneTablesWithDecay(
        self.data_path, str(self.iteration), p, self.total_stage
    )
```

---

## 五、阶段四：疾病动态标志物发现

### 5.1 动态标志物（Dynamic Markers）

动态标志物是在细胞轨迹上**单调**变化（持续上升或持续下降）的基因，代表疾病进展的驱动因素。

`runGetProgressionMarker_one_dist` 是核心入口：

```python
# UNAGI/marker_discovery/dynamic_markers.py
def runGetProgressionMarker_one_dist(directory, background, size, cutoff=0.05, topN=None):
    """
    directory: iDREM 结果目录
    background: 随机背景基因表达变化分布
    cutoff:  显著性阈值（默认 0.05）
    """
    one_dist = []
    for each in background.keys():
        one_dist.append(np.array(background[each]))
    background = np.array(one_dist).reshape(-1, size)
    out = getTopMarkersFromIDREM(directory, background, cutoff=cutoff, one_dist=True)
    return out
```

核心打分逻辑 `getTopMarkers` 从 iDREM 的 DREM.json 中读取各期表达值，筛选单调变化基因并与随机背景比较：

```python
# UNAGI/marker_discovery/dynamic_markers.py（getTopMarkers 内）
# 1. 读取各期表达值并计算单调性
tendency = temp[1:, 1].astype(float)
for i in range(2, stages):
    tendency *= temp[1:, i].astype(float)  # 乘积为正 → 单调
tendency[tendency < 0] = 0  # 非单调基因清零

# 2. 计算 log2FC（末期 - 首期）
change = temp[1:, stages].astype(float) - temp[1:, 1].astype(float)
change[index] = 0  # 非单调基因不计入

# 3. 与随机背景比较（p 值）
if one_dist:
    increasing_pval = scoreAgainstBackground(
        temp_background, temp_change[each],
        mean=mean, std=std  # 使用合并背景分布
    )
```

p 值计算基于**正态分布 CDF**（`scipy.stats.norm.cdf`），背景分布由随机采样基因的表达变化构成：

```python
def scoreAgainstBackground(background, input, all=False, mean=None, std=None):
    if mean is None:
        cdf = norm.cdf(input, loc=background.mean(axis=0),
                       scale=background.std(axis=0))
    else:
        cdf = norm.cdf(input, loc=mean, scale=std)
    return cdf
```

动态标志物最终存储于 `adata.uns['progressionMarkers']`，包含每条轨迹的上调/下调基因列表及对应 q 值。

### 5.2 层级静态标志物（Hierarchical Markers）

层级静态标志物是各时期特定细胞类型（簇）的特征基因，由 `get_dataset_hcmarkers` 计算：

```python
# UNAGI/UNAGI_analyst.py
hcmarkers = get_dataset_hcmarkers(
    self.adata,
    stage_key='stage',
    cluster_key='leiden',
    use_rep='umaps'
)
```

---

## 六、阶段五：通路扰动分析

通路分析用于识别哪些信号通路在反转疾病进展方面最为关键。

### 6.1 通路基因重叠计算

```python
# UNAGI/utils/analysis_helper.py
def calculateDataPathwayOverlapGene(adata, customized_pathway=None):
    if customized_pathway is None:
        data_path = get_data_file_path('gesa_pathways.npy')  # 内置 GSEA 数据库
    else:
        data_path = customized_pathway

    pathways = dict(np.load(data_path, allow_pickle=True).tolist())
    genenames = adata.var.index.tolist()
    out = {}
    for each in list(pathways.keys()):
        temp = [gene for gene in pathways[each] if gene in genenames]
        if len(temp) > 0:
            out[each] = temp
    # 去重：完全相同基因集的通路合并
    tmp = {}
    for key, value in out.items():
        value = '!'.join(value)
        if value in tmp:
            tmp[value].append(key)
        else:
            tmp[value] = [key]
    adata.uns['data_pathway_overlap_genes'] = {
        ','.join(keys): value.split('!') for value, keys in tmp.items()
    }
    return adata
```

### 6.2 通路基因权重排名

根据模型学到的 gene weights（存于 `adata.layers['geneWeight']`），对每条通路的基因重要性进行排名，优先测试与疾病最相关的通路：

```python
# UNAGI/utils/analysis_helper.py
def calculateTopPathwayGeneRanking(adata):
    pathway_gene = adata.uns['data_pathway_overlap_genes']
    avg_ranking = {}
    for i in range(1, 4):  # 遍历疾病进展期
        for clusterid in set(stageadata.obs['leiden']):
            cluster_gene_weight_table = clusteradata.layers['geneWeight']
            avg_cluster_gene_weight_table = np.mean(
                cluster_gene_weight_table, axis=0
            ).reshape(-1)
            avg_geneWeightTable_ranking = scipy.stats.rankdata(
                avg_cluster_gene_weight_table
            )
            for pathway in list(pathway_gene.keys()):
                # 通路内基因权重排名之和，除以基因数 → 通路重要性分数
                avg_ranking[i][clusterid][pathway] = sum(
                    avg_geneWeightTable_ranking[gene_idx]
                    for gene_idx in pathway_gene_indices
                ) / len(pathway_gene[pathway])
    adata.uns['pathway_ranking'] = new_av_ranking
    return adata
```

### 6.3 通路扰动执行

对每条通路，在基因表达空间执行上调/下调扰动（log2fc / 1/log2fc），将扰动后的表达编码到潜变量空间，计算 ΔD：

```python
# UNAGI/perturbations/perturbation.py（run 方法，mode='pathway' 分支）
elif mode == 'pathway':
    pathway_gene = self.adata.uns['data_pathway_overlap_genes']
    for each_direction in [log2fc, 1/log2fc]:  # 测试两个方向
        for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes)):
            gene_input_dict = dict.fromkeys(genes, each_direction)

            perturb_cells = self.adata.obs.index
            self.pb.perturb_input_data(perturb_cells, gene_input_dict, mode='direct')

            for lastCluster in self.tracks.keys():
                track_indices = track_cache[track_name]
                deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                self.adata.uns['pathway_perturbation_deltaD'][str(each_direction)][track_name][pathway_name] = deltaD
```

---

## 七、阶段六：药物扰动与 IPF 靶点发现（核心）

这是 UNAGI 发现 IPF 靶点的核心阶段。

### 7.1 药物-基因数据库准备

CMAP（Connectivity Map）数据库记录了每种药物调控的靶基因及方向（+/- 表示上调/下调）：

```python
# cmap_drug_target.npy 的格式示例
{
    'MG-132': ['CDKN1A:+', 'TP53:+', 'CASP9:-', ...],
    'Doxorubicin': ['TP53:+', 'MDM2:-', ...],
    ...
}
```

`process_customized_drug_database` 将药物数据库与数据集基因取交集：

```python
# UNAGI/UNAGI_analyst.py
def perturbation_analyse_customized_drug(self, customized_drug, ...):
    self.adata = process_customized_drug_database(self.adata, customized_drug=customized_drug)
    a = perturbation(self.adata, model_name, idrem_dir)
    a.run('drug', bound, ...)
    a.run('random_background', bound, ...)
    a.analysis('drug', bound, ...)
```

### 7.2 药物扰动执行

对每种药物，UNAGI 按靶基因方向缩放基因表达，再通过训练好的 VAE 编码器编码到潜变量空间：

```python
# UNAGI/perturbations/perturbation.py（run 方法，mode='drug' 分支）
if mode == 'drug':
    drug_gene = self.adata.uns['data_drug_overlap_genes']
    for each_direction in [log2fc, 1/log2fc]:  # 正向与反向扰动
        for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes)):
            gene_input_dict = {}
            plus  = each_direction
            minus = 1 / each_direction

            for item in genes:            # 例如 "NAT2:-"
                name, _, sign = item.partition(':')
                gene_input_dict[name] = plus if sign == '+' else minus

            perturb_cells = self.adata.obs.index
            self.pb.perturb_input_data(perturb_cells, gene_input_dict, mode='direct')

            for lastCluster in self.tracks.keys():
                track_indices = track_cache[track_name]
                deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                self.adata.uns['drug_perturbation_deltaD'][str(each_direction)][track_name][drug_name] = deltaD
```

### 7.3 基因表达缩放（直接扰动）

实际的基因表达修改通过 Numba 加速的 CSR 矩阵行缩放实现：

```python
# UNAGI/perturbations/perturbation.py
@nb.njit(cache=True)
def scale_csr_rows(data, indices, indptr, rows, factors):
    """
    对 CSR 稀疏矩阵的指定行按 factors 进行原地缩放
    factors[col] = log2fc 表示上调，1/log2fc 表示下调
    """
    for r in rows:
        row_start = indptr[r]
        row_end   = indptr[r + 1]
        for p in range(row_start, row_end):
            col = indices[p]
            f   = factors[col]
            if f != 1.0:
                data[p] *= f

def change_direct_targets(self, adata, selected_idx, gene_input_dict):
    X = adata.X          # 必须是 CSR 格式
    factors = np.ones(adata.n_vars, dtype=data.dtype)
    cols = adata.var_names.get_indexer(gene_input_dict.keys())
    factors[cols] = np.fromiter(gene_input_dict.values(), dtype=data.dtype, ...)
    data *= factors[indices]  # 向量化缩放整个矩阵
    return adata
```

### 7.4 ΔD 计算：潜变量空间中的距离变化

ΔD（Delta Distance）是 UNAGI 评估药物效果的核心指标，衡量扰动后细胞簇在潜变量空间中的移动方向：

```python
# UNAGI/perturbations/perturbation.py
def calculate_deltaD_for_tracks(self, list_of_indices):
    """
    list_of_indices: 每条轨迹各期的细胞 ID 列表
    返回: (n_tracks × n_tracks) 的 ΔD 矩阵
    """
    P_all = adata.obsm['perturbed_embedding']   # 扰动后嵌入
    O_all = adata.obsm['embdedding']             # 原始嵌入

    # 计算每条轨迹（每期）的平均嵌入
    for k, idx in enumerate(idx_arrays):
        P_means[k] = P_all[idx].mean(axis=0)
        O_means[k] = O_all[idx].mean(axis=0)

    # 计算两两距离矩阵
    dist_PO = distance.cdist(P_means, O_means, metric='euclidean')
    dist_OO = distance.cdist(O_means, O_means, metric='euclidean')

    # ΔD[i,j] = 扰动后 i 期到 j 期的距离 - 原始 i 期到 j 期的距离
    deltaD = dist_PO - dist_OO
    np.fill_diagonal(deltaD, 0.0)
    return deltaD
```

> **物理意义**：若 ΔD[疾病期, 对照期] < 0，说明扰动使疾病期细胞更接近健康对照期，即该药物/通路可能具有治疗潜力。

### 7.5 扰动得分计算

ΔD 矩阵通过 Sigmoid 函数转换为扰动得分，编码"向健康状态移动"的方向信息：

```python
# UNAGI/perturbations/analysis_perturbation.py
def calculateScore(self, delta, flag, weight=1):
    """
    delta: 某期与其他各期的 ΔD 向量
    flag:  当前期的索引
    """
    out = 0
    for i, each in enumerate(delta):
        if i != flag:
            # Sigmoid 变换：sign(i-flag) 编码"应向哪个方向移动"
            out += (1 - 1 / (1 + np.exp(weight * each * np.sign(i - flag))) - 0.5) / 0.5
    return out / (len(delta) - 1), ...
```

最终综合得分（考虑轨迹包含的细胞比例权重）：

```python
# UNAGI/perturbations/analysis_perturbation.py（load 方法内）
temp_track_score.append(
    np.array(track_percentage[each_track]) *
    np.abs(
        self.calculateScore(deltaD_direction1[i], i)[0] -
        self.calculateScore(deltaD_direction2[i], i)[0]
    ) / 2
)
out[drug]['overall'] = np.sum(np.array(unit_score))
```

### 7.6 随机背景构建与统计显著性

为消除偶然性，UNAGI 对随机基因集执行相同扰动，构建零假设分布：

```python
# UNAGI/perturbations/perturbation.py（mode='random_background' 分支）
elif mode == 'random_drug_background':
    for perturbation_item_idx in tqdm(range(random_times)):
        genes = get_random_genes(self.adata, random_genes, perturbation_item_idx)
        gene_input_dict = dict.fromkeys(genes, each_direction)
        self.pb.perturb_input_data(perturb_cells, gene_input_dict, mode='direct')
        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
        # 存储随机背景 ΔD ...
```

最终通过正态 CDF 计算 p 值：

```python
# UNAGI/perturbations/analysis_perturbation.py
def fitlerOutNarrowPathway(self, scores, sanity_scores, ...):
    sanity_scores = np.array(sanity_scores)
    for i, each in enumerate(scores):
        cdf = norm.cdf(each, sanity_scores.mean(), sanity_scores.std())
        if (1.000 - cdf) < 0.05:   # 单侧检验，p < 0.05
            final_top_compounds.append(top_compounds[i])
```

---

## 八、完整调用示例

### 8.1 训练阶段

```python
from UNAGI import UNAGI

unagi = UNAGI()

# 步骤 1：准备数据
unagi.setup_data(
    data_path='ipf_dataset.h5ad',
    stage_key='stage',         # obs 列：0=对照, 1/2/3=纤维化进展期
    total_stage=4,
    neighbors=25,
    threads=20
)

# 步骤 2：配置模型
unagi.setup_training(
    task='IPF_target_discovery',
    dist='ziln',
    device='cuda:0',
    GPU=True,
    latent_dim=64,
    hidden_dim=256,
    graph_dim=1024,
    epoch_initial=20,
    epoch_iter=10,
    max_iter=10
)

# 步骤 3：训练（含动态图 + iDREM）
unagi.run_UNAGI(idrem_dir='/path/to/idrem')
```

### 8.2 分析阶段（发现 IPF 靶点）

```python
unagi.analyse_UNAGI(
    data_path='results/2/stagedata/org_dataset.h5ad',
    iteration=2,
    progressionmarker_background_sampling_times=10,
    run_pertubration=True,
    cmap_dir='cmap_drug_target.npy',    # CMAP 药物-靶基因数据库
    defulat_perturb_change=0.5,          # log2FC = 0.5
    overall_perturbation_analysis=True,
    perturbed_tracks='all'
)
```

### 8.3 自定义药物数据库扰动

```python
unagi.customize_drug_perturbation(
    data_path='results/2/stagedata/org_dataset.h5ad',
    iteration=2,
    customized_drug='my_drug_targets.npy',   # 自定义 {药物: [基因:方向]} 字典
    bound=0.5,
    CUDA=True,
    save_csv='ipf_drug_rankings.csv'
)
```

---

## 九、关键数据结构

```
AnnData 对象（adata）
├── .X                         基因表达矩阵（细胞 × 基因）
├── .obs
│   ├── 'stage'                疾病分期（0=对照，1/2/3=进展期）
│   ├── 'leiden'               细胞簇 ID
│   └── 'ident'                细胞类型注释
├── .obsm
│   ├── 'z'                    VAE 潜变量嵌入
│   ├── 'embdedding'           原始细胞嵌入（扰动分析用）
│   └── 'perturbed_embedding'  扰动后细胞嵌入
├── .obsp
│   └── 'gcn_connectivities'   KNN 细胞图邻接矩阵
├── .layers
│   └── 'geneWeight'           基因重要性权重（迭代训练更新）
└── .uns
    ├── 'edges'                          时序动态图边 {period: [[from, to]]}
    ├── 'data_pathway_overlap_genes'     通路与数据集基因的交集
    ├── 'data_drug_overlap_genes'        药物靶基因与数据集基因的交集
    ├── 'pathway_ranking'                通路基因权重排名
    ├── 'progressionMarkers'             动态标志物（各轨迹上下调基因）
    ├── 'hcmarkers'                      层级静态标志物
    ├── 'drug_perturbation_deltaD'       各药物的 ΔD 矩阵
    ├── 'drug_perturbation_score'        药物综合得分排名
    ├── 'pathway_perturbation_deltaD'    各通路的 ΔD 矩阵
    └── 'pathway_perturbation_score'     通路综合得分排名
```

---

## 十、IPF 靶点发现的输出与解读

最终输出 `adata.uns['drug_perturbation_score']` 包含每种药物的综合得分，**得分越高代表该药物在计算机模拟中越能使 IPF 细胞（尤其是成纤维细胞）向健康对照状态移动**。结果可导出为 CSV 文件，配合 `fitlerOutNarrowPathway` 方法完成 p < 0.05 的统计显著性过滤，得到最终 IPF 候选治疗靶点列表。

对于 IPF 的 In-Silico 药物验证，代码还支持对特定细胞类型（如成纤维细胞）进行针对性扰动，通过 `in_silico_drug_discovery.ipynb` 教程可以模拟不同剂量的药物效应，并结合实验数据进行验证：

```python
# tutorials/in_silico_drug_discovery.ipynb
for i in range(len(candidate_drugs)):
    for change in [0.2, 0.3, 0.4]:  # 测试不同扰动幅度
        adata_copy = adata.copy()
        # 仅针对成纤维细胞（IPF 的关键病理细胞类型）
        fibroblast_cells = adata.obs[
            adata.obs['name.simple'].str.startswith('Fibroblast')
        ].index.tolist()

        for each_target in target_genes[i]:
            gene_name, direction = each_target.split(':')
            gene_idx = adata.var.index.tolist().index(gene_name)
            if direction == '+':
                adata_copy.layers['simu'][fibroblast_cells, gene_idx] += \
                    (4 - stage_number) * change
            else:
                adata_copy.layers['simu'][fibroblast_cells, gene_idx] -= \
                    (stage_number + 1) * change
        adata_copy.write(f'drug_{i}_change_{change}/dataset.h5ad')
```

---

## 参考文件索引

| 模块 | 文件路径 | 主要功能 |
|------|---------|---------|
| 主控类 | `UNAGI/UNAGI_tool.py` | UNAGI 类：数据准备、训练、分析入口 |
| 分析类 | `UNAGI/UNAGI_analyst.py` | analyst 类：标志物发现、扰动分析 |
| 深度学习模型 | `UNAGI/model/models.py` | VAE、GCN 编码器、判别器 |
| 概率分布 | `UNAGI/model/distributions.py` | 零膨胀分布族 |
| 训练流程 | `UNAGI/train/runner.py` | UNAGI_runner：单轮迭代完整流程 |
| 训练器 | `UNAGI/train/trainer.py` | UNAGI_trainer：模型训练实现 |
| 动态图 | `UNAGI/dynamic_graphs/buildGraph.py` | KL 散度连边、时序图构建 |
| 调控网络 | `UNAGI/dynamic_regulatory_networks/processIDREM.py` | iDREM 结果处理 |
| 动态标志物 | `UNAGI/marker_discovery/dynamic_markers.py` | 轨迹驱动基因发现 |
| 层级标志物 | `UNAGI/marker_discovery/hierachical_static_markers.py` | 细胞类型特异标志物 |
| 扰动执行 | `UNAGI/perturbations/perturbation.py` | 药物/通路扰动、ΔD 计算 |
| 扰动分析 | `UNAGI/perturbations/analysis_perturbation.py` | 得分计算、统计检验 |
| 工具函数 | `UNAGI/utils/analysis_helper.py` | 通路重叠、基因权重排名 |

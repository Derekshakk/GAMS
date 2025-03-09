import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, QED
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import requests
import networkx as nx
import random
import logging
import re
import math
import pickle
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from networkx.algorithms import community
import os
import time
import pickle
from rdkit.Chem import Draw
import pandas as pd
from sklearn.manifold import TSNE
from networkx.algorithms import community
import matplotlib.pyplot as plt
import gzip
import json
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Generate, DefaultSigFactory
from rdkit import RDConfig
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import  rdkit

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SA Score计算相关
_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

def readFragmentScores(name="fpscores.pkl.gz"):
    global _fscores
    if name == "fpscores.pkl.gz":
        name = op.join(op.dirname(__file__), name)
    with gzip.open(name) as f:
        data = pickle.load(f)
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculatesaScore(m):
    if not m.GetNumAtoms():
        return None
    if _fscores is None:
        readFragmentScores()
    

    sfp = mfpgen.GetSparseCountFingerprint(m)
    score1 = 0.
    nf = 0
    nze = sfp.GetNonzeroElements()
    for id, count in nze.items():
        nf += count
        score1 += _fscores.get(id, -4) * count
    score1 /= nf
    

    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1
    
    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = math.log10(nMacrocycles + 1)
    
    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    
    # 指纹密度校正
    score3 = 0.
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * .5
    
    sascore = score1 + score2 + score3
    
    # 转换为1-10分制
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    return sascore


import requests

def get_smiles(smiles:str="",num:int=1):   
    url = "http://127.0.0.1:8000/predict"   
    data = {
        "prompt":smiles ,
        "num":num
    }    
    response = requests.post(url, json=data)
    return response.json()
class MoleculeOptimizer:
    def __init__(self, api_key):
        """初始化分子优化器"""
        self.api_key = api_key
        self.thought_graph = nx.DiGraph()
        self.node_counter = 0
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2) 

    def calculate_similarity(self, mol1, mol2):
        """计算两个分子的双重相似性"""
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
        fp1 = self.morgan_generator.GetSparseCountFingerprint(mol1)
        fp2 = self.morgan_generator.GetSparseCountFingerprint(mol2)
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        props1 = self.get_property_vector(mol1)
        props2 = self.get_property_vector(mol2)
        cosine = np.dot(props1, props2) / (np.linalg.norm(props1) * np.linalg.norm(props2))
        alpha, beta = -0.5, 0.5
        similarity = alpha * tanimoto + beta * cosine
        # logging.info(f"Structural similarity (Tanimoto): {tanimoto:.3f}")
        # logging.info(f"Property similarity (Cosine): {cosine:.3f}")
        # logging.info(f"Combined similarity: {similarity:.3f}")
        return similarity

    def get_property_vector(self, mol):
        """获取分子属性向量"""
        MW = Descriptors.ExactMolWt(mol)
        LogP = Descriptors.MolLogP(mol)
        TPSA = Descriptors.TPSA(mol)
        HBD = Descriptors.NumHDonors(mol)
        HBA = Descriptors.NumHAcceptors(mol)
        SA = calculatesaScore(mol)
        QED_value = QED.default(mol)
        # print("----------------------------------------------", QED_value)
        RotB = Descriptors.NumRotatableBonds(mol)
        # 计算每个属性的分数
        MW_score = 1.0 if 250 <= MW <= 500 else 0.0
        LogP_score = 1.0 if 1 <= LogP <= 5 else 0.0
        TPSA_score = 1.0 if 20 <= TPSA <= 130 else 0.0
        HBD_score = 1.0 if 0 <= HBD <= 5 else 0.0
        HBA_score = 1.0 if 1 <= HBA <= 10 else 0.0
        RotB_score = 1.0 if 0 <= RotB <= 10 else 0.0
        SA_score = max(0, 1 - SA / 4.0)
        QED_score = max(0, (QED_value - 0.5) / 0.5)
        return np.array([
            MW_score, LogP_score, TPSA_score, HBD_score, HBA_score,
            SA_score, QED_score, RotB_score
        ])

    def calculate_select_score(self, node, original_mol):
        """计算节点扩展评分"""
        sim_score = self.calculate_similarity(node['mol'], original_mol)
        delta_property = self.calculate_property_improvement(node['mol'], original_mol)
        lambda1, lambda2 = 0.5, 0.5
        score = lambda1 * sim_score + lambda2 * delta_property
        # logging.info(f"Node expansion score: {score:.3f}")
        return score

    def calculate_property_improvement(self, current_mol, target_mol):
        """计算属性改进评分"""
        current_props = self.get_property_vector(current_mol)
        target_props = self.get_property_vector(target_mol)
        epsilon = 1e-10  # 添加一个很小的非零值
        improvements = (current_props - target_props) / (target_props + epsilon)
        improvement_score = np.mean(np.clip(improvements, -1, 1))
        # logging.info(f"Property improvement score: {improvement_score:.3f}")
        return improvement_score

    def calculate_path_score(self, path):
        """计算路径评分"""
        node_scores = sum(self.thought_graph.nodes[node]['score'] for node in path)
        efficiency = self.calculate_path_efficiency(path)
        novelty = self.calculate_path_novelty(path)
        total_score = node_scores + efficiency + novelty
        # logging.info(f"\nPath scoring details:")
        # logging.info(f"Node total score: {node_scores:.3f}")
        # logging.info(f"Efficiency score: {efficiency:.3f}")
        # logging.info(f"Novelty score: {novelty:.3f}")
        # logging.info(f"Total path score: {total_score:.3f}")
        return total_score

    def calculate_path_efficiency(self, path):
        """计算路径效率评分"""
        if len(path) <= 1:
            return 0
        start_props = self.get_property_vector(self.thought_graph.nodes[path[0]]['mol'])
        end_props = self.get_property_vector(self.thought_graph.nodes[path[-1]]['mol'])
        delta_props = np.abs(end_props - start_props)
        return np.mean(delta_props) / len(path)

    def calculate_path_novelty(self, path):
        """计算路径新颖性评分"""
        final_mol = self.thought_graph.nodes[path[-1]]['mol']
        ref_mols = [self.thought_graph.nodes[n]['mol'] for n in self.thought_graph.nodes()]
        max_sim = max(self.calculate_similarity(final_mol, ref_mol) 
                     for ref_mol in ref_mols if ref_mol != final_mol)
        return 1 - max_sim

    def send_request(self, prompt):
        """发送请求到GPT API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def create_expansion_prompt(self, node):
        """创建分子修饰提示"""
        return f"""
        Current molecule SMILES: {Chem.MolToSmiles(node['mol'])}
        Current properties:
        - Molecular Weight: {Descriptors.ExactMolWt(node['mol']):.2f}
        - LogP: {Descriptors.MolLogP(node['mol']):.2f}
        - TPSA: {Descriptors.TPSA(node['mol']):.2f}
        - H-bond donors: {Descriptors.NumHDonors(node['mol'])}
        - H-bond acceptors: {Descriptors.NumHAcceptors(node['mol'])}

        Please provide 3 chemically feasible molecular modifications.Be sure not to have existing smiles on the market, just brand new ones. For each suggestion:
        1. SMILES of modified molecule
        2. Modification rationale
        3. Expected property changes

        Format:
        1. SMILES: [modified SMILES]
           Rationale: [explanation]
        2. SMILES: [modified SMILES]
           Rationale: [explanation]
        3. SMILES: [modified SMILES]
           Rationale: [explanation]
        """

    def parse_llm_response(self, response):
        """解析GPT API响应"""
        suggestions = []
        lines = response.split('\n')
        current_smiles = ""
        current_reasoning = ""
        smiles_pattern = re.compile(r'\d*\.\s*SMILES:\s*(.*)')
        reasoning_pattern = re.compile(r'\s*Rationale:\s*(.*)')
        for line in lines:
            line = line.strip()
            smiles_match = smiles_pattern.match(line)
            reasoning_match = reasoning_pattern.match(line)
            if smiles_match:
                if current_smiles:
                    suggestions.append((current_smiles, current_reasoning.strip()))
                current_smiles = smiles_match.group(1).strip()
                current_reasoning = ""
            elif reasoning_match:
                current_reasoning += reasoning_match.group(1) + " "
        if current_smiles:
            suggestions.append((current_smiles, current_reasoning.strip()))

        return suggestions






    class UnionFind:
        """并查集数据结构"""
        def __init__(self):
            self.parent = {}
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
            elif self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, x, y):
            self.parent[self.find(x)] = self.find(y)

    def identify_quality_clusters(self, quality_threshold=6.5):
        """识别高质量分子簇（使用无向图）"""
        quality_nodes = []
        for node in self.thought_graph.nodes():
            mol = self.thought_graph.nodes[node]['mol']
            smiles = Chem.MolToSmiles(mol)
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", smiles)
            property_scores = self.get_property_vector(mol)
            avg_score = np.sum(property_scores)
            self.thought_graph.nodes[node]['score'] = avg_score
            # print("￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥", avg_score)
            if avg_score >= quality_threshold:
                quality_nodes.append(node)
        # logging.info(f"\nFound {len(quality_nodes)} high-quality molecular nodes")
        if not quality_nodes:
            # logging.info("No high-quality molecules found")
            return []
        # 将有向图转换为无向图
        quality_subgraph = nx.Graph(self.thought_graph.subgraph(quality_nodes))
        connected_components = list(nx.connected_components(quality_subgraph))
        largest_component = max(connected_components, key=len) if connected_components else set()
        # logging.info(f"Largest cluster size: {len(largest_component)}")
        return list(largest_component)

    # def identify_quality_clusters(self, quality_threshold=6.5):
    #     """识别高质量分子簇（使用有向图的弱连通组件）"""
    #     quality_nodes = []
    #     for node in self.thought_graph.nodes():
    #         mol = self.thought_graph.nodes[node]['mol']
    #         smiles = Chem.MolToSmiles(mol)
    #         print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", smiles)
    #         property_scores = self.get_property_vector(mol)
    #         avg_score = np.sum(property_scores)
    #         self.thought_graph.nodes[node]['score'] = avg_score
    #         print("￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥", avg_score)
    #         if avg_score >= quality_threshold:
    #             quality_nodes.append(node)
    #     logging.info(f"\nFound {len(quality_nodes)} high-quality molecular nodes")
    #     if not quality_nodes:
    #         logging.info("No high-quality molecules found")
    #         return []
    #     # 使用有向图的弱连通组件
    #     quality_subgraph = self.thought_graph.subgraph(quality_nodes)
    #     connected_components = list(nx.weakly_connected_components(quality_subgraph))
    #     largest_component = max(connected_components, key=len) if connected_components else set()
    #     logging.info(f"Largest cluster size: {len(largest_component)}")
    #     return list(largest_component)
    def optimize_molecule(self,  iterations=3, num_parallel=3):
        """分子优化主函数，支持并行优化多个分子"""
        logging.info("\nStarting molecule optimization process...")
        # logging.info(f"Task: {task}")
        # logging.info(f"Initial molecule: {initial_smiles}")
        initial_smiles=get_smiles(num=1)[0]
        initial_mol = Chem.MolFromSmiles(initial_smiles)
        self.thought_graph.add_node(0, mol=initial_mol, smiles=initial_smiles, score=0)
        
        # 使用集合存储当前活跃节点
        active_nodes = {0}
        best_score = 5.0
        best_molecule = initial_smiles

        for i in range(iterations):
            logging.info(f"\nIteration {i+1}:")
            new_active_nodes = set()
            
            # 对每个活跃节点进行优化
            for node_id in active_nodes:
                # prompt = self.create_expansion_prompt(self.thought_graph.nodes[node_id])
                try:
                    response = get_smiles(1)
                    # suggestions = self.parse_llm_response(response['choices'][0]['message']['content'])
                    # logging.info(f"Received {len(suggestions)} modification suggestions for node {node_id}")
                    
                    for smiles in response:
                        try:
                            new_mol = Chem.MolFromSmiles(smiles)
                            if new_mol is None:
                                continue
                                
                            new_node_id = self.node_counter + 1
                            score = self.calculate_select_score({'mol': new_mol}, initial_mol)
                            
                            self.thought_graph.add_node(
                                new_node_id, 
                                mol=new_mol, 
                                smiles=smiles, 
                                score=0
                                
                            )
                            self.thought_graph.add_edge(node_id, new_node_id)
                            
                            new_active_nodes.add(new_node_id)
                            self.node_counter += 1
                            
                            if score > best_score:
                                best_score = score
                                best_molecule = smiles
                                
                        except Exception as e:
                            # logging.error(f"Error processing modification: {str(e)}")
                            pass
                            
                except Exception as e:
                    # logging.error(f"API call error: {str(e)}")
                    pass
                    
            # 从新生成的节点中选择top-k作为下一轮的活跃节点
            if new_active_nodes:
                active_nodes = set(sorted(
                    new_active_nodes,
                    key=lambda x: self.thought_graph.nodes[x]['score'],
                    reverse=True
                )[:num_parallel])
                
            # logging.info(f"Active nodes for next iteration: {active_nodes}")

        quality_clusters = self.identify_quality_clusters()
        # logging.info("\nOptimization completed!")
        # logging.info(f"Best molecule SMILES: {best_molecule}")
        # logging.info(f"Final score: {best_score:.3f}")
        # logging.info(f"Total nodes generated: {self.node_counter}")

        new_node_id=4
        for node_id in range(1):
           
            node = self.thought_graph.nodes[node_id]['smiles']
            smiles=get_smiles(node,1)[0]
            new_mol = Chem.MolFromSmiles(smiles)
            score=self.calculate_select_score( {"mol":new_mol}, self.thought_graph.nodes[node_id]["mol"])

            self.thought_graph.add_node(
                                    new_node_id, 
                                    mol=new_mol, 
                                    smiles=smiles, 
                                    score=score
                                    
                                )
            
            self.thought_graph.add_edge(node_id, new_node_id)
            new_node_id+=1
        
        return best_molecule, best_score, quality_clusters

    def analyze_graph_features(self, save_dir='graph_analysis'):
        """执行图结构特性分析"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 基础数据准备
        G = self.thought_graph
        paths = list(nx.all_simple_paths(G, 0, max(G.nodes)))
        
        # 1. 模块度演化分析
        self._plot_modularity_evolution(G, f'{save_dir}/modularity_evolution.png')
        
        # 2. 路径结构分析
        self._plot_path_efficiency_matrix(G, paths, f'{save_dir}/path_efficiency_matrix.png')
        
        # 3. 族群密度分析
        self._plot_cluster_density(G, f'{save_dir}/cluster_density.png')
        
        # 4. 节点中心性分析
        self._plot_centrality_correlation(G, f'{save_dir}/centrality_correlation.png')

    def _plot_modularity_evolution(self, G, save_path):
        """模块度与属性协同演化图"""
        # 分阶段采样
        steps = np.linspace(0, len(G.nodes), 5, dtype=int)[1:]  # 分5个阶段
        modularities = []
        avg_qeds = []
        
        for n_nodes in steps:
            subgraph = G.subgraph(list(G.nodes)[:n_nodes])
            communities = nx.algorithms.community.greedy_modularity_communities(subgraph)
            mod = nx.algorithms.community.modularity(subgraph, communities)
            qeds = [QED.qed(node[1]['mol']) for node in subgraph.nodes(data=True)]
            
            modularities.append(mod)
            avg_qeds.append(np.mean(qeds))
        
        # 绘制双轴曲线
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(steps, modularities, 'b-o', markersize=8, linewidth=2)
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Modularity', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(steps, avg_qeds, 'r--s', markersize=8, linewidth=2)
        ax2.set_ylabel('Average QED', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 统计标注
        r, p = stats.pearsonr(modularities, avg_qeds)
        plt.title(f'Modularity-QED Co-evolution (Pearson r={r:.2f}, p={p:.2e})', fontsize=14)
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_path_efficiency_matrix(self, G, paths, save_path):
        """路径效率-连通性矩阵"""
        path_metrics = []
        
        for path in paths:
            if len(path) < 2:
                continue
                
            # 计算路径指标
            length = len(path)
            start_qed = G.nodes[path[0]]['score']
            end_qed = G.nodes[path[-1]]['score']
            efficiency = (end_qed - start_qed) / length
            connectivity = nx.node_connectivity(G.subgraph(path))
            
            path_metrics.append({
                'Length': length,
                'Efficiency': efficiency,
                'Connectivity': connectivity
            })
        
        # 创建热力图
        df = pd.DataFrame(path_metrics)
        pivot_table = df.pivot_table(values='Efficiency',
                                    index='Connectivity',
                                    columns=pd.cut(df['Length'], bins=3),
                                    aggfunc=np.mean)
        
        plt.figure(figsize=(10,8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu",
                  cbar_kws={'label': 'Efficiency (ΔQED/step)'})
        plt.title('Path Efficiency vs Connectivity and Length', fontsize=14)
        plt.xlabel('Path Length Quantiles', fontsize=12)
        plt.ylabel('Node Connectivity', fontsize=12)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cluster_density(self, G, save_path):
        """族群密度-属性关联图"""
        undirected_G = G.to_undirected()
        clusters = list(nx.connected_components(undirected_G))
        
        cluster_data = []
        for cluster in clusters:
            subgraph = undirected_G.subgraph(cluster)
            density = nx.density(subgraph)
            qeds = [G.nodes[n]['score'] for n in cluster]
            
            cluster_data.append({
                'Density': density,
                'Avg_QED': np.mean(qeds),
                'Size': len(cluster)
            })
        
        # 回归分析
        df = pd.DataFrame(cluster_data)
        X = df[['Density']]
        y = df['Avg_QED']
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        
        # 绘制散点图
        plt.figure(figsize=(10,8))
        sns.scatterplot(data=df, x='Density', y='Avg_QED', 
                       hue='Size', palette='viridis', 
                       size='Size', sizes=(50, 300))
        
        x_range = np.linspace(df['Density'].min(), df['Density'].max(), 100)
        plt.plot(x_range, model.predict(x_range.reshape(-1,1)), 
                'r--', linewidth=2)
        
        plt.title(f'Cluster Density vs QED (R²={r2:.2f})', fontsize=14)
        plt.xlabel('Cluster Density', fontsize=12)
        plt.ylabel('Average QED', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(title='Cluster Size')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_centrality_correlation(self, G, save_path):
        """节点中心性-属性关联图"""
        # 计算中心性指标
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # 构建数据框
        nodes = []
        for n in G.nodes:
            nodes.append({
                'Betweenness': betweenness[n],
                'Closeness': closeness[n],
                'PageRank': pagerank[n],
                'QED': G.nodes[n]['score'],
                'SA': calculatesaScore(G.nodes[n]['mol'])
            })
        df = pd.DataFrame(nodes)
        
        # 绘制相关矩阵
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                  vmin=-1, vmax=1, mask=np.triu(np.ones_like(df.corr())))
        plt.title('Node Centrality-Property Correlation Matrix', fontsize=14)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    def visualize_evolution(self, save_dir='visualization_results'):
        """分子优化过程可视化，每个图单独显示并保存"""
        # 创建保存目录
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 获取共用的布局和数据
        pos = nx.spring_layout(self.thought_graph)
        best_node = max(self.thought_graph.nodes(), 
                    key=lambda x: self.thought_graph.nodes[x]['score'])
        paths = list(nx.all_simple_paths(self.thought_graph, 0, best_node))
        best_path = max(paths, key=lambda p: sum(self.thought_graph.nodes[n]['score'] for n in p)) if paths else None

        # 1. 分子进化网络图
        plt.figure(figsize=(12, 10))
        node_scores = [self.thought_graph.nodes[node]['score'] for node in self.thought_graph.nodes()]
        node_sizes = [500 * calculatesaScore(self.thought_graph.nodes[node]['mol']) 
                    for node in self.thought_graph.nodes()]
        
        edge_weights = []
        for (u, v) in self.thought_graph.edges():
            sim = self.calculate_similarity(
                self.thought_graph.nodes[u]['mol'],
                self.thought_graph.nodes[v]['mol']
            )
            edge_weights.append(sim * 2)
        
        nodes = nx.draw_networkx_nodes(
            self.thought_graph, pos,
            node_size=node_sizes,
            node_color=node_scores,
            cmap='viridis'
        )
        nx.draw_networkx_nodes(
            self.thought_graph, pos,
            nodelist=[0],
            node_size=700,
            node_color='red'
        )
        edges = nx.draw_networkx_edges(
            self.thought_graph, pos,
            width=edge_weights,
            edge_color='gray',
            alpha=0.6,
            arrows=True
        )
        plt.colorbar(nodes, label='Score')
        plt.title('Molecular Evolution Network')
        plt.savefig(os.path.join(save_dir, '1_evolution_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 最优路径展示
        if best_path:
            plt.figure(figsize=(12, 10))
            # 先画完整的分子进化图作为背景
            nodes = nx.draw_networkx_nodes(
                self.thought_graph, pos,
                node_size=node_sizes,
                node_color=node_scores,
                cmap='viridis',
                alpha=0.1  # 淡化背景
            )
            edges = nx.draw_networkx_edges(
                self.thought_graph, pos,
                width=edge_weights,
                edge_color='gray',
                alpha=0.1,  # 淡化背景
                arrows=True
            )
            
            # 突出显示最佳路径
            best_path_graph = self.thought_graph.subgraph(best_path)
            path_pos = {node: pos[node] for node in best_path}
            
            nx.draw_networkx_nodes(
                best_path_graph, path_pos,
                node_color=[self.thought_graph.nodes[n]['score'] for n in best_path],
                node_size=600,
                cmap='viridis'
            )
            nx.draw_networkx_edges(
                best_path_graph, path_pos,
                edge_color='blue',
                width=2,
                arrows=True
            )
            plt.colorbar(nodes, label='Score')
            plt.title('Best Evolution Path')
            plt.savefig(os.path.join(save_dir, '2_best_path.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 高质量族群展示
        quality_clusters = self.identify_quality_clusters()
        if quality_clusters:
            plt.figure(figsize=(12, 10))
            # 先画完整的分子进化图作为背景
            nodes = nx.draw_networkx_nodes(
                self.thought_graph, pos,
                node_size=node_sizes,
                node_color=node_scores,
                cmap='viridis',
                alpha=0.1  # 淡化背景
            )
            edges = nx.draw_networkx_edges(
                self.thought_graph, pos,
                width=edge_weights,
                edge_color='gray',
                alpha=0.1,  # 淡化背景
                arrows=True
            )
            
            # 突出显示高质量族群
            cluster_graph = self.thought_graph.subgraph(quality_clusters)
            cluster_pos = {node: pos[node] for node in quality_clusters}
            
            nx.draw_networkx_nodes(
                cluster_graph, cluster_pos,
                node_color=[self.thought_graph.nodes[n]['score'] for n in quality_clusters],
                node_size=600,
                cmap='viridis'
            )
            nx.draw_networkx_edges(
                cluster_graph, cluster_pos,
                edge_color='green',
                width=2,
                arrows=True
            )
            plt.colorbar(nodes, label='Score')
            plt.title('High Quality Molecular Cluster')
            plt.savefig(os.path.join(save_dir, '3_quality_cluster.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 属性雷达图
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        
        categories = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'SA', 'QED', 'Lipinski']
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        initial_props = self.get_property_vector(self.thought_graph.nodes[0]['mol']).tolist()
        best_props = self.get_property_vector(self.thought_graph.nodes[best_node]['mol']).tolist()
        initial_props += initial_props[:1]
        best_props += best_props[:1]
        
        ax.plot(angles, initial_props, 'o-', label='Initial', 
                color='red', linewidth=2.5, alpha=0.7, zorder=2)
        ax.plot(angles, best_props, 'o-', label='Optimized', 
                color='blue', linewidth=2.5, alpha=0.7, zorder=1)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.legend()
        plt.title('Property Comparison')
        plt.savefig(os.path.join(save_dir, '4_property_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 优化路径分析
        if paths:
            plt.figure(figsize=(10, 6))
            # 为每个路径在每个步骤画散点
            for path in paths:
                scores = [self.thought_graph.nodes[n]['score'] for n in path]
                steps = range(len(scores))
                plt.scatter(steps, scores, alpha=0.5, color='blue', s=50)
            
            plt.xlabel('Iteration Step')
            plt.ylabel('Score')
            plt.title('All Evolution Paths Scores')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, '5_optimization_path.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 6. 效率分析
        if paths:
            plt.figure(figsize=(10, 6))
            efficiencies = []
            for path in paths:
                delta_props = np.abs(self.get_property_vector(self.thought_graph.nodes[path[-1]]['mol']) -
                                self.get_property_vector(self.thought_graph.nodes[path[0]]['mol']))
                efficiency = np.mean(delta_props) / len(path)
                efficiencies.append(efficiency)
            
            plt.hist(efficiencies, bins=20, color='skyblue', alpha=0.7)
            plt.axvline(np.mean(efficiencies), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(efficiencies):.3f}')
            plt.xlabel('Path Efficiency')
            plt.ylabel('Count')
            plt.title('Evolution Path Efficiency Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, '6_path_efficiency.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"All visualization results have been saved to '{save_dir}' directory")
        self.analyze_graph_features(save_dir)
        
        print(f"All visualizations saved to {save_dir}")


class QualityAwareOptimizer(MoleculeOptimizer):
    def __init__(self, api_key, qed_threshold=0.58, sa_threshold=4.5):
        super().__init__(api_key)
        self.qed_threshold = qed_threshold
        self.sa_threshold = sa_threshold
        self.quality_graph = nx.Graph()
        self.sim_cache = {}
        self.save_path = None
    
    def _is_high_quality(self, mol):
        """质量筛选函数"""
        try:
            qed = QED.qed(mol)
            sa = calculatesaScore(mol)
            return qed > self.qed_threshold and sa <= self.sa_threshold
        except:
            return False
    def plot_pharmacophore_heatmap(self, save_dir='visualization_results'):
        """创建增强版药效团分布热图分析"""
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 构建特征工厂
        feature_factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        # 定义详细的药效团类型
        pharm_types = {
            'Donor': 'HBD',           # 氢键供体
            'Acceptor': 'HBA',        # 氢键受体
            'Aromatic': 'Ar',         # 芳香环
            'Hydrophobe': 'Hy',       # 疏水基团
            'LumpedHydrophobe': 'LHy', # 疏水区域
            'PosIonizable': 'PI',     # 正离子化位点
            'NegIonizable': 'NI'      # 负离子化位点
        }
        
        # 收集药效团数据
        feature_data = []
        
        # 遍历分子进行药效团识别和统计
        for node in self.quality_graph.nodes():  # 使用 quality_graph 而不是 thought_graph
            mol = self.quality_graph.nodes[node]['mol']
            mol_features = feature_factory.GetFeaturesForMol(mol)
            
            # 初始化特征计数
            counts = {pharm_type: 0 for pharm_type in pharm_types.values()}
            
            # 统计药效团出现频次
            for feature in mol_features:
                feature_family = feature.GetFamily()
                if feature_family in pharm_types:
                    counts[pharm_types[feature_family]] += 1
                    
            # 添加其他关键描述符
            counts['MW'] = Descriptors.ExactMolWt(mol)  
            counts['LogP'] = Descriptors.MolLogP(mol)   
            counts['TPSA'] = Descriptors.TPSA(mol)      
            
            feature_data.append(counts)
        
        # 转换为DataFrame进行数据分析
        df = pd.DataFrame(feature_data)
        
        # 创建相关性热图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(df.corr()))
        
        sns.heatmap(df.corr(), 
                    mask=mask,
                    annot=True,           
                    fmt='.2f',            
                    cmap='RdYlBu_r',      
                    vmin=-1, vmax=1,      
                    center=0,             
                    square=True,          
                    linewidths=0.5,       
                    cbar_kws={'label': 'Pearson Correlation Coefficient'})
        
        plt.title('Pharmacophore Feature Correlation Analysis', pad=20, fontsize=14)
        plt.savefig(os.path.join(save_dir, 'pharmacophore_correlation.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
        
        # 创建药效团分布分析图
        plt.figure(figsize=(15, 8))
        
        # 仅分析药效团特征
        pharm_features = list(pharm_types.values())
        feature_stats = df[pharm_features].mean()
        feature_stds = df[pharm_features].std()
        
        # 创建条形图并修复索引问题
        x = np.arange(len(pharm_features))
        bars = plt.bar(x, feature_stats.values, 
                    yerr=feature_stds.values,
                    capsize=5,
                    alpha=0.8)
        
        # 修复标签添加方式
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(i, height,
                    f'{height:.2f}±{feature_stds.values[i]:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Pharmacophore Distribution Profile', fontsize=14)
        plt.xlabel('Pharmacophore Type', fontsize=12)
        plt.ylabel('Average Occurrence per Molecule (±SD)', fontsize=12)
        plt.xticks(x, pharm_features, rotation=45)
        plt.grid(True, alpha=0.3)

        
        # 保存分布图
        plt.savefig(os.path.join(save_dir, 'pharmacophore_distribution.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
        
        return df
    

    def visualize_pharmacophores_3d(self, mol, save_dir='visualization_results'):
        """创建分子药效团的3D可视化
        
        使用不同颜色标记不同类型的药效团:
        - 氢键供体(HBD): 深蓝色
        - 氢键受体(HBA): 红色
        - 芳香环(Ar): 橙色
        - 疏水基团(Hy): 黄色
        - 正离子化位点(PI): 绿色
        - 负离子化位点(NI): 紫色
        """
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 构建特征工厂
        feature_factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        # 为分子生成3D构象
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # 获取药效团特征
        features = feature_factory.GetFeaturesForMol(mol)
        # print(f"Features for {Chem.MolToSmiles(mol)}: {features}")  # 调试输出
        
        # 设置可视化参数
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        
        # 定义药效团颜色映射
        feature_colors = {
            'Donor': (0, 0, 1),      # 深蓝色
            'Acceptor': (1, 0, 0),   # 红色
            'Aromatic': (1, 0.65, 0),# 橙色
            'Hydrophobe': (1, 1, 0), # 黄色
            'PosIonizable': (0, 1, 0),# 绿色
            'NegIonizable': (0.5, 0, 0.5) # 紫色
        }
        
        # 为每个特征创建高亮
        highlights = {}
        for feature in features:
            family = feature.GetFamily()
            if family in feature_colors:
                atoms = feature.GetAtomIds()
                color = feature_colors[family]
                for atom_id in atoms:
                    if atom_id not in highlights:
                        highlights[atom_id] = color
        # print(f"Highlights: {highlights}")  # 调试输出
        
        # 设置原子颜色
        atom_cols = {}
        for atom_id, color in highlights.items():
            atom_cols[atom_id] = color
        # print(f"Atom Colors: {atom_cols}")  # 调试输出
        
        # 绘制分子
        img_path = os.path.join(save_dir, f'pharmacophore_3d_{Chem.MolToSmiles(mol)}.png')
        
        # 使用 MolToImage 生成图片
        img = Draw.MolToImage(
            mol,
            size=(800, 800),
            highlightAtoms=list(highlights.keys()),
            highlightAtomColors=atom_cols  # 明确传递 highlightAtomColors
        )
        # img.save(img_path)  # 保存图片
        
        return img_path
    def visualize_top_qed_pharmacophores(self, save_dir='visualization_results'):
        """为 quality_graph 中 QED 分数前 6 的分子生成药效团 3D 可视化图，排列成两行三列"""
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 获取 QED 分数前 6 的分子
        top_nodes = sorted(self.quality_graph.nodes,
                        key=lambda x: self.quality_graph.nodes[x]['qed'],
                        reverse=True)[:6]
        
        # 创建两行三列的子图布局
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pharmacophore 3D Visualization for Top 6 QED Molecules', fontsize=16, y=1.02)
        
        # 遍历每个分子并生成可视化
        for idx, node in enumerate(top_nodes):
            mol = self.quality_graph.nodes[node]['mol']
            smiles = self.quality_graph.nodes[node]['smiles']
            qed = self.quality_graph.nodes[node]['qed']
            
            # 生成药效团 3D 可视化
            img_path = self.visualize_pharmacophores_3d(mol, save_dir)
            # img = plt.imread(img_path)
            
            # # 在子图中显示
            # ax = axes[idx // 3, idx % 3]
            # ax.imshow(img)
            # ax.set_title(f'SMILES: {smiles}\nQED: {qed:.2f}', fontsize=10)
            # ax.axis('off')
        
        # 调整布局并保存
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, 'top_qed_pharmacophores.png'), dpi=300, bbox_inches='tight')
        # plt.close()
        
        return os.path.join(save_dir, 'top_qed_pharmacophores.png')




    def _add_to_quality_graph(self, mol):
        """添加到高质量分子网络（添加存在性检查）"""
        # 生成规范SMILES用于查重
        canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        
        # 检查是否已存在
        existing_smiles = {self.quality_graph.nodes[n]['smiles'] 
                          for n in self.quality_graph.nodes()}
        if canon_smiles in existing_smiles:
            # logging.info(f"Duplicate molecule found: {canon_smiles}, skipping addition")
            return
        
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        fp=self.morgan_generator.GetFingerprint(mol)
        new_id = len(self.quality_graph.nodes)
        
        # 添加节点属性
        self.quality_graph.add_node(new_id,
                                  mol=mol,
                                  fp=fp,
                                  qed=QED.qed(mol),
                                  sa=calculatesaScore(mol),
                                  smiles=canon_smiles,  # 使用规范SMILES
                                  mw=Descriptors.ExactMolWt(mol),
                                  logp=Descriptors.MolLogP(mol),
                                  tpsa=Descriptors.TPSA(mol))
        
        # 建立相似性连接
        for existing_id in self.quality_graph.nodes:
            if existing_id == new_id:
                continue
            
            cache_key = tuple(sorted([existing_id, new_id]))
            if cache_key in self.sim_cache:
                similarity = self.sim_cache[cache_key]
            else:
                existing_fp = self.quality_graph.nodes[existing_id]['fp']
                similarity = DataStructs.TanimotoSimilarity(existing_fp, fp)
                self.sim_cache[cache_key] = similarity
            
            if similarity > 0.4:
                self.quality_graph.add_edge(existing_id, new_id, weight=similarity)
    
    def optimize_molecule(self,  max_attempts=200, target_count=20, save_dir="results"):
        """增强的优化流程"""
        # 初始化保存路径
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.save_path = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(self.save_path, exist_ok=True)
        
        # 初始化原始图
        super().optimize_molecule( iterations=1, num_parallel=3)
        
        # 质量筛选循环
        attempt = 0
        new_node_id =7
        while len(self.quality_graph.nodes) < target_count and attempt < max_attempts:
            # 从原始图中随机选择节点进行扩展
            
            score_dict={}
            for node_id in self.thought_graph.nodes:
                score_dict[node_id] = self.thought_graph.nodes[node_id]['score']
            top_three_nodes = sorted(score_dict, key=score_dict.get, reverse=True)[:3]
            # nodes_to_expand = [ self.thought.graph.nodes[id]["smiles"] for id in top_three_nodes]
            for node in top_three_nodes:
                 
            
                try:
                    response = get_smiles(self.thought_graph.nodes[node]["smiles"],1)

                    
                    
                    if any(self.thought_graph.nodes[node_id]['smiles'] == response[0] for node_id in self.thought_graph.nodes):
                        continue  

                                    
            
                    new_mol = Chem.MolFromSmiles(response[0])
                    score=self.calculate_select_score( {"mol":new_mol}, self.thought_graph.nodes[node]["mol"])

                    self.thought_graph.add_node(
                                    new_node_id, 
                                    mol=new_mol, 
                                    smiles=response[0], 
                                    score=score
                                    
                                )
            
                    self.thought_graph.add_edge(node, new_node_id)
                    new_node_id+=1
                    mol = Chem.MolFromSmiles(response[0])
                    if mol and self._is_high_quality(mol):
                        self._add_to_quality_graph(mol)
                        # logging.info(f"Collected {len(self.quality_graph.nodes)} HQ molecules")
                        
                        # 达到目标数量立即停止
                        if len(self.quality_graph.nodes) >= target_count:
                            break
                    
                except Exception as e:
                    # logging.error(f"Generation error: {str(e)}")
                    pass
                
                attempt += 1
        
        # 保存结果
        self._save_results()
        logging.info(f"Optimization completed. Results saved to: {self.save_path}")


        # 新增代码：保存所有quality graph的SMILES到CSV
        smiles_list = []
        for node in self.quality_graph.nodes:
            smiles = self.quality_graph.nodes[node].get('smiles', '')
            if smiles:  # 确保有效SMILES
                smiles_list.append(smiles)
        
        # 创建DataFrame并保存
        df = pd.DataFrame({'smiles': smiles_list})
        csv_path = os.path.join(self.save_path, "optimized_molecules.csv")
        df.to_csv(csv_path, index=False)
        
        logging.info(f"Optimization completed. Results saved to: {self.save_path}")
        return self.quality_graph


    def _save_results(self):
        """保存所有结果"""
        # 保存网络数据
        with open(os.path.join(self.save_path, "quality_graph.pkl"), "wb") as f:
            pickle.dump(self.quality_graph, f)


        # 计算统计指标
        if len(self.quality_graph.nodes) > 0:
            total_qed = sum(self.quality_graph.nodes[n]['qed'] 
                        for n in self.quality_graph.nodes)
            total_sa = sum(self.quality_graph.nodes[n]['sa'] 
                       for n in self.quality_graph.nodes)
            avg_qed = total_qed / len(self.quality_graph.nodes)
            avg_sa = total_sa / len(self.quality_graph.nodes)
            
            print(f"\nQuality Graph Statistics:")
            print(f"Total Molecules: {len(self.quality_graph.nodes)}")
            print(f"Average QED: {avg_qed:.3f}")
            print(f"Average SA Score: {avg_sa:.3f}")
        
        # 保存分子数据
        self._save_molecule_data()
        
        # 生成可视化
        self.visualize_chemical_space(save=True)
        self.analyze_clusters(save=True)
        self._plot_property_distribution()
        self.plot_pharmacophore_heatmap(save_dir=self.save_path)
        self.visualize_top_qed_pharmacophores(save_dir=self.save_path)

    def _save_molecule_data(self):
        """保存分子数据到CSV"""
        data = []
        for node in self.quality_graph.nodes:
            mol_data = self.quality_graph.nodes[node]
            data.append({
                "SMILES": mol_data["smiles"],
                "QED": mol_data["qed"],
                "SA_Score": mol_data["sa"],
                "MolecularWeight": mol_data["mw"],
                "LogP": mol_data["logp"],
                "TPSA": mol_data["tpsa"]
            })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_path, "molecules.csv"), index=False)
        # logging.info(f"Saved molecular data to {os.path.join(self.save_path, 'molecules.csv')}")

    def visualize_chemical_space(self, save=False):
        """化学空间映射"""
        fps = [self.quality_graph.nodes[n]['fp'] for n in self.quality_graph.nodes]
        X = np.array([list(fp) for fp in fps])
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=min(5, len(X)-1))
        coords = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            coords[:, 0], coords[:, 1],
            c=[self.quality_graph.nodes[n]['qed'] for n in self.quality_graph.nodes],
            s=[200*(10 - self.quality_graph.nodes[n]['sa']) for n in self.quality_graph.nodes],
            cmap='viridis',
            alpha=0.7
        )
        
        # 标注最优分子
        best_node = max(self.quality_graph.nodes, key=lambda x: self.quality_graph.nodes[x]['qed'])
        plt.annotate(f"★ QED: {self.quality_graph.nodes[best_node]['qed']:.2f}",
                    xy=coords[best_node],
                    xytext=(coords[best_node][0]+5, coords[best_node][1]+5),
                    arrowprops=dict(facecolor='red', shrink=0.05))
        
        plt.colorbar(scatter, label='QED Score')
        plt.title("High Quality Molecular Space")
        
        if save:
            plt.savefig(os.path.join(self.save_path, "chemical_space.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def analyze_clusters(self, save=False):
        """族群分析"""
        communities = community.louvain_communities(self.quality_graph, weight='weight')
        
        # 绘制族群
        pos = nx.spring_layout(self.quality_graph, seed=42)
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab20.colors
        for i, comm in enumerate(communities):
            nx.draw_networkx_nodes(
                self.quality_graph, pos, nodelist=list(comm),
                node_color=[colors[i%20]]*len(comm),
                node_size=[300*self.quality_graph.nodes[n]['qed'] for n in comm],
                edgecolors='grey',
                label=f'Cluster {i+1}'
            )
        
        # 绘制边
        nx.draw_networkx_edges(self.quality_graph, pos, alpha=0.2)
        
        plt.legend()
        plt.title("Molecular Clusters (Colored by Community)")
        plt.axis('off')
        
        if save:
            plt.savefig(os.path.join(self.save_path, "molecular_clusters.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_property_distribution(self):
        """属性分布直方图"""
        plt.figure(figsize=(10, 6))
        
        properties = ['qed', 'sa', 'mw', 'logp']
        titles = ['QED Distribution', 'SA Score Distribution', 
                 'Molecular Weight Distribution', 'LogP Distribution']
                
        for i, (prop, title) in enumerate(zip(properties, titles), 1):
            plt.subplot(2, 2, i)
            values = [self.quality_graph.nodes[n][prop] for n in self.quality_graph.nodes]
            sns.histplot(values, bins=15, kde=True)
            plt.title(title)
            plt.xlabel('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "property_distributions.png"), dpi=300)
        plt.close()

    def show_top_molecules(self, top_n=5):
        """显示最优分子"""
        top_nodes = sorted(self.quality_graph.nodes,
                          key=lambda x: self.quality_graph.nodes[x]['qed'],
                          reverse=True)[:top_n]
        
        print(f"\nTop {top_n} Molecules:")
        for idx, node in enumerate(top_nodes, 1):
            data = self.quality_graph.nodes[node]
            print(f"{idx}. SMILES: {data['smiles']}")
            print(f"   QED: {data['qed']:.2f} | SA: {data['sa']:.1f}")
            print(f"   MW: {data['mw']:.1f} | LogP: {data['logp']:.2f} | TPSA: {data['tpsa']:.1f}")
            print("-"*60)

if __name__ == "__main__":
    # 初始化优化器
    optimizer = QualityAwareOptimizer(
        qed_threshold=0.58,
        sa_threshold=4.5
    )
    
    # 运行优化流程
    result_graph = optimizer.optimize_molecule(

        max_attempts=500,
        target_count=100,
        save_dir="./optimization_results"
    )
    
    # 查看结果
    optimizer.show_top_molecules(top_n=5)
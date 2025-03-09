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
import rdkit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    score3 = 0.
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * .5
    
    sascore = score1 + score2 + score3
    
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

def get_smiles(smiles: str = "", num: int = 1):   
    url = "http://127.0.0.1:8000/predict"   
    data = {
        "prompt": smiles,
        "num": num
    }    
    response = requests.post(url, json=data)
    return response.json()

class MoleculeOptimizer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.thought_graph = nx.DiGraph()
        self.node_counter = 0
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2) 

    def calculate_similarity(self, mol1, mol2):
        fp1 = self.morgan_generator.GetSparseCountFingerprint(mol1)
        fp2 = self.morgan_generator.GetSparseCountFingerprint(mol2)
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        props1 = self.get_property_vector(mol1)
        props2 = self.get_property_vector(mol2)
        cosine = np.dot(props1, props2) / (np.linalg.norm(props1) * np.linalg.norm(props2))
        alpha, beta = -0.5, 0.5
        similarity = alpha * tanimoto + beta * cosine
        return similarity

    def get_property_vector(self, mol):
        MW = Descriptors.ExactMolWt(mol)
        LogP = Descriptors.MolLogP(mol)
        TPSA = Descriptors.TPSA(mol)
        HBD = Descriptors.NumHDonors(mol)
        HBA = Descriptors.NumHAcceptors(mol)
        SA = calculatesaScore(mol)
        QED_value = QED.default(mol)
        RotB = Descriptors.NumRotatableBonds(mol)
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
        sim_score = self.calculate_similarity(node['mol'], original_mol)
        delta_property = self.calculate_property_improvement(node['mol'], original_mol)
        lambda1, lambda2 = 0.5, 0.5
        score = lambda1 * sim_score + lambda2 * delta_property
        return score

    def calculate_property_improvement(self, current_mol, target_mol):
        current_props = self.get_property_vector(current_mol)
        target_props = self.get_property_vector(target_mol)
        epsilon = 1e-10
        improvements = (current_props - target_props) / (target_props + epsilon)
        improvement_score = np.mean(np.clip(improvements, -1, 1))
        return improvement_score

    def calculate_path_score(self, path):
        node_scores = sum(self.thought_graph.nodes[node]['score'] for node in path)
        efficiency = self.calculate_path_efficiency(path)
        novelty = self.calculate_path_novelty(path)
        total_score = node_scores + efficiency + novelty
        return total_score

    def calculate_path_efficiency(self, path):
        if len(path) <= 1:
            return 0
        start_props = self.get_property_vector(self.thought_graph.nodes[path[0]]['mol'])
        end_props = self.get_property_vector(self.thought_graph.nodes[path[-1]]['mol'])
        delta_props = np.abs(end_props - start_props)
        return np.mean(delta_props) / len(path)

    def calculate_path_novelty(self, path):
        final_mol = self.thought_graph.nodes[path[-1]]['mol']
        ref_mols = [self.thought_graph.nodes[n]['mol'] for n in self.thought_graph.nodes()]
        max_sim = max(self.calculate_similarity(final_mol, ref_mol) 
                     for ref_mol in ref_mols if ref_mol != final_mol)
        return 1 - max_sim

    def send_request(self, prompt):
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
        quality_nodes = []
        for node in self.thought_graph.nodes():
            mol = self.thought_graph.nodes[node]['mol']
            smiles = Chem.MolToSmiles(mol)
            property_scores = self.get_property_vector(mol)
            avg_score = np.sum(property_scores)
            self.thought_graph.nodes[node]['score'] = avg_score
            if avg_score >= quality_threshold:
                quality_nodes.append(node)
        if not quality_nodes:
            return []
        quality_subgraph = nx.Graph(self.thought_graph.subgraph(quality_nodes))
        connected_components = list(nx.connected_components(quality_subgraph))
        largest_component = max(connected_components, key=len) if connected_components else set()
        return list(largest_component)

    def optimize_molecule(self, iterations=3, num_parallel=3):
        logging.info("\nStarting molecule optimization process...")
        initial_smiles = get_smiles(num=1)[0]
        initial_mol = Chem.MolFromSmiles(initial_smiles)
        self.thought_graph.add_node(0, mol=initial_mol, smiles=initial_smiles, score=0)
        
        active_nodes = {0}
        best_score = 5.0
        best_molecule = initial_smiles

        for i in range(iterations):
            logging.info(f"\nIteration {i+1}:")
            new_active_nodes = set()
            for node_id in active_nodes:
                try:
                    response = get_smiles(1)
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
                            pass
                except Exception as e:
                    pass
            if new_active_nodes:
                active_nodes = set(sorted(
                    new_active_nodes,
                    key=lambda x: self.thought_graph.nodes[x]['score'],
                    reverse=True
                )[:num_parallel])
        quality_clusters = self.identify_quality_clusters()
        new_node_id = 4
        for node_id in range(1):
            node = self.thought_graph.nodes[node_id]['smiles']
            smiles = get_smiles(node, 1)[0]
            new_mol = Chem.MolFromSmiles(smiles)
            score = self.calculate_select_score({"mol": new_mol}, self.thought_graph.nodes[node_id]["mol"])
            self.thought_graph.add_node(
                new_node_id, 
                mol=new_mol, 
                smiles=smiles, 
                score=score
            )
            self.thought_graph.add_edge(node_id, new_node_id)
            new_node_id += 1
        return best_molecule, best_score, quality_clusters

    def analyze_graph_features(self, save_dir='graph_analysis'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        G = self.thought_graph
        paths = list(nx.all_simple_paths(G, 0, max(G.nodes)))
        self._plot_modularity_evolution(G, f'{save_dir}/modularity_evolution.png')
        self._plot_path_efficiency_matrix(G, paths, f'{save_dir}/path_efficiency_matrix.png')
        self._plot_cluster_density(G, f'{save_dir}/cluster_density.png')
        self._plot_centrality_correlation(G, f'{save_dir}/centrality_correlation.png')

    def _plot_modularity_evolution(self, G, save_path):
        steps = np.linspace(0, len(G.nodes), 5, dtype=int)[1:]
        modularities = []
        avg_qeds = []
        for n_nodes in steps:
            subgraph = G.subgraph(list(G.nodes)[:n_nodes])
            communities = nx.algorithms.community.greedy_modularity_communities(subgraph)
            mod = nx.algorithms.community.modularity(subgraph, communities)
            qeds = [QED.qed(node[1]['mol']) for node in subgraph.nodes(data=True)]
            modularities.append(mod)
            avg_qeds.append(np.mean(qeds))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(steps, modularities, 'b-o', markersize=8, linewidth=2)
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Modularity', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(steps, avg_qeds, 'r--s', markersize=8, linewidth=2)
        ax2.set_ylabel('Average QED', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        r, p = stats.pearsonr(modularities, avg_qeds)
        plt.title(f'Modularity-QED Co-evolution (Pearson r={r:.2f}, p={p:.2e})', fontsize=14)
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_path_efficiency_matrix(self, G, paths, save_path):
        path_metrics = []
        for path in paths:
            if len(path) < 2:
                continue
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
        df = pd.DataFrame(path_metrics)
        pivot_table = df.pivot_table(values='Efficiency',
                                    index='Connectivity',
                                    columns=pd.cut(df['Length'], bins=3),
                                    aggfunc=np.mean)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu",
                  cbar_kws={'label': 'Efficiency (ΔQED/step)'})
        plt.title('Path Efficiency vs Connectivity and Length', fontsize=14)
        plt.xlabel('Path Length Quantiles', fontsize=12)
        plt.ylabel('Node Connectivity', fontsize=12)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cluster_density(self, G, save_path):
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
        df = pd.DataFrame(cluster_data)
        X = df[['Density']]
        y = df['Avg_QED']
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        plt.figure(figsize=(10, 8))
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
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
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
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                  vmin=-1, vmax=1, mask=np.triu(np.ones_like(df.corr())))
        plt.title('Node Centrality-Property Correlation Matrix', fontsize=14)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_evolution(self, save_dir='visualization_results'):
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pos = nx.spring_layout(self.thought_graph)
        best_node = max(self.thought_graph.nodes(), 
                    key=lambda x: self.thought_graph.nodes[x]['score'])
        paths = list(nx.all_simple_paths(self.thought_graph, 0, best_node))
        best_path = max(paths, key=lambda p: sum(self.thought_graph.nodes[n]['score'] for n in p)) if paths else None
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
        if best_path:
            plt.figure(figsize=(12, 10))
            nodes = nx.draw_networkx_nodes(
                self.thought_graph, pos,
                node_size=node_sizes,
                node_color=node_scores,
                cmap='viridis',
                alpha=0.1
            )
            edges = nx.draw_networkx_edges(
                self.thought_graph, pos,
                width=edge_weights,
                edge_color='gray',
                alpha=0.1,
                arrows=True
            )
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
        quality_clusters = self.identify_quality_clusters()
        if quality_clusters:
            plt.figure(figsize=(12, 10))
            nodes = nx.draw_networkx_nodes(
                self.thought_graph, pos,
                node_size=node_sizes,
                node_color=node_scores,
                cmap='viridis',
                alpha=0.1
            )
            edges = nx.draw_networkx_edges(
                self.thought_graph, pos,
                width=edge_weights,
                edge_color='gray',
                alpha=0.1,
                arrows=True
            )
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
        if paths:
            plt.figure(figsize=(10, 6))
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
        try:
            qed = QED.qed(mol)
            sa = calculatesaScore(mol)
            return qed > self.qed_threshold and sa <= self.sa_threshold
        except:
            return False

    def plot_pharmacophore_heatmap(self, save_dir='visualization_results'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        feature_factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        pharm_types = {
            'Donor': 'HBD',
            'Acceptor': 'HBA',
            'Aromatic': 'Ar',
            'Hydrophobe': 'Hy',
            'LumpedHydrophobe': 'LHy',
            'PosIonizable': 'PI',
            'NegIonizable': 'NI'
        }
        feature_data = []
        for node in self.quality_graph.nodes():
            mol = self.quality_graph.nodes[node]['mol']
            mol_features = feature_factory.GetFeaturesForMol(mol)
            counts = {pharm_type: 0 for pharm_type in pharm_types.values()}
            for feature in mol_features:
                feature_family = feature.GetFamily()
                if feature_family in pharm_types:
                    counts[pharm_types[feature_family]] += 1
            counts['MW'] = Descriptors.ExactMolWt(mol)  
            counts['LogP'] = Descriptors.MolLogP(mol)   
            counts['TPSA'] = Descriptors.TPSA(mol)      
            feature_data.append(counts)
        df = pd.DataFrame(feature_data)
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
        plt.figure(figsize=(15, 8))
        pharm_features = list(pharm_types.values())
        feature_stats = df[pharm_features].mean()
        feature_stds = df[pharm_features].std()
        x = np.arange(len(pharm_features))
        bars = plt.bar(x, feature_stats.values, 
                    yerr=feature_stds.values,
                    capsize=5,
                    alpha=0.8)
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
        plt.savefig(os.path.join(save_dir, 'pharmacophore_distribution.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
        return df
    
    def visualize_pharmacophores_3d(self, mol, save_dir='visualization_results'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        feature_factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        features = feature_factory.GetFeaturesForMol(mol)
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        feature_colors = {
            'Donor': (0, 0, 1),
            'Acceptor': (1, 0, 0),
            'Aromatic': (1, 0.65, 0),
            'Hydrophobe': (1, 1, 0),
            'PosIonizable': (0, 1, 0),
            'NegIonizable': (0.5, 0, 0.5)
        }
        highlights = {}
        for feature in features:
            family = feature.GetFamily()
            if family in feature_colors:
                atoms = feature.GetAtomIds()
                color = feature_colors[family]
                for atom_id in atoms:
                    if atom_id not in highlights:
                        highlights[atom_id] = color
        atom_cols = {}
        for atom_id, color in highlights.items():
            atom_cols[atom_id] = color
        img_path = os.path.join(save_dir, f'pharmacophore_3d_{Chem.MolToSmiles(mol)}.png')
        img = Draw.MolToImage(
            mol,
            size=(800, 800),
            highlightAtoms=list(highlights.keys()),
            highlightAtomColors=atom_cols
        )
        return img_path

    def visualize_top_qed_pharmacophores(self, save_dir='visualization_results'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        top_nodes = sorted(self.quality_graph.nodes,
                        key=lambda x: self.quality_graph.nodes[x]['qed'],
                        reverse=True)[:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pharmacophore 3D Visualization for Top 6 QED Molecules', fontsize=16, y=1.02)
        for idx, node in enumerate(top_nodes):
            mol = self.quality_graph.nodes[node]['mol']
            smiles = self.quality_graph.nodes[node]['smiles']
            qed = self.quality_graph.nodes[node]['qed']
            img_path = self.visualize_pharmacophores_3d(mol, save_dir)
        return os.path.join(save_dir, 'top_qed_pharmacophores.png')

    def _add_to_quality_graph(self, mol):
        canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        existing_smiles = {self.quality_graph.nodes[n]['smiles'] 
                          for n in self.quality_graph.nodes()}
        if canon_smiles in existing_smiles:
            return
        fp = self.morgan_generator.GetFingerprint(mol)
        new_id = len(self.quality_graph.nodes)
        self.quality_graph.add_node(new_id,
                                  mol=mol,
                                  fp=fp,
                                  qed=QED.qed(mol),
                                  sa=calculatesaScore(mol),
                                  smiles=canon_smiles,
                                  mw=Descriptors.ExactMolWt(mol),
                                  logp=Descriptors.MolLogP(mol),
                                  tpsa=Descriptors.TPSA(mol))
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
    
    def optimize_molecule(self, max_attempts=200, target_count=20, save_dir="results"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.save_path = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(self.save_path, exist_ok=True)
        super().optimize_molecule(iterations=1, num_parallel=3)
        attempt = 0
        new_node_id = 7
        while len(self.quality_graph.nodes) < target_count and attempt < max_attempts:
            score_dict = {}
            for node_id in self.thought_graph.nodes:
                score_dict[node_id] = self.thought_graph.nodes[node_id]['score']
            top_three_nodes = sorted(score_dict, key=score_dict.get, reverse=True)[:3]
            for node in top_three_nodes:
                try:
                    response = get_smiles(self.thought_graph.nodes[node]["smiles"], 1)
                    if any(self.thought_graph.nodes[node_id]['smiles'] == response[0] for node_id in self.thought_graph.nodes):
                        continue  
                    new_mol = Chem.MolFromSmiles(response[0])
                    score = self.calculate_select_score({"mol": new_mol}, self.thought_graph.nodes[node]["mol"])
                    self.thought_graph.add_node(
                                    new_node_id, 
                                    mol=new_mol, 
                                    smiles=response[0], 
                                    score=score
                                )
                    self.thought_graph.add_edge(node, new_node_id)
                    new_node_id += 1
                    mol = Chem.MolFromSmiles(response[0])
                    if mol and self._is_high_quality(mol):
                        self._add_to_quality_graph(mol)
                        if len(self.quality_graph.nodes) >= target_count:
                            break
                except Exception as e:
                    pass
                attempt += 1
        self._save_results()
        logging.info(f"Optimization completed. Results saved to: {self.save_path}")
        smiles_list = []
        for node in self.quality_graph.nodes:
            smiles = self.quality_graph.nodes[node].get('smiles', '')
            if smiles:
                smiles_list.append(smiles)
        df = pd.DataFrame({'smiles': smiles_list})
        csv_path = os.path.join(self.save_path, "optimized_molecules.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Optimization completed. Results saved to: {self.save_path}")
        return self.quality_graph

    def _save_results(self):
        with open(os.path.join(self.save_path, "quality_graph.pkl"), "wb") as f:
            pickle.dump(self.quality_graph, f)
        if len(self.quality_graph.nodes) > 0:
            total_qed = sum(self.quality_graph.nodes[n]['qed'] for n in self.quality_graph.nodes)
            total_sa = sum(self.quality_graph.nodes[n]['sa'] for n in self.quality_graph.nodes)
            avg_qed = total_qed / len(self.quality_graph.nodes)
            avg_sa = total_sa / len(self.quality_graph.nodes)
            print(f"\nQuality Graph Statistics:")
            print(f"Total Molecules: {len(self.quality_graph.nodes)}")
            print(f"Average QED: {avg_qed:.3f}")
            print(f"Average SA Score: {avg_sa:.3f}")
        self._save_molecule_data()
        self.visualize_chemical_space(save=True)
        self.analyze_clusters(save=True)
        self._plot_property_distribution()
        self.plot_pharmacophore_heatmap(save_dir=self.save_path)
        self.visualize_top_qed_pharmacophores(save_dir=self.save_path)

    def _save_molecule_data(self):
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

    def visualize_chemical_space(self, save=False):
        fps = [self.quality_graph.nodes[n]['fp'] for n in self.quality_graph.nodes]
        X = np.array([list(fp) for fp in fps])
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
        communities = community.louvain_communities(self.quality_graph, weight='weight')
        pos = nx.spring_layout(self.quality_graph, seed=42)
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab20.colors
        for i, comm in enumerate(communities):
            nx.draw_networkx_nodes(
                self.quality_graph, pos, nodelist=list(comm),
                node_color=[colors[i % 20]] * len(comm),
                node_size=[300 * self.quality_graph.nodes[n]['qed'] for n in comm],
                edgecolors='grey',
                label=f'Cluster {i+1}'
            )
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
        top_nodes = sorted(self.quality_graph.nodes,
                          key=lambda x: self.quality_graph.nodes[x]['qed'],
                          reverse=True)[:top_n]
        print(f"\nTop {top_n} Molecules:")
        for idx, node in enumerate(top_nodes, 1):
            data = self.quality_graph.nodes[node]
            print(f"{idx}. SMILES: {data['smiles']}")
            print(f"   QED: {data['qed']:.2f} | SA: {data['sa']:.1f}")
            print(f"   MW: {data['mw']:.1f} | LogP: {data['logp']:.2f} | TPSA: {data['tpsa']:.1f}")
            print("-" * 60)

if __name__ == "__main__":
    optimizer = QualityAwareOptimizer(
        qed_threshold=0.58,
        sa_threshold=4.5
    )
    result_graph = optimizer.optimize_molecule(
        max_attempts=500,
        target_count=100,
        save_dir="./optimization_results"
    )
    optimizer.show_top_molecules(top_n=5)

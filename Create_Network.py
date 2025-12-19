import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
import os
from datetime import datetime

# ============================================================
# CẤU HÌNH
# ============================================================
LOAD_ALL_EDGES = True
MAX_EDGES = 50000 

# Tạo folder để lưu kết quả
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'results_coauthor_network_{timestamp}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHÂN TÍCH ĐỒ THỊ ĐỒNG TÁC GIẢ")
print("="*60)
print(f"Load toàn bộ edges: {LOAD_ALL_EDGES}")
print(f"Thư mục kết quả: {OUTPUT_DIR}")
print()

# ============================================================
# XÂY DỰNG ĐỒ THỊ
# ============================================================
print("Đang tải dữ liệu...")
df_nodes = pd.read_csv('author node/author_nodes_csLG_v1_2024_2025.csv')
print(f" Đã tải {len(df_nodes)} tác giả")

G = nx.Graph()
# Thêm nodes trước
G.add_nodes_from(df_nodes['Id'])

# Bước 1: Load edges từ file
print("\nĐang tải edges từ file...")
if LOAD_ALL_EDGES:
    df_edges = pd.read_csv('edge/static_total_edges_csLG_v1_2024_2025.csv')
else:
    df_edges = pd.read_csv('edge/static_total_edges_csLG_v1_2024_2025.csv', nrows=MAX_EDGES)

print(f" Đã tải {len(df_edges)} edges từ file")

# Thêm edges với trọng số
for _, row in df_edges.iterrows():
    u, v, w = row['Source_Id'], row['Target_Id'], row['Weight']
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

# Thêm thuộc tính label cho nodes
authors = dict(zip(df_nodes['Id'], df_nodes['Label']))
nx.set_node_attributes(G, authors, 'label')

print(f"\n Đồ thị cuối cùng có {G.number_of_nodes()} nodes và {G.number_of_edges()} edges")

# Lưu đồ thị
print("\nĐang lưu đồ thị...")
nx.write_gexf(G, os.path.join(OUTPUT_DIR, 'coauthor_graph.gexf'))
print(f" Đã lưu: {OUTPUT_DIR}/coauthor_graph.gexf (mở bằng Gephi)")

nx.write_graphml(G, os.path.join(OUTPUT_DIR, 'coauthor_graph.graphml'))
print(f" Đã lưu: {OUTPUT_DIR}/coauthor_graph.graphml")

with open(os.path.join(OUTPUT_DIR, 'coauthor_graph.pkl'), 'wb') as f:
    pickle.dump(G, f)
print(f" Đã lưu: {OUTPUT_DIR}/coauthor_graph.pkl (Python pickle)")

# ============================================================
# 1. DEGREE DISTRIBUTION
# ============================================================
print("\n" + "="*60)
print("1. DEGREE DISTRIBUTION")
print("="*60)

degrees = [d for n, d in G.degree()]
degree_values = np.array(degrees)

print(f"Degree trung bình: {np.mean(degree_values):.2f}")
print(f"Degree trung vị: {np.median(degree_values):.2f}")
print(f"Degree min: {np.min(degree_values)}")
print(f"Degree max: {np.max(degree_values)}")
print(f"Độ lệch chuẩn: {np.std(degree_values):.2f}")

# Top 10 nodes có degree cao nhất
degree_dict = dict(G.degree())
top_degree_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 tác giả có nhiều kết nối nhất:")
for i, (node_id, degree) in enumerate(top_degree_nodes, 1):
    author_name = authors.get(node_id, f"ID_{node_id}")
    print(f"  {i:2d}. {author_name[:50]:50s} | Degree: {degree}")

# Vẽ degree distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(degrees, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Degree', fontsize=12)
axes[0].set_ylabel('Số lượng nodes', fontsize=12)
axes[0].set_title('Degree Distribution (Histogram)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Log-log plot
degree_count = {}
for d in degrees:
    degree_count[d] = degree_count.get(d, 0) + 1

x = sorted(degree_count.keys())
y = [degree_count[k] for k in x]

axes[1].loglog(x, y, 'bo-', markersize=4)
axes[1].set_xlabel('Degree (log scale)', fontsize=12)
axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
axes[1].set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'degree_distribution.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Đã lưu: {OUTPUT_DIR}/degree_distribution.png")

# Lưu degree distribution ra CSV
degree_df = pd.DataFrame([
    {'Author_ID': node_id, 'Author_Name': authors.get(node_id, ''), 'Degree': degree}
    for node_id, degree in degree_dict.items()
]).sort_values('Degree', ascending=False)
degree_df.to_csv(os.path.join(OUTPUT_DIR, 'degree_distribution.csv'), index=False, encoding='utf-8-sig')
print(f" Đã lưu: {OUTPUT_DIR}/degree_distribution.csv")

# ============================================================
# 2. CONNECTED COMPONENTS
# ============================================================
print("\n" + "="*60)
print("2. CONNECTED COMPONENTS")
print("="*60)

components = list(nx.connected_components(G))
component_sizes = [len(c) for c in components]

print(f"Số lượng connected components: {len(components)}")
print(f"Kích thước component lớn nhất: {max(component_sizes)} nodes")
print(f"Kích thước component nhỏ nhất: {min(component_sizes)} nodes")
print(f"Kích thước trung bình: {np.mean(component_sizes):.2f} nodes")

# Top 10 components lớn nhất
print("\nTop 10 components lớn nhất:")
sorted_components = sorted(components, key=len, reverse=True)[:10]
for i, comp in enumerate(sorted_components, 1):
    size = len(comp)
    percentage = (size / G.number_of_nodes()) * 100
    print(f"  {i:2d}. {size:6d} nodes ({percentage:5.2f}%)")

# Vẽ phân bố kích thước components
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(component_sizes, bins=50, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Kích thước component', fontsize=12)
plt.ylabel('Số lượng components', fontsize=12)
plt.title('Phân bố kích thước Connected Components', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
top_10_sizes = [len(c) for c in sorted_components]
plt.bar(range(1, len(top_10_sizes)+1), top_10_sizes, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Component rank', fontsize=12)
plt.ylabel('Kích thước', fontsize=12)
plt.title('Top 10 Components lớn nhất', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'connected_components.png'), dpi=150, bbox_inches='tight')
print(f"\n Đã lưu: {OUTPUT_DIR}/connected_components.png")

# Lưu components ra CSV
components_data = []
for comp_id, comp in enumerate(sorted_components):
    for node_id in comp:
        components_data.append({
            'Component_ID': comp_id,
            'Component_Size': len(comp),
            'Author_ID': node_id,
            'Author_Name': authors.get(node_id, '')
        })

components_df = pd.DataFrame(components_data)
components_df.to_csv(os.path.join(OUTPUT_DIR, 'connected_components.csv'), index=False, encoding='utf-8-sig')
print(f" Đã lưu: {OUTPUT_DIR}/connected_components.csv")

# ============================================================
# 3. CLUSTERING COEFFICIENT
# ============================================================
print("\n" + "="*60)
print("3. CLUSTERING COEFFICIENT")
print("="*60)

# Clustering coefficient toàn đồ thị
avg_clustering = nx.average_clustering(G)
print(f"Average clustering coefficient: {avg_clustering:.4f}")

# Clustering coefficient cho từng node
clustering_coeffs = nx.clustering(G)
clustering_values = list(clustering_coeffs.values())

print(f"Clustering coefficient trung vị: {np.median(clustering_values):.4f}")
print(f"Clustering coefficient min: {np.min(clustering_values):.4f}")
print(f"Clustering coefficient max: {np.max(clustering_values):.4f}")
print(f"Độ lệch chuẩn: {np.std(clustering_values):.4f}")

# Số nodes có clustering = 0 (không có tam giác)
zero_clustering = sum(1 for c in clustering_values if c == 0)
print(f"Số nodes có clustering = 0: {zero_clustering} ({zero_clustering/len(clustering_values)*100:.2f}%)")

# Top 10 nodes có clustering cao nhất (và degree > 1)
top_clustering = sorted(
    [(n, c, G.degree(n)) for n, c in clustering_coeffs.items() if G.degree(n) > 1],
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTop 10 tác giả có clustering coefficient cao nhất:")
for i, (node_id, clustering, degree) in enumerate(top_clustering, 1):
    author_name = authors.get(node_id, f"ID_{node_id}")
    print(f"  {i:2d}. {author_name[:50]:50s} | CC: {clustering:.4f} | Degree: {degree}")

# Vẽ clustering coefficient distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(clustering_values, bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Clustering Coefficient', fontsize=12)
plt.ylabel('Số lượng nodes', fontsize=12)
plt.title('Clustering Coefficient Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Clustering vs Degree
plt.subplot(1, 2, 2)
node_degrees = [G.degree(n) for n in G.nodes()]
node_clustering = [clustering_coeffs[n] for n in G.nodes()]
plt.scatter(node_degrees, node_clustering, alpha=0.3, s=10)
plt.xlabel('Degree', fontsize=12)
plt.ylabel('Clustering Coefficient', fontsize=12)
plt.title('Clustering Coefficient vs Degree', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_coefficient.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Đã lưu: {OUTPUT_DIR}/clustering_coefficient.png")

# Lưu clustering coefficient ra CSV
clustering_df = pd.DataFrame([
    {
        'Author_ID': node_id,
        'Author_Name': authors.get(node_id, ''),
        'Clustering_Coefficient': clustering_coeffs[node_id],
        'Degree': G.degree(node_id)
    }
    for node_id in G.nodes()
]).sort_values('Clustering_Coefficient', ascending=False)
clustering_df.to_csv(os.path.join(OUTPUT_DIR, 'clustering_coefficient.csv'), index=False, encoding='utf-8-sig')
print(f" Đã lưu: {OUTPUT_DIR}/clustering_coefficient.csv")

# ============================================================
# TỔNG KẾT
# ============================================================
print("\n" + "="*60)
print("TỔNG KẾT")
print("="*60)
print(f"Số nodes: {G.number_of_nodes()}")
print(f"Số edges: {G.number_of_edges()}")
print(f"Mật độ đồ thị: {nx.density(G):.6f}")
print(f"Degree trung bình: {np.mean(degrees):.2f}")
print(f"Average clustering coefficient: {avg_clustering:.4f}")
print(f"Số connected components: {len(components)}")
print(f"Component lớn nhất: {max(component_sizes)} nodes ({max(component_sizes)/G.number_of_nodes()*100:.2f}%)")

print("\n" + "="*60)
print("CÁC FILE ĐÃ TẠO")
print("="*60)
print(f"\nThư mục: {OUTPUT_DIR}/")
print("\nĐồ thị:")
print("  - coauthor_graph.gexf (GEXF - mở bằng Gephi)")
print("  - coauthor_graph.graphml (GraphML)")
print("  - coauthor_graph.pkl (Python pickle)")
print("\nDữ liệu CSV:")
print("  - degree_distribution.csv")
print("  - connected_components.csv")
print("  - clustering_coefficient.csv")
print("\nHình ảnh:")
print("  - degree_distribution.png")
print("  - connected_components.png")
print("  - clustering_coefficient.png")

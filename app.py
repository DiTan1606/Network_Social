import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# --- CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="Co-author Network Analysis", page_icon="🌐")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; text-align: center; }
    .main-header p { color: #b8d4ff; text-align: center; margin: 0.5rem 0 0 0; font-size: 0.95rem; }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label { color: #e0e0e0 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: white !important; font-weight: bold; }
    
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #4fc3f7 !important; }
    
    .tooltip-box {
        background: rgba(30, 60, 114, 0.95);
        border: 1px solid #4fc3f7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .tooltip-box h4 { color: #4fc3f7; margin: 0 0 0.5rem 0; }
    .tooltip-box p { color: #e0e0e0; margin: 0.3rem 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🌐 Co-author Network Analysis Dashboard</h1>
    <p>Multi-level Visualization | Community Detection | Bridge Analysis | Link Prediction</p>
</div>
""", unsafe_allow_html=True)

# --- LOAD DỮ LIỆU ---
@st.cache_data
def load_graph():
    try:
        G = nx.read_gexf('graph_with_time.gexf')
        return G
    except FileNotFoundError:
        st.error("Không tìm thấy file 'graph_with_time.gexf'!")
        return None

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv('predictions.csv')
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def compute_community_stats(_G):
    """Tính toán thống kê cho từng community"""
    comm_stats = defaultdict(lambda: {
        'nodes': [], 'size': 0, 'internal_edges': 0,
        'external_edges': 0, 'top_bridges': [], 'avg_betweenness': 0
    })
    
    # Gom nodes theo community
    for n, d in _G.nodes(data=True):
        comm = d.get('louvain_community', 0)
        comm_stats[comm]['nodes'].append(n)
        comm_stats[comm]['size'] += 1
    
    # Tính edges và bridges
    for comm_id, stats in comm_stats.items():
        nodes_set = set(stats['nodes'])
        betweenness_list = []
        
        for n in stats['nodes']:
            node_data = _G.nodes[n]
            betweenness_list.append((n, node_data.get('betweenness', 0), node_data.get('label', n)))
            
            for neighbor in _G.neighbors(n):
                neighbor_comm = _G.nodes[neighbor].get('louvain_community', 0)
                if neighbor_comm == comm_id:
                    stats['internal_edges'] += 1
                else:
                    stats['external_edges'] += 1
        
        stats['internal_edges'] //= 2  # Đếm 2 lần
        stats['avg_betweenness'] = sum(b for _, b, _ in betweenness_list) / len(betweenness_list) if betweenness_list else 0
        stats['top_bridges'] = sorted(betweenness_list, key=lambda x: -x[1])[:5]
    
    return dict(comm_stats)

@st.cache_data
def build_meta_graph(_G, comm_stats):
    """Xây dựng meta-graph: mỗi community là 1 node"""
    meta_G = nx.Graph()
    
    # Thêm community nodes
    for comm_id, stats in comm_stats.items():
        meta_G.add_node(comm_id, 
                        size=stats['size'],
                        internal_edges=stats['internal_edges'],
                        external_edges=stats['external_edges'],
                        avg_betweenness=stats['avg_betweenness'],
                        top_bridges=stats['top_bridges'])
    
    # Thêm edges giữa communities
    comm_edges = defaultdict(int)
    for u, v in _G.edges():
        comm_u = _G.nodes[u].get('louvain_community', 0)
        comm_v = _G.nodes[v].get('louvain_community', 0)
        if comm_u != comm_v:
            key = tuple(sorted([comm_u, comm_v]))
            comm_edges[key] += 1
    
    for (c1, c2), weight in comm_edges.items():
        meta_G.add_edge(c1, c2, weight=weight)
    
    return meta_G

G_full = load_graph()
df_pred = load_predictions()

if G_full:
    comm_stats = compute_community_stats(G_full)
    meta_G = build_meta_graph(G_full, comm_stats)
    
    # --- METRICS (Số liệu thực tế của toàn bộ dataset) ---
    total_nodes = 166314
    total_edges = 2206369
    total_communities = 9345
    avg_degree = 26.53
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Tổng Tác giả", f"{total_nodes:,}")
    m2.metric("🔗 Tổng Kết nối", f"{total_edges:,}")
    m3.metric("🏘️ Số Cộng đồng", f"{total_communities:,}")
    m4.metric("📈 Degree TB", f"{avg_degree:.2f}")
    
    st.markdown("---")

    # --- SIDEBAR ---
    st.sidebar.header("🎛️ Điều khiển")
    
    # Chọn Level hiển thị
    view_level = st.sidebar.radio(
        "📊 Chế độ xem:",
        ["🌍 Level 1: Tổng quan Communities", 
         "🏘️ Level 2: Chi tiết Community", 
         "👤 Level 3: Focus Tác giả"],
        help="Chọn mức độ chi tiết để khám phá mạng lưới"
    )
    
    # --- GIẢI THÍCH CHỈ SỐ ---
    with st.sidebar.expander("📖 Giải thích chỉ số", expanded=False):
        st.markdown("""
        <div class="tooltip-box">
            <h4>🔗 Betweenness Centrality</h4>
            <p>Đo lường mức độ "cầu nối" của một tác giả. Giá trị cao = nằm trên nhiều đường đi ngắn nhất giữa các tác giả khác → quan trọng trong việc kết nối các nhóm nghiên cứu.</p>
        </div>
        <div class="tooltip-box">
            <h4>🏘️ Louvain Community</h4>
            <p>Thuật toán phát hiện cộng đồng dựa trên tối ưu hóa modularity. Các tác giả trong cùng community có xu hướng hợp tác chặt chẽ với nhau hơn.</p>
        </div>
        <div class="tooltip-box">
            <h4>📊 Modularity</h4>
            <p>Đo chất lượng phân chia community. Giá trị cao (gần 1) = cấu trúc community rõ ràng, các nhóm tách biệt tốt.</p>
        </div>
        <div class="tooltip-box">
            <h4>🌉 Bridge Authors</h4>
            <p>Tác giả có betweenness cao, kết nối nhiều community khác nhau. Họ thường là những người có nghiên cứu liên ngành hoặc hợp tác rộng.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Lọc thời gian
    all_years = set()
    for u, v, data in G_full.edges(data=True):
        y_str = data.get('years', '')
        if y_str and y_str != 'Unknown':
            for y in y_str.split(','):
                try:
                    all_years.add(int(y))
                except:
                    pass
    sorted_years = sorted(list(all_years))
    
    with st.sidebar.expander("⏰ Lọc thời gian", expanded=False):
        time_filter = st.select_slider(
            "Chọn năm:",
            options=["Tất cả"] + sorted_years,
            value="Tất cả"
        )
    
    # Áp dụng filter thời gian
    if time_filter != "Tất cả":
        edges_in_year = [(u, v) for u, v, d in G_full.edges(data=True) 
                         if str(time_filter) in d.get('years', '').split(',')]
        G_filtered = G_full.edge_subgraph(edges_in_year).copy()
    else:
        G_filtered = G_full
    
    # Build name mapping
    name_to_id = {d.get('label', n): n for n, d in G_filtered.nodes(data=True)}
    id_to_name = {n: d.get('label', n) for n, d in G_filtered.nodes(data=True)}

    # ==========================================
    # LEVEL 1: TỔNG QUAN COMMUNITIES (Meta-graph)
    # ==========================================
    if "Level 1" in view_level:
        st.subheader("🌍 Tổng quan: Mỗi node = 1 Community")
        
        # Option hiển thị bridges ở giữa
        show_bridge_center = st.sidebar.checkbox("🌉 Hiện Top Bridges ở giữa", value=True,
                                                  help="Hiển thị các tác giả cầu nối quan trọng nhất ở trung tâm")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
            colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
            
            if show_bridge_center:
                # === CHẾ ĐỘ BRIDGES Ở GIỮA ===
                import math
                
                # Lấy top 10 bridges từ toàn bộ graph
                all_bridges = sorted(
                    [(n, d.get('betweenness', 0), d.get('label', n), d.get('louvain_community', 0)) 
                     for n, d in G_filtered.nodes(data=True)],
                    key=lambda x: -x[1]
                )[:10]
                
                # Tính các communities mà mỗi bridge kết nối tới
                bridge_connections = {}
                for node_id, betw, name, own_comm in all_bridges:
                    connected_comms = set()
                    connected_comms.add(own_comm)  # Community của chính họ
                    # Tìm tất cả communities của đồng tác giả
                    for neighbor in G_filtered.neighbors(node_id):
                        neighbor_comm = G_filtered.nodes[neighbor].get('louvain_community', 0)
                        connected_comms.add(neighbor_comm)
                    bridge_connections[node_id] = {
                        'name': name,
                        'betweenness': betw,
                        'own_comm': own_comm,
                        'connected_comms': connected_comms
                    }
                
                # Tính vị trí: Communities xếp vòng tròn, Bridges ở giữa
                num_comms = len(meta_G.nodes())
                radius = 400
                
                # Vẽ community nodes theo vòng tròn
                for i, node in enumerate(meta_G.nodes()):
                    angle = 2 * math.pi * i / num_comms
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    
                    data = meta_G.nodes[node]
                    size = min(data['size'] * 2, 100)
                    color = colors[node % len(colors)]
                    
                    bridges_info = "\n".join([f"  • {name}: {score:.4f}" 
                                              for _, score, name in data['top_bridges'][:3]])
                    
                    title = f"""Community {node}
━━━━━━━━━━━━━━━━━━━━
👥 Số tác giả: {data['size']}
🔗 Kết nối nội bộ: {data['internal_edges']}
🌉 Kết nối ra ngoài: {data['external_edges']}
📊 Betweenness TB: {data['avg_betweenness']:.6f}

🏆 Top Bridges:
{bridges_info}"""
                    
                    net.add_node(f"comm_{node}", 
                                label=f"C{node}\n({data['size']})",
                                title=title,
                                size=size,
                                color=color,
                                shape='dot',
                                x=x, y=y,
                                font={'size': 14, 'color': 'white'})
                
                # Vẽ bridges ở giữa (cluster nhỏ quanh tâm)
                for i, (node_id, betw, name, comm) in enumerate(all_bridges):
                    angle = 2 * math.pi * i / len(all_bridges)
                    x = 80 * math.cos(angle)
                    y = 80 * math.sin(angle)
                    
                    conn_info = bridge_connections[node_id]
                    num_connected = len(conn_info['connected_comms'])
                    comms_list = ", ".join([f"C{c}" for c in sorted(conn_info['connected_comms'])])
                    
                    title = f"""🌉 BRIDGE AUTHOR
━━━━━━━━━━━━━━━━━━━━
👤 {name}
🏘️ Community gốc: {comm}
🔗 Betweenness: {betw:.6f}
🌐 Kết nối {num_connected} communities:
   {comms_list}

Tác giả này là cầu nối giữa
{num_connected} nhóm nghiên cứu khác nhau."""
                    
                    net.add_node(f"bridge_{node_id}",
                                label=f"⭐{name}",
                                title=title,
                                size=20 + betw * 800,
                                color={'background': '#FFD700', 'border': '#FF4500'},
                                shape='star',
                                x=x, y=y,
                                borderWidth=3,
                                font={'size': 11, 'color': 'white', 'strokeWidth': 2, 'strokeColor': 'black'})
                    
                    # Kết nối bridge với TẤT CẢ communities mà họ có đồng tác giả
                    for connected_comm in conn_info['connected_comms']:
                        if f"comm_{connected_comm}" in [n['id'] for n in net.nodes]:
                            # Màu khác nhau: vàng cho community gốc, cam cho các community khác
                            edge_color = '#FFD700' if connected_comm == comm else '#FF6B6B'
                            edge_width = 3 if connected_comm == comm else 2
                            net.add_edge(f"bridge_{node_id}", f"comm_{connected_comm}",
                                        color={'color': edge_color, 'opacity': 0.7},
                                        width=edge_width,
                                        dashes=True,
                                        title=f"{'Community gốc' if connected_comm == comm else 'Có đồng tác giả'}")
                
                # Edges giữa communities
                max_weight = max((d['weight'] for _, _, d in meta_G.edges(data=True)), default=1)
                for u, v, d in meta_G.edges(data=True):
                    width = (d['weight'] / max_weight) * 8
                    net.add_edge(f"comm_{u}", f"comm_{v}", 
                                width=width,
                                title=f"Kết nối C{u} ↔ C{v}: {d['weight']} edges",
                                color={'color': '#ffffff', 'opacity': 0.2})
                
                net.set_options("""
                {
                    "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true, "dragView": true},
                    "physics": {
                        "enabled": true,
                        "barnesHut": {"gravitationalConstant": -2000, "springLength": 150, "damping": 0.9},
                        "maxVelocity": 3, "minVelocity": 0.1,
                        "stabilization": {"enabled": true, "iterations": 150}
                    }
                }
                """)
            
            else:
                # === CHẾ ĐỘ BÌNH THƯỜNG (chỉ communities) ===
                for node in meta_G.nodes():
                    data = meta_G.nodes[node]
                    size = min(data['size'] * 2, 100)
                    color = colors[node % len(colors)]
                    
                    bridges_info = "\n".join([f"  • {name}: {score:.4f}" 
                                              for _, score, name in data['top_bridges'][:3]])
                    
                    title = f"""Community {node}
━━━━━━━━━━━━━━━━━━━━
👥 Số tác giả: {data['size']}
🔗 Kết nối nội bộ: {data['internal_edges']}
🌉 Kết nối ra ngoài: {data['external_edges']}
📊 Betweenness TB: {data['avg_betweenness']:.6f}

🏆 Top Bridges:
{bridges_info}"""
                    
                    net.add_node(node, 
                                label=f"C{node}\n({data['size']})",
                                title=title,
                                size=size,
                                color=color,
                                shape='dot',
                                font={'size': 14, 'color': 'white'})
                
                max_weight = max((d['weight'] for _, _, d in meta_G.edges(data=True)), default=1)
                for u, v, d in meta_G.edges(data=True):
                    width = (d['weight'] / max_weight) * 10
                    net.add_edge(u, v, 
                                width=width,
                                title=f"Kết nối giữa C{u} ↔ C{v}: {d['weight']} edges",
                                color={'color': '#ffffff', 'opacity': 0.3})
                
                net.barnes_hut(gravity=-3000, spring_length=200)
                net.set_options("""
                {
                    "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true, "dragView": true},
                    "physics": {
                        "enabled": true,
                        "barnesHut": {"gravitationalConstant": -3000, "springLength": 200, "damping": 0.95},
                        "maxVelocity": 5, "minVelocity": 0.1,
                        "stabilization": {"enabled": true, "iterations": 200}
                    }
                }
                """)
            
            html = net.generate_html()
            components.html(html, height=620)
        
        with col2:
            st.markdown("### 📊 Thống kê Communities")
            
            # Bảng top communities
            comm_df = pd.DataFrame([
                {
                    'Community': f"C{cid}",
                    'Số tác giả': stats['size'],
                    'Edges nội bộ': stats['internal_edges'],
                    'Edges ra ngoài': stats['external_edges']
                }
                for cid, stats in sorted(comm_stats.items(), key=lambda x: -x[1]['size'])[:10]
            ])
            st.dataframe(comm_df, hide_index=True, use_container_width=True)
            
            # Pie chart
            st.markdown("### 🥧 Phân bố kích thước")
            sizes = [stats['size'] for stats in comm_stats.values()]
            labels = [f"C{cid}" for cid in comm_stats.keys()]
            
            fig = px.pie(values=sizes[:15], names=labels[:15], hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20),
                             showlegend=True, legend=dict(orientation="h", y=-0.2))
            fig.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Tip:** Hover vào node để xem chi tiết. Click và kéo để di chuyển. Scroll để zoom.")

    # ==========================================
    # LEVEL 2: CHI TIẾT COMMUNITY
    # ==========================================
    elif "Level 2" in view_level:
        st.subheader("🏘️ Chi tiết Community")
        
        # Chọn community
        comm_options = sorted(comm_stats.keys(), key=lambda x: -comm_stats[x]['size'])
        selected_comm = st.sidebar.selectbox(
            "Chọn Community:",
            options=comm_options,
            format_func=lambda x: f"Community {x} ({comm_stats[x]['size']} tác giả)"
        )
        
        # Lấy subgraph của community
        comm_nodes = [n for n, d in G_filtered.nodes(data=True) 
                      if d.get('louvain_community') == selected_comm]
        G_comm = G_filtered.subgraph(comm_nodes).copy()
        
        # Thêm bridge connections (edges ra ngoài community)
        show_bridges = st.sidebar.checkbox("Hiện kết nối ra ngoài (bridges)", value=True)
        
        if show_bridges:
            bridge_nodes = set()
            for n in comm_nodes:
                for neighbor in G_filtered.neighbors(n):
                    if G_filtered.nodes[neighbor].get('louvain_community') != selected_comm:
                        bridge_nodes.add(neighbor)
            
            # Thêm bridge nodes (giới hạn để không quá nặng)
            bridge_nodes = list(bridge_nodes)[:50]
            extended_nodes = comm_nodes + bridge_nodes
            G_comm = G_filtered.subgraph(extended_nodes).copy()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
            colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
            
            for n, d in G_comm.nodes(data=True):
                label = d.get('label', str(n))
                comm = d.get('louvain_community', 0)
                betweenness = d.get('betweenness', 0)
                
                title = f"""👤 {label}
━━━━━━━━━━━━━━━━━━━━
🏘️ Community: {comm}
🔗 Betweenness: {betweenness:.6f}
📊 Degree: {G_comm.degree(n)}

{'🌉 BRIDGE AUTHOR' if comm != selected_comm else ''}
{'(Kết nối từ community khác)' if comm != selected_comm else ''}"""
                
                # Styling
                if comm == selected_comm:
                    # Node trong community chính
                    size = max(betweenness * 5000, 15)
                    color = colors[comm % len(colors)]
                    
                    # Highlight top bridges
                    top_bridge_ids = [bid for bid, _, _ in comm_stats[selected_comm]['top_bridges']]
                    if n in top_bridge_ids:
                        net.add_node(n, label=f"⭐{label}", title=title, size=size*1.2,
                                    color={'background': '#FFD700', 'border': '#FF4500'},
                                    borderWidth=3, font={'size': 12, 'color': 'white'})
                    else:
                        net.add_node(n, label=label, title=title, size=size, color=color)
                else:
                    # Bridge node từ community khác
                    net.add_node(n, label=label, title=title, size=20,
                                color={'background': '#555555', 'border': '#888888'},
                                shape='diamond', font={'size': 10, 'color': '#aaaaaa'})
            
            # Edges
            for u, v, d in G_comm.edges(data=True):
                comm_u = G_comm.nodes[u].get('louvain_community')
                comm_v = G_comm.nodes[v].get('louvain_community')
                
                if comm_u == selected_comm and comm_v == selected_comm:
                    # Internal edge
                    net.add_edge(u, v, color={'color': colors[selected_comm % len(colors)], 'opacity': 0.5})
                else:
                    # Bridge edge
                    net.add_edge(u, v, color={'color': '#ff6b6b', 'opacity': 0.8}, 
                                dashes=True, width=2,
                                title="🌉 Kết nối liên community")
            
            net.barnes_hut(gravity=-2000, spring_length=150)
            html = net.generate_html()
            components.html(html, height=620)
        
        with col2:
            stats = comm_stats[selected_comm]
            
            st.markdown(f"### 📈 Community {selected_comm}")
            st.metric("Số tác giả", stats['size'])
            st.metric("Kết nối nội bộ", stats['internal_edges'])
            st.metric("Kết nối ra ngoài", stats['external_edges'])
            
            # Tỷ lệ kết nối
            total_conn = stats['internal_edges'] + stats['external_edges']
            if total_conn > 0:
                internal_ratio = stats['internal_edges'] / total_conn * 100
                st.progress(internal_ratio / 100, text=f"Nội bộ: {internal_ratio:.1f}%")
            
            st.markdown("### 🏆 Top Bridges")
            bridges_df = pd.DataFrame([
                {'Tên': name, 'Betweenness': f"{score:.6f}"}
                for _, score, name in stats['top_bridges']
            ])
            st.dataframe(bridges_df, hide_index=True, use_container_width=True)
            
            st.markdown("""
            <div class="tooltip-box">
                <h4>💡 Gợi ý</h4>
                <p>⭐ = Top bridge trong community</p>
                <p>◆ = Tác giả từ community khác</p>
                <p>--- = Kết nối liên community</p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # LEVEL 3: FOCUS TÁC GIẢ (Ego Network)
    # ==========================================
    elif "Level 3" in view_level:
        st.subheader("👤 Focus Tác giả - Ego Network")
        
        # Search tác giả
        all_names = sorted(name_to_id.keys())
        selected_author = st.sidebar.selectbox(
            "🔍 Tìm tác giả:",
            options=["-- Chọn tác giả --"] + all_names,
            help="Gõ tên để tìm kiếm"
        )
        
        # Depth của ego network
        ego_depth = st.sidebar.slider("Độ sâu mạng lưới:", 1, 3, 1,
                                      help="1 = chỉ kết nối trực tiếp, 2 = bạn của bạn, ...")
        
        if selected_author != "-- Chọn tác giả --":
            center_id = name_to_id.get(selected_author)
            
            if center_id and center_id in G_filtered:
                # Build ego network
                G_ego = nx.ego_graph(G_filtered, center_id, radius=ego_depth)
                
                # Thêm predicted edges
                predicted_edges = []
                if not df_pred.empty:
                    my_preds = df_pred[df_pred['Source'] == selected_author]
                    for _, row in my_preds.iterrows():
                        target_name = row['Target']
                        target_id = name_to_id.get(target_name)
                        if target_id and target_id in G_filtered.nodes():
                            if not G_ego.has_node(target_id):
                                G_ego.add_node(target_id, **G_filtered.nodes[target_id])
                            if not G_ego.has_edge(center_id, target_id):
                                G_ego.add_edge(center_id, target_id, 
                                              type='predicted',
                                              score=row['Score'],
                                              model=row['Model'])
                                predicted_edges.append((target_name, row['Score'], row['Model']))
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
                    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
                    
                    # Tính distance từ center
                    distances = nx.single_source_shortest_path_length(G_ego, center_id)
                    
                    for n, d in G_ego.nodes(data=True):
                        label = d.get('label', str(n))
                        comm = d.get('louvain_community', 0)
                        betweenness = d.get('betweenness', 0)
                        dist = distances.get(n, 99)
                        
                        title = f"""👤 {label}
━━━━━━━━━━━━━━━━━━━━
🏘️ Community: {comm}
🔗 Betweenness: {betweenness:.6f}
📏 Khoảng cách: {dist} bước
📊 Degree (trong view): {G_ego.degree(n)}"""
                        
                        if n == center_id:
                            # Center node - highlight đặc biệt
                            net.add_node(n, 
                                        label=f"⭐ {label}",
                                        title=title,
                                        size=45,
                                        color={'background': '#FFD700', 'border': '#FF4500',
                                               'highlight': {'background': '#FFFF00', 'border': '#FF0000'}},
                                        shape='star',
                                        borderWidth=5,
                                        font={'size': 18, 'color': 'white', 'strokeWidth': 2, 'strokeColor': 'black'})
                        else:
                            # Các node khác - size theo distance
                            size = max(30 - dist * 8, 10)
                            opacity = 1 - dist * 0.2
                            net.add_node(n, label=label, title=title, size=size,
                                        color=colors[comm % len(colors)],
                                        font={'size': 10, 'color': f'rgba(255,255,255,{opacity})'})
                    
                    # Edges
                    for u, v, d in G_ego.edges(data=True):
                        if d.get('type') == 'predicted':
                            # Predicted edge - nét đứt đỏ
                            score = d.get('score', 0)
                            model = d.get('model', 'Unknown')
                            net.add_edge(u, v, 
                                        color='#ff4757',
                                        dashes=True,
                                        width=3,
                                        title=f"🔮 DỰ BÁO\nModel: {model}\nScore: {score:.6f}")
                        else:
                            # Existing edge
                            years = d.get('years', '')
                            net.add_edge(u, v, 
                                        color={'color': '#4fc3f7', 'opacity': 0.5},
                                        title=f"Năm hợp tác: {years}" if years else "")
                    
                    net.barnes_hut(gravity=-2500, spring_length=180)
                    html = net.generate_html()
                    components.html(html, height=620)
                
                with col2:
                    # Thông tin tác giả
                    author_data = G_filtered.nodes[center_id]
                    
                    st.markdown(f"### 👤 {selected_author}")
                    st.metric("Community", author_data.get('louvain_community', 'N/A'))
                    st.metric("Betweenness", f"{author_data.get('betweenness', 0):.6f}")
                    st.metric("Số đồng tác giả", G_filtered.degree(center_id))
                    
                    # Danh sách đồng tác giả
                    st.markdown("### 👥 Đồng tác giả")
                    coauthors = []
                    for neighbor in G_filtered.neighbors(center_id):
                        n_data = G_filtered.nodes[neighbor]
                        coauthors.append({
                            'Tên': n_data.get('label', neighbor),
                            'Community': n_data.get('louvain_community', 'N/A')
                        })
                    
                    if coauthors:
                        st.dataframe(pd.DataFrame(coauthors[:15]), hide_index=True, use_container_width=True)
                    
                    # Dự báo
                    if predicted_edges:
                        st.markdown("### 🔮 Dự báo kết nối")
                        pred_df = pd.DataFrame([
                            {'Tác giả': name, 'Score': f"{score:.4f}", 'Model': model}
                            for name, score, model in predicted_edges[:10]
                        ])
                        st.dataframe(pred_df, hide_index=True, use_container_width=True)
                        
                        st.markdown("""
                        <div class="tooltip-box">
                            <h4>🔮 Về Link Prediction</h4>
                            <p>Dự báo khả năng hợp tác trong tương lai dựa trên cấu trúc mạng lưới hiện tại.</p>
                            <p>Score cao = khả năng cao sẽ có bài báo chung.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy tác giả trong dữ liệu.")
        else:
            st.info("👈 Chọn một tác giả từ sidebar để xem ego network.")
            
            # Hiển thị top bridges khi chưa chọn ai
            st.markdown("### 🏆 Top Bridge Authors (Gợi ý)")
            top_bridges = sorted(
                [(n, d.get('betweenness', 0), d.get('label', n), d.get('louvain_community', 0)) 
                 for n, d in G_filtered.nodes(data=True)],
                key=lambda x: -x[1]
            )[:20]
            
            bridges_df = pd.DataFrame([
                {'Tên': name, 'Betweenness': f"{score:.6f}", 'Community': comm}
                for _, score, name, comm in top_bridges
            ])
            st.dataframe(bridges_df, hide_index=True, use_container_width=True)

else:
    st.error("Không thể load dữ liệu. Kiểm tra file graph_with_time.gexf")

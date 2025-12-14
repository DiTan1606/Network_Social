import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px

# --- CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="Co-author Communities & Bridges Dashboard", page_icon="🌐")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        text-align: center;
    }
    .main-header p {
        color: #b8d4ff;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #e0e0e0 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #4fc3f7 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(79, 195, 247, 0.1);
        border-radius: 8px;
    }
    
    /* Card container */
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🌐 Co-author Communities & Bridges Dashboard</h1>
    <p>Phân tích mạng lưới đồng tác giả | Khám phá cộng đồng | Dự báo kết nối</p>
</div>
""", unsafe_allow_html=True)

# --- 1. LOAD DỮ LIỆU ---
@st.cache_data
def load_graph():
    try:
        G = nx.read_gexf('graph_with_time.gexf')
        return G
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file 'graph_with_time.gexf'. Hãy chạy script xử lý dữ liệu trước!")
        return None

@st.cache_data
def load_predictions():
    try:
        # Đọc file CSV dự báo
        df = pd.read_csv('predictions.csv')
        return df
    except FileNotFoundError:
        return pd.DataFrame() # Trả về bảng rỗng nếu chưa có file

G_full = load_graph()
df_pred = load_predictions()

if G_full:
    # ==========================================
    # 📊 METRICS ROW - THỐNG KÊ TỔNG QUAN
    # ==========================================
    total_nodes = G_full.number_of_nodes()
    total_edges = G_full.number_of_edges()
    total_communities = len(set(d.get('louvain_community', 0) for _, d in G_full.nodes(data=True)))
    avg_degree = sum(dict(G_full.degree()).values()) / total_nodes if total_nodes > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Tổng Tác giả", f"{total_nodes:,}")
    m2.metric("🔗 Tổng Kết nối", f"{total_edges:,}")
    m3.metric("🏘️ Số Cộng đồng", total_communities)
    m4.metric("📈 Degree TB", f"{avg_degree:.1f}")
    
    st.markdown("---")
    
    # ==========================================
    # 🌩️ SIDEBAR: CẤU HÌNH THEO THỨ TỰ MỚI
    # ==========================================
    st.sidebar.header("🎛️ Bộ lọc hiển thị")

    # ----------------------------------------
    # 1. LỌC THỜI GIAN (Năm)
    # ----------------------------------------
    all_years = set()
    for u, v, data in G_full.edges(data=True):
        y_str = data.get('years', '')
        if y_str and y_str != 'Unknown':
            for y in y_str.split(','):
                all_years.add(int(y))

    sorted_years = sorted(list(all_years))
    time_options = ["Toàn thời gian"] + [str(y) for y in sorted_years]

    with st.sidebar.expander("⏰ 1. Mốc Thời Gian", expanded=True):
        selected_time = st.radio("Thời gian:", options=time_options, horizontal=True, label_visibility="collapsed")

    # -> XỬ LÝ LOGIC LỌC NĂM
    if selected_time != "Toàn thời gian":
        edges_in_year = []
        for u, v, data in G_full.edges(data=True):
            y_str = data.get('years', '')
            if selected_time in y_str.split(','):
                edges_in_year.append((u, v))
        G_time = G_full.edge_subgraph(edges_in_year).copy()
    else:
        G_time = G_full.copy()

    # ----------------------------------------
    # 2. LỌC CỘNG ĐỒNG
    # ----------------------------------------
    if G_time.number_of_nodes() > 0:
        available_comms = set()
        for n, d in G_time.nodes(data=True):
            if 'louvain_community' in d:
                available_comms.add(d['louvain_community'])
        sorted_comms = sorted(list(available_comms))
    else:
        sorted_comms = []

    with st.sidebar.expander("🏘️ 2. Chọn Cộng đồng", expanded=True):
        all_comms_selected = st.checkbox("Chọn tất cả cộng đồng", value=True)

        if all_comms_selected:
            selected_comms = sorted_comms
        else:
            selected_comms = st.multiselect(
                "Chọn nhóm cụ thể:",
                options=sorted_comms,
                default=sorted_comms[:3] if len(sorted_comms) > 3 else sorted_comms
            )

    # -> XỬ LÝ LOGIC LỌC CỘNG ĐỒNG
    nodes_in_comm = [n for n, d in G_time.nodes(data=True) if d.get('louvain_community') in selected_comms]
    G_comm = G_time.subgraph(nodes_in_comm).copy()

    # ----------------------------------------
    # 3. LỌC TÁC GIẢ (Focus Mode)
    # ----------------------------------------
    name_to_id = {}
    current_names = []
    for n, data in G_comm.nodes(data=True):
        label = data.get('label', str(n))
        name_to_id[label] = n
        current_names.append(label)

    list_names = ["-- Xem Tổng Quan --"] + sorted(list(set(current_names)))

    with st.sidebar.expander("🔍 3. Tìm & Focus Tác giả", expanded=True):
        selected_author = st.selectbox("Gõ tên để Focus:", list_names)

    # ----------------------------------------
    # 4. CHỌN HIỂN THỊ TOP N (Chỉ dùng cho Tổng quan)
    # ----------------------------------------
    if selected_author == "-- Xem Tổng Quan --":
        with st.sidebar.expander("📊 4. Giới hạn hiển thị", expanded=True):
            top_n = st.slider("Số lượng tác giả (Top Betweenness)",
                              min_value=10, max_value=1000, value=100, step=10)
    else:
        st.sidebar.info("🎯 Đang ở chế độ Focus Tác giả")

    # ==========================================
    # ⚙️ XỬ LÝ GRAPH CUỐI CÙNG ĐỂ VẼ (G_VIZ)
    # ==========================================

    G_viz = None

    # TH1: Chế độ Focus Tác giả
    if selected_author != "-- Xem Tổng Quan --":
        center_id = name_to_id.get(selected_author)

        if center_id and center_id in G_comm:
            # 1. Lấy mạng lưới hiện tại (Quá khứ/Hiện tại)
            neighbors = list(G_comm.neighbors(center_id))
            ego_nodes = neighbors + [center_id]
            G_viz = G_comm.subgraph(ego_nodes).copy()
            
            # 2. Lấy dữ liệu DỰ BÁO (Tương lai)
            if not df_pred.empty:
                # Tìm các dòng mà Source là tác giả đang chọn
                my_preds = df_pred[df_pred['Source'] == selected_author]
                
                for _, row in my_preds.iterrows():
                    target_name = row['Target']
                    score = row['Score']
                    model_name = row['Model']
                    
                    # Tìm ID của người được dự báo
                    target_id = name_to_id.get(target_name)
                    
                    if target_id:
                        # Nếu node chưa có trong G_viz thì thêm vào
                        if not G_viz.has_node(target_id):
                            # Copy thông tin node từ G_full để có đủ label, group...
                            if G_full.has_node(target_id):
                                G_viz.add_node(target_id, **G_full.nodes[target_id])
                            else:
                                G_viz.add_node(target_id, label=target_name, group=99) # Fallback
                        
                        # THÊM CẠNH DỰ BÁO (Đánh dấu type='future')
                        if not G_viz.has_edge(center_id, target_id):
                            G_viz.add_edge(center_id, target_id, 
                                           title=f"Dự báo: {model_name}\nScore: {score:.4f}", 
                                           type='future')

            st.success(f"🔍 Đang focus vào: **{selected_author}**")
        else:
            st.warning("Tác giả không tìm thấy trong bộ lọc hiện tại.")
            G_viz = nx.Graph()

    # TH2: Chế độ Tổng quan (Áp dụng Top N)
    else:
        nodes_sorted = sorted(G_comm.nodes(data=True),
                              key=lambda x: x[1].get('betweenness', 0),
                              reverse=True)
        top_node_ids = [n[0] for n in nodes_sorted[:top_n]]
        G_viz = G_comm.subgraph(top_node_ids).copy()

    # ==========================================
    # 🎨 VẼ GIAO DIỆN CHÍNH
    # ==========================================
    col1, col2 = st.columns([3, 1])

    with col1:
        if G_viz and G_viz.number_of_nodes() > 0:
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

            for n, d in G_viz.nodes(data=True):
                label = d.get('label', str(n))
                group = d.get('louvain_community', 0)
                title = f"{label}\nGroup: {group}\nScore: {d.get('betweenness', 0):.4f}"

                # ⭐ HIGHLIGHT NODE CHÍNH KHI FOCUS
                if selected_author != "-- Xem Tổng Quan --" and label == selected_author:
                    net.add_node(n,
                                 label=f"⭐ {label}",
                                 title=title,
                                 shape='star',
                                 size=70,
                                 color={
                                     'background': '#FFD700',
                                     'border': '#FF4500',
                                     'highlight': {'background': '#FFFF00', 'border': '#FF0000'}
                                 },
                                 borderWidth=5,
                                 font={'size': 20, 'color': 'white', 'strokeWidth': 3, 'strokeColor': 'black'},
                                 group=group)
                else:
                    size = d.get('betweenness', 0.01) * 3000
                    if size < 10:
                        size = 10
                    net.add_node(n, label=label, title=title, value=size, group=group)

            # --- VẼ CẠNH ---
            for u, v, d in G_viz.edges(data=True):
                # Kiểm tra xem đây là cạnh thường hay dự báo
                if d.get('type') == 'future':
                    # Cấu hình nét đứt (dashes) và màu nổi bật
                    net.add_edge(u, v, 
                                 title=d.get('title', ''), 
                                 color='red', 
                                 dashes=True,  # <--- NÉT ĐỨT
                                 width=2)
                else:
                    # Cạnh bình thường
                    net.add_edge(u, v, value=1, color={'inherit': 'from', 'opacity': 0.6})

            net.barnes_hut(gravity=-2000, spring_length=150)

            html_string = net.generate_html()
            components.html(html_string, height=620)
        else:
            st.info("Không có dữ liệu. Hãy nới lỏng bộ lọc.")

    with col2:
        st.subheader("📈 Thống kê View")
        if G_viz:
            st.metric("Tác giả hiển thị", G_viz.number_of_nodes())
            # Tách số liệu mối quan hệ
            num_edges = G_viz.number_of_edges()
            num_future = sum(1 for u,v,d in G_viz.edges(data=True) if d.get('type') == 'future')
            st.metric("Mối quan hệ", num_edges, delta=f"+{num_future} Dự báo" if num_future > 0 else None)

        if selected_author == "-- Xem Tổng Quan --" and G_viz and G_viz.number_of_nodes() > 0:
            # --- PIE CHART: PHÂN BỐ CỘNG ĐỒNG ---
            st.markdown("#### 🥧 Phân bố Cộng đồng")
            comm_counts = {}
            for n, d in G_viz.nodes(data=True):
                comm = str(d.get('louvain_community', 0))
                comm_counts[comm] = comm_counts.get(comm, 0) + 1
            
            df_pie = pd.DataFrame([
                {'Cộng đồng': f"Nhóm {k}", 'Số lượng': v} 
                for k, v in sorted(comm_counts.items(), key=lambda x: -x[1])
            ])
            
            fig_pie = px.pie(df_pie, values='Số lượng', names='Cộng đồng', 
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                margin=dict(t=20, b=20, l=20, r=20),
                height=250
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # --- BAR CHART: XẾP HẠNG ---
            st.markdown("#### 🏆 Top Bridges")
            data_chart = []
            for n, d in G_viz.nodes(data=True):
                data_chart.append({
                    'Tên': d.get('label', str(n)),
                    'Điểm': d.get('betweenness', 0),
                    'Nhóm': str(d.get('louvain_community', 0))
                })
            df_chart = pd.DataFrame(data_chart).sort_values('Điểm', ascending=False).head(10)

            fig = px.bar(df_chart, x='Điểm', y='Tên', color='Nhóm', orientation='h',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'}, 
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_author != "-- Xem Tổng Quan --" and G_viz:
            st.markdown("### 👥 Kết nối trực tiếp")
            if selected_author in name_to_id:
                center_id = name_to_id[selected_author]
                neighbors_list = []
                for neighbor_id in G_viz.neighbors(center_id):
                    edge_data = G_viz.get_edge_data(center_id, neighbor_id)
                    if edge_data.get('type') != 'future':
                        neighbors_list.append(G_viz.nodes[neighbor_id].get('label', str(neighbor_id)))
                
                if neighbors_list:
                    st.dataframe(pd.DataFrame(neighbors_list, columns=["Đồng tác giả"]), hide_index=True)
                else:
                    st.info("Chưa có kết nối nào trong bộ lọc này.")

            # BẢNG DỰ BÁO
            if not df_pred.empty:
                st.markdown("### 🔮 Dự báo tiềm năng")
                my_preds = df_pred[df_pred['Source'] == selected_author][['Target', 'Score', 'Model']].copy()
                if not my_preds.empty:
                    my_preds['Score'] = my_preds['Score'].apply(lambda x: f"{x:.6f}")
                    st.dataframe(my_preds.head(10), hide_index=True)
                else:
                    st.info("Chưa có dự báo cho tác giả này.")
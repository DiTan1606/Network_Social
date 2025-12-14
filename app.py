import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px

# --- CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="Social Network Analysis")
st.title("🕸️ Phân tích Mạng lưới & Cộng đồng Tác giả")

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

    st.sidebar.subheader("1. Chọn Mốc Thời Gian")
    selected_time = st.sidebar.radio("Thời gian:", options=time_options, horizontal=True, label_visibility="collapsed")

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

    st.sidebar.subheader("2. Chọn Cộng đồng")

    all_comms_selected = st.sidebar.checkbox("Chọn tất cả cộng đồng", value=True)

    if all_comms_selected:
        selected_comms = sorted_comms
    else:
        selected_comms = st.sidebar.multiselect(
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
    st.sidebar.subheader("3. Tìm & Focus Tác giả")

    name_to_id = {}
    current_names = []
    for n, data in G_comm.nodes(data=True):
        label = data.get('label', str(n))
        name_to_id[label] = n
        current_names.append(label)

    list_names = ["-- Xem Tổng Quan --"] + sorted(list(set(current_names)))

    selected_author = st.sidebar.selectbox("Gõ tên để Focus:", list_names)

    # ----------------------------------------
    # 4. CHỌN HIỂN THỊ TOP N (Chỉ dùng cho Tổng quan)
    # ----------------------------------------
    if selected_author == "-- Xem Tổng Quan --":
        st.sidebar.subheader("4. Giới hạn hiển thị")
        top_n = st.sidebar.slider("Số lượng tác giả (Top Betweenness)",
                                  min_value=10, max_value=500, value=100, step=10)
    else:
        st.sidebar.info("Đang ở chế độ Focus Tác giả. Bộ lọc Top N tạm ẩn.")

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
        st.subheader("Thống kê")
        if G_viz:
            st.metric("Tác giả hiển thị", G_viz.number_of_nodes())
            # Tách số liệu mối quan hệ
            num_edges = G_viz.number_of_edges()
            num_future = sum(1 for u,v,d in G_viz.edges(data=True) if d.get('type') == 'future')
            st.metric("Mối quan hệ", num_edges, delta=f"+{num_future} Dự báo" if num_future > 0 else None)

        if selected_author == "-- Xem Tổng Quan --" and G_viz and G_viz.number_of_nodes() > 0:
            data_chart = []
            for n, d in G_viz.nodes(data=True):
                data_chart.append({
                    'Tên': d.get('label', str(n)),
                    'Điểm': d.get('betweenness', 0),
                    'Nhóm': str(d.get('louvain_community', 0))
                })
            df_chart = pd.DataFrame(data_chart).sort_values('Điểm', ascending=False).head(15)

            fig = px.bar(df_chart, x='Điểm', y='Tên', color='Nhóm', orientation='h', title="Xếp hạng")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_author != "-- Xem Tổng Quan --" and G_viz:
            st.markdown("### Kết nối trực tiếp")
            if selected_author in name_to_id:
                center_id = name_to_id[selected_author]
                neighbors_list = []
                # Chỉ lấy các neighbor "thật" (không phải future)
                for neighbor_id in G_viz.neighbors(center_id):
                    edge_data = G_viz.get_edge_data(center_id, neighbor_id)
                    if edge_data.get('type') != 'future':
                        neighbors_list.append(G_viz.nodes[neighbor_id].get('label', str(neighbor_id)))
                
                if neighbors_list:
                    st.dataframe(pd.DataFrame(neighbors_list, columns=["Đồng tác giả"]), hide_index=True)
                else:
                    st.info("Chưa có kết nối nào trong bộ lọc này.")

            # THÊM BẢNG DỰ BÁO
            if not df_pred.empty:
                st.markdown("### 🔮 Dự báo tiềm năng")
                my_preds = df_pred[df_pred['Source'] == selected_author][['Target', 'Score', 'Model']].copy()
                if not my_preds.empty:
                    # Format Score với 6 chữ số thập phân
                    my_preds['Score'] = my_preds['Score'].apply(lambda x: f"{x:.6f}")
                    st.dataframe(my_preds.head(10), hide_index=True)
                else:
                    st.info("Chưa có dự báo cho tác giả này.")
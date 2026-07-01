import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import folium
from io import BytesIO

st.set_page_config(page_title="Beat Planner", layout="wide")

st.title("🗺️ Beat Planning Tool")
st.markdown("Upload your outlet data, set your parameters, and generate geographically compact and equal clusters and beats.")

# ── SIDEBAR CONTROLS ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=5)
    beats_per_cluster = st.slider("Beats per Cluster", min_value=2, max_value=15, value=6)
    st.markdown("---")
    st.markdown(f"**Total beats:** {n_clusters * beats_per_cluster}")
    st.markdown("**Required columns:**")
    st.markdown("- Customer Code\n- Customer Name\n- Latitude\n- Longitude")

# ── FILE UPLOAD ───────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# ── CLUSTERING FUNCTIONS ──────────────────────────────────────────
def to_meters(lat, lon):
    R = 6371000
    lat = np.array(lat)
    lon = np.array(lon)
    x = np.radians(lon) * R * np.cos(np.radians(lat.mean()))
    y = np.radians(lat) * R
    return np.column_stack([x, y])

def strict_balance(df, coords, label_col, n, target, tolerance=5, max_iter=300):
    for _ in range(max_iter):
        counts = df[label_col].value_counts().to_dict()
        for i in range(1, n+1):
            counts.setdefault(i, 0)

        over  = [b for b,c in counts.items() if c > target + tolerance]
        under = [b for b,c in counts.items() if c < target - tolerance]
        if not over or not under:
            break

        ob = max(over,  key=lambda b: counts[b])
        ub = min(under, key=lambda b: counts[b])

        centroid_ub = coords[df[label_col] == ub].mean(axis=0)
        centroid_ob = coords[df[label_col] == ob].mean(axis=0)

        pts_mask    = (df[label_col] == ob).values
        pts_idx     = df[df[label_col] == ob].index
        pts_coords  = coords[pts_mask]

        dist_to_ub  = cdist(pts_coords, centroid_ub.reshape(1,-1)).flatten()
        dist_to_ob  = cdist(pts_coords, centroid_ob.reshape(1,-1)).flatten()

        border_mask = dist_to_ub < dist_to_ob
        border_idx  = pts_idx[border_mask]
        border_dist = dist_to_ub[border_mask]

        need   = (target - tolerance) - counts[ub]
        excess = counts[ob] - (target + tolerance)
        n_move = min(excess, need, len(border_idx))

        if n_move <= 0:
            n_move   = min(excess, need, len(pts_idx))
            move_idx = pts_idx[np.argsort(dist_to_ub)[:max(1, n_move)]]
        else:
            move_idx = border_idx[np.argsort(border_dist)[:n_move]]

        df.loc[move_idx, label_col] = ub
    return df

def angular_cluster(df, coords_m, n_clusters):
    center = coords_m.mean(axis=0)
    dx = coords_m[:, 0] - center[0]
    dy = coords_m[:, 1] - center[1]
    angles = np.arctan2(dy, dx)
    angle_order = np.argsort(angles)

    n = len(df)
    labels = np.zeros(n, dtype=int)
    chunk  = n // n_clusters

    for i in range(n_clusters):
        start = i * chunk
        end   = (i+1)*chunk if i < n_clusters-1 else n
        labels[angle_order[start:end]] = i + 1

    return labels

def run_clustering(df, n_clusters, beats_per_cluster):
    coords_m = to_meters(df['Latitude'].values, df['Longitude'].values)

    df['Cluster_No'] = angular_cluster(df, coords_m, n_clusters)

    centroids = np.array([
        coords_m[df['Cluster_No'] == c].mean(axis=0)
        for c in range(1, n_clusters+1)
    ])
    km = KMeans(n_clusters=n_clusters, init=centroids,
                n_init=1, max_iter=500, random_state=42)
    df['Cluster_No'] = km.fit_predict(coords_m) + 1

    target_cluster = len(df) // n_clusters
    df = strict_balance(df, coords_m, 'Cluster_No',
                        n_clusters, target_cluster, tolerance=5)

    df['Beat_No'] = 0

    for cl in range(1, n_clusters+1):
        sub_idx    = df[df['Cluster_No'] == cl].index
        sub        = df.loc[sub_idx].copy()
        sub_coords = to_meters(sub['Latitude'].values, sub['Longitude'].values)
        t_beat     = len(sub) // beats_per_cluster

        sub_center = sub_coords.mean(axis=0)
        dx = sub_coords[:,0] - sub_center[0]
        dy = sub_coords[:,1] - sub_center[1]
        sub_angles = np.arctan2(dy, dx)
        sub_order  = np.argsort(sub_angles)

        beat_labels = np.zeros(len(sub), dtype=int)
        chunk_b = len(sub) // beats_per_cluster
        for i in range(beats_per_cluster):
            start = i * chunk_b
            end   = (i+1)*chunk_b if i < beats_per_cluster-1 else len(sub)
            beat_labels[sub_order[start:end]] = i + 1

        sub['Beat_No'] = beat_labels

        beat_centroids = np.array([
            sub_coords[sub['Beat_No'] == b].mean(axis=0)
            for b in range(1, beats_per_cluster+1)
            if (sub['Beat_No'] == b).any()
        ])
        if len(beat_centroids) == beats_per_cluster:
            km2 = KMeans(n_clusters=beats_per_cluster, init=beat_centroids,
                         n_init=1, max_iter=300, random_state=42)
            sub['Beat_No'] = km2.fit_predict(sub_coords) + 1

        sub = strict_balance(sub, sub_coords, 'Beat_No',
                             beats_per_cluster, t_beat, tolerance=4)
        df.loc[sub_idx, 'Beat_No'] = sub['Beat_No']

    df['Final_Beat'] = ('Cluster ' + df['Cluster_No'].astype(str) +
                        ' - Beat '  + df['Beat_No'].astype(str))
    return df

def build_folium_map(df_result, color_by):
    cluster_palette = ['blue','red','green','purple','orange',
                       'darkred','darkblue','darkgreen','cadetblue','darkpurple',
                       'lightred','beige','lightblue','lightgreen','gray',
                       'black','pink','white','lightgray','darkpurple']

    beat_palette = [
        '#e6194b','#f58231','#ffe119','#bfef45','#3cb44b','#42d4f4',
        '#4363d8','#911eb4','#f032e6','#a9a9a9','#800000','#9A6324',
        '#469990','#000075','#e6beff','#ffd8b1','#aaffc3','#808000',
        '#dc143c','#ff8c00','#00ced1','#8b008b','#006400','#1e90ff',
        '#ff1493','#7fff00','#ff6347','#4682b4','#d2691e','#20b2aa'
    ]

    m = folium.Map(
        location=[df_result['Latitude'].mean(), df_result['Longitude'].mean()],
        zoom_start=12
    )

    if color_by == "Cluster":
        color_map = {c: cluster_palette[i % len(cluster_palette)]
                     for i, c in enumerate(sorted(df_result['Cluster_No'].unique()))}
        for _, row in df_result.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color_map[row['Cluster_No']],
                fill=True,
                fill_color=color_map[row['Cluster_No']],
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{row['Customer Name']}</b><br>"
                    f"Cluster {row['Cluster_No']}<br>{row['Final_Beat']}",
                    max_width=200)
            ).add_to(m)
    else:
        beats_sorted = sorted(df_result['Final_Beat'].unique())
        beat_colors  = {b: beat_palette[i % len(beat_palette)]
                        for i, b in enumerate(beats_sorted)}
        for _, row in df_result.iterrows():
            color = beat_colors[row['Final_Beat']]
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{row['Customer Name']}</b><br>{row['Final_Beat']}",
                    max_width=200)
            ).add_to(m)

    return m

# ── MAIN APP ──────────────────────────────────────────────────────
if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    df_raw = df_raw.dropna(subset=['Latitude','Longitude']).reset_index(drop=True)

    st.success(f"✅ File loaded: {len(df_raw)} outlets")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Outlets", len(df_raw))
    col2.metric("Clusters", n_clusters)
    col3.metric("Total Beats", n_clusters * beats_per_cluster)

    if st.button("🚀 Generate Beat Plan", type="primary", use_container_width=True):

        with st.spinner("Clustering outlets... please wait"):
            df_result = run_clustering(df_raw.copy(), n_clusters, beats_per_cluster)

        st.success("✅ Beat plan generated!")

        # ── SUMMARY TABLE ─────────────────────────────────────────
        st.subheader("📊 Cluster Summary")
        summary = df_result.groupby(['Cluster_No','Beat_No']).size().reset_index(name='Outlets')
        summary['Cluster'] = 'Cluster ' + summary['Cluster_No'].astype(str)
        summary['Beat']    = 'Beat '    + summary['Beat_No'].astype(str)
        pivot = summary.pivot(index='Cluster', columns='Beat', values='Outlets').fillna(0).astype(int)
        pivot['TOTAL'] = pivot.sum(axis=1)
        st.dataframe(pivot, use_container_width=True)

        # ── BUILD DOWNLOADABLE FILES ────────────────────────────────
        st.subheader("📥 Download Results")

        # Excel export
        out = df_result[['Customer Code','Customer Name','Latitude','Longitude',
                          'Cluster_No','Beat_No','Final_Beat']].copy()
        out.columns = ['Customer Code','Customer Name','Latitude','Longitude',
                       'Cluster','Beat No','Final Beat']
        out = out.sort_values(['Cluster','Beat No','Customer Name']).reset_index(drop=True)

        excel_buffer = BytesIO()
        out.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        # HTML maps (both cluster view and beat view) — built once, saved to string
        map_cluster = build_folium_map(df_result, "Cluster")
        map_beat    = build_folium_map(df_result, "Beat")

        html_cluster = map_cluster.get_root().render()
        html_beat    = map_beat.get_root().render()

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                label="⬇️ Download Excel (Beat Plan)",
                data=excel_buffer,
                file_name="beat_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                label="⬇️ Download Map (by Cluster).html",
                data=html_cluster,
                file_name="map_by_cluster.html",
                mime="text/html",
                use_container_width=True
            )
        with dl3:
            st.download_button(
                label="⬇️ Download Map (by Beat).html",
                data=html_beat,
                file_name="map_by_beat.html",
                mime="text/html",
                use_container_width=True
            )

        st.info("Open the downloaded .html map files directly in any browser — they're fully interactive (zoom, pan, click pins for details).")
else:
    st.info("👆 Upload an Excel file to get started")

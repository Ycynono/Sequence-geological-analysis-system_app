import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy import signal
import io
import chardet

# 1. 页面基础配置 - 适配 2025 标准
st.set_page_config(page_title="地层层序解析系统Pro", layout="wide")


# --- 核心算法 (保持高效) ---
def get_inpefa(series, order=1):
    clean = series.interpolate().ffill().bfill()
    data = (clean - clean.mean()) / clean.std()
    for _ in range(order):
        data = data.cumsum()
    return data


def get_wavelet_analysis(series, max_scale=128):
    data = series.interpolate().ffill().bfill().values
    if len(data) < 10:
        return np.zeros((max_scale, len(data))), np.zeros(len(data))
    widths = np.arange(1, max_scale + 1)
    cwtmatr, _ = pywt.cwt(data, widths, 'mexh')
    energy = np.abs(cwtmatr)
    return energy, np.sum(np.square(energy), axis=0)


def get_astro_cycles(series, low, high):
    data = series.interpolate().ffill().bfill().values
    if len(data) < 30: return np.zeros(len(data))
    try:
        sos = signal.butter(10, [low, high], 'bandpass', fs=1, output='sos')
        return signal.sosfiltfilt(sos, data)
    except:
        return np.zeros(len(data))


def load_data(file):
    try:
        rb = file.read()
        det = chardet.detect(rb)
        enc = det['encoding'] or 'utf-8'
        file.seek(0)
        return pd.read_csv(io.StringIO(rb.decode(enc, errors='ignore')),
                           sep=None, engine='python', on_bad_lines='skip')
    except:
        return None


# --- UI 顶部 ---
st.title("🏹 地层层序地质解析系统")

with st.sidebar:
    st.header("📁 数据配置与参数")
    uploaded_file = st.file_uploader("上传测井数据 (CSV/TXT)", type=["csv", "txt", "xlsx"])

    depth_col, target_col = None, None
    in_order, m_scale, f_range = 1, 128, (0.01, 0.08)

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            cols = df_raw.columns.tolist()
            depth_col = st.selectbox("维度列 (Depth)", cols, index=0)
            target_col = st.selectbox("分析目标 (Log)", cols, index=min(1, len(cols) - 1))
            st.markdown("---")
            st.subheader("⚙️ 算法精调")
            in_order = st.slider("INPEFA 阶数", 1, 15, 1)
            m_scale = st.slider("小波分析尺度", 32, 512, 128)
            f_range = st.slider("天文旋回频带", 0.001, 0.499, (0.01, 0.08))

# --- 主展示逻辑 ---
if uploaded_file and (depth_col and target_col):
    df = df_raw.copy()
    df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[depth_col, target_col]).sort_values(by=depth_col).reset_index(drop=True)

    if not df.empty:
        # 计算阶段
        with st.spinner('模型计算中...'):
            df['INPEFA'] = get_inpefa(df[target_col], order=in_order)
            w_mat, df['Wave_Energy'] = get_wavelet_analysis(df[target_col], m_scale)
            df['Astro'] = get_astro_cycles(df[target_col], f_range[0], f_range[1])

        # 绘图阶段
        fig = make_subplots(
            rows=1, cols=5, shared_yaxes=True, horizontal_spacing=0.02,
            subplot_titles=("Raw Log", "INPEFA Trend", "CWT Spectrum", "Energy", "Astro Cycle"),
            column_widths=[0.12, 0.12, 0.38, 0.18, 0.20]
        )

        d_min, d_max = float(df[depth_col].min()), float(df[depth_col].max())
        fig.add_trace(go.Scatter(x=df[target_col], y=df[depth_col], name="Raw", line=dict(color='#2c3e50')), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=df['INPEFA'], y=df[depth_col], name="Trend", line=dict(color='#e67e22', width=2)),
                      row=1, col=2)
        fig.add_trace(
            go.Heatmap(z=w_mat.T, x=np.arange(1, m_scale + 1), y=df[depth_col], colorscale='Jet', showscale=False),
            row=1, col=3)
        fig.add_trace(
            go.Scatter(x=df['Wave_Energy'], y=df[depth_col], name="Energy", fill='tozerox', line=dict(color='purple')),
            row=1, col=4)
        fig.add_trace(go.Scatter(x=df['Astro'], y=df[depth_col], name="Cycle", line=dict(color='red', width=1)), row=1,
                      col=5)

        fig.update_yaxes(range=[d_max, d_min], title="Depth (m)")
        fig.update_layout(height=850, template="plotly_white", margin=dict(t=60, b=40, l=60, r=40))

        # 核心改进：使用 1.52 标准参数 width="stretch"
        st.plotly_chart(fig, width="stretch")

        # 下载区
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("💾 下载解析数据 (CSV)", data=df.to_csv(index=False).encode('utf-8'),
                               file_name="sequence_data.csv", width="stretch")
        with c2:
            buf = io.StringIO()
            fig.write_html(buf, include_plotlyjs='cdn')
            st.download_button("🌐 下载交互图表 (HTML)", data=buf.getvalue(),
                               file_name="sequence_chart.html", width="stretch")

        # --- 重点：下降到页面尾部的解析解析说明 ---
        st.markdown("<br><br>", unsafe_allow_html=True)  # 增加一些间距
        st.divider()  # 视觉分割线

        st.subheader("🎓 深度解析指南")

        # 使用 Expander 避免页面过长，但默认展开
        with st.expander("点击展开参数及曲线含义说明", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.info("**🛠 参数调节逻辑**")
                st.markdown("""
                - **INPEFA 阶数**: 低阶识别局部小旋回，高阶识别区域性大趋势。如果曲线太乱，请调大此值。
                - **小波尺度**: 反映地层的响应窗口。数值越大，捕捉到的沉积单元厚度越大。
                - **天文频带**: 请根据地质背景设置。对应偏心率周期的信号通常频率极低。
                """)
            with col_b:
                st.info("**📈 曲线地质解释**")
                st.markdown("""
                - **INPEFA Trend**: 曲线由左向右转（波谷）常对应**最大海泛面 (MFS)**；由右向左转（波峰）常对应**层序界面 (SB)**。
                - **CWT Spectrum**: 红色亮带表示该深度处地层具有极强的周期性规律。
                - **Astro Cycle**: 模拟的米兰科维奇旋回，用于更精确的地层对比和旋回标定。
                """)
            st.caption("注：以上分析基于统计学算法，建议结合野外露头或岩心描述进行最终标定。")

    else:
        st.warning("⚠️ 选中的列中没有足够的数据。")
else:
    # 未上传时的欢迎页
    st.info("👋 欢迎！请在左侧上传 CSV 或 TXT 文件以开始分析。")
    st.markdown("""
    ### 快速开始步骤：
    1. 准备包含 **Depth**（深度）和 **Logging**（如 GR/SP）的 CSV 文件。
    2. 在侧边栏上传并选择对应的列。
    3. 系统将自动生成层序界面的预测趋势和小波时频谱。
    ---
    """)

    #streamlit run  well_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy import signal
import io
import chardet

# 1. 页面基础配置：设置为宽屏模式
st.set_page_config(page_title="层序地质解析系统", layout="wide")


# --- 核心算法函数 ---
def get_inpefa(series, order=1):
    """计算 INPEFA 曲线"""
    # 显式处理缺失值以避免计算警告
    clean_series = series.interpolate().ffill().bfill()
    data = (clean_series - clean_series.mean()) / clean_series.std()
    for _ in range(order):
        data = data.cumsum()
    return data


def get_wavelet_analysis(series, max_scale=128):
    """执行连续小波变换 (CWT) 并计算能量"""
    # 使用新版推荐的填充方式
    data = series.interpolate().ffill().bfill().values
    if len(data) < 10:
        return np.zeros((max_scale, len(data))), np.zeros(len(data))
    widths = np.arange(1, max_scale + 1)
    cwtmatr, _ = pywt.cwt(data, widths, 'mexh')
    energy_matrix = np.abs(cwtmatr)
    wavelet_energy_curve = np.sum(np.square(energy_matrix), axis=0)
    return energy_matrix, wavelet_energy_curve


def get_astro_cycles(series, low_freq, high_freq):
    """提取天文旋回（带通滤波）"""
    data = series.interpolate().ffill().bfill().values
    if len(data) < 30: return np.zeros(len(data))
    try:
        sos = signal.butter(10, [low_freq, high_freq], 'bandpass', fs=1, output='sos')
        return signal.sosfiltfilt(sos, data)
    except:
        return np.zeros(len(data))


def load_data(file):
    """加载测井数据并自动检测编码"""
    try:
        raw_bytes = file.read()
        det = chardet.detect(raw_bytes)
        encoding = det['encoding'] or 'utf-8'
        file.seek(0)
        df = pd.read_csv(io.StringIO(raw_bytes.decode(encoding, errors='ignore')),
                         sep=None, engine='python', on_bad_lines='skip')
        return df.dropna(how='all').dropna(axis=1, how='all')
    except:
        return None


# --- UI 界面布局 ---

st.title("🏹 层序地质解析系统")

# --- A. 侧边栏：参数配置与文件上传 ---
with st.sidebar:
    st.header("📁 数据与参数")
    uploaded_file = st.file_uploader("上传测井数据", type=["csv", "txt", "xlsx", "xls", "las"])

    depth_col, target_col = None, None
    inpefa_order, max_scale, freq_range = 1, 128, (0.01, 0.08)

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            cols = df_raw.columns.tolist()
            depth_col = st.selectbox("选择深度列 (Depth)", cols, index=0)
            target_col = st.selectbox("选择分析曲线 (Log)", cols, index=min(1, len(cols) - 1))

            st.markdown("---")
            st.subheader("⚙️ 算法微调")
            inpefa_order = st.slider("INPEFA 阶数", 1, 15, 1)
            max_scale = st.slider("小波尺度", 32, 512, 128)
            freq_range = st.slider("旋回频带", 0.001, 0.499, (0.01, 0.08))

# --- B. 主界面：图表展示区域 ---
if uploaded_file and (depth_col and target_col):
    df = df_raw.copy()
    df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[depth_col, target_col]).sort_values(by=depth_col).reset_index(drop=True)

    if not df.empty:
        with st.spinner('计算分析中...'):
            df['INPEFA'] = get_inpefa(df[target_col], order=inpefa_order)
            w_matrix, df['Wavelet_Energy'] = get_wavelet_analysis(df[target_col], max_scale)
            df['Astro_Cycle'] = get_astro_cycles(df[target_col], freq_range[0], freq_range[1])

        d_min, d_max = float(df[depth_col].min()), float(df[depth_col].max())

        fig = make_subplots(
            rows=1, cols=5,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            subplot_titles=("Raw Log", "INPEFA Trend", "CWT Spectrum", "Wavelet Energy", "Astro Cycle"),
            column_widths=[0.12, 0.12, 0.38, 0.18, 0.20]
        )

        # 绘图轨迹
        fig.add_trace(go.Scatter(x=df[target_col], y=df[depth_col], name="Log", line=dict(color='#2c3e50', width=1)),
                      row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['INPEFA'], y=df[depth_col], name="INPEFA", line=dict(color='darkblue', width=1.5)), row=1,
            col=2)
        # 这里保留了关键的 .T 转置修复
        fig.add_trace(
            go.Heatmap(z=w_matrix.T, x=np.arange(1, max_scale + 1), y=df[depth_col], colorscale='Jet', showscale=False),
            row=1, col=3)
        fig.add_trace(
            go.Scatter(x=df['Wavelet_Energy'], y=df[depth_col], name="Energy", line=dict(color='purple', width=1.2)),
            row=1, col=4)
        fig.add_trace(go.Scatter(x=df['Astro_Cycle'], y=df[depth_col], name="Astro", line=dict(color='red', width=1)),
                      row=1, col=5)

        fig.update_yaxes(range=[d_max, d_min], title="Depth (m)")
        fig.update_layout(height=1000, template="plotly_white", margin=dict(t=50, b=50, l=80, r=40),
                          hovermode="y unified")

        # 按照 2025 新规范，将 use_container_width 替换为 width="stretch"
        st.plotly_chart(fig, width="stretch")

        # 4. 数据导出按钮
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            csv = df.to_csv(index=False).encode('utf-8')
            # 按钮也更新为新规范 width="stretch"
            st.download_button("💾 下载分析数据 (CSV)", data=csv, file_name="analysis_results.csv", width="stretch")
        with col_btn2:
            html_buf = io.StringIO()
            fig.write_html(html_buf, include_plotlyjs='cdn')
            st.download_button("🌐 下载交互式 HTML 图表", data=html_buf.getvalue(), file_name="geology_chart.html",
                               width="stretch")

    else:
        st.error("❌ 数据处理出错，请确认所选列包含有效的数值。")
else:
    st.info("👈 系统就绪！请在左侧侧边栏上传测井数据文件。")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy import signal
import io
import chardet

# 1. 页面配置
st.set_page_config(page_title="层序地质解析系统", layout="wide")


# --- 核心算法函数 ---
def get_inpefa(series, order=1):
    """计算 INPEFA 曲线，改用新版 Pandas 填充语法"""
    clean_series = series.interpolate().ffill().bfill()
    data = (clean_series - clean_series.mean()) / clean_series.std()
    for _ in range(order):
        data = data.cumsum()
    return data


def get_wavelet_analysis(series, max_scale=128):
    """执行连续小波变换 (CWT)"""
    # 彻底修复 fillna 弃用警告
    data = series.interpolate().ffill().bfill().values
    if len(data) < 10:
        return np.zeros((max_scale, len(data))), np.zeros(len(data))
    widths = np.arange(1, max_scale + 1)
    cwtmatr, _ = pywt.cwt(data, widths, 'mexh')
    energy_matrix = np.abs(cwtmatr)
    wavelet_energy_curve = np.sum(np.square(energy_matrix), axis=0)
    return energy_matrix, wavelet_energy_curve


def get_astro_cycles(series, low_freq, high_freq):
    """提取天文旋回"""
    data = series.interpolate().ffill().bfill().values
    if len(data) < 30: return np.zeros(len(data))
    try:
        sos = signal.butter(10, [low_freq, high_freq], 'bandpass', fs=1, output='sos')
        return signal.sosfiltfilt(sos, data)
    except:
        return np.zeros(len(data))


def load_data(file):
    """加载数据逻辑"""
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


# --- UI 界面 ---
st.title("🏹 层序地质解析系统")

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

        # 核心绘图区
        fig = make_subplots(
            rows=1, cols=5,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            subplot_titles=("Raw Log", "INPEFA Trend", "CWT Spectrum", "Wavelet Energy", "Astro Cycle"),
            column_widths=[0.12, 0.12, 0.38, 0.18, 0.20]
        )

        fig.add_trace(go.Scatter(x=df[target_col], y=df[depth_col], name="Log", line=dict(color='#2c3e50', width=1)),
                      row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['INPEFA'], y=df[depth_col], name="INPEFA", line=dict(color='darkblue', width=1.5)), row=1,
            col=2)

        # 修正 CWT 填充问题：使用 w_matrix.T
        fig.add_trace(go.Heatmap(
            z=w_matrix.T,
            x=np.arange(1, max_scale + 1),
            y=df[depth_col],
            colorscale='Jet',
            showscale=False
        ), row=1, col=3)

        fig.add_trace(
            go.Scatter(x=df['Wavelet_Energy'], y=df[depth_col], name="Energy", line=dict(color='purple', width=1.2)),
            row=1, col=4)
        fig.add_trace(go.Scatter(x=df['Astro_Cycle'], y=df[depth_col], name="Astro", line=dict(color='red', width=1)),
                      row=1, col=5)

        fig.update_yaxes(range=[d_max, d_min], title="Depth (m)")
        fig.update_layout(height=900, template="plotly_white", margin=dict(t=50, b=50, l=80, r=40),
                          hovermode="y unified")

        # 【重点修改】不再显式设置宽度参数，Streamlit 将自动使用当前容器的最佳宽度
        st.plotly_chart(fig)

        st.markdown("---")
        # 下载按钮也不再手动设置宽度相关的参数
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 下载 CSV 数据", data=csv_data, file_name="results.csv")
        with col_btn2:
            html_buf = io.StringIO()
            fig.write_html(html_buf, include_plotlyjs='cdn')
            st.download_button("🌐 下载 HTML 图表", data=html_buf.getvalue(), file_name="chart.html")
    else:
        st.error("❌ 数据无效")

        # --- 原理说明区 ---
        st.markdown("---")
        st.header("📘 曲线原理与调节指南")
        exp1, exp2 = st.columns(2)
        with exp1:
            st.subheader("曲线地质含义")
            st.markdown("""
                - **Raw Log**: 原始曲线，反映岩性或物性的基础波动。
                - **INPEFA Trend**: 趋势线。上升段对应水退/供应增加，下降段对应水侵/可容空间增加。
                - **CWT Spectrum**: 展现不同尺度旋回的强度。
                - **Wavelet Energy**: 识别地层能量剧变，用于确定关键层序界面。
                - **Astro Cycle**: 天文旋回。反映受轨道力控制的周期性信号，用于高频旋回划分与精细对比。
                """)
        with exp2:
            st.subheader("参数调节说明")
            st.markdown(f"""
                - **INPEFA 阶数 ({inpefa_order})** : 增加阶数会使曲线更平滑，帮助识别二级或三级层序长周期趋势；降低阶数则保留更多局部细节。
                - **小波尺度 ({max_scale})** : 尺度越大，系统越能识别出超厚叠置组的地学旋回。
                - **频率带范围**: 根据地质经验微调，可剔除干扰杂波，精准锁定受天文驱动的沉积节拍
                """)
else:
    st.info("👈 请在左侧上传数据文件开始分析。")

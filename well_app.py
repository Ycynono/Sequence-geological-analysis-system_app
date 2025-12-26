import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy import signal
import io
import chardet

# 页面基础配置
st.set_page_config(page_title="层序地质解析系统", layout="wide")


# --- 核心算法 ---

def get_inpefa(series, order=1):
    data = (series - series.mean()) / series.std()
    for _ in range(order):
        data = data.cumsum()
    return data


def get_wavelet_analysis(series, max_scale=128):
    data = series.interpolate().fillna(method='ffill').fillna(method='bfill').values
    widths = np.arange(1, max_scale + 1)
    cwtmatr, _ = pywt.cwt(data, widths, 'mexh')
    energy_matrix = np.abs(cwtmatr)
    # 计算小波能量谱曲线（Energy Curve）
    wavelet_energy_curve = np.sum(np.square(energy_matrix), axis=0)
    return energy_matrix, wavelet_energy_curve


def get_astro_cycles(series, low_freq, high_freq):
    data = series.interpolate().fillna(method='ffill').fillna(method='bfill').values
    sos = signal.butter(10, [low_freq, high_freq], 'bandpass', fs=1, output='sos')
    cycle_curve = signal.sosfiltfilt(sos, data)
    return cycle_curve


def load_data(file):
    try:
        raw_bytes = file.read()
        det = chardet.detect(raw_bytes)
        encoding = det['encoding'] or 'utf-8'
        file.seek(0)
        ext = file.name.split('.')[-1].lower()
        if ext in ['xlsx', 'xls']:
            return pd.read_excel(file)
        df = pd.read_csv(io.StringIO(raw_bytes.decode(encoding)), sep=None, engine='python', on_bad_lines='skip')
        return df.dropna(how='all').dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"读取失败: {e}")
        return None


# --- UI 界面 ---

st.title("🏹  层序地质解析系统")
st.markdown("---")

# 侧边栏：中文控件
st.sidebar.header("📁 数据管理")
uploaded_file = st.sidebar.file_uploader("上传测井数据 (CSV, XLSX, LAS)", type=["csv", "txt", "xlsx", "xls", "las"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None and not df_raw.empty:
        cols = df_raw.columns.tolist()

        # 1. 字段配置面板
        with st.sidebar.expander("📍坐标与字段配置", expanded=True):
            depth_col = st.selectbox("选择深度列 (Depth)", cols, index=0)
            target_col = st.selectbox("选择分析曲线 (GR/Log)", cols, index=min(1, len(cols) - 1))

            df = df_raw.copy()
            df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df = df.dropna(subset=[depth_col, target_col]).sort_values(by=depth_col).reset_index(drop=True)

            d_min, d_max = float(df[depth_col].min()), float(df[depth_col].max())
            st.info(f"当前深度区间: {d_min} - {d_max} m")

        # 2. 算法控制面板
        with st.sidebar.expander("⚙️ 算法参数微调", expanded=True):
            inpefa_order = st.slider("INPEFA 阶数 (控制趋势平滑度)", 1, 15, 1)
            max_scale = st.slider("小波尺度 (控制频谱精细度)", 32, 512, 128)
            freq_range = st.slider("旋回频带 (提取天文旋回)", 0.001, 0.499, (0.01, 0.08))

        # 执行计算
        with st.spinner('XSimple 正在解析...'):
            df['INPEFA'] = get_inpefa(df[target_col], order=inpefa_order)
            w_matrix, df['Wavelet_Energy'] = get_wavelet_analysis(df[target_col], max_scale)
            df['Astro_Cycle'] = get_astro_cycles(df[target_col], freq_range[0], freq_range[1])

        # --- 可视化图表 ---
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
        fig.add_trace(
            go.Heatmap(z=w_matrix.T, x=np.arange(1, max_scale + 1), y=df[depth_col], colorscale='Jet', showscale=False),
            row=1, col=3)
        fig.add_trace(go.Scatter(x=df['Wavelet_Energy'], y=df[depth_col], name="Wavelet Energy",
                                 line=dict(color='purple', width=1.2)), row=1, col=4)
        fig.add_trace(
            go.Scatter(x=df['Astro_Cycle'], y=df[depth_col], name="Astro Cycle", line=dict(color='red', width=1)),
            row=1, col=5)

        fig.update_yaxes(range=[d_max, d_min], title="Depth (m)", showgrid=True)
        fig.update_layout(height=1000, template="plotly_white", margin=dict(t=150, b=50), hovermode="y unified")

        # 展示图表
        st.plotly_chart(fig, use_container_width=True)

        # --- 按钮区：导出成果 (核心更新：独立按钮) ---
        st.markdown("### 📥 成果导出与下载")
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            # 下载 CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 下载计算结果清单 (CSV)",
                data=csv_data,
                file_name=f"XSimple_Data_{uploaded_file.name}.csv",
                mime="text/csv",
                help="下载包含 INPEFA、能量谱等计算值的表格数据"
            )

        with col_btn2:
            # 下载 HTML 图表
            html_buf = io.StringIO()
            fig.write_html(html_buf, include_plotlyjs='cdn')
            st.download_button(
                label="🌐 下载交互式图表 (HTML)",
                data=html_buf.getvalue(),
                file_name=f"XSimple_Plot_{uploaded_file.name}.html",
                mime="text/html",
                help="下载后可用浏览器直接打开，支持旋转、放大、查看特定深度值"
            )

        with col_btn3:
            st.info("💡 提示：如需保存图片，请将鼠标悬停在上方图表右上角，点击【相机图标 (Download plot as png)】。")

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
            - **INPEFA 阶数 ({inpefa_order})**: 增加阶数会使曲线更平滑，帮助识别二级或三级层序长周期趋势；降低阶数则保留更多局部细节。
            - **小波尺度 ({max_scale})**: 尺度越大，系统越能识别出超厚叠置组的地学旋回。
            - **频率带范围**: 根据地质经验微调，可剔除干扰杂波，精准锁定受天文驱动的沉积节拍
            """)

    else:
        st.error("数据加载失败，请检查文件格式。")
else:
    st.info("👋 请在左侧上传您的数据文件开始分析。")
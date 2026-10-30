"""
U字型分布（両外側が大きい分布）から標本平均が正規分布に収束するシミュレーション (Streamlit版)
========================================================================================
実行方法:
    pip install streamlit numpy plotly
    streamlit run u_dist_app.py
"""

import math
import time
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. ページ設定 & カスタムCSS（横画面2カラム固定・見切れ防止）
# ============================================================================
st.set_page_config(
    page_title="中心極限定理シミュレーター（U字型分布）",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* 1. Streamlitの上部ヘッダーバーを透明化＆高さを最小化 */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }

    /* 2. ページ全体のスクロールを許可 */
    html, body, [data-testid="stAppViewContainer"], .main {
        overflow-y: auto !important;
    }

    /* 3. コンテナ上部の余白を確保して見切れを防止 */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* サイドバー開閉ボタンを非表示 */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* PC横画面レイアウト：左右2カラムを折り返さず横並びで固定 */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }

    /* ヘッダーバナー */
    .sim-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 12px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .sim-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.3;
    }
    .sim-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin: 3px 0 0 0;
    }

    /* 左カラム：標本平均バナー（中央揃え） */
    .result-banner {
        background: linear-gradient(90deg, #ecfdf5 0%, #f0fdfa 50%, #ecfdf5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 10px 16px;
        margin-bottom: 10px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .result-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 2px;
    }
    .result-val-emerald {
        font-size: 1.9rem;
        font-weight: 800;
        color: #059669;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.1;
    }

    /* 10x10 グリッド（下に標本ヒストグラムを置くためコンパクトに設計） */
    .sampling-grid-wrapper {
        background: #f1f5f9;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        max-width: 310px;
        margin: 0 auto 8px auto;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }
    .sampling-grid {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 4px;
        aspect-ratio: 1 / 1;
    }
    .cell-num {
        font-weight: 800;
        font-size: 0.86rem;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        user-select: none;
        aspect-ratio: 1 / 1;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    /* 両端の 1 と 5 を強調し、中央の 3 を控えめにした配色 */
    .val-1 { background: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc; }
    .val-2 { background: #f8fafc; color: #475569; border: 1px solid #cbd5e1; }
    .val-3 { background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; }
    .val-4 { background: #f8fafc; color: #475569; border: 1px solid #cbd5e1; }
    .val-5 { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

    .sub-hist-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #475569;
        text-align: center;
        margin: 4px 0 0 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 2. 定数・母集団の確率分布（1〜5のU字型分布）・セッション状態の初期化
# ============================================================================
N_CELLS = 100  # 10x10 マス = 1標本あたり100個
VALUES = np.array([1, 2, 3, 4, 5], dtype=int)

# 1 - (反復試行の確率) 型のU字分布: P(X = k+1) = 0.36 - 0.05 * C(4, k)
# 両外側(1, 5)が31%、内側(2, 4)が16%、中央(3)が6%
PROBS = np.array([0.31, 0.16, 0.06, 0.16, 0.31], dtype=float)
# ※純粋な (1 - C(4,k)/16) / 4 を使う場合は以下の行のコメントを外してください:
# PROBS = np.array([15/64, 12/64, 10/64, 12/64, 15/64], dtype=float)

# 母平均 μ と 母分散 σ^2、標本平均(n=100)の標準偏差 σ_xbar
POP_MEAN = float(np.sum(VALUES * PROBS))                          # 3.00
POP_VAR = float(np.sum(((VALUES - POP_MEAN) ** 2) * PROBS))       # 2.80
SAMPLE_MEAN_VAR = POP_VAR / N_CELLS                               # 0.028
SAMPLE_MEAN_STD = math.sqrt(SAMPLE_MEAN_VAR)                      # ≒ 0.1673

# 100個の合計値は 100(すべて1) 〜 500(すべて5) の整数（標本平均 1.00 〜 5.00）
MIN_SUM = 100
MAX_SUM = 500

# 連続実行の高速設定
SLEEP_INTERVAL = 0.03
STEP_BATCH_SIZE = 5


def sample_one_grid() -> np.ndarray:
    """母集団の確率分布 PROBS に従って100個の整数(1〜5)をサンプリングする"""
    return np.random.choice(VALUES, size=N_CELLS, p=PROBS)


def init_session_state() -> None:
    """セッション状態を初期化する"""
    if "bin_width" not in st.session_state:
        st.session_state.bin_width = 0.10  # 初期階級幅: 0.10
    if "sum_counts" not in st.session_state:
        st.session_state.sum_counts = np.zeros(MAX_SUM + 1, dtype=int)
    if "total_trials" not in st.session_state:
        st.session_state.total_trials = 0
    if "latest_sum" not in st.session_state:
        st.session_state.latest_sum = None
    if "current_grid" not in st.session_state:
        st.session_state.current_grid = sample_one_grid()
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "show_normal_curve" not in st.session_state:
        st.session_state.show_normal_curve = True


init_session_state()


# ============================================================================
# 3. サンプリング & 階級（ビン）構築ロジック
# ============================================================================
def reset_simulation() -> None:
    """累積データをリセットして初期状態に戻す"""
    st.session_state.is_running = False
    st.session_state.sum_counts = np.zeros(MAX_SUM + 1, dtype=int)
    st.session_state.total_trials = 0
    st.session_state.latest_sum = None
    st.session_state.current_grid = sample_one_grid()


def run_trials(num_samples: int = 1) -> None:
    """指定回数のサンプリングを実行し、10×10グリッドおよび度数分布を更新する"""
    if num_samples <= 0:
        return

    last_grid = sample_one_grid()
    last_sum = int(np.sum(last_grid))

    if num_samples == 1:
        st.session_state.sum_counts[last_sum] += 1
        st.session_state.total_trials += 1
    else:
        batch_grids = np.random.choice(VALUES, size=(num_samples - 1, N_CELLS), p=PROBS)
        batch_sums = np.sum(batch_grids, axis=1)
        all_sums = np.append(batch_sums, last_sum)

        bincount = np.bincount(all_sums, minlength=MAX_SUM + 1)
        st.session_state.sum_counts += bincount
        st.session_state.total_trials += num_samples

    st.session_state.current_grid = last_grid
    st.session_state.latest_sum = last_sum


def build_bins(bin_width: float, sum_counts: np.ndarray, latest_sum: Optional[int]) -> List[Dict]:
    """
    合計値 S (100〜500) の刻み幅 step_s = round(bin_width * 100) に基づいて
    母平均 3.00 (S=300) が中央階級の中心付近になるよう階級を構築する。
    """
    step_s = int(round(bin_width * 100))  # 10, 5, 2
    bins = []

    if step_s == 10:
        # S = 300 が 295〜304 の中央になるようオフセット 5 で分割
        starts = [100] + list(range(105, MAX_SUM + 1, 10))
    elif step_s == 5:
        # S = 300 が 298〜302 のど真ん中になるようオフセット 3 で分割
        starts = [100] + list(range(103, MAX_SUM + 1, 5))
    else:
        # step_s == 2
        starts = list(range(100, MAX_SUM + 1, 2))

    for min_s in starts:
        if step_s == 10 and min_s == 100:
            max_s = 104
        elif step_s == 5 and min_s == 100:
            max_s = 102
        else:
            max_s = min(MAX_SUM, min_s + step_s - 1)

        c = int(np.sum(sum_counts[min_s : max_s + 1]))
        actual_w = (max_s - min_s + 1) / 100.0
        center_mean = (min_s + max_s) / 2.0 / 100.0
        min_mean = min_s / 100.0
        max_mean = max_s / 100.0

        r_lbl = f"{min_mean:.2f}" if min_s == max_s else f"{min_mean:.2f}〜{max_mean:.2f}"
        bins.append({
            "min_s": min_s,
            "max_s": max_s,
            "actual_width": actual_w,
            "center": center_mean,
            "count": c,
            "range_label": r_lbl,
            "contains_latest": (latest_sum is not None and min_s <= latest_sum <= max_s),
        })

    return bins


# ============================================================================
# 4. HTML & チャート描画ヘルパー
# ============================================================================
def render_10x10_grid_html(grid_vals: np.ndarray) -> str:
    """10×10の数値(1〜5)グリッドをHTMLとして生成する"""
    cells_html = [f'<div class="cell-num val-{v}">{v}</div>' for v in grid_vals]
    return f'<div class="sampling-grid-wrapper"><div class="sampling-grid">{"".join(cells_html)}</div></div>'


def render_left_result_banner_html(latest_sum: Optional[int]) -> str:
    """左カラム上部の「標本平均」バナーHTML（中央揃え）"""
    mean_str = "--" if latest_sum is None else f"{latest_sum / N_CELLS:.2f}"
    return f"""
    <div class="result-banner">
        <div class="result-label">標本平均</div>
        <div class="result-val-emerald">{mean_str}</div>
    </div>
    """


def create_sample_histogram_figure(grid_vals: np.ndarray, has_trial: bool) -> go.Figure:
    """
    左カラム下部：今回の標本（100個）そのもののヒストグラム（横軸 1〜5）
    """
    if has_trial:
        counts = [int(np.sum(grid_vals == v)) for v in VALUES]
    else:
        counts = [0, 0, 0, 0, 0]

    bar_colors = ["#6366f1", "#94a3b8", "#cbd5e1", "#94a3b8", "#10b981"]
    border_colors = ["#4338ca", "#64748b", "#94a3b8", "#64748b", "#059669"]
    text_vals = [f"{c}" if c > 0 else "" for c in counts]

    fig = go.Figure()

    # 100個中の実測個数バー
    fig.add_trace(
        go.Bar(
            x=VALUES,
            y=counts,
            width=0.58,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=1.2)),
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11, color="#334155"),
            hovertemplate="値 %{x}: <b>%{y}個</b> / 100個<extra></extra>",
        )
    )

    # 母集団の理論期待個数（31個, 16個, 6個, 16個, 31個）のガイド点線
    expected_counts = PROBS * N_CELLS
    fig.add_trace(
        go.Scatter(
            x=VALUES,
            y=expected_counts,
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=6, color="#f59e0b"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=175,
        margin=dict(l=32, r=16, t=12, b=28),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
            fixedrange=True,
            gridcolor="#f8fafc",
            linecolor="#cbd5e1",
            tickfont=dict(size=11, color="#334155"),
        ),
        yaxis=dict(
            range=[0, 50],
            dtick=10,
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickfont=dict(size=10, color="#64748b"),
        ),
    )
    return fig


def create_mean_histogram_figure(bins: List[Dict]) -> go.Figure:
    """
    右カラム：標本平均の相対度数ヒストグラム（縦軸目盛り固定・横軸 1〜5）
    """
    N = st.session_state.total_trials
    w = st.session_state.bin_width

    mean = POP_MEAN          # 3.00
    var = SAMPLE_MEAN_VAR    # 0.028
    std = SAMPLE_MEAN_STD    # ≒ 0.1673

    centers = [b["center"] for b in bins]
    counts = [b["count"] for b in bins]
    rel_freqs = [(c / N) if N > 0 else 0.0 for c in counts]
    widths = [b["actual_width"] * 0.88 for b in bins]
    range_labels = [b["range_label"] for b in bins]

    # 相対度数（Y軸）の上限と目盛りを完全固定
    y_max = 0.30
    y_dtick = 0.05

    fig = go.Figure()

    bar_colors = [
        "#0d9488" if b["contains_latest"] else "rgba(16, 185, 129, 0.82)"
        for b in bins
    ]
    border_colors = [
        "#065f46" if b["contains_latest"] else "#059669"
        for b in bins
    ]
    border_widths = [2.2 if b["contains_latest"] else 1.0 for b in bins]

    expected_freqs = []
    for b in bins:
        c_val = b["center"]
        z = (c_val - mean) / std
        pdf = (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)
        expected_freqs.append(b["actual_width"] * pdf)

    custom_data = np.column_stack([range_labels, counts, expected_freqs])

    fig.add_trace(
        go.Bar(
            x=centers,
            y=rel_freqs,
            width=widths,
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=border_widths),
            ),
            customdata=custom_data,
            hovertemplate=(
                "<b>標本平均: %{customdata[0]}</b><br>"
                "相対度数: <b>%{y:.3f}</b> (%{customdata[1]:,}回)<br>"
                "理論相対度数: 約 %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # 正規分布の理論曲線オーバーレイ
    if st.session_state.show_normal_curve:
        x_curve = np.linspace(1.0, 5.0, 400)
        z_curve = (x_curve - mean) / std
        pdf_curve = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z_curve**2)
        y_curve = w * pdf_curve
        line_dash = "solid" if N > 0 else "dash"

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                line=dict(color="#f59e0b", width=3.2, dash=line_dash),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=440,
        margin=dict(l=48, r=20, t=16, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(
            range=[1, 5],
            dtick=0.5,
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#cbd5e1",
            tickfont=dict(size=11, color="#475569"),
        ),
        yaxis=dict(
            range=[0, y_max],
            dtick=y_dtick,
            tickformat=".2f",
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
        ),
        bargap=0.06,
        hovermode="closest",
    )

    return fig


# ============================================================================
# 5. ヘッダー & トップコントロールバー
# ============================================================================
st.markdown(
    """
    <div class="sim-header">
        <div class="sim-title">
            標本平均が正規分布に収束するシミュレーション（U字型分布）
        </div>
        <p class="sim-subtitle">
            1～5の整数を100個を抽出し、その標本平均を右のヒストグラムに記録します。
            ただし、今回の母集団分布はU字型です。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

ctrl_cols = st.columns([1.2, 1.6, 1.0, 1.0, 1.0, 1.0])

with ctrl_cols[0]:
    if st.button("1回実行", use_container_width=True):
        st.session_state.is_running = False
        run_trials(1)

with ctrl_cols[1]:
    run_btn_label = "一時停止" if st.session_state.is_running else "連続実行"
    run_btn_type = "secondary" if st.session_state.is_running else "primary"
    if st.button(run_btn_label, type=run_btn_type, use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running:
            run_trials(STEP_BATCH_SIZE)

with ctrl_cols[2]:
    if st.button("+10回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(10)

with ctrl_cols[3]:
    if st.button("+100回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(100)

with ctrl_cols[4]:
    if st.button("+1000回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(1000)

with ctrl_cols[5]:
    if st.button("リセット", use_container_width=True):
        reset_simulation()
        st.rerun()


# ============================================================================
# 6. メイン2カラムレイアウト（左: 標本グリッド＆標本ヒストグラム / 右: 標本平均ヒストグラム）
# ============================================================================
all_bins = build_bins(
    st.session_state.bin_width,
    st.session_state.sum_counts,
    st.session_state.latest_sum,
)

col_left, col_right = st.columns([4.2, 6.8], gap="large")

# ---------------- 左カラム: 10×10 標本グリッド & 標本そのもののヒストグラム ----------------
with col_left:
    with st.container(border=True):
        # 1. 標本平均バナー（中央揃え）
        st.markdown(
            render_left_result_banner_html(st.session_state.latest_sum),
            unsafe_allow_html=True,
        )

        # 2. 10x10 数値グリッド
        st.markdown(
            render_10x10_grid_html(st.session_state.current_grid),
            unsafe_allow_html=True,
        )

        # 3. 標本そのもののヒストグラム（1〜5の出現個数）
        st.markdown(
            '<div class="sub-hist-title">今回の標本（100個）の度数分布</div>',
            unsafe_allow_html=True,
        )
        sample_fig = create_sample_histogram_figure(
            st.session_state.current_grid,
            has_trial=(st.session_state.latest_sum is not None),
        )
        st.plotly_chart(sample_fig, use_container_width=True, config={"displayModeBar": False})


# ---------------- 右カラム: 標本平均の相対度数ヒストグラム ----------------
with col_right:
    with st.container(border=True):
        BIN_WIDTH_LABELS = {
            0.10: "幅 0.10",
            0.05: "幅 0.05",
            0.02: "幅 0.02",
        }

        top_opt_cols = st.columns([6, 4])
        with top_opt_cols[0]:
            selected_bw = st.radio(
                "階級幅の選択",
                options=[0.10, 0.05, 0.02],
                format_func=lambda x: BIN_WIDTH_LABELS[x],
                index=[0.10, 0.05, 0.02].index(st.session_state.bin_width),
                horizontal=True,
            )
            if selected_bw != st.session_state.bin_width:
                st.session_state.bin_width = selected_bw
                st.rerun()

        with top_opt_cols[1]:
            st.write("")
            st.session_state.show_normal_curve = st.toggle(
                "正規分布の理論曲線を表示",
                value=st.session_state.show_normal_curve,
            )

        fig = create_mean_histogram_figure(all_bins)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# 7. 連続実行モードの自動更新ループ
# ============================================================================
if st.session_state.is_running:
    time.sleep(SLEEP_INTERVAL)
    run_trials(STEP_BATCH_SIZE)
    st.rerun()

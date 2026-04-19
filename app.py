import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import urllib.parse
import requests
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xgboost import XGBClassifier
import anthropic
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="주식 AI 리포트", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

try:
    DART_API_KEY = st.secrets["DART_API_KEY"]
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception as e:
    st.error("⚠️ API 키가 설정되지 않았습니다.")
    st.stop()

STOCK_OPTIONS = {
    "삼성전자": {"ticker": "005930.KS", "corp_code": "00126380"},
    "SK하이닉스": {"ticker": "000660.KS", "corp_code": "00164779"},
    "LG전자": {"ticker": "066570.KS", "corp_code": "00401731"},
    "네이버": {"ticker": "035420.KS", "corp_code": "00266961"},
    "카카오": {"ticker": "035720.KS", "corp_code": "00258801"},
}

MACRO_TICKERS = {"USDKRW": "KRW=X", "NASDAQ": "^IXIC", "VIX": "^VIX",
    "SP500": "^GSPC", "DXY": "DX-Y.NYB", "WTI": "CL=F", "GOLD": "GC=F"}

FEATURES = ["Samsung_Ret", "USDKRW_Ret", "NASDAQ_Ret_Lag1", "VIX_Ret_Lag1",
    "SP500_Ret_Lag1", "DXY_Ret_Lag1", "WTI_Ret_Lag1", "GOLD_Ret_Lag1",
    "Disparity5", "Disparity20", "Disparity60", "MA_Ratio",
    "RSI_f", "MACD_Hist", "Vol5", "VolRatio", "Mom5", "Mom20"]

BEST_PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
               "subsample": 0.8, "colsample_bytree": 0.8}

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
    model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_data(ttl=600)
def load_stock_data(ticker, period="3y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain/loss))
    return df

@st.cache_data(ttl=600)
def load_macro_data():
    macro = {}
    for name, tk in MACRO_TICKERS.items():
        d = yf.download(tk, period="3y", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        macro[name] = d["Close"]
    return pd.DataFrame(macro)

@st.cache_data(ttl=1800)
def load_news(company_name):
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(company_name)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:15]:
        parts = entry.title.rsplit(" - ", 1)
        items.append({"title": parts[0],
                      "press": parts[1] if len(parts) > 1 else "?",
                      "url": entry.link})
    return pd.DataFrame(items)

@st.cache_data(ttl=3600)
def load_disclosures(corp_code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    try:
        resp = requests.get("https://opendart.fss.or.kr/api/list.json",
            params={"crtfc_key": DART_API_KEY, "corp_code": corp_code,
                    "bgn_de": start_date, "end_de": end_date, "page_count": 100},
            timeout=10)
        data = resp.json()
        if data.get("status") == "000":
            return pd.DataFrame(data.get("list", []))
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "PER": info.get("trailingPE"), "PBR": info.get("priceToBook"),
            "PER_Forward": info.get("forwardPE"), "ROE": info.get("returnOnEquity"),
            "Dividend_Yield": info.get("dividendYield"), "Market_Cap": info.get("marketCap"),
            "EPS": info.get("trailingEps"), "Book_Value": info.get("bookValue"),
            "Debt_To_Equity": info.get("debtToEquity"), "Profit_Margin": info.get("profitMargins"),
            "Revenue_Growth": info.get("revenueGrowth"), "52w_High": info.get("fiftyTwoWeekHigh"),
            "52w_Low": info.get("fiftyTwoWeekLow"),
        }
    except:
        return {}

# 사이드바
st.sidebar.title("⚙️ 설정")
selected_name = st.sidebar.selectbox("📊 종목", list(STOCK_OPTIONS.keys()))
selected_info = STOCK_OPTIONS[selected_name]
selected_ticker = selected_info["ticker"]
selected_corp = selected_info["corp_code"]

period_options = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y", "3년": "3y"}
selected_period = st.sidebar.radio("📅 기간", list(period_options.keys()), index=3)
use_claude = st.sidebar.checkbox("🎓 Claude AI 분석", value=True)

if st.sidebar.button("🔄 새로고침", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""### 📌 사용법
**단기 관점**: AI 예측 탭
**장기 관점**: 장기 가치 탭
**둘 다 보기**: 🔮 종합 예측 탭

⚠️ 참고용. 최종 판단은 본인.
""")

# 메인
st.title(f"🤖 {selected_name} AI 리포트")
st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with st.spinner("📊 데이터 로드 중..."):
    df = load_stock_data(selected_ticker, period_options[selected_period])
    df_3y = load_stock_data(selected_ticker, "3y")

latest = df.iloc[-1]
prev = df.iloc[-2]
chg = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
rsi = latest["RSI"]
rsi_status = "과매수 ⚠️" if rsi > 70 else ("과매도 💡" if rsi < 30 else "중립")

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 현재가", f"{latest['Close']:,.0f}원", f"{chg:+.2f}%")
col2.metric("📊 MA20", f"{latest['MA20']:,.0f}원",
            f"{(latest['Close']/latest['MA20']-1)*100:+.1f}%")
col3.metric("📈 MA60", f"{latest['MA60']:,.0f}원",
            f"{(latest['Close']/latest['MA60']-1)*100:+.1f}%")
col4.metric("⚡ RSI", f"{rsi:.1f}", rsi_status)

# 7개 탭으로 확장 (종합 예측 추가)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 차트", "📰 뉴스 AI", "📋 공시", "🤖 AI 예측", "💎 장기 가치", "🔮 종합 예측", "🎓 Claude"
])

# 탭 1: 차트
with tab1:
    st.markdown("### 📈 주가 차트")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="종가",
                              line=dict(color="#1D3557", width=2.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                              line=dict(color="#F77F00", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], name="MA60",
                              line=dict(color="#E63946", width=1.5)))
    fig.update_layout(height=500, hovermode="x unified", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚡ RSI")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                  line=dict(color="#6A4C93", width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    fig_rsi.update_layout(height=300, plot_bgcolor="white", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

# 탭 2: 뉴스 AI
with tab2:
    st.markdown("### 📰 뉴스 AI 감성 분석")
    with st.spinner("뉴스 수집 중..."):
        news_df = load_news(selected_name)
        tokenizer, sent_model, device = load_sentiment_model()

        def analyze(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=128, padding=True).to(device)
            with torch.no_grad():
                probs = F.softmax(sent_model(**inputs).logits, dim=-1)[0].cpu().numpy()
            return {"label": ["부정","중립","긍정"][probs.argmax()],
                    "score": float(probs[2] - probs[0])}

        results = [analyze(t) for t in news_df["title"]]
        news_df["label"] = [r["label"] for r in results]
        news_df["score"] = [r["score"] for r in results]

    pos = (news_df["label"] == "긍정").sum()
    neg = (news_df["label"] == "부정").sum()
    neu = (news_df["label"] == "중립").sum()
    avg_score = news_df["score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 긍정", f"{pos}건")
    c2.metric("⚪ 중립", f"{neu}건")
    c3.metric("🔴 부정", f"{neg}건")
    c4.metric("평균", f"{avg_score:+.3f}")

    for _, row in news_df.iterrows():
        emoji = {"긍정": "🟢", "중립": "⚪", "부정": "🔴"}[row["label"]]
        with st.expander(f"{emoji} [{row['label']}] {row['title'][:70]}"):
            st.markdown(f"**언론사:** {row['press']} | **스코어:** {row['score']:+.3f}")
            st.markdown(f"[원문]({row['url']})")

    st.session_state["news_df"] = news_df
    st.session_state["news_stats"] = {"pos": pos, "neg": neg, "neu": neu, "avg": avg_score}

# 탭 3: 공시
with tab3:
    st.markdown("### 📋 DART 공시 (최근 30일)")
    with st.spinner("공시 수집 중..."):
        disc_df = load_disclosures(selected_corp)

    if len(disc_df) == 0:
        st.warning("공시 없음")
    else:
        st.metric("공시 건수", f"{len(disc_df)}건")
        display_df = disc_df[["rcept_dt", "report_nm"]].head(15).copy()
        display_df.columns = ["접수일", "공시명"]
        display_df["접수일"] = pd.to_datetime(display_df["접수일"], format="%Y%m%d").dt.strftime("%m/%d")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# 탭 4: AI 예측 (단기)
with tab4:
    st.markdown("### 🤖 XGBoost AI 단기 예측 (1주)")
    st.warning("""**⚠️ 올바른 해석**
- 상승 **확률**이지 상승 **폭**이 아닙니다
- AI 정확도 약 55% → 참고 자료로만 사용""")

    with st.spinner("AI 학습 중..."):
        macro_df = load_macro_data()
        merged = pd.concat(
            [df_3y[["Close","Volume"]].rename(columns={"Close":"Samsung","Volume":"SV"}),
             macro_df], axis=1).dropna()
        d = merged.copy()
        d["Target"] = (d["Samsung"].shift(-1) > d["Samsung"]).astype(int)
        d["Samsung_Ret"] = d["Samsung"].pct_change()
        d["USDKRW_Ret"] = d["USDKRW"].pct_change()
        for c in ["NASDAQ","VIX","SP500","DXY","WTI","GOLD"]:
            d[f"{c}_Ret_Lag1"] = d[c].pct_change().shift(1)
        for w in [5, 20, 60]:
            d[f"MA{w}_f"] = d["Samsung"].rolling(w).mean()
            d[f"Disparity{w}"] = d["Samsung"] / d[f"MA{w}_f"]
        d["MA_Ratio"] = d["MA5_f"] / d["MA20_f"]
        dd = d["Samsung"].diff()
        g = (dd.where(dd>0,0)).rolling(14).mean()
        l = (-dd.where(dd<0,0)).rolling(14).mean()
        d["RSI_f"] = 100 - (100/(1 + g/l))
        e12 = d["Samsung"].ewm(span=12, adjust=False).mean()
        e26 = d["Samsung"].ewm(span=26, adjust=False).mean()
        d["MACD_Hist"] = (e12-e26) - (e12-e26).ewm(span=9, adjust=False).mean()
        d["Vol5"] = d["Samsung_Ret"].rolling(5).std()
        d["VolRatio"] = d["SV"] / d["SV"].rolling(20).mean()
        d["Mom5"] = d["Samsung"].pct_change(5)
        d["Mom20"] = d["Samsung"].pct_change(20)
        d = d.dropna()
        X = d[FEATURES]
        y = d["Target"]
        weights = np.linspace(0.5, 1.5, len(X))
        xgb_model = XGBClassifier(**BEST_PARAMS, random_state=42, verbosity=0)
        xgb_model.fit(X, y, sample_weight=weights)
        chart_prob = xgb_model.predict_proba(X.iloc[-1:])[0, 1]

        price = merged["Samsung"]
        ma20_s = price.rolling(20).mean()
        ma60_s = price.rolling(60).mean()
        mom20 = price.pct_change(20)
        disp60 = price / ma60_s - 1
        score = 0
        if ma20_s.iloc[-1] > ma60_s.iloc[-1] * 1.02: score += 1
        elif ma20_s.iloc[-1] < ma60_s.iloc[-1] * 0.98: score -= 1
        if mom20.iloc[-1] > 0.05: score += 1
        elif mom20.iloc[-1] < -0.05: score -= 1
        if disp60.iloc[-1] > 0.05: score += 1
        elif disp60.iloc[-1] < -0.05: score -= 1
        if score >= 2: regime, re = "강세장", "🚀"
        elif score <= -2: regime, re = "약세장", "📉"
        else: regime, re = "횡보장", "🔀"

    prob_pct = chart_prob * 100
    if prob_pct >= 70: signal, sc = "🟢🟢 강한 상승 신호", "#06A77D"
    elif prob_pct >= 55: signal, sc = "🟢 약한 상승", "#558B2F"
    elif prob_pct >= 45: signal, sc = "⚪ 중립", "#666666"
    elif prob_pct >= 30: signal, sc = "🔴 약한 하락", "#C62828"
    else: signal, sc = "🔴🔴 강한 하락 신호", "#E63946"

    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("🤖 상승 확률", f"{prob_pct:.1f}%")
    cp2.metric(f"{re} 체제", regime)
    cp3.metric("스코어", f"{score:+d}")

    st.markdown(f"""<div style="background-color: {sc}20; padding: 20px; border-radius: 10px; 
                border-left: 5px solid {sc};"><h3 style="color: {sc}; margin: 0;">{signal}</h3>
                </div>""", unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_pct,
        title={"text": "상승 확률"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1D3557"},
            "steps": [
                {"range": [0, 30], "color": "#E63946"},
                {"range": [30, 45], "color": "#F77F00"},
                {"range": [45, 55], "color": "#FFE66D"},
                {"range": [55, 70], "color": "#9FCA2E"},
                {"range": [70, 100], "color": "#06A77D"}],
            "threshold": {"line": {"color": "black", "width": 4}, "value": 50}}))
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.session_state["chart_prob"] = chart_prob
    st.session_state["regime"] = regime
    st.session_state["short_signal"] = signal
    st.session_state["short_color"] = sc

# 탭 5: 장기 가치
with tab5:
    st.markdown("### 💎 장기 투자 지표")
    with st.spinner("재무 데이터..."):
        fund = load_fundamentals(selected_ticker)

    if not fund:
        st.error("데이터 로드 실패")
    else:
        st.markdown("#### 💰 밸류에이션")
        cv1, cv2, cv3, cv4 = st.columns(4)
        per = fund.get("PER")
        pbr = fund.get("PBR")
        div_yield = fund.get("Dividend_Yield")
        roe = fund.get("ROE")
        
        cv1.metric("PER", f"{per:.1f}배" if per else "N/A",
                  "저평가 ✅" if per and per < 10 else ("고평가 ⚠️" if per and per > 25 else ("적정" if per else "")))
        cv2.metric("PBR", f"{pbr:.2f}배" if pbr else "N/A",
                  "저평가 ✅" if pbr and pbr < 1 else ("고평가 ⚠️" if pbr and pbr > 3 else ("적정" if pbr else "")))
        if div_yield:
            div_pct = div_yield * 100 if div_yield < 1 else div_yield
            cv3.metric("배당수익률", f"{div_pct:.2f}%")
        else:
            cv3.metric("배당수익률", "N/A")
        if roe:
            cv4.metric("ROE", f"{roe*100:.1f}%",
                      "우수 ✅" if roe*100 > 15 else "")
        else:
            cv4.metric("ROE", "N/A")

        st.markdown("#### 📊 52주 범위")
        w_high = fund.get("52w_High")
        w_low = fund.get("52w_Low")
        if w_high and w_low:
            current = latest["Close"]
            position = (current - w_low) / (w_high - w_low) * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("최저", f"{w_low:,.0f}원")
            c2.metric("현재 위치", f"{position:.0f}%")
            c3.metric("최고", f"{w_high:,.0f}원")

        # 장기 투자 점수 계산
        score_long = 0
        reasons_long = []
        
        if per and per < 15:
            score_long += 1
            reasons_long.append(f"✅ PER {per:.1f}배 (저평가)")
        elif per and per > 25:
            score_long -= 1
            reasons_long.append(f"⚠️ PER {per:.1f}배 (고평가)")

        if pbr and pbr < 1.5:
            score_long += 1
            reasons_long.append(f"✅ PBR {pbr:.2f}배")
        elif pbr and pbr > 3:
            score_long -= 1

        if roe and roe * 100 > 15:
            score_long += 1
            reasons_long.append(f"✅ ROE {roe*100:.1f}% (수익성 우수)")
        elif roe and roe * 100 > 8:
            reasons_long.append(f"👍 ROE {roe*100:.1f}% (양호)")
        elif roe and roe * 100 < 5:
            score_long -= 1
            reasons_long.append(f"⚠️ ROE {roe*100:.1f}% (부진)")

        if div_yield:
            dp = div_yield * 100 if div_yield < 1 else div_yield
            if dp > 2:
                score_long += 1
                reasons_long.append(f"✅ 배당 {dp:.2f}%")

        # 52주 위치도 반영
        if w_high and w_low:
            pos = (latest["Close"] - w_low) / (w_high - w_low) * 100
            if pos < 30:
                score_long += 1
                reasons_long.append(f"✅ 52주 저점권 (하위 {pos:.0f}%)")
            elif pos > 85:
                score_long -= 1
                reasons_long.append(f"⚠️ 52주 고점권 (상위 {100-pos:.0f}%)")

        # 장기 투자 장기 확률 계산 (점수 → 확률 변환)
        # 점수 범위: -4 ~ +5 정도
        # 0점 = 50%, +1점당 +10%, -1점당 -10%
        long_prob = max(10, min(90, 50 + score_long * 10))

        if score_long >= 3: verdict_long, vc_long = "💎 장기 투자 매력적", "#06A77D"
        elif score_long >= 1: verdict_long, vc_long = "✅ 장기 투자 고려 가능", "#558B2F"
        elif score_long >= -1: verdict_long, vc_long = "⚪ 중립", "#666666"
        else: verdict_long, vc_long = "⚠️ 장기 투자 주의", "#E63946"

        st.markdown("#### 🏆 장기 투자 적합도")
        st.markdown(f"""<div style="background-color: {vc_long}20; padding: 15px; border-radius: 10px; 
                    border-left: 5px solid {vc_long};">
                    <h3 style="color: {vc_long}; margin: 0;">{verdict_long} (점수: {score_long:+d})</h3>
                    </div>""", unsafe_allow_html=True)

        st.markdown("**평가 근거:**")
        for r in reasons_long:
            st.markdown(r)

        # 세션에 저장 (종합 예측 탭에서 사용)
        st.session_state["long_prob"] = long_prob
        st.session_state["long_score"] = score_long
        st.session_state["long_verdict"] = verdict_long
        st.session_state["long_color"] = vc_long
        st.session_state["long_reasons"] = reasons_long

# 탭 6: 🔮 종합 예측 (신규!)
with tab6:
    st.markdown("### 🔮 종합 예측: 단기 vs 장기")
    st.caption("단기(1주)와 장기(1년) 관점을 한눈에 비교")

    # 세션 체크
    if "chart_prob" not in st.session_state or "long_prob" not in st.session_state:
        st.info("⚠️ 먼저 **🤖 AI 예측** 탭과 **💎 장기 가치** 탭을 방문해주세요. 데이터 로드가 필요합니다.")
    else:
        short_prob = st.session_state["chart_prob"] * 100
        long_prob = st.session_state["long_prob"]
        short_signal = st.session_state.get("short_signal", "중립")
        long_verdict = st.session_state.get("long_verdict", "중립")
        short_color = st.session_state.get("short_color", "#666666")
        long_color = st.session_state.get("long_color", "#666666")

        # 메인 비교 - 좌우 나란히
        col_s, col_l = st.columns(2)
        
        with col_s:
            st.markdown("#### 📊 단기 (1주일)")
            st.markdown(f"""<div style="background-color: {short_color}15; padding: 20px; 
                        border-radius: 10px; border: 2px solid {short_color}; text-align: center;">
                        <h1 style="color: {short_color}; margin: 0; font-size: 48px;">{short_prob:.1f}%</h1>
                        <p style="margin: 10px 0 0 0; font-size: 13px; color: #666;">상승 확률</p>
                        <p style="margin: 5px 0 0 0; font-weight: bold; color: {short_color};">{short_signal}</p>
                        </div>""", unsafe_allow_html=True)
            st.caption("🎯 XGBoost AI 기반 기술적 예측")

        with col_l:
            st.markdown("#### 📅 장기 (1년+)")
            st.markdown(f"""<div style="background-color: {long_color}15; padding: 20px; 
                        border-radius: 10px; border: 2px solid {long_color}; text-align: center;">
                        <h1 style="color: {long_color}; margin: 0; font-size: 48px;">{long_prob:.0f}%</h1>
                        <p style="margin: 10px 0 0 0; font-size: 13px; color: #666;">상승 확률</p>
                        <p style="margin: 5px 0 0 0; font-weight: bold; color: {long_color};">{long_verdict}</p>
                        </div>""", unsafe_allow_html=True)
            st.caption("💎 PER/ROE/재무 기반 가치 평가")

        st.markdown("---")

        # 듀얼 게이지 차트
        st.markdown("#### 📊 시각화")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_short = go.Figure(go.Indicator(
                mode="gauge+number", value=short_prob,
                title={"text": "단기 (1주)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": short_color},
                    "steps": [
                        {"range": [0, 30], "color": "#ffcccc"},
                        {"range": [30, 45], "color": "#ffe4cc"},
                        {"range": [45, 55], "color": "#fff4cc"},
                        {"range": [55, 70], "color": "#d4f4cc"},
                        {"range": [70, 100], "color": "#b4e8b4"}],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            fig_short.update_layout(height=300)
            st.plotly_chart(fig_short, use_container_width=True)

        with col_g2:
            fig_long = go.Figure(go.Indicator(
                mode="gauge+number", value=long_prob,
                title={"text": "장기 (1년+)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": long_color},
                    "steps": [
                        {"range": [0, 30], "color": "#ffcccc"},
                        {"range": [30, 45], "color": "#ffe4cc"},
                        {"range": [45, 55], "color": "#fff4cc"},
                        {"range": [55, 70], "color": "#d4f4cc"},
                        {"range": [70, 100], "color": "#b4e8b4"}],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            fig_long.update_layout(height=300)
            st.plotly_chart(fig_long, use_container_width=True)

        # 4분면 종합 전략 추천
        st.markdown("---")
        st.markdown("#### ⚡ 종합 투자 전략 추천")

        if short_prob >= 55 and long_prob >= 60:
            strategy = "🔥 강력 매수 기회"
            strategy_color = "#06A77D"
            strategy_desc = """
            **단기 ✅ + 장기 ✅ = 올인 시그널**
            - 단기에도 오를 가능성 높고, 장기 매력도 높음
            - 적극적 매수 고려 가능
            - 다만 분할 매수로 리스크 관리
            """
        elif short_prob < 45 and long_prob >= 60:
            strategy = "💎 분할 매수 기회"
            strategy_color = "#558B2F"
            strategy_desc = """
            **단기 🔴 + 장기 ✅ = 조정 후 매수**
            - 단기 조정 가능성 있으나 장기 매력도 높음
            - 급하지 않게 분할 매수 접근
            - "단기 하락 = 할인 기회" 관점
            - 가치투자 스타일 (버핏 방식)
            """
        elif short_prob >= 55 and long_prob < 40:
            strategy = "⚡ 단기 트레이딩만"
            strategy_color = "#F77F00"
            strategy_desc = """
            **단기 ✅ + 장기 🔴 = 빠른 익절**
            - 단기 반등 가능성 있지만 장기 매력도 낮음
            - 장기 보유 비추, 짧게 치고 빠지기
            - 손절선 미리 설정 필수
            - 초보자에겐 추천 안 함
            """
        elif short_prob < 45 and long_prob < 40:
            strategy = "🚫 관망 / 회피"
            strategy_color = "#E63946"
            strategy_desc = """
            **단기 🔴 + 장기 🔴 = 피하기**
            - 양쪽 다 부정적 시그널
            - 현재는 매수 타이밍 아님
            - 보유 중이면 비중 축소 고려
            - 다른 종목 탐색 추천
            """
        else:
            strategy = "🤔 관망 권장"
            strategy_color = "#666666"
            strategy_desc = """
            **단기 ⚪ + 장기 ⚪ = 신호 불분명**
            - 뚜렷한 매수/매도 신호 없음
            - 추가 정보 수집 필요
            - 뉴스, 공시 확인 후 판단
            - Claude 리포트 참고 추천
            """

        st.markdown(f"""<div style="background-color: {strategy_color}20; padding: 25px; 
                    border-radius: 10px; border-left: 5px solid {strategy_color};">
                    <h2 style="color: {strategy_color}; margin: 0;">{strategy}</h2>
                    </div>""", unsafe_allow_html=True)
        st.markdown(strategy_desc)

        # 예측 매트릭스 시각화
        st.markdown("---")
        st.markdown("#### 🗺️ 현재 위치 (매트릭스)")
        
        fig_matrix = go.Figure()
        # 4분면 배경
        fig_matrix.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100,
                             fillcolor="#d4f4cc", opacity=0.3, line=dict(width=0))
        fig_matrix.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                             fillcolor="#fff4cc", opacity=0.3, line=dict(width=0))
        fig_matrix.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50,
                             fillcolor="#ffe4cc", opacity=0.3, line=dict(width=0))
        fig_matrix.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50,
                             fillcolor="#ffcccc", opacity=0.3, line=dict(width=0))
        # 중앙선
        fig_matrix.add_shape(type="line", x0=50, y0=0, x1=50, y1=100,
                             line=dict(color="gray", width=1, dash="dash"))
        fig_matrix.add_shape(type="line", x0=0, y0=50, x1=100, y1=50,
                             line=dict(color="gray", width=1, dash="dash"))
        # 현재 위치
        fig_matrix.add_trace(go.Scatter(x=[short_prob], y=[long_prob],
            mode="markers+text",
            marker=dict(size=30, color=strategy_color, line=dict(color="black", width=2)),
            text=[selected_name], textposition="top center",
            textfont=dict(size=14, color="black"), showlegend=False))
        # 4분면 라벨
        fig_matrix.add_annotation(x=75, y=75, text="🔥 강력 매수", showarrow=False, font=dict(size=12))
        fig_matrix.add_annotation(x=25, y=75, text="💎 분할 매수", showarrow=False, font=dict(size=12))
        fig_matrix.add_annotation(x=75, y=25, text="⚡ 단기만", showarrow=False, font=dict(size=12))
        fig_matrix.add_annotation(x=25, y=25, text="🚫 회피", showarrow=False, font=dict(size=12))
        
        fig_matrix.update_layout(
            xaxis=dict(title="← 하락  |  단기 상승 확률 (%)  |  상승 →", range=[0, 100]),
            yaxis=dict(title="← 부정적  |  장기 매력도 (%)  |  긍정적 →", range=[0, 100]),
            height=500, plot_bgcolor="white",
            title="📍 현재 포지션")
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.info("""💡 **매트릭스 해석법**
        - **오른쪽 위**: 단기도 좋고 장기도 좋음 → 매수 적극 고려
        - **왼쪽 위**: 단기 약하지만 장기 매력적 → 조정 시 매수
        - **오른쪽 아래**: 단기 좋지만 장기 약함 → 단타만
        - **왼쪽 아래**: 양쪽 다 안 좋음 → 피하기""")

# 탭 7: Claude
with tab7:
    st.markdown("### 🎓 Claude 전문가 리포트")
    if not use_claude:
        st.info("사이드바에서 Claude AI 체크하세요.")
    else:
        if st.button("🎬 분석 시작", type="primary"):
            if "news_df" not in st.session_state:
                st.error("뉴스 AI 탭 먼저 방문")
            elif "chart_prob" not in st.session_state:
                st.error("AI 예측 탭 먼저 방문")
            else:
                ns = st.session_state["news_df"]
                st2 = st.session_state["news_stats"]
                cp = st.session_state["chart_prob"]
                rg = st.session_state["regime"]
                long_p = st.session_state.get("long_prob", 50)
                long_v = st.session_state.get("long_verdict", "중립")
                top_p = ns.nlargest(3, "score")[["title"]].to_dict("records")
                top_n = ns.nsmallest(3, "score")[["title"]].to_dict("records")

                prompt = f"""당신은 15년 경력 한국 주식 애널리스트입니다.
{selected_name} 리포트:

현재가: {int(latest['Close']):,}원 ({chg:+.2f}%)
RSI: {rsi:.1f}
체제: {rg}
단기 예측(1주): 상승 {cp*100:.1f}%
장기 예측(1년+): 상승 {long_p:.0f}% ({long_v})
뉴스: 평균 {st2['avg']:+.3f} | 긍정 {st2['pos']}/중립 {st2['neu']}/부정 {st2['neg']}

호재:
{chr(10).join([f"- {n['title'][:60]}" for n in top_p])}

악재:
{chr(10).join([f"- {n['title'][:60]}" for n in top_n])}

6섹션 리포트:
### 1. 한 줄 요약
### 2. 주가 해석
### 3. 주의 신호
### 4. 긍정 시그널
### 5. 단기 vs 장기 관점
### 6. 주목 포인트

각 2-4줄 간결하게."""
                try:
                    with st.spinner("Claude 분석..."):
                        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                        msg = client.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=2000,
                            messages=[{"role": "user", "content": prompt}])
                        report = msg.content[0].text
                        cost = (msg.usage.input_tokens*3 + msg.usage.output_tokens*15)/1_000_000
                    st.success(f"완료! ${cost:.4f} (~{cost*1400:.0f}원)")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"오류: {e}")

st.markdown("---")
st.caption(f"⚠️ 투자 참고용 | v4.0 | {datetime.now().strftime('%Y-%m-%d')}")

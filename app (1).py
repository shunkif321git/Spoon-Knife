import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob

st.set_page_config(page_title="感情分析アプリ", layout="centered")
st.title("🧠 簡単！感情分析アプリ")
st.markdown("このアプリでは、入力されたテキストの感情（ポジティブ／ネガティブ）を自動判定します。")

st.divider()
st.header("① 単一テキストの感情分析")

text_input = st.text_area("感情を判定したいテキストを入力してください(English)", height=150)

if st.button("感情を判定する"):
    with st.spinner("分析中..."):
        blob = TextBlob(text_input)
        polarity = blob.sentiment.polarity
        sentiment = "ポジティブ 😊" if polarity > 0 else "ネガティブ 😠" if polarity < 0 else "中立 😐"
        st.success(f"判定結果：{sentiment}（スコア: {polarity:.2f}）")

st.divider()
st.header("② CSVアップロードによるバッチ判定")

uploaded_file = st.file_uploader("CSVファイルをアップロード（'text'列を含む）", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSVに'text'という列が必要です。")
    else:
        st.write("アップロードされたデータ:")
        st.dataframe(df.head())

        if st.button("バッチ判定を実行"):
            with st.spinner("一括処理中..."):
                df["polarity"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                df["sentiment"] = df["polarity"].apply(
                    lambda p: "ポジティブ" if p > 0 else "ネガティブ" if p < 0 else "中立"
                )
                st.success("判定が完了しました。")
                st.dataframe(df[["text", "sentiment", "polarity"]])

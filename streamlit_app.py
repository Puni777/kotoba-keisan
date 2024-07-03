import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import gensim

modelName = st.selectbox(
    "モデルを選択",
    ["日本語 Wikipedia エンティティベクトル2017",
     "chiVe"]
)

modelDict = {"日本語 Wikipedia エンティティベクトル2017":0, "chiVe":1}


st.title("言葉の足し算＆引き算")
st.text("\
 言葉の足し算引き算を行います。\n\
 足し算したい言葉と引き算したい言葉を入力してください。\n\
 スペースを開けることで複数個入力することができます。")


positiveWord = st.text_input("足し算したい言葉を入力")
negativeWord = st.text_input("引き算したい言葉を入力")

@st.cache_data

def modelInit(num):
    if num == 0:
        model = gensim.models.KeyedVectors.load_word2vec_format('./models/entity_vector.model.bin', binary=True)
    if num == 1:
        model = gensim.models.KeyedVectors.load("./models/chive-1.3-mc5.kv")
    return model

model = modelInit(modelDict[modelName])

if st.button("計算"):
    p = positiveWord.split("　")
    n = negativeWord.split("　")
    flag = 0
    try:
        if not(p[0] == "") and not(n[0] == "") :
            st.write(model.most_similar(positive=p.copy(), negative=n.copy()))
        if not(p[0] == "") and n[0] == "" :
            st.write(model.most_similar(positive=p.copy()))
        if  p[0] == "" and not(n[0] == "") :
            st.write(model.most_similar(negative=n.copy()))
        if p[0] == "" and n[0] == "" :
            st.text("何も入力されていません")
    except:
        st.write("学習データに無い言葉のようです……")
        

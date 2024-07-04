import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import gensim

modelName = st.selectbox(
    "モデルを選択",
    ["東北大学_Wikipedia 2017",
     "chiVe_CommonCrawl",
     "@Hironsan_Wikipedia 2017",
     "@Frq09_Wikipedia 2023"],
)

modelDict = {"東北大学_Wikipedia 2017":0, 
             "chiVe_CommonCrawl":1, 
             "@Hironsan_Wikipedia 2017":2,
             "@Frq09_Wikipedia 2023":3}


st.title("言葉の足し算＆引き算")
st.text("\
 言葉の足し算引き算を行います。\n\
 足し算したい言葉と引き算したい言葉を入力してください。\n\
 スペースを開けることで複数個入力することができます。")


positiveWord = st.text_input("足し算したい言葉を入力")
negativeWord = st.text_input("引き算したい言葉を入力")

@st.cache_resource(max_entries=1)
def modelInit(num):
    if num == 0:
        model = gensim.models.KeyedVectors.load_word2vec_format('./models/entity_vector.model.bin', binary=True)
    if num == 1:
        model = gensim.models.KeyedVectors.load("./models/chive-1.3-mc5.kv")
    if num == 2:
        model = gensim.models.KeyedVectors.load_word2vec_format('./models/fastText_model.vec', binary=False)
    if num == 3:
        model = gensim.models.KeyedVectors.load('./models/2023-03-01-word2vec.model')
    return model



model = modelInit(modelDict[modelName])

if st.button("計算"):
    p = positiveWord.split("　")
    n = negativeWord.split("　")
    flag = 0
    try:
        if modelDict[modelName] == 3:
            if not(p[0] == "") and not(n[0] == "") :
                st.write(model.wv.most_similar(positive=p.copy(), negative=n.copy()))
            if not(p[0] == "") and n[0] == "" :
                st.write(model.wv.most_similar(positive=p.copy()))
            if  p[0] == "" and not(n[0] == "") :
                st.write(model.wv.most_similar(negative=n.copy()))
            if p[0] == "" and n[0] == "" :
                st.text("何も入力されていません")
        else:
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
        

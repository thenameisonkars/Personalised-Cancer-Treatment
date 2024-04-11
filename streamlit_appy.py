import streamlit as st
import pandas as pd
import plotly.express as px
from main import predict

st.write(""" 
# Personalized Cancer Treatment

You can use this tool, given some Info, to detect the Class.  
"""
         )

Variation = st.text_input("Enter a Variation", key=1)
Gene = st.text_input("Enter a Gene", key=2)
TEXT = st.text_area("Enter a TEXT")

# input=np.array([[Variation , Gene, TEXT]]).astype(np.float64)
df = pd.DataFrame({"Variation": [Variation], "Gene": [Gene], "TEXT": [TEXT]})

if st.button("Class ?"):
    result = predict(df)
    st.success(f"{result}")

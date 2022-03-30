import streamlit as st

from flawsleuth.ai import predict
from flawsleuth.timeseries import fetch_data_frame


def run():
    st.write("Here's an entity series from Quantum Leap:")
    df = fetch_data_frame()
    st.write(df)

    st.write("Is there a flaw?")
    answer = predict(12.3)
    st.write(answer)

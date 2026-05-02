import streamlit as st


def render():
    st.header("Live Attendance")

    session_id = st.text_input("Session ID", placeholder="e.g. CS401-2025-05-12")

    col1, col2 = st.columns([3, 1])
    with col1:
        # TODO (Person 3, Week 3): replace with streamlit-webrtc stream + bbox overlay
        st.info("Live webcam stream with bounding boxes will appear here.")
    with col2:
        st.metric("Present", "—")
        st.metric("Unknown", "—")
        st.metric("Spoof attempts", "—")

    if st.button("End Session", disabled=not session_id):
        st.success("Session ended. Go to Attendance History to export the CSV.")

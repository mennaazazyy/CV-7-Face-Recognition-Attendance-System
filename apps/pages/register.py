import streamlit as st


def render():
    st.header("Register Student")

    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", placeholder="e.g. 22-101003")
    with col2:
        full_name = st.text_input("Full Name", placeholder="e.g. Menna Azazy")

    st.subheader("Webcam Preview")
    # TODO (Person 3, Week 2): wire streamlit-webrtc here
    img_placeholder = st.empty()
    img_placeholder.info("Webcam preview — connect streamlit-webrtc component here.")

    n_frames = st.slider("Frames to capture", min_value=10, max_value=40, value=25)

    if st.button("Start Enrollment", disabled=not (student_id and full_name)):
        with st.spinner(f"Capturing {n_frames} frames for {full_name}…"):
            try:
                from src.pipeline.enroll import enroll_student
                enroll_student(student_id, full_name, n_frames=n_frames)
                st.success(f"✅ {full_name} enrolled successfully!")
            except Exception as exc:
                st.error(f"Enrollment failed: {exc}")

import streamlit as st

_ADMIN_PASSWORD = "cv7admin"   # override via config.py in production


def render():
    st.header("Admin")

    if "admin_auth" not in st.session_state:
        st.session_state.admin_auth = False

    if not st.session_state.admin_auth:
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == _ADMIN_PASSWORD:
                st.session_state.admin_auth = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    st.success("Logged in as admin.")

    st.subheader("Delete Student")
    del_id = st.text_input("Student ID to delete")
    if st.button("Delete", disabled=not del_id):
        confirm = st.checkbox(f"Confirm permanent deletion of {del_id}?")
        if confirm:
            from src.database.db import delete_student
            delete_student(del_id)
            st.success(f"{del_id} deleted.")

    if st.button("Logout"):
        st.session_state.admin_auth = False
        st.rerun()

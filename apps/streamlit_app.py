import streamlit as st

st.set_page_config(
    page_title="CV-7 Face Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "🏠 Home": "home",
    "📋 Register Student": "register",
    "📸 Live Attendance": "live_attendance",
    "📊 Attendance History": "history",
    "🔒 Admin": "admin",
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]

if page == "home":
    st.title("CV-7 Face Attendance System")
    st.markdown(
        """
        Welcome! Use the sidebar to navigate:
        - **Register Student** — enrol a new student via webcam
        - **Live Attendance** — run a live attendance session
        - **Attendance History** — browse and export records
        - **Admin** — manage students and settings
        """
    )

elif page == "register":
    from apps.pages import register as reg_page
    reg_page.render()

elif page == "live_attendance":
    from apps.pages import live_attendance as live_page
    live_page.render()

elif page == "history":
    from apps.pages import history as hist_page
    hist_page.render()

elif page == "admin":
    from apps.pages import admin as admin_page
    admin_page.render()

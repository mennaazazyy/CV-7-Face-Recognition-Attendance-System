import streamlit as st
import pandas as pd


def render():
    st.header("Attendance History")

    session_id = st.text_input("Session ID to view", placeholder="CS401-2025-05-12")

    if st.button("Load") and session_id:
        from src.database.db import query_attendance
        rows = query_attendance(session_id)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Export CSV",
                data=csv_bytes,
                file_name=f"attendance_{session_id}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No records found for that session ID.")

import streamlit as st
import requests
from utils import BACKEND_URL


def send_feedback_email(issue_type: str, description: str) -> bool:
    """
    Send feedback to the backend email service.

    Args:
        issue_type: Type of feedback (Bug Report, Feature Request, General Feedback)
        description: Description of the feedback

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Prepare session state information (alphabetically sorted)
        session_info = {}
        sorted_keys = sorted(st.session_state.keys())
        for key in sorted_keys:
            value = st.session_state[key]
            # Skip sensitive values
            if key not in ['access_token', 'token_type']:
                session_info[key] = str(value)

        # Send request to backend
        response = requests.post(
            f"{BACKEND_URL}/email/send-feedback",
            json={
                "issue_type": issue_type,
                "description": description,
                "session_info": session_info
            },
            timeout=10
        )

        return response.status_code == 200

    except requests.exceptions.RequestException as e:
        print(f"Error sending feedback to backend: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error sending email: {str(e)}")
        return False


# Modal dialog for feedback
@st.dialog("Report Issue or Suggestion")
def feedback_modal():
    # Get current page from session state
    current_page = st.session_state.get("Page", "Unknown")

    issue_type = st.selectbox(
        "What would you like to report?",
        ["Bug Report", "Feature Request", "General Feedback"]
    )
    
    description = st.text_area(
        "Description",
        placeholder="Please describe the issue or suggestion in detail...",
        height=150
    )
    
    col1, col2 = st.columns([1, 1])
    if col1.button("Cancel", use_container_width=True):
        st.rerun()
    if col2.button("Submit", type="primary", use_container_width=True):
        # Send feedback via email
        if send_feedback_email(issue_type, description):
            st.success("✅ Feedback submitted successfully!")
        else:
            st.error("❌ Failed to send feedback. Please try again.")
        st.rerun()


def render_sidebar_footer():
    # Logout button in sidebar
    with st.sidebar:
        # col1, col2 = st.columns(2)

        # TODO: fix email
        # # Feedback button (primary action)
        # with col1:
        #     if st.button("📝 Feedback", use_container_width=True, type="primary", help="Report issues or suggest features"):
        #         feedback_modal()

        # Logout button
        # with col2:
        if st.button("👋 Logout", use_container_width=True, help="Sign out"):
            # Clear session state
            for key in ['access_token', 'token_type', 'user_email']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Compact warning
        st.caption("🚧 Under development — data saved to cloud")

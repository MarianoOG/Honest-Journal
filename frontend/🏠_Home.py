from typing import Optional
from email_validator import validate_email, EmailNotValidError
from datetime import datetime
import streamlit as st
from utils import login_user, register_user, get_reflections, get_reflection_emoji
from footer import render_sidebar_footer
    

def format_time(date_str: str) -> str:
    """Format datetime string to show just the time"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%H:%M")
    except:
        return "--:--"

def get_date_group_header(date_str: str) -> str:
    """Get a readable date header for grouping entries"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now()
        
        # Calculate time difference
        diff = now - dt.replace(tzinfo=None)
        
        # If today
        if diff.days == 0:
            return "**📅 Today**"
        # If yesterday
        elif diff.days == 1:
            return "**📅 Yesterday**"
        # If this week
        elif diff.days < 7:
            return f"**📅 {dt.strftime('%A')}**"
        # If this year
        elif dt.year == now.year:
            return f"**📅 {dt.strftime('%B %d')}**"
        # Otherwise, show full date
        else:
            return f"**📅 {dt.strftime('%B %d, %Y')}**"
    except:
        return "**📅 Unknown Date**"

def group_reflections_by_date(reflections):
    """Group reflections by date for display"""
    from collections import defaultdict
    
    grouped = defaultdict(list)
    
    for reflection in reflections:
        try:
            dt = datetime.fromisoformat(reflection.get("created_at", "").replace('Z', '+00:00'))
            # Use just the date part for grouping
            date_key = dt.date()
            grouped[date_key].append(reflection)
        except:
            # Fallback for entries with invalid dates
            grouped[None].append(reflection)
    
    # Sort groups by date (most recent first)
    sorted_groups = []
    for date_key in sorted(grouped.keys(), reverse=True, key=lambda x: x if x else datetime.min.date()):
        reflections_for_date = grouped[date_key]
        # Sort reflections within each date group by time (most recent first)
        reflections_for_date.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        sorted_groups.append((date_key, reflections_for_date))
    
    return sorted_groups

def check_email(user_email: str, check_deliverability: bool = False) -> Optional[str]:
    try:
        email_info = validate_email(user_email, check_deliverability=check_deliverability)
        return email_info.normalized
    except EmailNotValidError:
        st.error("Email is not valid")
        return None

def render_login():
    st.title("📔 Honest Journal - Login")
    
    # Toggle between login and register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.write("Please enter your email and password")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", disabled=not email or not password):
            validated_email = check_email(email)
            if validated_email:
                token_response = login_user(validated_email, password)
                if token_response:
                    st.session_state.access_token = token_response["access_token"]
                    st.session_state.token_type = token_response["token_type"]
                    st.session_state.user_email = validated_email
                    st.success("Login successful!")
                    st.switch_page("pages/1_✍️_Journal.py")
    
    with tab2:
        st.write("Create a new account")
        name = st.text_input("Name", key="register_name")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password", 
                                 help="Password must be at least 8 characters long")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
        
        # Check if passwords match
        passwords_match = password == confirm_password if password and confirm_password else True
        if not passwords_match and confirm_password:
            st.error("Passwords do not match")
        
        register_disabled = (not name or not email or not password or not confirm_password or 
                           len(password) < 8 or not passwords_match)
        
        if st.button("Register", disabled=register_disabled):
            validated_email = check_email(email)
            if validated_email:
                user_response = register_user(name, validated_email, password)
                if user_response:
                    st.success("Account created successfully! Logging you in...")
                    # Auto-login after successful registration
                    token_response = login_user(validated_email, password)
                    if token_response:
                        st.session_state.access_token = token_response["access_token"]
                        st.session_state.token_type = token_response["token_type"]
                        st.session_state.user_email = validated_email
                        st.switch_page("pages/1_✍️_Journal.py")
            else:
                st.error("Email verification failed")


def render_reflections():    
    # Initialize pagination state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("All entries")
    
    with col2:
        if st.button("New entry", key="create_new_entry", type="primary", use_container_width=True):
            st.session_state.current_reflection_id = None
            st.session_state.mode = "edit"
            st.switch_page("pages/1_✍️_Journal.py")
    
    # Get reflections with pagination (request 11 to check if there's a next page)
    items_per_page = 10
    offset = (st.session_state.current_page - 1) * items_per_page
    reflections = get_reflections(limit=items_per_page + 1, offset=offset)

    if reflections:
        # Check if there are more pages
        has_next = len(reflections) > items_per_page
        current_reflections = reflections[:items_per_page]  # Show only 10

        # Group entries by date
        grouped_reflections = group_reflections_by_date(current_reflections)
        
        # Display grouped entries
        for date_key, reflections_for_date in grouped_reflections:
            # Show date header
            if date_key:
                header = get_date_group_header(reflections_for_date[0].get("created_at", ""))
            else:
                header = "**📅 Unknown Date**"
                
            st.write(header)
            
            # Show entries for this date
            for reflection in reflections_for_date:
                render_entry(reflection)
                
            # Add some space between date groups
            st.markdown("")
        
        # Pagination controls
        st.markdown("---")
        pcol1, _, pcol3 = st.columns([1, 3, 1])
        
        with pcol1:
            if st.session_state.current_page > 1:
                if st.button("← Previous", key="prev_page", use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()
        
        with pcol3:
            if has_next:
                if st.button("Next →", key="next_page", use_container_width=True):
                    st.session_state.current_page += 1
                    st.rerun()
                    
    else:
        st.info("No reflections found, create a new entry to get started.")
        

def render_entry(reflection: dict):
    """Render a single reflection entry with time and question"""
    time_formatted = format_time(reflection.get("created_at", ""))

    # Create a clean entry layout
    col1, col2, col3 = st.columns([1, 8, 1])

    with col1:
        st.caption(f"🕐 {time_formatted}")

    with col2:
        emoji = get_reflection_emoji(reflection)
        st.write(f"{emoji} {reflection["question"]}")

    with col3:
        # Make the question clickable to navigate to journal
        if st.button("➤", 
                    key=f"entry_{reflection['id']}", 
                    use_container_width=True,
                    type="secondary"):
            # Navigate to journal page with this reflection
            st.session_state.current_reflection_id = reflection["id"]
            st.session_state.mode = "view"
            st.switch_page("pages/1_✍️_Journal.py")


def main():
    st.session_state["Page"] = "Home"
    
    # Login form
    if "access_token" not in st.session_state:
        render_login()
        return
    
    # Title
    st.title("📔 Honest Journal")
    
    # Render reflections
    render_reflections()
    render_sidebar_footer()

if __name__ == "__main__":
    st.set_page_config(layout="centered", page_icon="📔", page_title="Honest Journal")
    main()

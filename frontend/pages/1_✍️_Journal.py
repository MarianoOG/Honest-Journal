from typing import Optional
import streamlit as st
from utils import (
    sentiment_emojis,
    get_reflection,
    get_reflections,
    get_reflection_parent,
    get_reflection_children,
    get_reflection_themes,
    save_reflection,
    delete_reflection,
    analyze_reflection,
    truncate_text,
    get_reflection_emoji
)
from footer import render_sidebar_footer

def render_edit_mode(reflection: Optional[dict] = None):
    """Render the edit interface"""

    with st.form("reflection_form", border=False):
        title = "✍️ Journal - New"
        if reflection and reflection.get("question"):
            title = "✍️ " + reflection['question']
        st.subheader(title)

        # Determine text area label based on context
        text_area_label = "Write your thoughts, reflections and memories"
        if reflection and reflection.get("context"):
            text_area_label = "Context: " + reflection.get('context', '')

        answer = st.text_area(
            text_area_label,
            value=reflection.get("answer", "") if reflection else "",
            max_chars=2000,
            height=400
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if reflection:
                cancel_clicked = st.form_submit_button("❌ Cancel", use_container_width=True, disabled=not reflection.get("answer", ""))
                if cancel_clicked:
                    if reflection:
                        st.session_state.mode = "view"
                        st.rerun()
                    else:
                        st.switch_page("🏠_Home.py")
        with col2:
            save_clicked = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
        with col3:
            if reflection:
                delete_clicked = st.form_submit_button("🗑️ Delete", type="secondary", use_container_width=True)
                if delete_clicked:
                    if delete_reflection(reflection["id"]):
                        st.success("Reflection deleted!")
                        st.session_state.current_reflection_id = None
                        st.rerun()
                    else:
                        st.error("Failed to delete reflection")
        
        if save_clicked:
            if not answer.strip():
                st.error("Answer is required!")
                return
            
            reflection_data = {
                "language": reflection.get("language", "") if reflection else "en",
                "sentiment": reflection.get("sentiment", "") if reflection else "Neutral",
                "parent_id": reflection.get("parent_id", "") if reflection else None,
                "context": reflection.get("context", "") if reflection else None,
                "question": reflection.get("question", "") if reflection else "",
                "answer": answer if answer.strip() else None
            }

            if reflection:
                reflection_data["id"] = reflection["id"]
            
            result = save_reflection(reflection_data)
            if result:
                st.success("Reflection saved successfully! 🎉")
                st.session_state.current_reflection_id = result["id"]

                # Auto-analyze the reflection
                with st.spinner("Analyzing your reflection..."):
                    analyze_reflection(result["id"])

                st.session_state.mode = "view"
                st.rerun()
            else:
                st.error("Failed to save reflection")


def render_view_mode(reflection: dict):
    """Render the view interface"""
    col1, col2 = st.columns([3, 1])

    with col1:
        # Subtitle
        emoji = get_reflection_emoji(reflection)
        st.subheader(f"{emoji} {reflection['question']}")
        if reflection.get("context"):
            st.write(f"**Context:** {reflection['context']}")
        
        # Answer
        if reflection.get("answer"):
            st.markdown(reflection["answer"])
        else:
            st.warning("No reflection content yet. Click Edit to add your thoughts.")

        # Render actions
        render_actions(reflection)

    with col2:   
        render_metadata(reflection)


def render_metadata(reflection: dict):
    # Render sentiment
    sentiment_emoji = sentiment_emojis.get(reflection["sentiment"], "😐")
    st.metric("Sentiment", f"{sentiment_emoji} {reflection['sentiment']}")

    # Render themes
    themes = get_reflection_themes(reflection["id"])
    if themes:
        with st.expander("🏷️ Themes", expanded=True):
            for theme in themes:
                st.markdown(f"• {theme['name']}")
    else:
        st.info("No themes assigned")

    # Render parent and children relationships
    st.markdown("### 🔗 Linked Reflections")
    
    # Render parent
    parent = get_reflection_parent(reflection["id"])
    if parent:
        if st.button(f"⬆️ Started From: {truncate_text(parent['question'], 60)}", key="parent_btn", use_container_width=True):
            st.session_state.current_reflection_id = parent["id"]
            st.rerun()

    # Render Children
    children = get_reflection_children(reflection["id"])
    if children:
        with st.expander(f"⬇️ Follow-up Questions ({len(children)})", expanded=True):
            for i, child in enumerate(children):
                emoji = get_reflection_emoji(child)
                if st.button(f"{emoji} {truncate_text(child['question'], 40)}", key=f"child_{i}", use_container_width=True):
                    st.session_state.current_reflection_id = child["id"]
                    # Open in edit mode if child has no answer, otherwise view mode
                    st.session_state.mode = "edit" if not child.get("answer") else "view"
                    st.rerun()
    else:
        st.info("No follow-up questions yet")

def render_actions(reflection: dict):
    """Render action buttons"""
    if st.button("✏️ Edit", type="primary", use_container_width=True):
        st.session_state.mode = "edit"
        st.rerun()

def render_reflection_list():
    """Render a sidebar with all reflections for navigation"""
    with st.sidebar:
        st.header("📝 Reflections")

        if st.button("New Entry", type="primary", use_container_width=True):
            st.session_state.current_reflection_id = None
            st.session_state.mode = "edit"
            st.rerun()

        st.subheader("💡 Suggested Follow-ups")
        reflections = get_reflections(5, with_answer=False)

        if reflections:
            for reflection in reflections:
                emoji = get_reflection_emoji(reflection)
                if st.button(f"{emoji} {truncate_text(reflection['question'], 25)}",
                           key=f"nav_{reflection['id']}",
                           use_container_width=True):
                    st.session_state.current_reflection_id = reflection["id"]
                    st.session_state.mode = "edit" if not reflection.get("answer") else "view"
                    st.rerun()
        else:
            st.info("No follow-ups yet")

        st.divider()

def main():
    # Initialize session state
    if "mode" not in st.session_state:
        st.session_state.mode = "edit"  # Start in edit mode for new entries
    
    if "current_reflection_id" not in st.session_state:
        st.session_state.current_reflection_id = None
    
    if "parent_id" not in st.session_state:
        st.session_state.parent_id = None
    
    # Render navigation sidebar
    render_reflection_list()

    # Render footer with logout and feedback
    render_sidebar_footer()

    # Handle URL parameters or session state
    current_id = st.session_state.current_reflection_id
    mode = st.session_state.mode
    
    if current_id and mode == "view":
        # View existing reflection
        reflection = get_reflection(current_id)
        if reflection:
            render_view_mode(reflection)
        else:
            st.error("Reflection not found")
            st.session_state.current_reflection_id = None
            st.rerun()
    
    elif current_id and mode == "edit":
        # Edit existing reflection
        reflection = get_reflection(current_id)
        if reflection:
            render_edit_mode(reflection)
        else:
            st.error("Reflection not found")
            st.session_state.current_reflection_id = None
            st.rerun()
            
    else:
        # Create new reflection
        base_reflection = {}
        if st.session_state.parent_id:
            base_reflection["parent_id"] = st.session_state.parent_id
            st.session_state.parent_id = None  # Clear after use
        
        render_edit_mode(base_reflection if base_reflection else None)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_icon="✍️", page_title="Journal")
    if "access_token" not in st.session_state:
        st.switch_page("🏠_Home.py")
    st.session_state["Page"] = "Journal"
    main()

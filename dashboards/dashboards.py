"""
TASK 2: AI-Powered Feedback Dashboard System
Two-dashboard web application with shared data source.
Uses Streamlit for simplicity and fast deployment.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(
    page_title="AI Feedback System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API
API_KEY =  os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Data file path
DATA_FILE = "submissions.json"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_submissions():
    """Load all submissions from JSON file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_submission(submission: dict):
    """Add new submission to JSON file"""
    submissions = load_submissions()
    submission['id'] = len(submissions) + 1
    submission['timestamp'] = datetime.now().isoformat()
    submissions.append(submission)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(submissions, f, indent=2)
    
    return submission

def generate_ai_response(review: str, rating: int) -> str:
    """Generate AI response to user's review using LLM"""
    prompt = f"""You are a friendly customer service representative. 
    
A customer left this review with {rating} stars:
"{review}"

Write a brief, empathetic response (2-3 sentences) that:
1. Acknowledges their feedback
2. Shows you care
3. Offers next steps if appropriate

Response:"""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Thank you for your feedback! We appreciate your input."

def generate_summary(review: str) -> str:
    """Generate AI summary of review for admin"""
    prompt = f"""Summarize this review in 1-2 sentences for internal use:

"{review}"

Summary:"""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Unable to generate summary"

def generate_recommended_action(review: str, rating: int) -> str:
    """Generate recommended action for admin"""
    sentiment_map = {
        1: "Very negative - escalation needed",
        2: "Negative - requires attention",
        3: "Neutral - standard response",
        4: "Positive - acknowledge and thank",
        5: "Very positive - collect testimonial"
    }
    
    prompt = f"""Based on this {rating}-star review, what action should customer service take?

Review: "{review}"

Provide ONE specific recommended action (max 1 sentence):"""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return sentiment_map.get(rating, "Review and respond appropriately")

# ============================================================================
# PAGE 1: USER DASHBOARD (Public-Facing)
# ============================================================================

def user_dashboard():
    """User-facing dashboard for submitting reviews"""
    
    st.title("📝 Share Your Feedback")
    st.markdown("We'd love to hear about your experience!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Your Review")
        review_text = st.text_area(
            "Tell us what you think...",
            placeholder="What was your experience like?",
            height=150,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### Rating")
        rating = st.select_slider(
            "How would you rate us?",
            options=[1, 2, 3, 4, 5],
            value=4,
            label_visibility="collapsed"
        )
        
        # Visual rating display
        star_display = "⭐" * rating + "☆" * (5 - rating)
        st.markdown(f"<h2 style='text-align: center'>{star_display}</h2>", 
                   unsafe_allow_html=True)
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📤 Submit Review", use_container_width=True):
            
            if not review_text.strip():
                st.error("Please enter your review!")
            else:
                with st.spinner("Processing your feedback..."):
                    
                    # Generate AI response
                    ai_response = generate_ai_response(review_text, rating)
                    
                    # Save to database
                    submission = {
                        "rating": rating,
                        "review": review_text,
                        "ai_response": ai_response,
                        "summary": generate_summary(review_text),
                        "recommended_action": generate_recommended_action(review_text, rating)
                    }
                    save_submission(submission)
                    
                    # Show success message
                    st.success("✓ Thank you for your feedback!")
                    st.markdown("### Our Response:")
                    st.info(ai_response)
    
    # Recent feedback section
    st.markdown("---")
    st.markdown("### Recent Feedback (Preview)")
    
    submissions = load_submissions()
    if submissions:
        # Show last 3 submissions
        recent = submissions[-3:][::-1]
        
        for sub in recent:
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**Rating:** {'⭐' * sub['rating']}")
                    st.markdown(f"*\"{sub['review'][:100]}...\"*")
                with col2:
                    st.caption(sub['timestamp'].split('T')[0])
    else:
        st.info("No feedback yet. Be the first to review!")

# ============================================================================
# PAGE 2: ADMIN DASHBOARD (Internal-Facing)
# ============================================================================

def admin_dashboard():
    """Admin-facing dashboard for reviewing submissions and taking action"""
    
    st.title("📊 Admin Dashboard")
    st.markdown("Review and manage all customer feedback")
    
    # Load submissions
    submissions = load_submissions()
    
    if not submissions:
        st.info("No submissions yet.")
        return
    
    # Convert to DataFrame for easier display
    df = pd.DataFrame([
        {
            'ID': sub.get('id'),
            'Date': sub.get('timestamp', '').split('T')[0],
            'Rating': '⭐' * sub.get('rating', 0),
            'Review': sub.get('review', '')[:50] + "...",
            'Summary': sub.get('summary', 'N/A'),
            'Action': sub.get('recommended_action', 'N/A')
        }
        for sub in submissions
    ])
    
    # Statistics Section
    st.markdown("## 📈 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    ratings = [sub.get('rating', 0) for sub in submissions]
    
    with col1:
        st.metric("Total Feedback", len(submissions))
    with col2:
        st.metric("Avg Rating", f"{sum(ratings)/len(ratings):.1f}")
    with col3:
        positive = sum(1 for r in ratings if r >= 4)
        st.metric("Positive (4-5★)", positive)
    with col4:
        negative = sum(1 for r in ratings if r <= 2)
        st.metric("Negative (1-2★)", negative)
    
    # Rating distribution chart
    st.markdown("## 📊 Rating Distribution")
    rating_counts = {i: ratings.count(i) for i in range(1, 6)}
    st.bar_chart(rating_counts)
    
    # Detailed submissions table
    st.markdown("## 📋 All Submissions")
    
    # Display as expandable rows
    for idx, (_, row) in enumerate(df.iterrows()):
        sub = submissions[-(idx+1)]  # Reverse order to show newest first
        
        with st.expander(f"#{sub.get('id')} - {sub.get('timestamp', '').split('T')[0]} - {row['Rating']}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Original Review:**")
                st.write(sub.get('review'))
            
            with col2:
                st.markdown("**AI Summary:**")
                st.write(sub.get('summary'))
            
            st.markdown("**Recommended Action:**")
            st.info(sub.get('recommended_action'))
            
            st.markdown("**AI Response Sent to User:**")
            st.success(sub.get('ai_response'))
            
            st.markdown("---")
    
    # Filters section
    st.markdown("## 🔍 Quick Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show Negative Reviews (1-2★)"):
            negative_subs = [s for s in submissions if s.get('rating', 0) <= 2]
            st.info(f"Found {len(negative_subs)} negative reviews")
            for sub in negative_subs:
                st.warning(f"**{sub.get('rating')}★ - {sub.get('review')[:100]}...**")
    
    with col2:
        if st.button("Show Positive Reviews (4-5★)"):
            positive_subs = [s for s in submissions if s.get('rating', 0) >= 4]
            st.success(f"Found {len(positive_subs)} positive reviews")
            for sub in positive_subs:
                st.success(f"**{sub.get('rating')}★ - {sub.get('review')[:100]}...**")
    
    with col3:
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="feedback_export.csv",
                mime="text/csv"
            )

# ============================================================================
# MAIN APP ROUTER
# ============================================================================

def main():
    """Main app with page routing"""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["👤 User Dashboard", "📊 Admin Dashboard"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **AI Feedback System**
    
    - Users submit reviews with ratings
    - AI generates responses & summaries
    - Admin reviews all feedback
    - Real-time synchronization
    """)
    
    # Route to page
    if page == "👤 User Dashboard":
        user_dashboard()
    else:
        # Simple password protection for admin
        st.sidebar.markdown("---")
        admin_password = st.sidebar.text_input("Admin Password", type="password")
        
        if admin_password == "admin123":  # Change this in production!
            admin_dashboard()
        elif admin_password:
            st.error("❌ Incorrect password")
        else:
            st.info("Enter admin password in sidebar to access admin dashboard")

if __name__ == "__main__":
    main()
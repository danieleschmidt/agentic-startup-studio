import streamlit as st

st.set_page_config(layout="wide")

st.title("Agentic Startup Studio")

st.header("Idea Management")

# Placeholder for idea creation form
with st.expander("Create New Idea"): 
    with st.form("new_idea_form"):
        title = st.text_input("Idea Title")
        description = st.text_area("Idea Description")
        category = st.selectbox("Category", ["AI/ML", "Fintech", "SaaS", "Healthcare", "Other"])
        problem = st.text_area("Problem Statement")
        solution = st.text_area("Solution Description")
        market = st.text_input("Target Market")
        evidence_links = st.text_area("Evidence Links (comma-separated)")
        
        submitted = st.form_submit_button("Submit Idea")
        if submitted:
            st.success(f"Idea '{title}' submitted successfully!")
            # In a real application, you would call your backend API here
            st.json({
                "title": title,
                "description": description,
                "category": category,
                "problem": problem,
                "solution": solution,
                "market": market,
                "evidence_links": [link.strip() for link in evidence_links.split(',') if link.strip()]
            })


st.header("Existing Ideas")

# Placeholder for listing existing ideas
st.write("Displaying a list of existing ideas here...")

# Example data for demonstration
ideas_data = [
    {"id": "1", "title": "AI-Powered Legal Assistant", "status": "RESEARCHING", "category": "AI/ML"},
    {"id": "2", "title": "Sustainable Urban Farming Platform", "status": "IDEATE", "category": "Other"},
    {"id": "3", "title": "Blockchain-based Supply Chain", "status": "VALIDATED", "category": "Fintech"},
]

for idea in ideas_data:
    st.subheader(f"Idea: {idea['title']}")
    st.write(f"Status: {idea['status']}")
    st.write(f"Category: {idea['category']}")
    st.button(f"View Details for {idea['id']}", key=f"view_{idea['id']}")
    st.markdown("---")
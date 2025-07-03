import streamlit as st
import openai

# --- Set your OpenAI key here or via OS environment variable ---
openai.api_key = st.secrets["api_key"]

st.set_page_config(page_title="Wazzup. I am your Business Analyst", layout="wide")

st.title("Wazzup!!! I am your Business Analyst")

st.write("""
Upload a text file (project brief, business requirement, etc).
This app will summarize **high-level IT requirements** for:
- External users: people who will use the system.
- Internal users: staff who will use or support the system.
""")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")

    st.header("Original Content")
    st.write(content)
    
    if st.button("BA Arvee go to work!"):
        # Formulate prompt for LLM
        prompt = f"""
        Read the following project description/text. 
        From it, identify high-level IT requirements necessary for a new IT system.
        Group requirements into:

        A) External Users: (users using the system from outside the organization)
        B) Internal Users: (staff needing system access, info, or administration)

        Write clear, actionable, high-level IT requirements for each group.

        -----
        {content}
        -----
        """

        with st.spinner("Extracting requirements..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            requirements = response['choices'][0]['message']['content']

        st.subheader("Extracted High-Level IT Requirements")
        st.write(requirements)

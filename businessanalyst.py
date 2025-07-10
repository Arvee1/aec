import streamlit as st
import openai

openai.api_key = st.secrets["api_key"]

st.set_page_config(page_title="IT Requirements Extractor", layout="wide")

st.title("Wazzup!!! It is BA Arvee ready to go!")

st.write("""
Upload a text file (project brief, business requirement, etc).
This app will summarize **high-level IT requirements** for:
- External users: people who will use the system.
- Internal users: staff who will use or support the system.
""")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

def chunk_text(text, max_chars=5000):
    paragraphs = text.split('\n')
    chunks = []
    chunk = ""
    for para in paragraphs:
        if len(chunk) + len(para) <= max_chars:
            chunk += para + '\n'
        else:
            chunks.append(chunk)
            chunk = para + '\n'
    if chunk:
        chunks.append(chunk)
    return chunks

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.header("Original Content")
    st.write(f"File contains {len(content)} characters. Showing first 3000 chars below:\n\n")
    st.write(content[:3000] + ("..." if len(content) > 3000 else ""))
    
    if st.button("Generate IT Requirements"):
        st.info("Large files will be processed in chunks and summarized in steps.")
        with st.spinner("Extracting requirements..."):

            # Split content into manageable pieces
            chunks = chunk_text(content, max_chars=5000)
            all_requirements = []

            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1}/{len(chunks)}")
                prompt = f"""
                Read the following text (part of a project description).
                You are an expert in IT-enabled legislative transformation. Carefully review the following legislation:
                Analyze the requirements, changes, and objectives. Based on your analysis, draft a cohesive target state 
                concept document structured under the following headings and subheadings. For each section, describe the 
                intended future state, including systems, processes, stakeholder experience, and organizational capabilities 
                needed to effectively implement and administer the legislation.

                -----
                {chunk}
                -----
                Output ONLY the requirements in bullet points for each group.
                """
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3,
                )
                requirements = response.choices[0].message.content
                all_requirements.append(requirements)

            # Optionally, ask the model to summarize all chunk results into a final set
            final_prompt = f"""
            Below are extracted high-level IT requirements from several sections. 
            Please consolidate into a single, clear set of bullet-point requirements for:
            (A) External Users
            (B) Internal Users

            TEXT:
            {' '.join(all_requirements)}
            """
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            final_requirements = response.choices[0].message.content
            st.subheader("Extracted High-Level IT Requirements")
            st.write(final_requirements)

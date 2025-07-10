import streamlit as st
import openai

openai.api_key = st.secrets["api_key"]

st.set_page_config(page_title="IT Requirements Extractor", layout="wide")

st.title("Wazzup!!! It is BA Arvee ready to go!")

st.write("""
Upload a text file (project brief, business requirement, etc).
This app will summarize **target state concepts** under the following headings:
1. Regulatory Framework
2. Digital Experience Platform 
   a. DXP Features
   b. Service Design
3. Admin and Operational Systems
   a. Legislative Reforms Process
   b. Customer Management Capabilities
   c. Enabling Capabilities (e.g. HR, service management, tech management)
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

def summarize_chunk(chunk):
    prompt = f"""
You are an expert in IT-enabled legislative transformation. Carefully review the following legislation:

{chunk}

Analyze the requirements, changes, and objectives. Based on your analysis, draft a cohesive target state 
concept document structured under the following headings and subheadings. For each section, describe the intended 
future state, including systems, processes, stakeholder experience, and organizational capabilities needed to effectively 
implement and administer the legislation.

Headings:
1. Regulatory Framework
2. Digital Experience Platform
    a. DXP Features
    b. Service Design
3. Admin and Operational Systems
    a. Legislative Reforms Process
    b. Customer Management Capabilities
    c. Enabling Capabilities (e.g. HR, service management, tech management)
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def consolidate_summaries(summaries):
    prompt = f"""
You are an expert in IT-enabled legislative transformation.
Below are extracted target state concept summaries from several sections of legislation and requirements. Please 
develop a detailed, cohesive target state concept document under the following headings and subheadings. 
For each section, summarize the intended future state (systems, processes, stakeholder experience, organizational 
capabilities) required to implement and administer the legislation.

Headings:
1. Regulatory Framework
2. Digital Experience Platform
    a. DXP Features
    b. Service Design
3. Admin and Operational Systems
    a. Legislative Reforms Process
    b. Customer Management Capabilities
    c. Enabling Capabilities (e.g. HR, service management, tech management)

TEXT:
{' '.join(summaries)}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def iterative_summarize(requirements_list, group_size=5):
    if len(requirements_list) == 1:
        return requirements_list[0]
    summaries = []
    for i in range(0, len(requirements_list), group_size):
        group = requirements_list[i:i+group_size]
        merged = consolidate_summaries(group)
        summaries.append(merged)
    return iterative_summarize(summaries, group_size)

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.header("Original Content")
    st.write(f"File contains {len(content)} characters. Showing first 3000 chars below:\n\n")
    st.text_area("Preview", content[:3000] + ("..." if len(content) > 3000 else ""), height=300)

    if st.button("Generate Target State Concepts"):
        st.info("Large files will be processed in chunks and summarized in steps.")
        with st.spinner("Extracting target state summary..."):

            # Split content into manageable pieces
            chunks = chunk_text(content, max_chars=4000)
            all_summaries = []

            for i, chunk in enumerate(chunks):
                st.write(f"Processing chunk {i+1}/{len(chunks)}")
                summary = summarize_chunk(chunk)
                all_summaries.append(summary)

            st.success("Initial extraction complete. Consolidating outputs so they fit within model limits...")

            # Hierarchical summarization
            final_summary = iterative_summarize(all_summaries, group_size=5)

            st.subheader("Target State Concept Summary")
            st.write(final_summary)

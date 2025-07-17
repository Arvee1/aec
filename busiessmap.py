import streamlit as st
import os
import replicate

# Set Replicate API Token from Streamlit secrets
os.environ["REPLICATE_API_TOKEN"] = st.secrets["replicate_api_token"]

st.set_page_config(page_title="Business Process Flowchart Mapper (Mermaid via Claude)", layout="wide")
st.title("Business Process Flowchart Mapper by User Type (Claude+Mermaid)")

st.write("""
Upload legislation or requirements text.
This will:
- Identify process steps for each user type/role in your document.
- **Render a Mermaid flowchart diagram for each user type and their process.**

> Diagrams are auto-generated from process steps using Claude, and rendered in-browser.
""")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

def chunk_text(text, max_chars=5000):
    paragraphs = text.split('\n')
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk) + len(para) <= max_chars:
            chunk += para + '\n'
        else:
            chunks.append(chunk)
            chunk = para + '\n'
    if chunk:
        chunks.append(chunk)
    return chunks

def process_map_mermaid_claude(chunk):
    prompt = f"""You are a business process analyst.

Review the following legislation or requirements fragment:

{chunk}

For every main user type or role (e.g., Customer, Staff, Regulator), do the following:

- List each user type you identify.
- For each, generate a **Mermaid flowchart (flowchart TD) diagram** of their main business process steps, showing the step sequence, key decisions as diamonds, and any important handoffs to other user types/roles. Label the nodes and connectors clearly for readability. Each user type's flow should be named and separated.

**Output only Mermaid code. For each user type, prefix with a Markdown Main Heading showing the user/role before the Mermaid code block.**

Example format:

# Staff
```mermaid
flowchart TD
    A[Start] --> B[First Step]
    B -->|Yes| C[Decision Branch]
    ...

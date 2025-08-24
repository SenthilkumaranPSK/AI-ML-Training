import streamlit as st
from llama_cpp import Llama

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---- Load model once ----
@st.cache_resource
def load_llama():
    return Llama(
        model_path = r"mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        use_mlock=True
    )

llm = load_llama()

# ---- Load and index PDF once ----
@st.cache_resource
def load_pdf_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

vector_store = load_pdf_vectorstore(r"data.pdf")  # <-- your PDF path here

# Session state variables
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI setup
st.title("Chat with Mistral LLM + PDF Context")

user_input = st.text_input("You:", key="input")

def generate_prompt(context_docs, question):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant answering questions using the following document context:\n{context_text}\n\nQuestion: {question}\nAnswer:"""
    return prompt

if user_input:
    user_input_lower = user_input.strip().lower()

    if user_input_lower == "thanks":
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Model", "You're welcome! Have a great day 😊"))
        st.session_state.chat_started = False
    elif user_input_lower == "hii":
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Model", "Hello! Ask me anything related to the PDF."))
        st.session_state.chat_started = True
    elif not st.session_state.chat_started:
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Model", "Please say 'hii' to start chatting."))
    else:
        # Retrieve top 3 relevant chunks from PDF vector store
        relevant_docs = vector_store.similarity_search(user_input, k=3)
        prompt = generate_prompt(relevant_docs, user_input)

        output = llm(prompt, max_tokens=200, temperature=0.7, stop=["###"])

        response = output["choices"][0]["text"].strip()

        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Model", response))

# Display chat messages
for sender, message in st.session_state.messages:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

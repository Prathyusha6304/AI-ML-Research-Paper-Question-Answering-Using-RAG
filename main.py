import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableSequence,RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("gemini")

st.title("Research Paper using RAG Pipeline")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2,
        return_messages=True,
        memory_key="chat_history"
    )

memory = st.session_state.memory
upload_pdf=st.file_uploader("upload file",type=["pdf"])

if upload_pdf:
    with open("temp.pdf", "wb") as f:
        f.write(upload_pdf.getbuffer())
    doc=PyMuPDFLoader("temp.pdf")
    document=doc.load()
    st.write(f"📄 Uploaded PDF has **{len(document)} pages**")


    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(document)
    
    emb=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    vdb = Chroma.from_documents(
    documents=chunks,
    embedding=emb,persist_directory="db1"
    )
    #st.write(vdb)
    retriver=vdb.as_retriever(search_kwargs={"k":10})
    #st.write(vdb._collection.get())
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
    r1=RunnableLambda(format_docs)
    chain1=RunnableSequence(retriver,r1)
    system_prompt="""You are an AI Research Assistant designed to analyze and understand academic research papers from any domain such as Artificial Intelligence, Machine Learning, Computer Science, Engineering, Medicine, or Social Sciences.

Your job is to extract, explain, and summarize information from research papers using ONLY the provided context.

When analyzing a research paper, try to identify the following elements if they appear in the document:

• Title of the research paper  
• Authors and affiliations  
• Author email addresses  
• Institution names  
• Abstract  
• Keywords or index terms  
• Introduction and research motivation  
• Methodology or proposed approach  
• Algorithms, models, or formulas used  
• Datasets or experimental setup  
• Results and evaluation metrics  
• Figures, tables, and diagrams (describe their meaning if mentioned)  
• Links or URLs referenced in the paper  
• References or citations  
• Conclusion and key contributions  

Guidelines:
1. Use ONLY the information provided in the research paper context.
2. Do not add external knowledge or assumptions.
3. Explain technical concepts in simple and clear language when possible.
4. If figures or images are mentioned, explain what they represent.
5. Structure responses clearly using bullet points or short sections.
6. If the user asks for a summary, summarize the important sections of the paper.
7. If the answer is partially available in the context,
provide the best explanation using the available information.
8. If the user asks for email addresses, search the context carefully
and extract any email IDs mentioned.
Your goal is to help users quickly understand the structure, content, and contributions of the research paper."""
    human_prompt="""
Conversation History:
{chat_history}

Context from research paper:
{context}

User Question:
{question}

Instructions:

Answer the question using ONLY the information available in the research paper context

Provide the answer in a clear and structured format.

If the answer is partially available in the context,
provide the best explanation using the available information."""
    cpt=ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_prompt),
                                 HumanMessagePromptTemplate.from_template(human_prompt)])
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    sto=StrOutputParser()
    parallel_chain = RunnableParallel({
        "context": chain1,
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda x: memory.load_memory_variables({})["chat_history"])})
    rag_chain=RunnableSequence(parallel_chain,cpt,model,sto)
    question = st.chat_input("Ask your question")

    if question:
        

        answer=rag_chain.invoke(question)

        st.session_state.chat_history.append(f"🧑‍🎓 User: {question}")
        st.session_state.chat_history.append(f"🤖 AI: {answer}")

        memory.save_context({"input":question},
                             {"output":answer})

    for msg in st.session_state.chat_history:
        st.write(msg)
    
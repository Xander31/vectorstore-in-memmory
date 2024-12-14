"""
pipenv install langchain pypdf langchain-openai langchain-community langchainhub
faiss-cpu
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

if __name__ == "__main__":

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    query = "Give me the gist of React in 2 sentences."

    #Loading data in vector Store (FAISS)
    pdf_path = "React Agent paper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    #Better control of the chunk size to avoid hitting the token limits
    text_spitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_spitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(folder_path="../vectorstore-in-memmory", index_name="faiss_index_react")

    #Retrieval
    new_vector_store = FAISS.load_local(
        folder_path="../vectorstore-in-memmory", index_name="faiss_index_react", embeddings=embeddings, allow_dangerous_deserialization=True  #Testing only
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        new_vector_store.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input":query})
    print(res["answer"])

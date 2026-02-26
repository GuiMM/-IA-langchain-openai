import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import faiss
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.5, 
    openai_api_key=api_key
)

embeddings = OpenAIEmbeddings()

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf"
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ],[]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
).split_documents(documentos)


vector_store = FAISS.from_documents(
    text_splitter,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate(
    [
        ("system", "Responda exclusivamente o conteudo fornecido."),
        ("human", "{query} \n\nContexto:\n{context} \n\nResposta:"),
    ]
)

cadeia = prompt | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = vector_store.invoke(pergunta)
    contexto = "\n\n".join([trecho.page_content for trecho in trechos])
    return cadeia.invoke({"query": pergunta, "context": contexto})


print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"))
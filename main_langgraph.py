import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from typing import Literal, TypedDict

set_debug(False)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.5, 
    openai_api_key=api_key)

prompt_consultor_cultural = ChatPromptTemplate([
    ("system", "Você é um consultor de viagens especialista em viagens com destinos culturais. Apresente-se como sra. Maroccas Cultura."),
    ("human", "{query}"),
]
)

prompt_consultor_montanha = ChatPromptTemplate([
    ("system", "Você é um consultor de viagens especialista em viagens com destinos para montanhas e atividades radicais. Apresente-se como sra. Maroccas Montanha."),
    ("human", "{query}"),
]
)

cadeia_cultura = prompt_consultor_cultural | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["cultural", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'cultural' ou 'montanha'."),
        ("human", "{query}"),
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config: RunnableConfig):
    return {"destino": await roteador.ainvoke({"query":estado["query"]}, config=config) }

async def no_cultural(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_cultura.ainvoke({"query":estado["query"]}, config=config) }

async def no_montanha(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query":estado["query"]}, config=config) }


def escolher_no(estado:Estado) -> Literal["cultural","montanha"]:
    return "cultural" if estado["destino"]["destino"] == "cultural" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("cultural", no_cultural)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("cultural", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {"query": "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?"}
    )
    print(resposta)

asyncio.run(
    main()
)
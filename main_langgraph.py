import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
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

def responda(pergunta:str):
    rota=roteador.invoke({"query": pergunta})["destino"]
    print(rota)
    if rota == "cultural":
        return cadeia_cultura.invoke({"query": pergunta})
    elif rota == "montanha":
        return cadeia_montanha.invoke

print(responda(
     "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?"
    )
)
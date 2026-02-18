
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

numero_dias = 7
numero_criancas = 2
atividades = "praia"

modelo_de_prompt = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {dias} dias,
    para uma família com {criancas} criancas,
    que gostam de {atividades}.
    """
)

prompt = modelo_de_prompt.format(
    dias=numero_dias, 
    criancas=numero_criancas, 
    atividades=atividades
)

modelo = ChatOpenAI(model="gpt-3.5-turbo",
                    temperature=0.5,  
                    api_key=api_key)

resposta = modelo.invoke(prompt)

print(resposta.content)

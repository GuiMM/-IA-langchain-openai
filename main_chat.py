import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key=api_key)

prompt_sugetao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como sra. Maroccas." ),
        ("placeholder", "{historico}"),
        ("human", "{query}"),
    ]
)

cadeia = prompt_sugetao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain_alura"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]


lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?",
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia, 
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
            {
                "query": pergunta,
            },
            config={"session_id": sessao},
        )
    print("usuario: ", pergunta)
    print("Ia: ", resposta)
    print("\n")


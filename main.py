from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

numero_dias = 7

numero_criancas = 2

atividades = "praia, parque, museus"

prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma família com {numero_criancas} crianças, que gosta de {atividades} "



cliente = OpenAI(api_key=api_key)

resposta = cliente.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        #caracterizacao
        {
            "role": "system", 
            "content": "Você é um assistente de roteiro de viagens."
        },
        #Quem está interagindo
        {
            "role": "user", 
            "content": prompt
        }
    ]
)

resposta_em_texto = resposta.choices[0].message.content
print(resposta)
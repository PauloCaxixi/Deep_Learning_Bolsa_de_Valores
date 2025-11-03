from fastapi import APIRouter
from app.application.get_data import GetData
from app.domain.entities.information import Info

#Instancia o roteador
router = APIRouter()

#Define a rota /home
@router.get("/home")
async def read_home():
    # Instancia o Info com os parâmetros desejados
    dt = Info('PETR4.SA', '2018-01-01', '2024-07-20')

    #Instantia o GetData com os parâmetros fornecidos e retorna os dados
    dados = GetData(dt).QueryData()
    
    return {"dados": dados}
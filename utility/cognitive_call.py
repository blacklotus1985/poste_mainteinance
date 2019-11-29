# Databricks notebook source
import requests

# COMMAND ----------


documents = {"documents": [
    {"id": "1", "language": "en", "text": "Wonderful hotel :)"},
    {"id": "2", "language": "en", "text": "Awful experience!"},
    {"id": '3', "language": "it",
     "text": "È un sistema 'nuovo e complesso di spaccio su Roma' spiega il procuratore capo Michele Prestipino in apertuta della conferenza che annuncia 51 ordinanze di custodia cautelare in carcere nei confronti di narcotrafficanti che approviggionavano la capitale di cocaina, hashish e marijuana. Ordinanze eseguite dal Comando provinciale della Finanza e dagli uomino del Gico. Ai vertici dell'organizzazione c'era Fabrizio Piscitelli, ucciso il 7 agosto al Parco degli Acquedotti, e con lui il suo braccio destro, il broker Fabrizio Fabietti."}

]}

# COMMAND ----------

documents

# COMMAND ----------

azure_key = "f9e186042007437d9e95bbb9c8f91fdd"
headers = {"Ocp-Apim-Subscription-Key": azure_key}
response = requests.post("https://westeurope.api.cognitive.microsoft.com/text/analytics/v2.1/keyPhrases",
                         headers=headers, json=documents)

# COMMAND ----------

resp = response.json()


type(resp)


resp['documents'][0]['score']




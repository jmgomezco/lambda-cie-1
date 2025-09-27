import uuid
import json
import boto3
import unicodedata
from datetime import datetime
import requests
import traceback
from pinecone import Pinecone

MAX_PINECONE_RESULTS = 50

def get_secrets():
    try:
        secrets_client = boto3.client('secretsmanager')
        secret_value = secrets_client.get_secret_value(SecretId="prod")
        secrets = json.loads(secret_value['SecretString'])
        return secrets
    except Exception as e:
        raise

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('utf-8', 'ignore').decode('utf-8').upper()

def get_openai_embedding(input_text, secrets):
    """Obtiene embedding de OpenAI usando requests, modelo text-embedding-3-large."""
    try:
        OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "text-embedding-3-large",
            "input": input_text
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            error_msg = f"Error en embedding OpenAI: {response.text}"
            raise Exception(error_msg)
        embedding = response.json()["data"][0]["embedding"]
        return embedding
    except Exception as e:
        raise

def search_pinecone(query_text, secrets):
    try:
        index_name = "cie2024"
        pc = Pinecone(api_key=secrets["PINECONE_API_KEY"])
        index = pc.Index(index_name)

        embedding = get_openai_embedding(query_text, secrets)

        pinecone_response = index.query(
            vector=embedding,
            top_k=MAX_PINECONE_RESULTS,
            include_metadata=True
        )

        candidates = []
        for match in pinecone_response.matches:
            codigo = str(match.id).upper()
            desc = str(match.metadata.get("desc", ""))
            candidates.append({"codigo": codigo, "desc": desc})

        return candidates
    except Exception as e:
        raise

def get_cie10_descriptions(codes):
    """
    Dado una lista de códigos, consulta la tabla cie2024 y retorna
    una lista de dicts {codigo, desc} en el mismo orden, excluyendo los que no existan.
    """
    if not codes:
        return []
    dynamodb = boto3.resource('dynamodb')
    cie_table = dynamodb.Table("cie2024")
    # DynamoDB batch_get_item solo acepta hasta 100 claves por lote
    keys = [{"codigo": code} for code in codes]
    # Puede que respondan menos, filtramos por los que existen
    response = cie_table.meta.client.batch_get_item(
        RequestItems={
            "cie2024": {"Keys": keys}
        }
    )
    cie_items = {item["codigo"]: item for item in response["Responses"].get("cie2024", [])}
    enriched = []
    for code in codes:
        if code in cie_items:
            desc = cie_items[code].get("desc", "")
            enriched.append({"codigo": code, "desc": desc})
    return enriched

def lambda_handler(event, context):
    try:
        secrets = get_secrets()

        body = json.loads(event["body"])
        texto = body.get("texto", "")

        sessionId = event.get("sessionId")
        if not sessionId:
            sessionId = str(uuid.uuid4())

        trimmed_texto = texto[:200]
        normalized_texto = normalize_text(trimmed_texto)

        pinecone_results = search_pinecone(normalized_texto, secrets)
        pinecone_results = pinecone_results[:MAX_PINECONE_RESULTS]
        
        # Llamada a GPT-4o para seleccionar SOLO CÓDIGOS (no objetos)
        gpt_api_url = "https://api.openai.com/v1/chat/completions"
        gpt_headers = {
            "Authorization": f"Bearer {secrets['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }

        pinecone_json = json.dumps([
            {"codigo": str(r.get("codigo", "")), "desc": str(r.get("desc", "")).upper()}
            for r in pinecone_results
        ], ensure_ascii=False)
        valid_codes = set(str(r.get("codigo", "")) for r in pinecone_results)

        reglas_json = json.dumps({
            "instrucciones": [
                "Si el texto es una cadena de carácteres sin sentido no generes ningun código en tú lista. Tampoco generes ningún código si la frase no contiene un término clínico"
                "Selecciona exactamente 8 códigos cuyo campo 'desc' tenga mayor concordancia semántica con el texto.",
                "Solo puedes elegir códigos que estén presentes en la lista de candidatos, no inventes ni modifiques ningún código.",
                "En candidatos con lateralidad, si el texto introducido no hace referencia a ella, elige solo lateralidad derecha"
                "En candidatos de fractura, elige fracturas cerradas, siempre que el introducido no especifique lo contrario"
                "Responde solo un array JSON con los 8 códigos más relevantes, sin explicaciones ni objetos, solo array de strings."
            ]
        }, ensure_ascii=False)

        messages = [
            {"role": "system", "content": "Eres un asistente experto en codificación CIE10."},
            {"role": "user", "content": f"Texto: {trimmed_texto.upper()}\nCandidatos: {pinecone_json}\nReglas: {reglas_json}\nResponde solo un array JSON con los 8 códigos más relevantes."}
        ]

        gpt_data = {
            "model": "gpt-4o",  # o el modelo apropiado disponible en tu cuenta
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.2
        }

        codigos_gpt = []
        try:
            response_gpt = requests.post(gpt_api_url, headers=gpt_headers, json=gpt_data)
            if response_gpt.status_code == 200:
                gpt_response_text = response_gpt.json()["choices"][0]["message"]["content"]
                # Intenta parsear como lista JSON
                try:
                    codigos_gpt_raw = json.loads(gpt_response_text)
                except Exception:
                    # Si GPT responde con texto extra, intenta extraer el array
                    import re
                    match = re.search(r'(\[.*\])', gpt_response_text, re.DOTALL)
                    if match:
                        codigos_gpt_raw = json.loads(match.group(1))
                    else:
                        codigos_gpt_raw = []
                if isinstance(codigos_gpt_raw, list):
                    # Filtra solo los códigos válidos que estén en los candidatos de Pinecone
                    codigos_gpt = [str(c) for c in codigos_gpt_raw if str(c) in valid_codes]
        except Exception as e:
            print("Error procesando respuesta de GPT:", str(e))
            codigos_gpt = []

        # Ahora consulta DynamoDB para obtener las descripciones REALES y filtrar los que no existan
        candidatos_gpt = get_cie10_descriptions(codigos_gpt)

        # Guardar en DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table("sesiones")
        table.put_item(
            Item={
                "sessionId": sessionId,
                "texto": trimmed_texto,  # texto literal del usuario, recortado a 200 caracteres
                "pinecone_results": [
                    {"codigo": r.get("codigo", ""), "desc": r.get("desc", "")}
                    for r in pinecone_results
                ],
                "candidatos_gpt": candidatos_gpt,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "candidatos_gpt": candidatos_gpt,
                "sessionId": sessionId
            })
        }
    except Exception as e:
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

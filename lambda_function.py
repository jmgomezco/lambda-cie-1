import uuid
import json
import boto3
import unicodedata
from datetime import datetime
import requests
import traceback
from pinecone import Pinecone

MAX_PINECONE_RESULTS = 40

def get_secrets():
    try:
        secrets_client = boto3.client('secretsmanager')
        secret_value = secrets_client.get_secret_value(SecretId="prod/pine")
        secrets = json.loads(secret_value['SecretString'])
        return secrets
    except Exception as e:
        raise

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
        # Convertir el texto a mayúsculas antes de obtener el embedding
        embedding = get_openai_embedding(query_text, secrets)
        pinecone_response = index.query(
            vector=embedding,
            top_k=MAX_PINECONE_RESULTS,
            include_metadata=True
        )
        candidates = []
        for match in pinecone_response.matches:
            codigo = str(match.id)
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
    keys = [{"codigo": code} for code in codes]
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

def get_client_ip(event):
    # Primero intenta 'X-Forwarded-For' del API Gateway, luego 'requestContext', luego headers comunes
    ip = None
    # AWS API Gateway v2 (HTTP API)
    if "requestContext" in event:
        rc = event["requestContext"]
        # Websocket
        ip = rc.get("identity", {}).get("sourceIp")
        # REST API
        if not ip:
            ip = rc.get("http", {}).get("sourceIp")
    # X-Forwarded-For header
    if not ip and "headers" in event:
        xff = event["headers"].get("X-Forwarded-For") or event["headers"].get("x-forwarded-for")
        if xff:
            ip = xff.split(",")[0].strip()
    # Direct sourceIp
    if not ip:
        ip = event.get("sourceIp")
    # Fallback
    if not ip:
        ip = "unknown"
    return ip

def lambda_handler(event, context):
    try:
        secrets = get_secrets()

        body = json.loads(event["body"])
        texto = body.get("texto", "")

        sessionId = event.get("sessionId")
        if not sessionId:
            sessionId = str(uuid.uuid4())

        trimmed_texto = texto[:150]
        trimmed_texto = trimmed_texto[:1].upper() + trimmed_texto[1:] if trimmed_texto else ""

        pinecone_results = search_pinecone(trimmed_texto, secrets)
        pinecone_results = pinecone_results[:MAX_PINECONE_RESULTS]

        # Corrige variable mal nombrada: candidatos_pencone -> pinecone_results
        candidatos_pinecone = [
            {"codigo": r.get("codigo", ""), "desc": r.get("desc", "")}
            for r in pinecone_results
        ]

        pinecone_json = json.dumps([
            {"codigo": str(r.get("codigo", "")), "desc": str(r.get("desc", ""))}
            for r in pinecone_results
        ], ensure_ascii=False)
        valid_codes = set(str(r.get("codigo", "")) for r in pinecone_results)


        reglas_json = json.dumps({
        "instrucciones": [
        "Selecciona hasta 8 códigos CIE-10 con mayor relevancia, solo escoge códigos de la lista de candidatos que se te ofrece.",
        "En códigos de traumatismos, envenenamientos, etc. (capítulo 19 de la CIE-10), selecciona primero códigos de 'contacto inicial'.",
        "En códigos de fracturas, elige primero fractura cerrada.",
        "Si el texto NO contiene ningún término clínico, NO generes ningún código y responde con un array vacío: [].",
        "NO ofrezcas códigos si el texto es absurdo, no tiene sentido clínico o es una conversación casual.",
        "Responde solo un array JSON con los códigos relevantes (máximo 8), solo array de strings, sin explicaciones."
        ]
        }, ensure_ascii=False)
        
        # CAMBIO: Enviamos el texto trimado (sin normalizar) a GPT
        messages = [
            {"role": "system", "content": "Eres un asistente experto en codificación CIE-10."},
            {"role": "user", "content": f"Texto: {trimmed_texto}\nCandidatos: {pinecone_json}\nReglas: {reglas_json}\nResponde solo un array JSON con los 8 códigos más relevantes."}
        ]

        gpt_api_url = "https://api.openai.com/v1/chat/completions"
        gpt_headers = {
            "Authorization": f"Bearer {secrets['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }
        gpt_data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.2
        }

        codigos_gpt = []
        tokens_entrada = 0
        tokens_salida = 0
        try:
            response_gpt = requests.post(gpt_api_url, headers=gpt_headers, json=gpt_data)
            if response_gpt.status_code == 200:
                gpt_json = response_gpt.json()
                gpt_response_text = gpt_json["choices"][0]["message"]["content"]
                # Intenta parsear como lista JSON
                try:
                    codigos_gpt_raw = json.loads(gpt_response_text)
                except Exception:
                    import re
                    match = re.search(r'(\[.*\])', gpt_response_text, re.DOTALL)
                    if match:
                        codigos_gpt_raw = json.loads(match.group(1))
                    else:
                        codigos_gpt_raw = []
                if isinstance(codigos_gpt_raw, list):
                    codigos_gpt = [str(c) for c in codigos_gpt_raw if str(c) in valid_codes]
                # Guardamos los tokens si están disponibles
                usage = gpt_json.get("usage", {})
                tokens_entrada = usage.get("prompt_tokens", 0)
                tokens_salida = usage.get("completion_tokens", 0)
        except Exception as e:
            print("Error procesando respuesta de GPT:", str(e))
            codigos_gpt = []

        candidatos_gpt = get_cie10_descriptions(codigos_gpt)

        # Obtener IP cliente moderna
        ip_cliente = get_client_ip(event)

        # Guardar en DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table("sesiones")
        table.put_item(
            Item={
                "sessionId": sessionId,
                "texto": trimmed_texto,  # texto literal del usuario, recortado a 200 caracteres
                "candidatos_pinecone": candidatos_pinecone,
                "candidatos_gpt": candidatos_gpt,
                "timestamp": datetime.utcnow().isoformat(),
                "tokens_entrada": tokens_entrada,
                "tokens_salida": tokens_salida,
                "ip_cliente": ip_cliente
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "candidatos_gpt": candidatos_gpt,
                "candidatos_pinecone": candidatos_pinecone,
                "sessionId": sessionId
            })
        }
    except Exception as e:
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

import uuid
import json
import boto3
import unicodedata
from datetime import datetime
import requests
import traceback
import re
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

def extract_client_ip(event):
    """
    Extrae la IP del cliente desde el evento de API Gateway (HTTP REST).
    Método moderno y sencillo.
    """
    # API Gateway HTTP REST: IP se suele encontrar en headers bajo X-Forwarded-For
    headers = event.get("headers", {})
    x_forwarded_for = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
    if x_forwarded_for:
        # Puede ser lista: "client-ip, proxy1, proxy2"
        return x_forwarded_for.split(",")[0].strip()
    # Fallback: busca en requestContext (aunque HTTP REST no siempre lo tiene)
    request_context = event.get("requestContext", {})
    identity = request_context.get("identity", {})
    ip = identity.get("sourceIp")
    if ip:
        return ip
    # Último recurso: no encontrada
    return None

def lambda_handler(event, context):
    try:
        secrets = get_secrets()

        body = json.loads(event["body"])
        texto = body.get("texto", "")

        sessionId = event.get("sessionId")
        if not sessionId:
            sessionId = str(uuid.uuid4())

        trimmed_texto = texto[:150]
        normalized_texto = normalize_text(trimmed_texto)

        pinecone_results = search_pinecone(normalized_texto, secrets)
        pinecone_results = pinecone_results[:MAX_PINECONE_RESULTS]
        
        # Formato compacto para GPT
        candidatos_gpt_json = [
            {"codigo": str(r.get("codigo", "")), "desc": str(r.get("desc", "")).upper()}
            for r in pinecone_results
        ]
        valid_codes = set(str(r.get("codigo", "")) for r in pinecone_results)

        reglas_json = [
            "Si el texto no es clínico, responde [].",
            "Elige hasta 8 códigos más relevantes.",
            "Solo puedes elegir códigos de la lista.",
            "Prefiere lateralidad derecha si no se indica.",
            "Prefiere fracturas cerradas si no se especifica.",
            "Responde solo con un array JSON de códigos.",
            "Ordena los códigos por relevancia."
        ]

        # Mensaje para GPT
        gpt_api_url = "https://api.openai.com/v1/chat/completions"
        gpt_headers = {
            "Authorization": f"Bearer {secrets['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }

        gpt_prompt_json = {
            "system": "Eres un codificador experto en CIE-10. Aplica las reglas y responde exclusivamente con los códigos más relevantes en formato JSON.",
            "texto": trimmed_texto.upper(),
            "candidatos": candidatos_gpt_json,
            "reglas": reglas_json
        }

        messages = [
            {"role": "system", "content": gpt_prompt_json["system"]},
            {"role": "user", "content": json.dumps({k: v for k, v in gpt_prompt_json.items() if k != "system"}, ensure_ascii=False)}
        ]
        gpt_data = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.2
        }

        codigos_gpt = []
        tokens_entrada = 0
        tokens_salida = 0
        try:
            response_gpt = requests.post(gpt_api_url, headers=gpt_headers, json=gpt_data)
            if response_gpt.status_code == 200:
                gpt_response = response_gpt.json()
                gpt_response_text = gpt_response["choices"][0]["message"]["content"]
                # Intenta parsear como lista JSON
                try:
                    codigos_gpt_raw = json.loads(gpt_response_text)
                except Exception:
                    match = re.search(r'(\[.*\])', gpt_response_text, re.DOTALL)
                    if match:
                        codigos_gpt_raw = json.loads(match.group(1))
                    else:
                        codigos_gpt_raw = []
                if isinstance(codigos_gpt_raw, list):
                    codigos_gpt = [str(c) for c in codigos_gpt_raw if str(c) in valid_codes]
                # Tokens usados (requiere que OpenAI devuelva usage)
                usage = gpt_response.get("usage", {})
                tokens_entrada = usage.get("prompt_tokens", 0)
                tokens_salida = usage.get("completion_tokens", 0)
        except Exception as e:
            print("Error procesando respuesta de GPT:", str(e))
            codigos_gpt = []
            tokens_entrada = 0
            tokens_salida = 0

        candidatos_gpt = get_cie10_descriptions(codigos_gpt)
        ip_cliente = extract_client_ip(event)

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
                "tokens_entrada": tokens_entrada,
                "tokens_salida": tokens_salida,
                "ip_cliente": ip_cliente,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "candidatos_gpt": candidatos_gpt,
                "sessionId": sessionId,
                "tokens_entrada": tokens_entrada,
                "tokens_salida": tokens_salida,
                "ip_cliente": ip_cliente
            })
        }
    except Exception as e:
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# FastAPI + Milvus (Vector DB) — BBDD Vectoriales

API REST para ingestar frases y hacer búsquedas:
- **Léxica** (por texto)
- **Semántica** (por embeddings + vector search)
- **Híbrida** (filtro léxico + ranking semántico)

El proyecto usa **Milvus** como base de datos vectorial y un sistema de embeddings configurable:
- `fake` (mock determinista, funciona siempre)
- `openai` (preparado, requiere permisos de embeddings)

---

## Requisitos

- Python 3.12+
- Docker Desktop (Windows) con Milvus levantado
- (Opcional) API key de OpenAI con permisos de embeddings

---

## Levantar Milvus (Docker)

> Si ya lo tienes levantado, puedes saltarte este paso.

Milvus standalone suele exponer:
- **gRPC:** `localhost:19530`
- **REST:** `localhost:9091` (dependiendo del compose)
- Attu (UI) si lo usas: `localhost:8000` (o el puerto que tengas)

---

## Instalación

### 1) Crear entorno virtual

```bash
python -m venv venv
```
Activar en Windows:
```bash
venv\Scripts\activate
```

### 2) Instalar dependencias
```bash
python -m pip install -r requirements.txt
```

## Configuración del entorno

La configuración se realiza mediante variables de entorno.

### Modo FAKE (recomendado para desarrollo)

- EMBEDDING_PROVIDER = fake  
- EMBEDDING_DIM = 64  

En este modo los embeddings se generan de forma determinista sin usar ningún servicio externo.  
Permite probar todo el flujo sin depender de permisos o APIs externas.

### Modo OpenAI (opcional)

- EMBEDDING_PROVIDER = openai  
- EMBEDDING_DIM = 1536  
- OPENAI_API_KEY = tu_api_key  
- OPENAI_EMBED_MODEL = text-embedding-3-small  

Este modo utiliza modelos reales de embeddings de OpenAI.  
Requiere que la API key tenga permisos para generar embeddings.

Cuando se usa OpenAI, la colección de Milvus se crea automáticamente con un nombre distinto para evitar conflictos de dimensión.

---

## Arrancar la API

```bash
uvicorn main:app --reload --port 8000
```

Y vamos a http://127.0.0.1:8000/docs para utilizar Swagger

---

## Endpoints disponibles

### Health check

GET /health  

Devuelve el estado de la API, la colección activa y el proveedor de embeddings.

---

### Crear colección

POST /collection/create  

Crea la colección vectorial si no existe.

---

### Ingesta de datos

POST /ingest  

Permite ingestar un conjunto de frases junto con metadatos.

Ejemplo de body:
```bash
{
  "frases": [
    "Hola.",
    "El gato duerme.",
    "La inteligencia artificial avanza rápido."
  ],
  "metadatos": {
    "origen": "clase",
    "idioma": "es"
  }
}
```
Cada frase se almacena con:
- Texto original
- Embedding vectorial
- Metadatos

---

### Búsqueda léxica

POST /search/lexical  

Busca documentos que contengan el texto indicado.

Ejemplo:
```bash
{
  "query": "gato",
  "top_k": 5
}
```
---

### Búsqueda semántica

POST /search/semantic  

Busca documentos semánticamente similares usando embeddings.

Ejemplo:
```bash
{
  "query": "animal descansando",
  "top_k": 5
}
```
---

### Búsqueda híbrida

POST /search/hybrid  

Combina búsqueda léxica para filtrar candidatos y búsqueda semántica para ordenar resultados.

Ejemplo:
```bash
{
  "query": "paella",
  "top_k": 5
}
```
---

### Eliminar documento

DELETE /documents/{doc_id}  

Elimina un documento por su ID.

---

### Actualizar documento

PUT /documents/{doc_id}  

Actualiza un documento existente.  
Internamente se realiza como delete + insert (el ID cambia).

Ejemplo de body:
```bash
{
  "text": "El gato duerme en el sofá.",
  "metadatos": {
    "updated": true
  }
}
```
---

## Sobre top_k

El parámetro top_k indica cuántos resultados devuelve una búsqueda.

- top_k = 3 → devuelve los 3 documentos más relevantes
- top_k = 10 → devuelve los 10 más relevantes

Se utiliza tanto en búsqueda semántica como híbrida.

---

## Notas importantes

- Milvus no evita duplicados automáticamente.
- Si se ingesta el mismo texto varias veces, se crearán registros distintos con IDs diferentes.
- El sistema está preparado para sustituir embeddings fake por embeddings reales sin modificar el resto del código.

---
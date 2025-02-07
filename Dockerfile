FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install langchain==0.1.15 flask neo4j python-decouple sentence-transformers gunicorn
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "fhir_rag.app:app", "--workers", "4", "--timeout", "360"]
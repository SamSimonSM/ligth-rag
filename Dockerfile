# Usar imagem base com Python
FROM python:3.11-slim

# Evita que o Python crie arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Garante que os logs saiam direto no console
ENV PYTHONUNBUFFERED=1

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto
COPY . .

# Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expõe a porta da aplicação
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
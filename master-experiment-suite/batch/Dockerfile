FROM python:3.9-slim-bullseye
WORKDIR /app
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src .
CMD ["python", "--version"]
FROM python:3.12-slim
WORKDIR /app
# Only the dependency list is baked in; the code is bind-mounted at /app (docker/compose.yaml).
COPY lab/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
EXPOSE 8766
ENTRYPOINT ["uvicorn", "lab.server:app", "--host", "0.0.0.0", "--port", "8766"]

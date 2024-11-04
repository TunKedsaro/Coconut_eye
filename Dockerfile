FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgl1 libglib2.0-0

RUN pip install uvicorn

COPY requirements.txt ./ 

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["uvicorn", "api1:app", "--host", "0.0.0.0", "--port", "3000","--reload"]

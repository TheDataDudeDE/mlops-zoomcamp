FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv

RUN pipenv install --deploy --system

COPY starter.py script.py

ENTRYPOINT ["python", "script.py"]
CMD ["2023", "03"]


# FROM python:3.8

# RUN pip install poetry
# RUN mkdir /app
# WORKDIR /app

# COPY poetry.lock pyproject.toml /app/
# RUN poetry config virtualenvs.create false \
#   && poetry install --no-dev --no-interaction --no-ansi

# COPY streamlit-app.py /app/streamlit-app.py
# COPY flawsleuth /app/flawsleuth
# COPY data /app/data

# ENV PYTHONPATH=$PWD:$PYTHONPATH

# EXPOSE 8501

# ENTRYPOINT ["streamlit", "run", "--server.headless", "true", \
#             "--server.port", "8501", "streamlit-app.py"]

FROM python:3.9

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry
RUN mkdir /app
WORKDIR /app


RUN if [ $(dpkg-query -W -f='${Status}' nvidia-cuda-toolkit 2>/dev/null | grep -c "ok installed") -eq 1 ]; then \
        echo "CUDA detected, installing CUDA-enabled PyTorch"; \
        pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html; \
    else \
        echo "No CUDA detected, installing CPU-only PyTorch"; \
        pip install torch torchvision torchaudio; \
    fi



COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi

COPY streamlit-app.py /app/streamlit-app.py
COPY flawsleuth /app/flawsleuth
COPY data /app/data
COPY .streamlit/config.toml .streamlit/config.toml

ENV PYTHONPATH=$PWD:$PYTHONPATH

EXPOSE 8500

ENTRYPOINT ["streamlit", "run", "--server.headless", "true", \
            "--server.port", "8500", "streamlit-app.py"]
FROM python:3.9-slim

# Метаданные
LABEL maintainer="headliners@example.com"
LABEL version="1.0"
LABEL description="CXRForeignViewer - AI for foreign object detection in chest X-rays"

# Устанавливаем переменные окружения
ENV APP_HOME /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Создаем рабочую директорию
WORKDIR $APP_HOME

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p demo_files output

# Экспонируем порт (если нужно для веб-интерфейса)
EXPOSE 8000

# Команда по умолчанию - запуск основного скрипта
CMD ["python", "main.py"]
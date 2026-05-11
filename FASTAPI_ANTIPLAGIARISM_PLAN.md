# План разработки FastAPI-сервиса антиплагиата

## 1. Цель

Сделать production-ready FastAPI сервис для проверки PDF-документов на заимствования по кастомной базе документов. Сервис должен уметь:

- принимать PDF на проверку через API;
- извлекать текст из PDF, включая OCR для сканов;
- очищать текст и выделять содержательную часть документа;
- разбивать документ на чанки;
- строить embeddings той же моделью, что использовалась в ноутбуках;
- сравнивать загруженный документ с документами из кастомной базы;
- возвращать понятный JSON-отчет с score, найденными источниками и совпавшими чанками;
- поддерживать пополнение базы новыми документами;
- хранить документы, чанки, embeddings, проверки и отчеты в базе данных.

## 2. Текущее состояние кода

Исторически логика проекта выросла из ноутбуков в `raw_jupyter_code`. Сейчас для сервиса важна не папка с экспортами, а сама последовательность шагов: PDF extraction, OCR, preprocessing, chunking, embeddings и проверка одного документа против корпуса.

При переносе в сервис нужно брать логику из кода, а текстовые сообщения, regex-паттерны и комментарии лучше восстановить/перепроверить в `.ipynb` или вручную.

## 3. Целевая архитектура

Рекомендуемая структура проекта:

```text
app/
  main.py
  api/
    v1/
      routes/
        health.py
        documents.py
        checks.py
        admin.py
  core/
    config.py
    logging.py
    errors.py
  db/
    session.py
    models.py
    migrations/
  schemas/
    documents.py
    checks.py
    common.py
  services/
    pdf_extractor.py
    ocr.py
    preprocessing.py
    chunking.py
    embeddings.py
    similarity.py
    plagiarism.py
    reports.py
  repositories/
    documents.py
    chunks.py
    checks.py
  workers/
    tasks.py
  tests/
```

Основная идея: вынести notebook-код в переиспользуемые сервисные модули, а FastAPI оставить тонким слоем для HTTP, валидации и постановки тяжелых задач в очередь.

## 4. База данных

### 4.1 Рекомендуемый вариант

Использовать PostgreSQL + `pgvector`.

Плюсы:

- embeddings хранятся рядом с метаданными;
- можно делать быстрый vector search;
- проще обновлять корпус без пересборки `.npy`;
- удобнее масштабировать API и фоновые задачи.

### 4.2 Минимальная альтернатива для MVP

Использовать SQLite/PostgreSQL для метаданных + FAISS или NumPy-файлы для векторного индекса.

Плюсы:

- быстрее запустить MVP;
- меньше инфраструктуры.

Минусы:

- сложнее синхронизировать БД и индекс;
- хуже подходит для параллельного пополнения базы;
- потребуется отдельная процедура rebuild индекса.

### 4.3 Предлагаемая схема таблиц

```text
documents
  id
  filename
  original_path
  source
  year
  status
  total_pages
  text_len
  clean_word_count
  created_at
  updated_at

document_texts
  id
  document_id
  raw_text
  body_text
  clean_text

chunks
  id
  document_id
  chunk_idx
  token_start
  token_end
  text_preview
  embedding vector
  created_at

checks
  id
  filename
  status
  total_pages
  clean_word_count
  n_chunks
  suspect_count
  top_score
  created_at
  completed_at
  error_message

check_matches
  id
  check_id
  matched_document_id
  matched_filename
  matched_year
  matched_query_chunks
  matched_corpus_chunks
  n_query_chunks
  n_corpus_chunks
  overlap_query
  overlap_corpus
  score
  max_similarity
  mean_top_similarity

check_match_chunks
  id
  check_match_id
  query_chunk_idx
  corpus_chunk_idx
  similarity
  query_token_start
  query_token_end
  corpus_token_start
  corpus_token_end
```

## 5. API endpoints

### 5.1 Health

```text
GET /health
GET /api/v1/health
```

Возвращает статус сервиса, БД, модели embeddings и OCR.

### 5.2 Документы базы

```text
POST /api/v1/documents
```

Загрузить PDF в кастомную базу.

```text
GET /api/v1/documents
GET /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
POST /api/v1/documents/{document_id}/reindex
```

Нужны для просмотра корпуса, удаления документа и пересчета embeddings.

### 5.3 Проверки

```text
POST /api/v1/checks
```

Загрузить PDF и запустить проверку на плагиат.

Параметры:

- `match_threshold`, по умолчанию `0.90`;
- `suspect_overlap`, по умолчанию `0.20`;
- `skip_intro_chunks`, по умолчанию `2`;
- `top_k_matches`, по умолчанию `10`.

```text
GET /api/v1/checks/{check_id}
GET /api/v1/checks/{check_id}/matches
```

Возвращают статус, итоговый отчет и детальные совпадения.

## 6. Алгоритм проверки

Алгоритм переносится из `06_check_file.txt`:

1. Получить PDF.
2. Извлечь страницы через PyMuPDF.
3. При необходимости прогнать OCR для пустых/сканированных страниц.
4. Удалить служебные части документа: титульные страницы, аннотацию, оглавление, библиографию, приложения.
5. Очистить текст:
   - lower-case;
   - удалить спецсимволы;
   - удалить стоп-слова `russian`, `english`, `kazakh`;
   - удалить слишком короткие токены.
6. Разбить текст на чанки:
   - модель: `ibm-granite/granite-embedding-311m-multilingual-r2`;
   - `MAX_LEN = 512`;
   - `OVERLAP = 64`.
7. Построить нормализованные embeddings.
8. Сравнить query chunks с chunks из базы:
   - cosine similarity через dot product;
   - исключить intro chunks через `skip_intro_chunks`;
   - считать `matched_q`, `matched_c`, `overlap_q`, `overlap_c`;
   - итоговый `score = max(overlap_q, overlap_c)`.
9. Отсортировать совпадения по `score`.
10. Вернуть подозрительные документы, где `score >= suspect_overlap`.

## 7. Этапы разработки

### Этап 1. Инициализация проекта

- Создать структуру `app/`.
- Добавить `pyproject.toml` или `requirements.txt`.
- Добавить зависимости:
  - `fastapi`;
  - `uvicorn`;
  - `pydantic-settings`;
  - `sqlalchemy`;
  - `alembic`;
  - `psycopg` или `asyncpg`;
  - `pgvector`, если выбран PostgreSQL + pgvector;
  - `pymupdf`;
  - `pillow`;
  - `pytesseract`;
  - `numpy`;
  - `pandas`;
  - `torch`;
  - `transformers`;
  - `nltk`;
  - `python-multipart`;
  - `pytest`.
- Настроить `.env`:
  - `DATABASE_URL`;
  - `MODEL_ID`;
  - `UPLOAD_DIR`;
  - `TESSERACT_CMD`;
  - `MATCH_THRESHOLD`;
  - `SUSPECT_OVERLAP`.

### Этап 2. Перенос notebook-кода в модули

- Перенести извлечение текста из `02_data_extracting.txt` в `services/pdf_extractor.py`.
- Перенести OCR-логику в `services/ocr.py`.
- Перенести `extract_body`, regex-паттерны и `clean` из `03_data_preprocessing.txt` / `06_check_file.txt` в `services/preprocessing.py`.
- Перенести `chunk_document`, `build_batch`, `mean_pool` в `services/chunking.py` и `services/embeddings.py`.
- Перенести расчет similarity из `05_plagiarism_detection.txt` / `06_check_file.txt` в `services/similarity.py`.
- Собрать orchestration-класс `PlagiarismService` в `services/plagiarism.py`.

### Этап 3. Модель embeddings

- Загружать tokenizer/model один раз на старте приложения.
- Хранить модель в dependency/container, а не загружать на каждый запрос.
- Поддержать CPU и CUDA:
  - `cuda`, если доступна;
  - `torch.autocast` только для CUDA;
  - batch size вынести в настройки.
- Проверять норму embeddings после расчета.
- Добавить прогрев модели на старте или первый lazy load.

### Этап 4. Хранилище корпуса

- Создать SQLAlchemy models и Alembic migrations.
- Реализовать repositories:
  - создание документа;
  - сохранение текстов;
  - сохранение чанков и embeddings;
  - выборка embeddings для поиска;
  - сохранение результата проверки.
- Проверить, что `doc_idx` из корпуса корректно мапится на новые `document_id`.

### Этап 5. Индексация документов

- Реализовать `POST /api/v1/documents`.
- Pipeline:
  - сохранить файл;
  - извлечь raw text;
  - очистить body/clean text;
  - создать chunks;
  - построить embeddings;
  - сохранить документ и чанки в БД;
  - отметить статус `ready`.
- Для долгих задач добавить очередь:
  - MVP: `FastAPI BackgroundTasks`;
  - лучше: Celery/RQ/Dramatiq + Redis.

### Этап 6. Проверка на плагиат

- Реализовать `POST /api/v1/checks`.
- Pipeline:
  - сохранить uploaded PDF;
  - обработать документ тем же pipeline, что и corpus documents;
  - построить query embeddings;
  - получить кандидатов из БД;
  - посчитать score и top matches;
  - сохранить `checks`, `check_matches`, `check_match_chunks`;
  - вернуть `check_id` и/или итоговый отчет.
- Для MVP можно делать синхронно, если корпус небольшой.
- Для реального использования лучше асинхронно:
  - `POST /checks` возвращает `202 Accepted`;
  - клиент опрашивает `GET /checks/{check_id}`.

### Этап 7. Отчеты

- JSON-отчет должен включать:
  - статус проверки;
  - имя файла;
  - количество страниц;
  - количество слов после очистки;
  - количество чанков;
  - количество подозрительных совпадений;
  - top matched documents;
  - `score`, `max_sim`, `overlap_q`, `overlap_c`;
  - top chunk matches.
- Дополнительно можно добавить экспорт:
  - `GET /checks/{check_id}/report.csv`;
  - `GET /checks/{check_id}/report.pdf`.

### Этап 8. Тесты

- Unit-тесты:
  - `clean_text`;
  - `extract_body`;
  - `chunk_document`;
  - `mean_pool`;
  - расчет `score`;
  - сортировка совпадений.
- Integration-тесты:
  - загрузка документа в базу;
  - проверка PDF против маленького тестового корпуса;
  - обработка пустого PDF;
  - обработка документа с малым количеством текста.
- API-тесты:
  - `/health`;
  - `/documents`;
  - `/checks`;
  - `/checks/{id}`.

### Этап 9. Производительность

- Не считать полную матрицу `N x N` в API-запросе.
- Для проверки нового документа сравнивать query chunks с индексом базы.
- Если используется pgvector:
  - получать top-N похожих chunks для каждого query chunk;
  - агрегировать результаты по документам.
- Если используется NumPy/FAISS:
  - держать индекс в памяти;
  - обновлять индекс после добавления документов;
  - сохранять snapshot индекса на диск.
- Добавить лимиты:
  - максимальный размер PDF;
  - максимальное число страниц;
  - таймаут OCR;
  - batch size embeddings.

### Этап 10. Надежность и эксплуатация

- Логировать каждую проверку с `check_id`.
- Возвращать понятные ошибки:
  - файл не PDF;
  - не удалось извлечь текст;
  - слишком мало текста;
  - модель не загружена;
  - OCR недоступен;
  - БД недоступна.
- Добавить README с командами запуска и списком переменных окружения.

## 8. MVP

Минимальная версия, которую стоит собрать первой:

1. FastAPI приложение с `/health`.
2. DB-first корпус в PostgreSQL.
3. Модульная версия функций проверки PDF.
4. Endpoint `POST /api/v1/checks`, который принимает PDF и возвращает JSON-результат.
5. Простые tests для preprocessing, chunking и scoring.
6. README с инструкцией запуска.

После MVP:

1. `POST /api/v1/documents` для пополнения кастомной базы.
2. PostgreSQL + pgvector.
3. Фоновые задачи.
4. История проверок.
5. Экспорт отчетов.

## 9. Риски

- OCR через Tesseract может быть медленным и зависит от установленных языков `kaz`, `rus`, `eng`.
- Модель Granite тяжелая; на CPU проверка может занимать заметное время.
- Полная матрица similarity из `05_plagiarism_detection.txt` не подходит для online API на большом корпусе.
- Regex-паттерны для удаления титульных страниц/библиографии нужно восстановить из-за проблем с кодировкой в `.txt`.
- Stopwords `kazakh` в NLTK могут быть недоступны в окружении без нужного корпуса.
- Нужно одинаково обрабатывать документы базы и загружаемые документы, иначе score будет нестабильным.

## 10. Критерии готовности

- Сервис запускается командой `uvicorn app.main:app --reload`.
- `/health` показывает, что БД и модель доступны.
- Можно загрузить PDF в кастомную базу.
- Можно проверить новый PDF через `/api/v1/checks`.
- Результат проверки сохраняется в БД.
- JSON-отчет содержит top совпадения и chunk-level детали.
- Для тестового документа из базы сервис находит высокий score с самим собой или близкой копией.
- Unit/integration tests проходят.
- README описывает установку Tesseract, переменные окружения и запуск API.

## 11. Предлагаемый порядок реализации

1. Создать базовую структуру FastAPI проекта.
2. Вынести чистые функции из ноутбуков в `services/`.
3. Добавить endpoint `/checks`.
4. Добавить БД и модели.
5. Добавить endpoint `/documents`.
6. Оптимизировать vector search.
7. Добавить очередь для тяжелых задач.
8. Добавить отчеты, тесты и README для запуска без Docker.

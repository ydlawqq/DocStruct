# DocStruct

Сервис извлечения структурированных данных из чеков и документов на базе мультимодальной модели **Qwen2-VL-2B-Instruct**, дообученной под задачу с помощью **LoRA (QLoRA)**.

Проект состоит из двух независимых частей:

- **Обучение** — дообучение модели на датасете чеков, результат — PEFT-адаптер;
- **Инференс** — HTTP-сервис на FastAPI, который по изображению возвращает структурированный JSON.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

---

## Содержание

- [Возможности](#возможности)
- [Стек технологий](#стек-технологий)
- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Обучение (QLoRA)](#обучение-qlora)
- [Быстрый старт (Docker)](#быстрый-старт-docker)
- [API](#api)
- [Конфигурация](#конфигурация)
- [Локальный запуск без Docker](#локальный-запуск-без-docker)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Возможности

- Извлечение данных из чеков: на вход — изображение, на выходе — JSON (позиции, суммы, итоги);
- LoRA-дообучение (QLoRA): 4-bit квантование + low-rank адаптеры, экономия памяти и времени;
- Инференс как отдельный сервис: FastAPI + асинхронная очередь, независимость от БД;
- Совместимость с CPU: автоматический fallback на `float16` при отсутствии CUDA;
- Готовая конфигурация Docker Compose: подъём сервиса одной командой (CPU/GPU).

---

## Стек технологий

| Компонент     | Версия                              |
|---------------|-------------------------------------|
| Python        | 3.12                                |
| PyTorch       | 2.9.1                               |
| TorchVision   | 0.24.1                              |
| Transformers  | 5.3.0                               |
| PEFT          | 0.20.0                              |
| qwen-vl-utils | 0.0.14                              |
| TRL (SFTTrainer) | latest                          |
| FastAPI / Uvicorn | 0.110+ / 0.29+                  |
| PostgreSQL    | 15 (сервис `db`)                   |
| Модель        | `Qwen/Qwen2-VL-2B-Instruct`        |

> **Примечание.** Версии зависимостей в `Dockerfile` зафиксированы под локальное окружение. Важно сохранять совместимость пары `transformers` ↔ `qwen-vl-utils` — см. [Troubleshooting](#troubleshooting).

---

## Архитектура

```
┌──────────────────────┐       ┌───────────────────────────────┐
│   Обучение (QLoRA)   │       │    Инференс (FastAPI)         │
│                      │       │                               │
│  dataset → trainer   │       │  POST /extract                │
│  → adapters/         │──────>│  └─> InferenceQueue ─> model  │
└──────────────────────┘       │  └─> JSON-ответ              │
                               └───────────────────────────────┘
```

- **Обучение** отделено от инференса: результатом обучения является каталог адаптеров (`adapters/qwen2vl-2b`), который инференс подхватывает при старте.
- **Инференс** — автономный сервис: принимает изображение, возвращает JSON и не выполняет запись в БД (персистентность — ответственность клиента/прослойки).
- **Очередь** (`InferenceQueue`) выполняет тяжёлую генерацию в отдельном потоке, не блокируя event loop FastAPI.

---

## Структура проекта

```
.
├── src/
│   ├── configs/
│   │   └── qwen2_2b.yaml          # конфиг модели (обучение + инференс)
│   ├── model/
│   │   ├── config.py              # загрузка YAML-конфига
│   │   ├── prompts.py             # промпт извлечения
│   │   ├── inference/
│   │   │   └── model.py           # загрузка модели + генерация
│   │   └── api/
│   │       ├── endpoint.py        # FastAPI приложение (/extract, /health)
│   │       ├── queue.py           # асинхронная очередь инференса
│   │       └── serve.py           # CLI-запуск сервиса
│   └── training/
│       ├── main.py                # entry point `train`
│       ├── train.py               # SFTTrainer + QLoRA
│       └── data/
│           └── prepare_dataset.py # подготовка датасета, коллаторы
├── adapters/
│   └── qwen2vl-2b/                # обученный PEFT-адаптер
├── db/                            # работа с PostgreSQL (репозитории)
├── Dockerfile                     # сборка образа инференса
├── docker-compose.yml             # сервисы: db + inference
└── docker-compose.gpu.yml         # оверлей для NVIDIA GPU
```

---

## Обучение (QLoRA)

### Пайплайн

1. **Датасет**: `mychen76/ds_receipts_v2_test` (HuggingFace) — колонки `image` и `text`.
   Код подготовки: `src/training/data/prepare_dataset.py`.
2. **Предобработка целевого поля**: из JSON удаляются поля `tax`, `subtotal`, `ignore`, `tips`, а из позиций — `item_key`.
3. **Разбиение**: 90% train / 10% val (`random_state=42`).
4. **Квантование**: 4-bit `BitsAndBytes` (nf4 + double quant) — режим QLoRA.
5. **LoRA**: адаптеры на `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`.
6. **Маскирование промпта**: токены пользовательской части исключаются из loss — модель обучается только на ответе ассистента.
7. **Early stopping**: `patience=1`, `threshold=0.1`.
8. **Результат**: PEFT-адаптер сохраняется в `adapters/qwen2vl-2b` и используется инференсом.

### Запуск

```bash
# из корня проекта
train --config src/configs/qwen2_2b.yaml
```

### Ключевые параметры конфига

| Секция        | Параметр           | Значение по умолчанию               | Описание                              |
|---------------|--------------------|-------------------------------------|---------------------------------------|
| `model`       | `name`             | `Qwen/Qwen2-VL-2B-Instruct`         | Базовая модель                       |
| `model`       | `adapters_dir`     | `adapters/qwen2vl-2b`               | Каталог сохранения/чтения адаптера   |
| `quantization`| `load_in_4bit`     | `true`                              | 4-bit квантование (только NVIDIA GPU)|
| `lora`        | `r` / `alpha`      | `16` / `32`                         | Ранг и масштаб LoRA                  |
| `lora`        | `target_modules`   | `[q,k,v,o,gate,up_proj]`            | Целевые слои                          |
| `training`    | `num_train_epochs` | `3`                                 | Число эпох                            |
| `training`    | `learning_rate`    | `2e-5`                              | Скорость обучения                     |
| `training`    | `output_dir`       | `adapters/qwen2vl-2b`               | Каталог адаптера                      |

> **Примечание.** QLoRA (4-bit) требует NVIDIA GPU. Обучение на CPU не предусмотрено.

---

## Быстрый старт (Docker)

### Требования

- Docker (или podman) с Compose;
- для GPU-режима: NVIDIA драйвер + nvidia-container-toolkit;
- файл `.env` в корне проекта с переменными БД:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASS=postgres
DB_NAME=docstruct
```

### Вариант 1 — только инференс (CPU)

```bash
cd /home/ydlawq/DocStruct
docker compose up -d --build --no-deps inference
```

### Вариант 2 — только инференс (GPU)

```bash
cd /home/ydlawq/DocStruct
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --no-deps inference
```

### Вариант 3 — весь стек (db + inference)

```bash
docker compose up -d --build
```

### Проверка

```bash
# статус контейнера — должен быть Up
docker ps

# health — отвечает {"status":"ok"} только после полной загрузки модели
curl localhost:8000/health

# логи загрузки модели
docker logs -f qwen-vlm-inference
```

Ожидаемые строки в логах:

```
INFO:src.model.inference.model:Загрузка модели Qwen/Qwen2-VL-2B-Instruct...
INFO:src.model.inference.model:Загрузка адаптеров PEFT из /app/adapters/qwen2vl-2b...
INFO:src.model.inference.model:Модель полностью загружена и готова к генерации
```

> **Примечание.** Повторяющиеся `{"status":"ok"}` в логах — результат работы HEALTHCHECK Docker (каждые 30 секунд), это нормальное поведение.

---

## API

### `GET /health`

Проверка готовности сервиса. Отвечает `200` только после загрузки модели.

```bash
curl localhost:8000/health
```

```json
{"status":"ok"}
```

### `POST /extract`

Принимает изображение чека (multipart/form-data) и возвращает распознанный JSON.

```bash
curl -X POST localhost:8000/extract \
  -F "file=@/путь/к/чеку.jpg" \
  -o result.json -w "HTTP %{http_code}\n"
```

Ответ — JSON с извлечёнными данными (состав полей зависит от датасета обучения).

> **Примечание.** На CPU генерация занимает десятки секунд — ожидаемое поведение. Для продакшена рекомендуется GPU.

---

## Конфигурация

Инференс управляется YAML-конфигом `src/configs/qwen2_2b.yaml` и переменными окружения:

| Переменная     | По умолчанию                     | Описание                                          |
|----------------|----------------------------------|---------------------------------------------------|
| `CONFIG_PATH`  | `src/configs/qwen2_2b.yaml`      | Путь к YAML-конфигу модели                       |
| `ADAPTERS_DIR` | из конфига (`adapters/qwen2vl-2b`) | Абсолютный путь к PEFT-адаптерам (важно в контейнере) |
| `HF_HOME`      | `~/.cache/huggingface`           | Кэш базовой модели HuggingFace                   |

Приоритет пути к конфигу:

1. явный флаг `--config`;
2. переменная `CONFIG_PATH`;
3. значение по умолчанию `src/configs/qwen2_2b.yaml`.

> **SELinux (Fedora/RHEL и производные):** bind-mount каталога адаптеров должен иметь суффикс `:z` — он уже указан в `docker-compose.yml` (`./adapters:/app/adapters:z`). Без него контейнер не сможет читать адаптеры (`Permission denied` → `Can't find adapter_config.json`).

---

## Локальный запуск без Docker

При условии, что на хосте установлены совместимые версии зависимостей:

```bash
cd /home/ydlawq/DocStruct
serve --config src/configs/qwen2_2b.yaml --host 0.0.0.0 --port 8000
```

Флаги команды `serve`:

| Флаг       | По умолчанию           | Описание                     |
|------------|------------------------|------------------------------|
| `--config` | `CONFIG_PATH` / дефолт | Путь к конфигу              |
| `--host`   | `0.0.0.0`              | Хост привязки                |
| `--port`   | `8000`                 | Порт                         |
| `--reload` | —                      | Автоперезагрузка (dev-режим) |

---

## Troubleshooting

| Проблема                                                       | Причина                                          | Решение                                                     |
|----------------------------------------------------------------|--------------------------------------------------|--------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'torchvision'`           | `qwen_vl_utils` требует torchvision              | Добавить `torchvision==<парная версия>` в `Dockerfile`       |
| `Could not import module 'BloomPreTrainedModel'`               | Несовместимость `transformers` и `qwen-vl-utils` | Зафиксировать совместимые версии (например `transformers==4.45.2` + `qwen-vl-utils==0.0.8`) |
| `Can't find 'adapter_config.json' ...`                         | SELinux блокирует bind-mount; относительный путь | В compose использовать `:z` и env `ADAPTERS_DIR=/app/adapters/qwen2vl-2b` |
| Повторяющийся `{"status":"ok"}` в `docker logs`                | HEALTHCHECK Docker (каждые 30 с)                 | Нормальное поведение, не ошибка                               |
| `Permission denied` на bind-mount                              | SELinux (контекст `user_home_t` → `container_t`) | Суффикс volume `:z` / `:Z`                                   |
| Контейнер `inference` не стартует с GPU-секцией                | Нет NVIDIA GPU / nvidia-container-toolkit        | Запускать CPU-вариант без `docker-compose.gpu.yml`            |

---

## Roadmap

Секция для планов развития проекта.

- [ ] Покрыть инференс нагрузочным тестом, рассмотреть несколько воркеров в очереди
- [ ] Метрики качества извлечения (валидность JSON, точность позиций)
- [ ] CI: автосборка образа и smoke-тест `/extract`
- [ ] Поддержка других форматов документов (PDF → изображение)
- [ ] Документация и примеры датасета
- [ ] Финальные правки пайплайна обучения
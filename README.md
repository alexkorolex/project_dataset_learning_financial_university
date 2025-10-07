# Бинарная классификация (Bank Marketing)

https://www.kaggle.com/datasets/adilshamim8/binary-classification-with-a-bank-dataset

---

## Быстрый старт

```bash
# 1) Установить зависимости
pip install -r requirements.txt

# 2) Запустить полный пайплайн через DVC (рекомендуется)
dvc repro

# Альтернатива: через Makefile
make all
```

**Входные данные** (положите заранее):
```
data/raw/train.csv
data/raw/test.csv
```

**Результаты** появятся в `artifacts/`:
- модели: `model_*.joblib` (sklearn‑пайплайны), `model_*.cbm` (CatBoost);
- метрики валидации: `metrics_valid.json`;
- вероятности на валидации: `preds_*_valid.csv`;
- кривые ROC/PR: `curves_*_{roc,pr}.png`;
- сабмиты: `submission_*_test.csv` (колонки `id,p`).

---

## Конфиг проекта: `config/config.yaml` (референс)

```yaml
data:
  raw_dir: data/raw
  target: y
  val_size: 0.2
  random_state: 42
  drop_cols: [id]           # удаляются до разделения на X/y
  yn_binary_cols: [default, housing, loan]   # 'yes'/'no' → 1/0
  use_duration: false       # защититься от утечки (duration)
  subsample_rows: null      # опционально сократить train для быстрой отладки

features:
  numeric: [age, balance, day, pdays, previous, duration]  # 'duration' будет отброшен, если use_duration=false
  categorical: [job, marital, education, contact, month, poutcome]
  add_pdays_indicator: true   # создаёт признак pdays_is_never (pdays == -1)

artifacts:
  dir: artifacts

models:
  logreg:
    type: logreg
    params: {max_iter: 1000, n_jobs: -1, random_state: 42}
  rf:
    type: rf
    params: {n_estimators: 400, max_depth: null, n_jobs: -1, random_state: 42}
  hgbt:
    type: hgbt
    params: {max_depth: null, learning_rate: 0.05, max_iter: 500, random_state: 42}
  cat:
    type: catboost
    params: {depth: 8, learning_rate: 0.05, iterations: 1000, loss_function: Logloss, random_seed: 42, verbose: false}

thresholds:
  default: 0.5
  tune_on_valid: true        # искать f1‑оптимальный порог на валидации
```

> CatBoost проходит **отдельной веткой** (нативные категориальные, формат `.cbm`), остальные модели идут как sklearn‑пайплайны через `ColumnTransformer`.

---

## Makefile (ключевые цели)

```bash
make venv       # создать .venv, обновить pip/setuptools/wheel
make install    # установить зависимости в .venv
make eda        # сгенерировать artifacts/eda_summary.json
make train      # обучение, метрики, кривые, модели
make infer      # сабмиты по всем сохранённым моделям
make lint       # ruff + black --check + isort --check-only
make test       # pytest -q
make dvc-init   # dvc init (однократно)
make dvc-repro  # dvc repro (полный DAG)
make reset      # удалить artifacts, .venv и прочие временные папки
```

Makefile кроссплатформенный (Windows/Unix), правильно прокидывает `PYTHONPATH` и разделители.

---

## Тестирование

Запуск:
```bash
pytest -q
```

Покрытие (минимум):
- `tests/test_preprocess.py` — корректная сборка фич (включая индикатор `pdays_is_never`), отсутствие пересечений списков;
- `tests/test_pipelines.py` — устойчивость препроцессоров к пропускам;
- `tests/test_metrics.py` — метрики и подбор порога.

---

## DVC‑пайплайн

В `dvc.yaml` описаны стадии:

- **eda** → `eda_summary.py` → `artifacts/eda_summary.json`
- **train** → `train.py` → модели, метрики, кривые
- **infer** → `infer.py` → сабмиты

```bash
dvc repro        # запустить все стадии с учётом зависимостей
dvc dag          # посмотреть граф стадий
```

При желании подключите удалённое хранилище для артефактов:
```bash
dvc remote add -d storage <s3|gdrive|ssh|...>
dvc push
```

---

## Качество кода

- **pre-commit**: ruff, black, isort (см. `.pre-commit-config.yaml`).
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) запускает линтеры и тесты на PR.

```bash
pre-commit install
pre-commit run --all-files
```

---

## Метрики и результаты

- Метрики валидации складываются в `artifacts/metrics_valid.json`;
- Порог классификации: либо `thresholds.default`, либо f1‑оптимальный (`tune_on_valid=true`);
- Визуализации ROC/PR сохраняются для каждой модели.

> На Kaggle, как правило, сабмит — это вероятность (`p`), поэтому в сабмитах мы **не применяем порог** — он используется только для отчётных метрик.

---

## Репродуцируемость

- Зафиксированы версии библиотек в `requirements.txt`;
- Сид: `data.random_state` + `random_state` у моделей;
- Сохранение снапшот конфига рядом с метриками/моделями (см. `utils.save_json`).

---

## Частые проблемы и советы

- **`duration`** — пост‑фактум признак, выключен по умолчанию (`use_duration: false`), чтобы не ловить утечку.
- **Пропуски** — для линейных: `SimpleImputer(median)` + `StandardScaler`; для деревьев: `SimpleImputer(median)` и `OrdinalEncoder` с `unknown_value=-1`.
- **Категориальные** — CatBoost использует индексы категориальных фич напрямую; sklearn‑модели — OneHot/Ordinal.
- **ID в сабмите** — берётся из `test.csv` (если нет — используется `range(n)`).
- **Windows/Unix различия** — Makefile учитывает обе ОС, пути и переменные среды прокидываются корректно.



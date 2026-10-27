# Гибридные методы NLP для казахского языка

Сравнение rule-based, нейросетевых и гибридных подходов на трёх задачах:
NER (KazNERD), POS-теггинг и морфологический анализ (UD Kazakh-KTB).

## Шаги методички AITU → код

Проект выстроен по табл. 7 «Data collection» и табл. 8 «Data processing».

| Шаг методички | Где в проекте | Результат |
|---|---|---|
| 7.1 Defining a data type | `docs/data.md` | описание данных и меток |
| 7.2 Careful planning | этапы 0–5 диссертации, decision log | план, ограничения |
| 7.3 Selecting data collection methods | `docs/data.md` | обоснование выбора корпусов |
| 7.4 Development of tools | `src/data`, `src/eval`, `tests` | конвейер и инструмент оценки |
| 7.5 Data collection | `src/data/collect.py` | данные по зафиксированным версиям, проверка SHA-256 |
| 7.6 Data processing | `src/pipeline.py` (шаги 8.1–8.10) | см. ниже |
| 7.7 Data analysis | `src/data/eda.py`, `src/eval/` | EDA; сравнение методов |
| 7.8 Documentation of results | `manifest.json`, `report.md`, `pipeline_log.json` | |
| 8.1 Data quality check | `src/data/quality.py`, `pipeline.step_quality` | `outputs/stage1_data/quality/` |
| 8.2 Coding and structuring | `src/data/readers.py`, `src/data/encoding.py` | единый JSONL, `label_maps.json` |
| 8.3 Data transformation | `src/data/clean.py`, `src/data/splits.py`, `encoding.align_to_subwords` | очищенные данные, сплиты |
| 8.4 Aggregation and summation | `src/data/eda.py`, `src/eval/breakdown.py` | `outputs/stage1_data/tables/` |
| 8.5 Statistical analysis | `src/data/eda.py`, `src/eval/significance.py` | описательная статистика; bootstrap, Холм |
| 8.6 Data visualisation | `src/data/plots.py` | `outputs/stage1_data/figures/` (PNG, 300 dpi) |
| 8.7 Interpretation of results | `src/eval/` + текст диссертации | |
| 8.8 Creating a report | `src/report.py`, `pipeline.step_report` | `report.html`, `report.md` |
| 8.9 Secrecy and confidentiality | `.gitignore`, Colab Secrets, `DATA_LICENSES.md` | данные вне git, лицензии |
| 8.10 Documentation | `manifest.json`, `cleaning_log.jsonl`, `pipeline_log.json` | воспроизводимость |
| 8.11 Publication of results | репозиторий после защиты | |

## Структура

```
src/pipeline.py   этап 1 целиком: шаги методички по порядку
src/report.py     сборка отчёта (Markdown + HTML)
src/data/         сбор, чтение, проверка качества, очистка, кодирование, сплиты, EDA, графики
src/eval/         единая оценка 9 комбинаций, значимость (bootstrap + Холм), разбивка ошибок
notebooks/        01_stage1_data — тот же этап 1 по шагам, с результатами
docs/data.md      описание данных и предобработки
tests/            pytest -q
```

## Этап 1: данные

Исходные файлы лежат на Google Drive (не в git):

```
thesis/data/kaznerd/IOB2_{train,valid,test}.txt
thesis/data/ud-ktb/kk_ktb-ud-{train,test}.conllu
```

Одна команда выполняет все шаги и пишет результаты в `thesis/outputs/stage1_data/`:

```bash
python -m src.pipeline --thesis /content/drive/MyDrive/thesis            # --download скачает недостающие файлы
```

Подготовленные данные — в `thesis/data/processed/` (см. `docs/data.md`).

## Оценка моделей

Каждый метод сохраняет предсказания в `results/preds/{task}_{method}[_{run}].jsonl`
(формат в `src/eval/io.py`), затем:

```bash
python -m src.eval.evaluate results/preds                  # таблица 3×3 (среднее ± std по seed)
python -m src.eval.evaluate results/preds --significance   # все сравнения: 95% ДИ, p, p с поправкой Холма
```

Разбивка результатов по OOV, длине предложения и источнику — `src/eval/breakdown.py`.

Проверка на игрушечных данных:

```bash
python scripts/make_demo_preds.py
python -m src.eval.evaluate demo/preds --out demo --significance
```

## Colab

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
token = userdata.get('GH_TOKEN')
!git clone -q https://{token}@github.com/zhannaa0/kazakh-hybrid-nlp.git /content/project 2>/dev/null || git -C /content/project pull -q
%cd /content/project
!pip install -q -r requirements.txt
!python -m src.pipeline --thesis /content/drive/MyDrive/thesis
```

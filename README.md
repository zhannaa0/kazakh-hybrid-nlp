# Гибридные методы NLP для казахского языка

Сравнение rule-based, нейросетевых и гибридных подходов на трёх задачах:
NER (KazNERD), POS-теггинг и морфологический анализ (UD Kazakh KTB).

## Структура

```
src/eval/      единая оценка для всех 9 комбинаций (задача × метод)
scripts/       вспомогательные скрипты
tests/         тесты (pytest -q)
results/       метрики и итоговые таблицы
```

## Оценка

Каждый метод сохраняет предсказания в `results/preds/{task}_{method}[_{run}].jsonl`
(формат описан в `src/eval/io.py`), затем:

```bash
python -m src.eval.evaluate results/preds                    # таблица 3×3
python -m src.eval.evaluate --compare results/preds/ner_neural_seed13.jsonl results/preds/ner_hybrid_seed13.jsonl
```

Проверка на игрушечных данных:

```bash
python scripts/make_demo_preds.py
python -m src.eval.evaluate demo/preds --out demo
```

## Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/zhannaa0/kazakh-hybrid-nlp.git /content/project 2>/dev/null || git -C /content/project pull
%cd /content/project
!pip install -q -r requirements.txt
```

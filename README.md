# auditenv — воспроизводимая RL-среда для ИИ-аудитора финансовых записей

Хакатон REC Genesis (FinTech). Агент получает одну синтетическую компанию за один отчётный период
(главная книга, счета, платежи, курсы валют), исследует записи read-only-действиями и один раз
отправляет выводы об ошибках с доказательствами. Без UI, без сети, без GPU, без реальных данных.

* Python 3.11, только стандартная библиотека (pytest — только для тестов, версии закреплены).
* Полная детерминированность: `seed + config + версия + действия` ⇒ побайтово одинаковые записи,
  наблюдения и награды (локальный `random.Random`, деньги — `int` в минорных единицах, курсы — `Decimal`-строки).
* Эпизод ≤ 100 шагов и ≤ 60 секунд.

## Быстрый старт

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
make check
```

`make check` (то же самое без `make`: `python scripts/check.py`) = проверка материализованных эпизодов
(`verify`) → прогон 4 базовых политик на 20 эпизодах в закрытом режиме + 20 парных проверок, генерация
`reports/report.md`, `reports/report.json` и `trajectories.jsonl` (`evaluate`) → все pytest-тесты (`test`),
включая `tests/test_artifacts.py` — сверку со списком сдачи.

Без доступа к PyPI зависимости ставятся из закреплённых wheel-файлов в репозитории:

```bash
.venv/bin/pip install --no-index --find-links vendor/wheels --require-hashes -r requirements.txt
```

### Контейнер (без сети во время работы)

```bash
docker build -t auditenv:1.0.0 .
```

```bash
docker run --rm --network none auditenv:1.0.0
```

Сеть нужна только чтобы скачать базовый образ `python:3.11.10-slim-bookworm`; pytest и зависимости ставятся
из `vendor/wheels/` (`--no-index`, проверка sha256), apt не используется. Контейнер запускает
`python scripts/check.py`. Проверено: в контейнере Python 3.11.10, сети нет, проходят все 178 тестов,
а `trajectories.jsonl` побайтово совпадает с файлом из репозитория.

### Демо со скриптовой политикой

```bash
make demo
```

## Использование API

```python
from auditenv.env import AuditEnv

env = AuditEnv(mode="open")              # "open" — обучение, "closed" — проверка
obs, info = env.reset(seed=123, config={"interactions": {"fx_partial_boundary": 1}})
obs, reward, terminated, truncated, info = env.step(
    {"action": "list_records", "table": "ledger", "cursor": None, "limit": 50})
obs, reward, terminated, truncated, info = env.step(
    {"action": "submit", "episode_id": obs["episode_id"], "findings": [...]})
```

| Действие | Аргументы | Результат |
|---|---|---|
| `list_records` | `table` ∈ ledger/invoices/payments/fx, `cursor` (из `next_cursor`), `limit` ≤ 50 | страница, `total`, `next_cursor` |
| `get_record` | `table`, `id` (row_id / invoice_id / payment_id / fx_id) | одна запись |
| `get_document` | `doc_id` | счёт или платёж |
| `calculator` | `expression` (`+ - * / ( )`, `round(x, n)`, decimal) | строка-значение |
| `submit` | `episode_id`, `findings` (≤ 30) | конец эпизода |

Вывод: `{type, affected_ids, claimed_amount_minor, currency, evidence_ids (≤10), explanation (≤1000)}`.
Типы: `duplicate_invoice`, `missing_or_partial_allocation`, `wrong_fx_date_or_rate`, `period_misattribution`.
Определения сумм, `affected_ids` и требуемых доказательств — в `policy.json` (виден агенту в первом наблюдении).

Награда за исследование 0. Терминальная: `clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)`; в закрытом режиме
агент всегда видит 0.0, оценка остаётся в журнале оценщика (`provider.log`). Неверная структура submit ⇒ −1.
Лимит шагов/времени ⇒ `truncated=True`, ответ считается пустым.

## Траектории (`trajectories.jsonl`)

Одна JSON-строка на шаг: `policy, episode_id, step, action, observation, reward, terminated, truncated`.
`step = 0` — наблюдение после `reset` (`action = null`). Записаны в открытом (обучающем) режиме для
политик random, rule_based и flag_all на 10 train-эпизодах (30 эпизодов, 404 строки): награда ненулевая
только на терминальном шаге. Oracle исключён, потому что его действия — это и есть эталон. Тест
`tests/test_leaks.py` сканирует наблюдения на ID и поля эталона. Файл детерминирован и пересоздаётся
`make evaluate`.

## Конфиги и версии

`configs/default-1.0.0.json` — конфиг по умолчанию, `configs/suite-1.0.0.json` — 20 конфигов
train/dev с хэшами (без seed), `configs/CHANGELOG.md` — история версий. Тест следит, чтобы файлы не
расходились с кодом.

## Структура

```
src/auditenv/
  generator/   чистая история → документы/проводки → ошибки + provenance; transforms (перестановка, смена ID)
  validator/   независимый валидатор по открытым файлам: равенства до, расхождения после, достаточность доказательств
  env/         AuditEnv (reset/step, open/closed), PublicBundle (только 6 открытых файлов), провайдеры эпизодов
  tools/       безопасный decimal-калькулятор, деньги/FX, время
  matcher/     нормализация, допуск, взаимно-однозначное сопоставление
  reward/      метрики, терминальная награда, Grader (журнал оценщика), экспериментальная награда отдельно
  policies/    random, rule_based, flag_all, oracle (только у оценщика) + worker для запуска агента в отдельном процессе
  schemas/     JSON-схемы действий, выводов, записей + валидатор подмножества draft-07
  evaluation/  набор train/dev, прогон, парные проверки, отчёт, траектории, материализация
episodes/{train,dev}/<episode_id>/   открытые файлы: manifest.json ledger.csv invoices.jsonl payments.jsonl fx.csv policy.json
grader/{train,dev}/<episode_id>/     закрытые файлы оценщика: truth.json provenance.json (+ index.json с seed)
configs/  vendor/wheels/  tests/  docs/design.md  scripts/{demo,check}.py  trajectories.jsonl  reports/
```

Каталог `grader/` — сторона оценщика; агенту его не передают (в демонстрационной сборке он лежит в репозитории,
чтобы результаты можно было проверить). Эпизоды восстанавливаются генератором: `make episodes`, проверка — `make verify`.

## Результаты базовых политик (закрытый режим, 20 эпизодов)

| политика | P | R | F1 | return | шаги |
|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | −0.901 | 17.2 |
| rule_based | 1.000 | 0.657 | 0.793 | +0.565 | 20.1 |
| flag_all | 0.002 | 0.014 | 0.003 | −0.994 | 3.7 |
| oracle | 1.000 | 1.000 | 1.000 | +1.000 | 1.0 |

Парные проверки (перестановка порядка, смена ID): 20/20. Подробности — `reports/report.md`, дизайн и
ограничения — [`docs/design.md`](docs/design.md).

**Важно:** высокая награда в этой среде не доказывает пригодность агента для реального аудита.

Лицензия MIT.

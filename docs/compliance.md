# Сверка с требованиями кейса

Каждое требование из PDF кейса «RL-среды для дообучения моделей финансового аудита» → где оно реализовано →
каким тестом проверено. Все тесты запускаются одной командой: `make check` (или `python scripts/check.py`).

## 01 · Суть вызова

| требование | реализация | проверка |
|---|---|---|
| Среда для дообучения, а не приложение/чат-бот/отчёт | `src/auditenv/env/` — Gymnasium-подобный контракт, без UI | `tests/test_env_contract.py` |
| Интерфейс для RL, записанные траектории, оценка качества | `AuditEnv.reset/step`; `trajectories.jsonl`; `reports/report.md` | `test_env_contract.py`, `test_artifacts.py`, `test_policies.py` |
| Без платных API и GPU; не заявлять улучшение модели | только stdlib; обучения весов нет и не заявляется | `Dockerfile` (`--network none`) |

## 02 · Сложность

| требование | реализация | проверка |
|---|---|---|
| 4 независимо настраиваемых типа ошибок | `generator/inject.py`, `config["errors"][type] = {enabled, count}` | `test_generator.py::test_each_type_independently_configurable` |
| 2 сценария взаимодействующих ошибок | `fx_partial_boundary`, `duplicate_cross_period` | `test_generator.py::test_interaction_scenarios` |
| Чистые компании и отвлекающие случаи (разбитые платежи, возвраты, повторяющиеся суммы) | `generator/history.py::build_clean`, 9 видов отвлекающих случаев | `test_generator.py::test_distractors_present_in_clean_company` |
| Сначала согласованная скрытая история, затем документы и ошибки с причинным изменением | `Builder.build_clean` → `assert_clean` → `Injector` → `provenance.json` | `test_generator.py::test_provenance_records_every_mutation` |
| Эталон = реальное расхождение + доступные доказательства, без недоступных фактов | независимый `validator/` | `test_generator.py::test_validator_on_many_seeds` (24 seed + 400 случайных конфигураций при разработке) |
| Вариативность имён, ID, порядка, сумм, валют, поясов, сочетаний | случайные hex-ID, 6 валют, 9 поясов, `row_order` | `test_generator.py::test_varies_across_seeds` |

## 03 · Эпизод и формат данных

| требование | реализация | проверка |
|---|---|---|
| 200–2000 проводок, 20–200 документов, 0–12 ошибок | `generator/config.py`, проверка размера в `generate` | `test_generator.py::test_validator_on_many_seeds`, `test_big_episode_and_clean_company` |
| `manifest.json` с schema_version, episode_id, base_currency, period_start/end, timezone, числом записей, округлением | `generator/__init__.py::generate` | `test_env_contract.py::test_reset_observation` |
| Время ISO-8601 со смещением, деньги — int + разрядность, курсы — decimal-строки | `tools/money.py`, `tools/timeutil.py` | `test_generator.py::test_formats_money_fx_time` |
| Поля ledger / invoices / payments / fx / policy | `schemas/__init__.py` (`*_SCHEMA`) | `validator.structural_errors` на каждом эпизоде |
| Отсутствующие ссылки — явный null | пустое поле CSV → `null` | `test_formats_money_fx_time` |
| Закрытые `truth.json`, `provenance.json`; агент их не видит, как и seed | `grader/<split>/<id>/`; среда держит только `PublicBundle` | `test_leaks.py` |
| Оценщик вне процесса агента; доступ к записям только на чтение | агент в отдельном процессе (`policies/worker.py`), наблюдения — копии | `test_policies.py::test_remote_policy_matches_in_process`, `test_env_contract.py::test_get_record_and_document` |
| Эталон не утекает через наблюдения, пути, ошибки, debug info, объяснения награды | фиксированные коды ошибок, закрытый режим без награды и диагностики | `test_leaks.py` (4 сценария + открытый режим + файлы + траектории) |

## 04 · Наблюдения, действия, завершение

| требование | реализация | проверка |
|---|---|---|
| `reset(seed, config) → (observation, info)`: задача, manifest, политика, бюджет | `AuditEnv.reset` | `test_env_contract.py::test_reset_observation` |
| `step(action) → (obs, reward, terminated, truncated, info)`, JSON-совместимо | `AuditEnv.step` | `test_env_contract.py::test_step_contract_and_paging` |
| Воспроизводимость; reset изолирует состояние | локальный `random.Random`, без глобального состояния | `test_reproducibility.py`, `test_env_contract.py::test_reset_isolates_state` |
| 5 действий; страница ≤ 50 строк; стабильные курсоры | `AuditEnv._dispatch` | `test_step_contract_and_paging` |
| Калькулятор — только арифметика и decimal | `tools/calculator.py` (рекурсивный спуск, без eval) | `test_calculator.py` |
| Неизвестный ID/действие → ограниченная ошибка, расход шага | словарь `ERRORS` | `test_env_contract.py::test_bad_actions_consume_step` |
| Нет shell, SQL, файловой системы, интернета | только 5 действий; лишние ключи отклоняются | `test_bad_actions_consume_step`, `test_leaks.py` |
| ≤ 100 шагов включая submit; submit → terminated | `max_steps=100` | `test_step_limit_truncates_with_empty_answer`, `test_submit_on_step_100_is_allowed` |
| Лимит шагов или времени → truncated и пустой ответ | проверка часов в `step`; жёсткий дедлайн агента в `run_episode` | `test_timeout_truncates`, `test_jury_scenarios.py::test_hung_agent_is_truncated_by_deadline` |
| Награда за исследование 0; при проверке награда и диагностика у оценщика | режимы `open` / `closed` | `test_leaks.py`, `test_adversarial.py::test_closed_mode_gives_no_signal_between_good_and_bad` |

## 05 · Выводы и награда

| требование | реализация | проверка |
|---|---|---|
| Формат submit `{episode_id, findings:[...]}` | `schemas.SUBMIT_ACTION_SCHEMA` | `test_invalid_submission_gives_minus_one` |
| Лимиты: 30 выводов, 10 доказательств, 1000 символов | `schemas.LIMITS` | `test_limits_are_exact`, `test_invalid_submission_gives_minus_one` |
| Неверная структура → −1 и запись в журнал оценщика | `reward.Grader.finalize` | `test_invalid_submission_gives_minus_one` |
| Взаимно однозначное сопоставление; TP = тип + канонические ID + сумма в допуске + валюта + полный набор доказательств | `matcher/` (паросочетание Куна, допуск 1 минорная единица) | `test_matcher_reward.py` |
| Повторы — FP, пропуски — FN | `matcher.match` | `test_duplicate_findings_give_no_benefit` |
| Генератор даёт записи, достаточные для проверки | `validator.prove` на каждом наборе доказательств | `test_validator_on_many_seeds` |
| Открытый код нормализации, допусков, выбора при неоднозначности | `src/auditenv/matcher/__init__.py` | `test_one_to_one_matching_is_order_independent`, `test_tolerance_and_normalization` |
| P, R, F1 и крайние случаи | `reward.metrics` | `test_metric_edge_cases` |
| Награда `clip(2·F1 − 1 − 0.002·steps, −1, 1)`; экспериментальная — отдельно | `reward.terminal_reward`; `experimental.partial_credit_f1` | `test_metric_edge_cases`, `test_guessed_ids_without_evidence_do_not_score` |
| Награда не зависит от текста | `explanation` не читается | `test_explanation_text_is_ignored` |
| Полный ответ лучше молчания на ошибочных книгах | — | `test_full_answer_beats_silence_on_error_company` |
| Молчание лучше ложных обвинений на чистых | — | `test_silence_beats_false_accusations_on_clean_company` |
| Дубли не дают выгоды | — | `test_duplicate_findings_give_no_benefit`, `test_padding_with_correct_findings_does_not_help` |
| «Пометить всё» даёт плохой результат | — | `test_flag_everything_scores_badly`, `test_brute_force_guessing_is_unprofitable` |
| Экономия шагов лишь немного различает равные ответы | — | `test_step_savings_only_slightly_separate_equal_answers` |
| Угаданные ID, неподтверждённые доказательства, валюты, повторная отправка, неверный ввод, тайм-аут | — | `test_guessed_ids_without_evidence_do_not_score`, `test_unsupported_evidence_does_not_score`, `test_wrong_currency_amount_type_ids`, `test_submit_terminates_and_repeat_submit_gains_nothing`, `test_bad_actions_consume_step`, `test_timeout_truncates` |

## 06 · Закрытая проверка среды

| требование | реализация | проверка |
|---|---|---|
| Открытые train/dev-конфиги и seed отделены от закрытых seed и новых сочетаний | `configs/` (без seed), `grader/*/index.json`; `evaluate --suite` для любых seed | `test_jury_scenarios.py::test_hidden_seeds_and_new_parameter_combinations` |
| Безвредные преобразования | `DirectoryProvider` читает изменённые файлы с диска | `test_jury_scenarios.py::test_file_level_reorder_and_rename_preserve_answers` (10 эпизодов, независимое преобразование текста) |
| 4 политики организаторов: random с фиксированным seed, простая сверка, «пометить всё», oracle только у оценщика | `policies/`; внешние политики через `модуль:Класс`; oracle агенту запрещён | `test_policies.py`, `test_jury_scenarios.py::test_agent_process_cannot_load_the_oracle_or_junk` |
| 20 эпизодов (5 чистых, 5 с взаимодействием) + 20 парных проверок | `evaluation/suite.py`, `evaluation.paired_checks` | `test_policies.py::test_suite_composition`, `test_paired_checks_all_pass` |
| Эталон и доказательства проверяются независимо по записям | `python -m auditenv.cli validate` | `test_jury_scenarios.py::test_cli_on_disk_episodes` |

## 07 · Минимальные требования

| требование | где |
|---|---|
| Генератор: 4 типа, 2 сценария, чистые и отвлекающие | разделы 02–03 выше |
| Среда reset/step, действия, 100 шагов | раздел 04 |
| Общий matcher и награда с атакующими тестами | раздел 05, `tests/test_adversarial.py` |
| Траектории без эталона в наблюдениях | `trajectories.jsonl`, `test_leaks.py::test_trajectories_file_is_clean` |
| Проверка без сети, платных API, GPU | `docker run --network none` — проверено, 196 тестов |
| Инструкции контейнера и единая команда | `README.md`, `make check` |
| Лимиты Python 3.11, 2 CPU, 4 ГБ, 1 ГБ, 60 с, 100 шагов | 3.11; один процесс; репозиторий ≈ 5 МБ; эпизод ≤ 0,7 с; 100 шагов |

## 09 · Состав сдачи

| материал | где |
|---|---|
| Исходный код с лицензией и зафиксированными зависимостями | `src/`, `LICENSE` (MIT), `requirements.txt` (версии + sha256), `vendor/wheels/` |
| README: контейнер + единая команда | `README.md` |
| Matcher и награда, схемы, политики | `src/auditenv/{matcher,reward,schemas,policies}` |
| 10 train + 10 dev эпизодов | `episodes/{train,dev}/` |
| Тесты среды, сопоставления, награды | `tests/` |
| Четыре адаптера базовых политик | `src/auditenv/policies/` |
| `trajectories.jsonl` с нужными полями, без эталона | корень репозитория |
| Записка о дизайне | `docs/design.md` |
| Демо со скриптовой политикой | `make demo` (`scripts/demo.py`), `python -m auditenv.cli run` |

## 10 · Критерии оценки

| критерий | чем подкреплён |
|---|---|
| Контракт и изоляция (25) | `test_env_contract.py`, `test_leaks.py`, `test_adversarial.py`, агент в отдельном процессе, тайм-аут и падение агента → truncated |
| Данные и эталон (25) | независимый валидатор: равенства до, расхождение после, достаточность каждого набора доказательств; `cli validate` |
| Надёжность награды (25) | `test_matcher_reward.py`, `test_adversarial.py` |
| Воспроизводимость и обобщение (15) | `test_reproducibility.py`, 20/20 парных проверок, независимые преобразования файлов, совпадение в Docker побайтово |
| Инженерия и объяснение (10) | одна команда; `configs/` + CHANGELOG; `trajectories.jsonl`; измеренный разбор сбоев и честные ограничения в `design.md` |
| P, R, F1, возврат, шаги, время отдельно | `reports/report.md` |

## 12 · Важно

| требование | где |
|---|---|
| Всё вымышлено, нет реальных данных, карт, персональных данных | только синтетические случайные ID, без имён людей и реальных компаний |
| Проверка без сети | Docker `--network none` |
| Не аудиторское заключение; UI нет | `docs/design.md`, раздел «Честные ограничения» |

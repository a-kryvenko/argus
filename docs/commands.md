# Команды: локально и на проде

Локальные команды выполняются из корня репозитория. На сервере используется
`/var/www/bin/argus`: он подставляет env-файлы и закреплённые версии образов релиза.

## Локальный запуск

Нужны Docker Compose v2, Node.js 20.9+ (Next.js 16), pnpm 9, uv и Python 3.12.11.
Для сбора данных и прогнозов нужен приватный checkout `packages/forecast-core`,
а также модели и входные файлы по путям из `configs/`.

```bash
pnpm install --frozen-lockfile
uv sync --project apps/api --frozen
uv sync --project apps/clio --frozen
uv sync --project apps/prophet --frozen
uv sync --project apps/intelligence --frozen
```

Подготовьте `.env` и `.env.local` (локальные значения имеют приоритет):

| Переменные | Локальное значение / назначение |
| --- | --- |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Начальная БД и администратор контейнера PostgreSQL |
| `DB_HOST`, `DB_PORT` | Подключение администратора для provisioning: `127.0.0.1`, `5432` |
| `API_DB_*`, `CLIO_DB_*` | Для каждого домена: `HOST`, `PORT`, `NAME`, `USER`, `PASSWORD` |
| `PROPHET_DB_*`, `INTELLIGENCE_DB_*` | Аналогичные параметры отдельных БД/владельцев |
| `OBSERVATIONS_URL` | `http://127.0.0.1:8001` |
| `OBSERVATIONS_SERVICE_TOKEN` | Общий секрет Clio, API и Prophet |
| `FORECASTS_URL` | `http://127.0.0.1:8002` |
| `FORECASTS_SERVICE_TOKEN` | Общий секрет Prophet, API и Intelligence |
| `DASHBOARD_ORIGINS`, `DASHBOARD_COOKIE_SECURE` | `http://localhost:3000`, `false` для локального HTTP |

Примеры настроек — в [domain-storage.md](domain-storage.md). Локальный хост — `127.0.0.1`,
на проде — `postgres`. Пароли передавайте без URL-кодирования.
Следующая последовательность предназначена
для новой БД; для существующей сначала прочитайте [правила provisioning](domain-storage.md).
Provisioning создаёт отдельные БД и владельцев, не меняя существующие пароли.

```bash
./scripts/argus compose up -d --wait
./scripts/argus db provision --apply
./scripts/argus db migrate
```

Запуск всех HTTP-сервисов и frontend одной командой:

```bash
pnpm dev
```

`pnpm dev` запускает web (`http://localhost:3000`), API (`http://localhost:8000`),
Clio (`127.0.0.1:8001`), Prophet HTTP (`127.0.0.1:8002`) и два коллектора.
Адреса `OBSERVATIONS_URL` и `FORECASTS_URL` из таблицы выше обязательны;
планировщики и Intelligence этим скриптом не запускаются. Первое наполнение:

```bash
./scripts/argus clio refresh
./scripts/argus clio aggregate
./scripts/argus prophet generate
```

Для непрерывной обработки запустите каждый процесс в отдельном терминале:

```bash
./scripts/argus clio schedule refresh
./scripts/argus clio schedule aggregate
./scripts/argus prophet worker
./scripts/argus intelligence worker
```

`./scripts/argus` выбирает установленную `.venv` приложения, задаёт рабочий
каталог и загружает корневые `.env`, затем `.env.local`. Зависимости автоматически
не устанавливаются: сначала выполните `uv sync` из setup выше. Справка:
`./scripts/argus --help`. Команда работает и при вызове по абсолютному пути из
другого каталога. Старые `scripts/clio`, `scripts/prophet`, `scripts/intelligence`,
`scripts/db`, `scripts/domain-db` остаются совместимыми алиасами.

Только frontend: `pnpm dev:web`. Только API: `pnpm dev:api`
(Clio и Prophet при этом нужно запустить отдельно).
Остановка процессов — `Ctrl+C`; инфраструктуры —
`./scripts/argus compose down`.
`down --volumes` удаляет локальные данные PostgreSQL и Redis.

## Общие команды

Аргументы доменных CLI одинаковы в обоих окружениях:

| Домен | Локальный префикс | Префикс на сервере |
| --- | --- | --- |
| Clio | `./scripts/argus clio` | `/var/www/bin/argus clio` |
| Prophet | `./scripts/argus prophet` | `/var/www/bin/argus prophet` |
| Intelligence | `./scripts/argus intelligence` | `/var/www/bin/argus intelligence` |

Добавьте к префиксу нужную команду:

| Домен | Команда | Результат |
| --- | --- | --- |
| Clio | `collect solar-wind` / `collect geomagnetic` | Разовый сбор; `--watch` включает цикл |
| Clio | `refresh` | Обновить нормализованные наблюдения и историю входов плотности |
| Clio | `aggregate --limit 240` | Обработать очередь агрегатов |
| Clio | `audit --json` | Проверить агрегаты за последние 7 дней без изменений БД |
| Clio | `cleanup --limit 24` | Показать проверенные raw-часы старше 90 дней; `--apply` удаляет их |
| Prophet | `generate [all\|wind\|kp\|hmf\|density]` | Рассчитать и опубликовать прогнозы; по умолчанию `all` |
| Prophet | `status solar-wind-speed` | Текущий релиз и диагностика входных данных |
| Prophet | `slots --limit 10` / `runs --limit 10` | Последние слоты / запуски |
| Prophet | `show-run <run-uuid>` | Детали запуска; `--inputs` добавляет сохранённые входы |
| Prophet | `export` | Повторить ожидающие CSV-экспорты; `--force` восстановит все текущие CSV |
| Intelligence | `check` | Проверить HTTP-контракт без записи в БД |
| Intelligence | `process` / `status` | Разовая обработка / история обработки |

Например: `./scripts/argus prophet generate wind` локально и
`/var/www/bin/argus prophet generate wind` на сервере. Справка: `<префикс> --help`
или `<префикс> <команда> --help`. Intelligence пока сохраняет stub-результаты;
по умолчанию обрабатывает `solar-wind-speed`.

На проде Compose уже запускает коллекторы и планировщики: Clio refresh в `:00 UTC`,
агрегацию каждые 5 минут, Prophet в `:10 UTC`, Intelligence каждые 60 секунд.
Дополнительный cron не нужен. Ручные refresh/aggregate/generate не отмечают
плановый слот выполненным. Удаление истории автоматически не запускается.

## Прод: состояние и обслуживание

На сервере с установленным релизом:

```bash
export PATH="/var/www/bin:$PATH"
argus compose ps
argus compose logs --tail 100 -f api clio prophet prophet-api intelligence
argus compose logs --tail 100 -f solar-wind geomagnetic clio-refresh clio-aggregate
argus compose exec solar-wind clio check-health solar-wind
argus compose exec geomagnetic clio check-health geomagnetic
argus prophet status solar-wind-speed
argus intelligence status
argus compose restart api
```

Проверки heartbeat выполняются через `exec` внутри работающего коллектора:
одноразовый `argus clio check-health ...` не видит его локальный heartbeat-файл.
`intelligence status` показывает историю БД, а не живость процесса.

Доменные команды `argus` берут блокировку от одновременного деплоя и запускают
одноразовый контейнер без старта зависимостей. PostgreSQL и нужные HTTP-сервисы
должны уже работать. `argus compose` блокировку не берёт: обслуживание выполняйте
отдельно от деплоя. Provisioning и миграции берут эксклюзивную блокировку
от других команд wrapper; работающие сервисы она не останавливает. `restart` не применяет новые env; после их изменения нужен
`argus compose up -d <service>`.

### Разовый перенос общей БД

После настройки доменных `*_DB_*` и сборки через **prepare_only=true**:

```bash
/var/www/bin/argus db transfer
```

Команда останавливает запись, делает резервную копию, переносит и проверяет
данные, затем применяет подготовленный релиз. При сбое повторите её после
устранения причины. Подробности и ограничения — в [domain-storage.md](domain-storage.md).
Это разовая продовая операция; обычный деплой перенос данных не запускает.

### Миграции

Все четыре БД: `./scripts/argus db migrate` локально, `argus db migrate` на проде.
При первой ошибке последовательность прекращается. Обычный деплой применяет
изменившиеся миграции сам. Ручное применение на проде —
после остановки писателей соответствующего домена и резервной копии БД:

| Домен | Локально | На проде |
| --- | --- | --- |
| API | `./scripts/argus api migrate upgrade head` | `argus api migrate upgrade head` |
| Clio | `./scripts/argus clio migrate upgrade head` | `argus clio migrate upgrade head` |
| Prophet | `./scripts/argus prophet migrate upgrade head` | `argus prophet migrate upgrade head` |
| Intelligence | `./scripts/argus intelligence migrate upgrade head` | `argus intelligence migrate upgrade head` |

Локально текущая ревизия API: `./scripts/argus api migrate current`; других доменов — `migrate current`.
Создание миграции API: `./scripts/argus api migrate revision --autogenerate -m "описание"`; откат на одну ревизию:
`./scripts/argus api migrate downgrade -1` (может удалить данные). Первичное provisioning прод-БД описано
в [domain-storage.md](domain-storage.md); приложение и Alembic используют одного владельца своей БД.

### Пользователи dashboard

```bash
# Локально
./scripts/argus api user create admin --group admins
./scripts/argus api user reset-password admin
# На сервере
argus api user create admin --group admins
argus api user reset-password admin
```

Пароль вводится интерактивно (12–256 символов). Сброс отзывает все сессии пользователя.

### Выпуск релиза

Из локального checkout: `./deploy.sh patch -m "Описание релиза"`
(`minor` / `major` для других типов версии). Скрипт загружает модели и метрики,
коммитит **все изменения** в notebooks, приватном backend и основном репозитории,
делает push и создаёт тег `v*`, запускающий GitHub Actions. Перед запуском проверьте
содержимое этих checkout. Параметры четырёх доменных БД и сервисные токены должны
быть настроены на сервере заранее. Существующую общую БД сначала перенесите
по [процедуре перехода](domain-storage.md#перенос-существующей-общей-бд).

При сбое деплоя исправьте причину и повторите применение того же release bundle:
`bash <release>/deploy.sh /var/www`. Писатели могли остаться остановленными;
проверьте `argus compose ps` и логи. Автоматического отката БД нет.
Подробности: [README_DEPLOY.md](../README_DEPLOY.md).

## Локальные проверки

```bash
uv sync --frozen
./scripts/test-python -q
pnpm lint
pnpm check-types
pnpm build
pnpm --filter web test:forecast
pnpm --filter web test:live
```

Общий Python runner требует приватный backend и не включает тесты Intelligence:
`uv run --project apps/intelligence --frozen --with pytest python -m pytest apps/intelligence/tests`.
Интеграционные проверки БД используют отдельную одноразовую БД через
`TEST_DATABASE_ADMIN_DSN`; без неё соответствующие тесты пропускаются.

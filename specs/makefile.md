---
title: Makefile como interfaz única para el flujo Docker
status: completed
created: 2026-08-04
updated: 2026-08-04
issue: #7
---

## Nota de reconciliación (2026-08-04)

Al pasar a implementación se encontró que la rama `diplomado-docker-first-cleanup` (pusheada el 2026-08-01, antes de esta auditoría) ya resolvía este mismo problema — referenciando `cuauhtemocbe/meta-projects#29`, un tracker externo de propagación de estándares — con un Makefile más simple (`help, up, down, shell, jupyter, prod-up, prod-down`, sin `lint`/`test`). Esta implementación se rebasó sobre esa rama en vez de crear una nueva: se conservaron los nombres de target ya pusheados (`jupyter` en vez de `notebook`, `up` ya en modo detached en vez de un par `up`/`up-d`) y se agregaron los que faltaban (`build`, `lint`, `test`, `jupyter-local`, `lint-local`, `test-local`). El resto de esta spec (requirements, arquitectura, testing strategy) sigue aplicando conceptualmente; los nombres de target abajo deben leerse con ese mapeo.

**Update 2026-08-05 — rebase sobre `main`:** al abrir el PR se descubrió que tanto `fix-pylint-ci` (PR #3) como el Makefile original de `diplomado-docker-first-cleanup` (PR #2) ya estaban mergeados a `main` desde el 01/08, pero esta rama seguía basada en el `main` viejo (previo a ambos merges) — por eso el PR quedó `CONFLICTING`. Se rebaseó sobre `origin/main` (en un worktree aislado para no pisar `.claude/`/`CLAUDE.md` locales no versionados); git reconoció el commit original de esta rama como ya aplicado (`skipped previously applied commit e1418b8`) y solo re-aplicó el commit nuevo. Con `main` real de base, `make lint` ya da **10.00/10** — el caveat original (documentado abajo, ya no aplica) quedaba resuelto por este rebase, no por esperar un merge futuro.


# Makefile como interfaz única para el flujo Docker

## Objective

Agregar un `Makefile` autodocumentado en la raíz del repo que envuelva `docker-compose.dev.yml`, de modo que levantar Jupyter Lab, correr lint y correr tests sea un solo comando no interactivo (`make notebook`, `make lint`, `make test`) en vez de la secuencia manual de 3 pasos que hoy documenta el README.

## Context

El `README.md` actual le pide al usuario memorizar y ejecutar a mano:

```bash
docker compose up -d
docker exec -it diplomado-ds bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```

Esto es interactivo por diseño (requiere `-it` + lanzar Jupyter a mano cada sesión) y no escala como interfaz de proyecto. `development-standards.md` (sección 2) pide que un Makefile sea la interfaz única del repo. El patrón de referencia (`agentic-evals/Makefile`) ya resuelve esto: Docker como flujo principal, con `help` autodocumentado como target por defecto.

Dato clave del repo: el servicio `diplomado-ds` en `docker-compose.dev.yml` ya se mantiene vivo (`command: "/bin/bash"` + `tty: true`), así que `docker compose exec diplomado-ds ...` funciona sin cambios al compose — el gap es puramente la falta del wrapper de Makefile.

Nota de corrección respecto al issue original: el borrador de `lint` en el issue proponía `poetry run pylint src`, pero el CI real (`.github/workflows/pylint.yml`) corre `poetry run pylint $(git ls-files '*.py')` sobre todo el repo (así lo documenta también `CLAUDE.md`). Para que `make lint` sea un espejo fiel de CI, debe usar el mismo comando sobre todo el repo, no solo `src`.

## Requirements

### Functional Requirements

- [ ] `make help` (target por defecto, `.DEFAULT_GOAL := help`) lista todos los targets con su descripción, generada por `grep`+`awk` sobre comentarios `##` en cada línea de target.
- [ ] `make build` — `docker compose -f docker-compose.dev.yml build`
- [ ] `make up` — `docker compose -f docker-compose.dev.yml up` (foreground, con logs)
- [ ] `make up-d` — `docker compose -f docker-compose.dev.yml up -d`
- [ ] `make down` — `docker compose -f docker-compose.dev.yml down`
- [ ] `make shell` — `docker compose -f docker-compose.dev.yml exec diplomado-ds bash`
- [ ] `make notebook` — depende de `up-d`, luego `docker compose -f docker-compose.dev.yml exec diplomado-ds jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --notebook-dir=notebooks`
- [ ] `make lint` — depende de `up-d`, luego `docker compose -f docker-compose.dev.yml exec diplomado-ds poetry run pylint $(git ls-files '*.py')` (idéntico al comando de CI, no solo `src`)
- [ ] `make test` — depende de `up-d`, luego `docker compose -f docker-compose.dev.yml exec diplomado-ds poetry run pytest tests -v`
- [ ] Targets `-local` opcionales como fallback sin Docker (Poetry directo en el host): `notebook-local`, `lint-local`, `test-local`.
- [ ] `README.md` actualizado: la sección "Configuración con Jupyter Lab (Opción Sencilla)" se simplifica a `make notebook` como camino principal; los pasos manuales de Docker quedan como detalle de implementación (opcional, en una sección aparte o eliminados).

### Non-Functional Requirements

- [ ] Todos los targets Docker son idempotentes: correr `make notebook` dos veces seguidas no falla (si el contenedor ya está arriba, `up -d` es no-op).
- [ ] `.PHONY` declarado para todos los targets (ninguno debe colisionar con un archivo/carpeta real del repo, p.ej. no hay carpeta `build/`, `test/` o `help/` en la raíz — verificar antes de mergear).

## Architecture

### Components

Un único archivo `Makefile` en la raíz del repo. No requiere cambios a `docker-compose.dev.yml` ni a `pyproject.toml`.

### Data Model

N/A — no hay estado ni esquema.

### External Dependencies

- `docker` / `docker compose` — ya es requisito documentado del repo.
- `make` (GNU Make) — nuevo requisito implícito; agregar a la lista de "Requisitos Previos" del README si no está ya asumido (típicamente preinstalado en Linux/macOS; en Windows requiere WSL o Git Bash, que el repo ya asume vía Docker Desktop).

## User Stories

Como estudiante/colaborador del diplomado, quiero correr `make notebook` y tener Jupyter Lab corriendo en `http://localhost:8889/lab/tree/notebooks` sin tener que recordar tres comandos distintos, para poder empezar a trabajar más rápido y sin errores de tipeo en los flags.

Como mantenedor del repo, quiero que `make lint` corra exactamente lo mismo que corre CI, para no descubrir fallos de lint recién en el pull request.

## Testing Strategy

### Manual verification (no hay test automatizado de un Makefile)

- Correr `make help` y confirmar que lista los 9 targets esperados con descripción.
- Correr `make notebook` desde cero (sin contenedor levantado) y confirmar que Jupyter Lab queda accesible en `http://localhost:8889/lab/tree/notebooks` sin prompt de token.
- Correr `make notebook` una segunda vez (contenedor ya arriba) y confirmar que no falla.
- Correr `make lint` y confirmar que el output es idéntico (mismos archivos analizados) al job de CI.
- Correr `make test` y confirmar que corre `tests/test_hello_world.py` dentro del contenedor.
- Correr `make down` y confirmar que el contenedor se detiene.

## Boundaries & Constraints

### In Scope

- Un `Makefile` en la raíz envolviendo `docker-compose.dev.yml`.
- Targets `-local` opcionales como fallback sin Docker.
- Actualización del README para reflejar `make notebook` como camino principal.

### Out of Scope

- Cambios a `docker-compose.dev.yml`, `docker-compose.prod.yml` o `Dockerfile.dev`.
- Un Makefile para el flujo de producción (`docker-compose.prod.yml` / gunicorn) — fuera del alcance de este issue, es un repo local-only (ver decisión de scope en issues #5/#6, cerrados won't-fix).
- CI (no se agrega ningún workflow nuevo ni se modifica `pylint.yml`).

### Technical Constraints

- Debe funcionar con GNU Make estándar (sin extensiones no portables).
- No debe requerir que el usuario tenga Poetry instalado en el host para los targets Docker (todo corre dentro del contenedor, que ya tiene Poetry).

## Success Criteria

- [ ] `make help` es el comportamiento por defecto de `make` sin argumentos.
- [ ] `make notebook` reemplaza los 3 pasos manuales del README con un solo comando.
- [ ] `make lint` produce el mismo resultado que el job `build (3.12)` de CI.
- [ ] El README ya no le pide al usuario memorizar flags de `docker exec`/`jupyter lab`.
- [ ] Issue #7 se cierra referenciando el PR que agrega el Makefile.

## Implementation Plan

Ver `specs/makefile-plan.md` (a crear en la fase PLAN, tras aprobación de este spec).

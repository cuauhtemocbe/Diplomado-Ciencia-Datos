---
title: Dependabot Socket Firewall workflow
status: in-progress
created: 2026-09-25
updated: 2026-09-25
issue: #61
---

# Dependabot Socket Firewall workflow

## Objective

Reinstalar las dependencias de cada PR de Dependabot a través de Socket Firewall
Free (`sfw`) y cerrar el PR automáticamente si esa instalación falla, para que
los PRs de Dependabot (pip, docker, github-actions) no queden mergeables sin
ninguna revisión automatizada.

## Context

Parte de la US global cuauhtemocbe/meta-projects#47 (extender el workflow a los
repos con Dependabot que no lo tienen). `.github/dependabot.yml` existe
(github-actions, docker, pip) y `.github/workflows/` solo tenía `pylint.yml`.

El repo gestiona dependencias con **Poetry** (`pyproject.toml` + `poetry.lock`)
aunque el ecosistema de Dependabot se llame `pip`. `sfw` no soporta Poetry y
`sfw poetry install` falla con `CERTIFICATE_VERIFY_FAILED` (spike de
DataScience-Docker, PR #38), lo que sería indistinguible de un bloqueo y
cerraría todos los PRs. Se replica el enfoque validado allí:
`poetry export` + `sfw pip install -r`.

## Requirements

### Functional Requirements

- [x] `.github/workflows/dependabot-socket-firewall.yml` con trigger
      `pull_request` a `main` y guarda `github.actor == 'dependabot[bot]'`.
- [x] `poetry export -f requirements.txt --without-hashes --with dev` seguido de
      `sfw pip install -r /tmp/req.txt` (nunca `sfw poetry install`).
- [x] `setup-python` con `3.14`, dentro de `>=3.13,<4.0` y consistente con
      `pylint.yml` y `Dockerfile.dev`.
- [x] Poetry pinneado y capaz de exportar este `poetry.lock` (ver Technical
      Constraints).
- [x] El paso de instalación es `continue-on-error: true`; si falla, `gh pr close`
      con comentario explicativo.

### Non-Functional Requirements

- [x] Security: solo cierra PRs de `dependabot[bot]`; nunca mergea ni toca PRs
      humanos.
- [x] Security: sin secretos nuevos (`SocketDev/action` en `firewall-free` +
      `github.token`).
- [x] Security: actions pinneadas por SHA con comentario de versión, mismos SHAs
      que el workflow de referencia (re-verificados contra los tags upstream el
      2026-09-25: checkout v7.0.1, setup-python v7.0.0, SocketDev/action v1.3.2).
- [x] Permisos a nivel de job: `pull-requests: write`, `contents: read`.
- [x] Reversible: borrar el archivo elimina todo el comportamiento.

## Architecture

### Components

Un único archivo, `.github/workflows/dependabot-socket-firewall.yml`, job
`socket-check`: checkout → setup-python → export del lock → `SocketDev/action`
→ instalación con `sfw` (continue-on-error) → cierre del PR si falló.

### Data Model

N/A.

### External Dependencies

- `actions/checkout`, `actions/setup-python`, `SocketDev/action`: por SHA.
- `poetry==2.5.1` + `poetry-plugin-export==1.10.1` (pip, antes del firewall).
- `sfw` (Socket Firewall Free), sin API key.

## User Stories

```gherkin
Feature: Dependabot PRs never sit mergeable with a malicious dependency

  Scenario: Dependabot PR carries a compromised package
    Given the dependabot-socket-firewall workflow is installed
    When Dependabot opens a PR and Socket Firewall blocks the install
    Then the workflow closes the PR with an explanatory comment

  Scenario: Dependabot PR is clean
    When Dependabot opens a PR and the firewalled install succeeds
    Then the PR stays open and follows the normal review flow

  Scenario: Human-authored PR
    When a human opens a PR touching dependencies
    Then the workflow does not run (`github.actor == 'dependabot[bot]'` guard)
```

## Testing Strategy

### Unit Tests
N/A — workflow YAML, sin código de aplicación.

### Integration Tests
Antes del PR: sintaxis YAML válida y ejecución en CI (rama desechable) del flujo
completo `poetry export` + `sfw pip install` sobre el `poetry.lock` actual, con
Poetry 1.8.4 y 2.5.1. Resultado 2026-09-25: ambos exportan e instalan sin error
(143/144 líneas, incluye grupo dev); 1.8.4 avisa de incompatibilidad del lock.

### E2E Tests
Pendiente: observar el workflow en al menos un PR real de Dependabot sin cierre
incorrecto (no reproducible en el PR de implementación: el actor es humano y el
job se omite por la guarda).

## Boundaries & Constraints

### In Scope
- El archivo de workflow y su spec/plan.

### Out of Scope
- Corregir la etiqueta `package-ecosystem: pip` de `dependabot.yml`.
- Distinguir bloqueos reales de otros fallos de instalación (ver Riesgo aceptado).
- Exportar los grupos opcionales (`tensorflow`, `nlp`, `geo`, `bio`, `explain`).

### Technical Constraints
- Poetry 2.5.1 + `poetry-plugin-export`, no 1.8.x: `poetry.lock` es
  `lock-version 2.1` generado por Poetry 2.5.1. Poetry 1.8.4 lo exporta pero
  avisa "The lock file might not be compatible", su export difiere (extras
  `bleach[css]`, `jsonschema[format-nongpl]`, orden de marcadores) y en Python
  3.14 compila `dulwich` desde fuente.
- `--with dev` obligatorio: `export` omite el grupo dev por defecto.
- Los grupos opcionales quedan sin revisar: tensorflow no tiene wheel cp314 (el
  repo lo construye sobre 3.13) y torch viene de un índice aparte; exportarlos
  cerraría PRs por fallos de instalación ajenos al firewall. Consecuencia: un PR
  que solo toca dependencias de esos grupos (p. ej. gunicorn, shap) pasa sin que
  `sfw` vea el paquete.
- Riesgo aceptado (heredado del diseño de referencia): el PR se cierra ante
  cualquier fallo del paso de instalación (red, PyPI caído, wheel que no
  compila), no solo ante bloqueos del firewall; el comentario del cierre lo
  advierte y el PR se puede reabrir. Un fallo de `poetry export` no cierra el PR.

## Success Criteria

- [ ] Workflow mergeado a `main` vía rama + PR.
- [x] Enfoque Poetry confirmado funcionando antes del merge (CI en rama
      desechable, 2026-09-25).
- [ ] Workflow observado en al menos un PR real de Dependabot sin cerrarlo
      incorrectamente.

## Implementation Plan

Ver `dependabot-socket-firewall-plan.md`.

## Changelog

<!-- Vacío hasta que el spec llegue a `completed` y se vuelva a tocar. -->

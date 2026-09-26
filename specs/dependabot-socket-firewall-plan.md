# Implementation Plan: Dependabot Socket Firewall workflow

**Spec**: [dependabot-socket-firewall.md](./dependabot-socket-firewall.md)
**Issue**: #61 (parte de cuauhtemocbe/meta-projects#47)
**Created**: 2026-09-25
**Status**: in-progress

## Components

### 1. Workflow `dependabot-socket-firewall.yml`
- **Purpose**: gatear PRs de Dependabot con `sfw` y cerrarlos si la instalación falla.
- **Files**: `.github/workflows/dependabot-socket-firewall.yml`
- **Effort**: S

### 2. Spec y plan
- **Purpose**: registrar decisiones (versión de Poetry, alcance de grupos, riesgo aceptado).
- **Files**: `specs/dependabot-socket-firewall.md`, `specs/dependabot-socket-firewall-plan.md`
- **Effort**: XS

## Dependencies

### Build Order
1. Verificar que la versión de Poetry lee/exporta el lock (CI, rama desechable).
2. Workflow (depende de 1 para elegir la versión).
3. Spec/plan (documentan la decisión).

### External Dependencies
- Enfoque de referencia: DataScience-Docker PR #38 (mismos SHAs de actions).

## Risks & Assumptions

### Risks
- **Cierre por cualquier fallo de instalación**: aceptado, documentado en spec y
  en el comentario del cierre.
- **Grupos opcionales sin revisar**: aceptado; seguimiento posible exportando
  `nlp`/`geo`/`bio`/`explain` una vez comprobado que instalan bajo 3.14 con `sfw`.
- **Lock 2.1 vs Poetry 1.8.x**: mitigado usando Poetry 2.5.1 + plugin.

### Assumptions
- Dependabot no actualiza las versiones de Poetry/plugin pinneadas en el `run`.
- El comportamiento en un PR real de Dependabot solo se puede observar tras el merge.

## Milestones

- [x] Export + `sfw pip install` validados en CI sobre el lock actual (Poetry 1.8.4 y 2.5.1).
- [ ] PR mergeado a `main`.
- [ ] Workflow observado en un PR real de Dependabot.

## Tasks

**Slicing strategy**: Vertical — un único slice (un archivo de workflow), sin dependencia bloqueante que justifique capas.

### Slice 1: Workflow de firewall para Dependabot
- [x] **Task 1**: validar Poetry export contra `poetry.lock` en CI.
  - **Acceptance**: export incluye grupo dev; `sfw pip install` termina sin error.
  - **Files**: ninguno (rama desechable, eliminada).
  - **Tests**: ejecución en GitHub Actions.
  - **Effort**: S
- [x] **Task 2**: crear el workflow.
  - **Acceptance**: YAML válido; SHAs idénticos al de referencia; guarda de actor, permisos y cierre presentes.
  - **Files**: `.github/workflows/dependabot-socket-firewall.yml`
  - **Tests**: parseo YAML; checks del PR (el job se omite por la guarda).
  - **Effort**: XS
- [ ] **Task 3**: observar el workflow en un PR real de Dependabot.
  - **Acceptance**: corre y no cierra el PR incorrectamente.
  - **Files**: ninguno.
  - **Tests**: manual, tras el merge.
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: ~0.25 día.

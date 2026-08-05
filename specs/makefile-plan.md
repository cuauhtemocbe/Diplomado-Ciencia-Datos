# Implementation Plan: Makefile como interfaz única para el flujo Docker

**Spec**: `specs/makefile.md`
**Created**: 2026-08-04
**Status**: draft

## Components

### 1. `Makefile` (raíz del repo)
- **Purpose**: Interfaz única para build/up/down/shell/notebook/lint/test sobre `docker-compose.dev.yml`, más targets `-local` de fallback sin Docker. `help` autodocumentado como `.DEFAULT_GOAL`.
- **Files**: `Makefile` (nuevo)
- **Effort**: S

### 2. Actualización de `README.md`
- **Purpose**: Reemplazar la sección "Configuración con Jupyter Lab (Opción Sencilla)" (3 pasos manuales) por `make notebook` como camino principal. Agregar `make` a "Requisitos Previos".
- **Files**: `README.md`
- **Effort**: XS

## Dependencies

### Build Order
1. `Makefile` (independiente, no depende de nada más)
2. README (depende del Makefile ya existente, para poder referenciar los targets reales)

No hay dependencias externas nuevas — Docker y `make` ya son (o pasan a ser) requisitos documentados.

## Risks & Assumptions

### Risks
- **Colisión de nombres con `.PHONY`**: si algún target (`build`, `test`, `help`) coincidiera con un archivo/carpeta real en la raíz, `make` no lo ejecutaría por default. Mitigación: verificar `ls` de la raíz antes de escribir el Makefile y declarar `.PHONY` explícito en todos los targets de todos modos (defensa en profundidad).
- **`make lint` diverge de CI silenciosamente en el futuro**: si alguien cambia `pylint.yml` sin actualizar el Makefile, dejan de estar en sync. Mitigación: no hay gate automático posible sin over-engineering para este repo; se acepta el riesgo y se deja documentado en la spec (comentario en el Makefile apuntando a `pylint.yml` como fuente de verdad).
- **Jupyter sin token en `0.0.0.0`**: el target `notebook` mapea `--ServerApp.token=''` (igual que el README actual), lo cual es aceptable solo porque el puerto se publica en localhost (`8889:8888`) y es un entorno de desarrollo local, no expuesto. No es un cambio de postura de seguridad respecto al estado actual — solo se documenta como asunción heredada, no se re-decide acá.

### Assumptions
- El contenedor `diplomado-ds` sigue vivo entre comandos (`command: "/bin/bash"` + `tty: true` en el compose) — confirmado leyendo `docker-compose.dev.yml`, no requiere validación adicional.
- `poetry` está disponible dentro de la imagen `diplomado-ds:latest` (usado hoy vía `poetry run pytest`/`pylint` en CLAUDE.md) — asumido correcto, se valida en Milestone 2 al correr `make lint`/`make test` de verdad.

## Milestones

- [ ] **Milestone 1**: `make help` corre y lista los 9 targets con descripción correcta.
- [ ] **Milestone 2**: `make notebook` desde cero deja Jupyter Lab accesible en `http://localhost:8889/lab/tree/notebooks` sin prompt de token, y es idempotente (correrlo dos veces no falla).
- [ ] **Milestone 3**: `make lint` y `make test` corren dentro del contenedor y producen el mismo resultado que correrlos manualmente hoy (`make lint` en particular debe coincidir con el output de CI).
- [ ] **Milestone 4**: README actualizado, `make down` limpia el entorno, y el issue #7 queda listo para cerrarse referenciando el PR.

## Tasks

### Foundation (Build First)
- [ ] **Task 1**: Crear el `Makefile` con targets Docker (`help`, `build`, `up`, `up-d`, `down`, `shell`, `notebook`, `lint`, `test`)
  - **Acceptance**: Los 9 targets existen, `.PHONY` declarado, `help` es `.DEFAULT_GOAL` y lista todos los targets vía `grep`+`awk` sobre comentarios `##`. `make notebook`/`lint`/`test` dependen de `up-d` (target prerequisito de Make, no llamada recursiva).
  - **Files**: `Makefile`
  - **Tests**: Manual — `make help`, `make notebook` (desde cero y repetido), `make lint`, `make test`, `make down`, según Testing Strategy de la spec.
  - **Effort**: S

### Features (Build Second)
- [ ] **Task 2**: Agregar targets `-local` de fallback sin Docker (`notebook-local`, `lint-local`, `test-local`)
  - **Acceptance**: Cada target corre el equivalente directo con `poetry run` en el host (sin Docker), documentado en `help` igual que los targets Docker.
  - **Files**: `Makefile`
  - **Tests**: Manual — correr cada `-local` target en un entorno con Poetry instalado (o documentar como no verificado si no hay entorno Poetry local disponible en esta sesión).
  - **Effort**: XS

### Integration (Build Third)
- [ ] **Task 3**: Actualizar `README.md`
  - **Acceptance**: La sección "Configuración con Jupyter Lab (Opción Sencilla)" queda reemplazada por `make notebook` como camino principal (con el link a `http://localhost:8889/lab/tree/notebooks`); `make` agregado a "Requisitos Previos"; los pasos Docker manuales quedan solo como referencia opcional o se eliminan.
  - **Files**: `README.md`
  - **Tests**: Revisión manual de que el README sigue siendo coherente de punta a punta (no quedan referencias rotas a pasos eliminados).
  - **Effort**: XS

### Polish
- [ ] **Task 4**: Verificación end-to-end + cierre de issue
  - **Acceptance**: Los 4 milestones arriba confirmados corriendo los comandos reales; PR abierto referenciando `Closes #7`; spec actualizada a `status: completed` tras merge.
  - **Files**: N/A (verificación), `specs/makefile.md` (status update)
  - **Tests**: Los del Testing Strategy de la spec, ejecutados de punta a punta.
  - **Effort**: XS

## Effort Estimate

**Total Estimated**: ~1 sesión de trabajo (S + XS + XS + XS)

| Phase | Effort |
|-------|--------|
| Foundation (Task 1) | S |
| Features (Task 2) | XS |
| Integration (Task 3) | XS |
| Polish (Task 4) | XS |

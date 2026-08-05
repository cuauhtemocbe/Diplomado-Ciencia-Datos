.DEFAULT_GOAL := help

COMPOSE := docker compose -f docker-compose.dev.yml
SERVICE := diplomado-core
JUPYTER_CMD := jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

.PHONY: help build up down shell jupyter lint test \
	verify-build verify-groups verify-dockerfile verify-compose \
	jupyter-local lint-local test-local \
	build-core up-core down-core shell-core jupyter-core \
	build-tensorflow up-tensorflow down-tensorflow shell-tensorflow jupyter-tensorflow \
	build-nlp up-nlp down-nlp shell-nlp jupyter-nlp \
	build-geo up-geo down-geo shell-geo jupyter-geo \
	build-bio up-bio down-bio shell-bio jupyter-bio \
	build-explain up-explain down-explain shell-explain jupyter-explain

help: ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# Atajos para el grupo core (el que casi todos los notebooks necesitan).
# Equivalentes a `make <target>-core`.
build: build-core ## Construir la imagen de desarrollo (alias de build-core)

up: up-core ## Levantar el contenedor de desarrollo (alias de up-core)

down: down-core ## Detener el contenedor de desarrollo (alias de down-core)

shell: shell-core ## Abrir una shell dentro del contenedor de desarrollo (alias de shell-core)

jupyter: jupyter-core ## Iniciar Jupyter Lab dentro del contenedor (alias de jupyter-core)

# Debe coincidir con .github/workflows/pylint.yml — si cambia ahí, cambiar acá también.
lint: up-core ## Correr pylint dentro del contenedor (mismo comando que CI)
	$(COMPOSE) exec $(SERVICE) poetry run pylint $$(git ls-files '*.py')

test: up-core ## Correr pytest dentro del contenedor
	$(COMPOSE) exec $(SERVICE) poetry run pytest tests -v

verify-build: ## Validar contexto de build, cache de poetry install y ausencia de .env en la imagen (issue #12)
	./scripts/verify-build-context.sh

verify-groups: ## Validar que cada grupo de Poetry instala e importa lo que necesita, aislado (issue #13)
	./scripts/verify-poetry-groups.sh

verify-dockerfile: ## Validar los targets multi-stage de Dockerfile.dev: tamaño e imports (issue #14)
	./scripts/verify-dockerfile-targets.sh

verify-compose: ## Validar que cada servicio de docker-compose construye e inicia (issue #15)
	./scripts/verify-compose-services.sh

jupyter-local: ## Iniciar Jupyter Lab con Poetry, sin Docker
	poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''

lint-local: ## Correr pylint con Poetry, sin Docker
	poetry run pylint $$(git ls-files '*.py')

test-local: ## Correr pytest con Poetry, sin Docker
	poetry run pytest tests -v

# --- core (puerto 8889) -------------------------------------------------
build-core: ## Construir la imagen del grupo core
	$(COMPOSE) build diplomado-core

up-core: ## Levantar el contenedor core (Jupyter/Poetry)
	$(COMPOSE) up -d diplomado-core

down-core: ## Detener el contenedor core
	$(COMPOSE) stop diplomado-core

shell-core: up-core ## Abrir una shell en el contenedor core (construye si hace falta)
	$(COMPOSE) exec diplomado-core bash

jupyter-core: up-core ## Iniciar Jupyter Lab en el contenedor core (http://localhost:8889/lab/tree/notebooks)
	$(COMPOSE) exec diplomado-core $(JUPYTER_CMD)

# --- tensorflow (puerto 8890) --------------------------------------------
build-tensorflow: ## Construir la imagen del grupo tensorflow
	$(COMPOSE) build diplomado-tensorflow

up-tensorflow: ## Levantar el contenedor tensorflow
	$(COMPOSE) up -d diplomado-tensorflow

down-tensorflow: ## Detener el contenedor tensorflow
	$(COMPOSE) stop diplomado-tensorflow

shell-tensorflow: up-tensorflow ## Abrir una shell en el contenedor tensorflow (construye si hace falta)
	$(COMPOSE) exec diplomado-tensorflow bash

jupyter-tensorflow: up-tensorflow ## Iniciar Jupyter Lab en el contenedor tensorflow (http://localhost:8890/lab/tree/notebooks)
	$(COMPOSE) exec diplomado-tensorflow $(JUPYTER_CMD)

# --- nlp (puerto 8891) ----------------------------------------------------
build-nlp: ## Construir la imagen del grupo nlp
	$(COMPOSE) build diplomado-nlp

up-nlp: ## Levantar el contenedor nlp
	$(COMPOSE) up -d diplomado-nlp

down-nlp: ## Detener el contenedor nlp
	$(COMPOSE) stop diplomado-nlp

shell-nlp: up-nlp ## Abrir una shell en el contenedor nlp (construye si hace falta)
	$(COMPOSE) exec diplomado-nlp bash

jupyter-nlp: up-nlp ## Iniciar Jupyter Lab en el contenedor nlp (http://localhost:8891/lab/tree/notebooks)
	$(COMPOSE) exec diplomado-nlp $(JUPYTER_CMD)

# --- geo (puerto 8892) -----------------------------------------------------
build-geo: ## Construir la imagen del grupo geo
	$(COMPOSE) build diplomado-geo

up-geo: ## Levantar el contenedor geo
	$(COMPOSE) up -d diplomado-geo

down-geo: ## Detener el contenedor geo
	$(COMPOSE) stop diplomado-geo

shell-geo: up-geo ## Abrir una shell en el contenedor geo (construye si hace falta)
	$(COMPOSE) exec diplomado-geo bash

jupyter-geo: up-geo ## Iniciar Jupyter Lab en el contenedor geo (http://localhost:8892/lab/tree/notebooks)
	$(COMPOSE) exec diplomado-geo $(JUPYTER_CMD)

# --- bio (puerto 8893) ------------------------------------------------------
build-bio: ## Construir la imagen del grupo bio
	$(COMPOSE) build diplomado-bio

up-bio: ## Levantar el contenedor bio
	$(COMPOSE) up -d diplomado-bio

down-bio: ## Detener el contenedor bio
	$(COMPOSE) stop diplomado-bio

shell-bio: up-bio ## Abrir una shell en el contenedor bio (construye si hace falta)
	$(COMPOSE) exec diplomado-bio bash

jupyter-bio: up-bio ## Iniciar Jupyter Lab en el contenedor bio (http://localhost:8893/lab/tree/notebooks)
	$(COMPOSE) exec diplomado-bio $(JUPYTER_CMD)

# --- explain (puerto 8894) ---------------------------------------------------
build-explain: ## Construir la imagen del grupo explain
	$(COMPOSE) build diplomado-explain

up-explain: ## Levantar el contenedor explain
	$(COMPOSE) up -d diplomado-explain

down-explain: ## Detener el contenedor explain
	$(COMPOSE) stop diplomado-explain

shell-explain: up-explain ## Abrir una shell en el contenedor explain (construye si hace falta)
	$(COMPOSE) exec diplomado-explain bash

jupyter-explain: up-explain ## Iniciar Jupyter Lab en el contenedor explain (http://localhost:8894/lab/tree/notebooks)
	$(COMPOSE) exec diplomado-explain $(JUPYTER_CMD)

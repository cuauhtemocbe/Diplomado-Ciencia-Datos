.DEFAULT_GOAL := help

COMPOSE := docker compose -f docker-compose.dev.yml
SERVICE := diplomado-ds

.PHONY: help build up down shell jupyter lint test verify-build verify-groups verify-dockerfile jupyter-local lint-local test-local

help: ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## Construir la imagen de desarrollo
	$(COMPOSE) build

up: ## Levantar el contenedor de desarrollo (Jupyter/Poetry)
	$(COMPOSE) up -d

down: ## Detener el contenedor de desarrollo
	$(COMPOSE) down

shell: up ## Abrir una shell dentro del contenedor de desarrollo
	$(COMPOSE) exec $(SERVICE) bash

jupyter: up ## Iniciar Jupyter Lab dentro del contenedor (http://localhost:8889/lab/tree/notebooks)
	$(COMPOSE) exec $(SERVICE) jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

# Debe coincidir con .github/workflows/pylint.yml — si cambia ahí, cambiar acá también.
lint: up ## Correr pylint dentro del contenedor (mismo comando que CI)
	$(COMPOSE) exec $(SERVICE) poetry run pylint $$(git ls-files '*.py')

test: up ## Correr pytest dentro del contenedor
	$(COMPOSE) exec $(SERVICE) poetry run pytest tests -v

verify-build: ## Validar contexto de build, cache de poetry install y ausencia de .env en la imagen (issue #12)
	./scripts/verify-build-context.sh

verify-groups: ## Validar que cada grupo de Poetry instala e importa lo que necesita, aislado (issue #13)
	./scripts/verify-poetry-groups.sh

verify-dockerfile: ## Validar los targets multi-stage de Dockerfile.dev: tamaño e imports (issue #14)
	./scripts/verify-dockerfile-targets.sh

jupyter-local: ## Iniciar Jupyter Lab con Poetry, sin Docker
	poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''

lint-local: ## Correr pylint con Poetry, sin Docker
	poetry run pylint $$(git ls-files '*.py')

test-local: ## Correr pytest con Poetry, sin Docker
	poetry run pytest tests -v

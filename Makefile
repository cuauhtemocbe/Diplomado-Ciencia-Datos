.DEFAULT_GOAL := help

help: ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Levantar el contenedor de desarrollo (Jupyter/Poetry)
	docker compose -f docker-compose.dev.yml up -d

down: ## Detener el contenedor de desarrollo
	docker compose -f docker-compose.dev.yml down

shell: ## Abrir una shell dentro del contenedor de desarrollo
	docker compose -f docker-compose.dev.yml exec diplomado-ds bash

jupyter: up ## Iniciar Jupyter Lab dentro del contenedor (http://localhost:8889/lab/tree/notebooks)
	docker compose -f docker-compose.dev.yml exec diplomado-ds jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

prod-up: ## Levantar el servicio de producción (api-clustering) localmente
	docker compose -f docker-compose.prod.yml up -d

prod-down: ## Detener el servicio de producción local
	docker compose -f docker-compose.prod.yml down

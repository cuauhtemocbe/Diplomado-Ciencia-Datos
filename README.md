## Acerca de este Repositorio

Espacio para subir las actividades realizadas durante el diplomado de Ciencia de Datos 2024-2025. 

Este repositorio utiliza Docker para configurar un entorno de Python 3.13 orientado a **Ciencia de Datos** con Jupyter, facilitando la gestión de bibliotecas mediante Poetry.

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalados los siguientes programas:

1. **Docker**: [Guía de instalación de Docker](https://docs.docker.com/engine/install/)
2. **Git**: [Guía de instalación de Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
3. **Make**: viene preinstalado en Linux/macOS; en Windows usá WSL o Git Bash.
4. **Visual Studio Code (VSC)**: [Descargar Visual Studio Code](https://code.visualstudio.com/download)

La instalación de Visual Studio Code es opcional, pero se recomienda especialmente si tienes experiencia programando. Si eres principiante, puedes optar por no instalarlo.

## Instrucciones de Instalación

### Clonar el Repositorio

1. Elige una ubicación en tu computadora para clonar el repositorio. Abre tu terminal y ejecuta el siguiente comando:

    ```bash
    git clone https://github.com/cuauhtemocbe/Diplomado-Ciencia-Datos.git
    ```
    Este comando creará una carpeta llamada **Diplomado-Ciencia-Datos** en tu máquina.

2. ⚠️**Importante:⚠️** Crear un archivo `.env` en la raíz del repositorio (junto a este README). Solo lo necesitan los notebooks/módulos que usan la YouTube API (variable `youtube_api_key`) — puede quedar vacío para el resto.

### Levantar el entorno con Docker (`make`)

Todo el flujo corre en Docker, sin depender de un editor específico. `make help` lista todos los comandos disponibles.

1. Desde la terminal, dentro de la carpeta **Diplomado-Ciencia-Datos**, levanta el contenedor de desarrollo:

    ```bash
    make up
    ```

2. Inicia Jupyter Lab dentro del contenedor:

    ```bash
    make jupyter
    ```

3. Abre el siguiente enlace en tu navegador: [http://localhost:8889/lab/tree/notebooks](http://localhost:8889/lab/tree/notebooks)

4. Navega en el explorador a la carpeta notebooks, y abre el notebook `0-Hello-Pandas.ipynb`.

5. Disfruta. Cuando termines, `make down` detiene el contenedor.

Si prefieres usar Visual Studio Code para editar los notebooks, ábrelo directamente sobre esta carpeta (`File > Open Folder`) — no requiere ninguna extensión ni configuración especial; sigue corriendo Jupyter vía `make jupyter` como arriba.

#### Un contenedor por grupo de dependencias

`make up`/`make jupyter`/`make shell` (sin sufijo) siempre apuntan al grupo **core** — el que cubre casi todos los notebooks (0 a 5, 7, 8, 12, 14, 15). Si el notebook que quieres correr necesita un grupo más pesado, usa el target con el sufijo del grupo; la imagen se construye sola la primera vez:

| Grupo | Notebooks | Comando | Puerto |
|---|---|---|---|
| `core` | La mayoría | `make jupyter-core` (= `make jupyter`) | 8889 |
| `tensorflow` | 16-Manuscrita, Clasificación_Pokémons(_Base), Predicción_precios_casas_CNN | `make jupyter-tensorflow` | 8890 |
| `nlp` | 13-Agrupamiento-texto, `src/app_clustering` | `make jupyter-nlp` | 8891 |
| `geo` | 11-Índice-de-marginalidad | `make jupyter-geo` | 8892 |
| `bio` | 9-Electrocardiograma | `make jupyter-bio` | 8893 |
| `explain` | 10-Proyecto-Hipertension-Mexico, 6-Whatsapp | `make jupyter-explain` | 8894 |

Cada grupo también tiene `build-<grupo>`, `up-<grupo>`, `down-<grupo>` y `shell-<grupo>` (por ejemplo `make shell-geo`). Puedes tener varios contenedores corriendo a la vez — cada uno en su propio puerto — así que no hace falta bajar uno para levantar otro.

Para contribuir código a `src/`, `make lint` y `make test` corren pylint/pytest dentro del contenedor **core** con el mismo comando que usa CI, así no hay diferencia entre "pasa en mi máquina" y "pasa en CI". `make test` no instala el grupo `nlp`, así que los tests de `app_clustering` (que necesitan flask) se reportan como *skipped* en vez de fallar por import; usa `make test-nlp` para correrlos de verdad dentro del contenedor **nlp**. CI también corre `poetry run black --check` y `poetry run isort --check-only` sobre los mismos archivos que pylint — usa `make lint` (o `poetry run black .` / `poetry run isort .` dentro del contenedor) para formatear antes de subir.

### Hook de pre-push con Trivy (activación única)

El repositorio incluye un hook de `pre-push` en `.githooks/` que corre [Trivy](https://trivy.dev/) sobre las dependencias (`poetry.lock`) y bloquea el `git push` si encuentra una vulnerabilidad **CRITICAL** con corrección disponible. La activación es manual y se hace una sola vez por clone — no ocurre automáticamente al clonar:

```bash
./scripts/install-hooks.sh
```

Este script configura `core.hooksPath` a `.githooks`. A diferencia del resto del flujo, el hook corre en tu máquina (no dentro de Docker), así que necesitas tener `trivy` instalado y en el `PATH`; si no lo encuentra, el push se bloquea con un mensaje que apunta a las instrucciones de instalación en [`.claude/skills/trivy-scan/setup.md`](.claude/skills/trivy-scan/setup.md).

## Enlaces de Interés

- **Poetry**: [Sitio oficial de Poetry](https://python-poetry.org/)

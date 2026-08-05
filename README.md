## Acerca de este Repositorio

Espacio para subir las actividades realizadas durante el diplomado de Ciencia de Datos 2024-2025. 

Este repositorio utiliza Docker para configurar un entorno de Python 3.12.3 orientado a **Ciencia de Datos** con Jupyter, facilitando la gestión de bibliotecas mediante Poetry.

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

Cada grupo también tiene `build-<grupo>`, `up-<grupo>`, `down-<grupo>` y `shell-<grupo>` (por ejemplo `make shell-geo`). Puedes tener varios contenedores corriendo a la vez — cada uno en su propio puerto — así que no hace falta bajar uno para levantar otro. `tensorflow`/`nlp` están definidos pero hoy no se pueden construir (falla la instalación de `torch`); pendiente de una refactorización de esos notebooks.

Para contribuir código a `src/`, `make lint` y `make test` corren pylint/pytest dentro del contenedor con el mismo comando que usa CI, así no hay diferencia entre "pasa en mi máquina" y "pasa en CI".

## Enlaces de Interés

- **Poetry**: [Sitio oficial de Poetry](https://python-poetry.org/)

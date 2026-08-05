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

Para contribuir código a `src/`, `make lint` y `make test` corren pylint/pytest dentro del contenedor con el mismo comando que usa CI, así no hay diferencia entre "pasa en mi máquina" y "pasa en CI".

## Enlaces de Interés

- **Poetry**: [Sitio oficial de Poetry](https://python-poetry.org/)

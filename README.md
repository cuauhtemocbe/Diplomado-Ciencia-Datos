## Acerca de este Repositorio

Espacio para subir las actividades realizadas durante el diplomado de Ciencia de Datos 2024-2025. 

Este repositorio utiliza Docker para configurar un entorno de Python 3.10.11 orientado a **Ciencia de Datos** con Jupyter, facilitando la gestión de bibliotecas mediante Poetry.

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalados los siguientes programas:

1. **Docker**: [Guía de instalación de Docker](https://docs.docker.com/engine/install/)
2. **Git**: [Guía de instalación de Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
3. **Visual Studio Code (VSC)**: [Descargar Visual Studio Code](https://code.visualstudio.com/download)

La instalación de Visual Studio Code es opcional, pero se recomienda especialmente si tienes experiencia programando. Si eres principiante, puedes optar por no instalarlo.

## Instrucciones de Instalación

### Clonar el Repositorio

1. Elige una ubicación en tu computadora para clonar el repositorio. Abre tu terminal y ejecuta el siguiente comando:

    ```bash
    git clone https://github.com/cuauhtemocbe/Diplomado-Ciencia-Datos.git
    ```
    Este comando creará una carpeta llamada **Diplomado-Ciencia-Datos** en tu máquina.

2. ⚠️**Importante :⚠️** Crear un archivo `.env` dentro de la carpeta **DataScience-Docker**

### Configuración con Visual Studio Code (Opción Avanzada)

Si prefieres usar Visual Studio Code para desarrollar o ejecutar los notebooks, sigue estos pasos:

1. Abre Visual Studio Code y selecciona `File > Open Folder`. Luego elige la carpeta **Diplomado-Ciencia-Datos** para abrir el repositorio.
2. Instala la extensión **Dev Containers** desde el Marketplace de VSC.
3. Abre la Paleta de Comandos (Command Palette) con `Shift + Ctrl + P` y escribe `Dev Containers: Rebuild and Reopen in Container`. Ejecútalo para construir y levantar el contenedor Docker.
4. En el explorador de archivos (`Ctrl + Shift + E`), navega hasta la carpeta `notebooks` y abre el archivo `0-Hello-Pandas.ipynb`.
5. Ejecuta la primer celda; al inicio te solicitará seleccionar un kernel, elige Python 3.10.11.
6. Disfruta

### Configuración con Jupyter Lab (Opción Sencilla)

Si prefieres utilizar Jupyter Lab con Docker, sigue estos pasos:

1. Desde la terminal, dentro de la carpeta **Diplomado-Ciencia-Datos**, ejecuta el siguiente comando para construir y levantar el contenedor:

    ```bash
    docker compose up -d
    ```

2. Luego, ejecuta el siguiente comando para ingresar al contenedor y utilizar la terminal:

    ```bash
    docker exec -it diplomado-ciencia-datos bash
    ```

3. Dentro del contenedor, inicia el servicio de Jupyter Lab con el siguiente comando:

    ```bash
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
    ```

4. Abre el siguiente enlace en tu navegador: [http://localhost:8888/lab/tree/notebooks](http://localhost:8888/lab/tree/notebooks)

5. Navega en el explorador a la carpeta notebooks, y abre prueba el notebook `0-Hello-Pandas.ipynb`.

6. Disfruta.

## Enlaces de Interés

- **Poetry**: [Sitio oficial de Poetry](https://python-poetry.org/)

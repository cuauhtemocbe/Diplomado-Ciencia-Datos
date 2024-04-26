# Diplomado Ciencia de Datos

Espacio para subir las actividades realizadas durante el diplomado de Ciencia de Datos 2024-2025. 

## Sobre el repositorio
El repo usa Docker para levantar un ambiente de python 3.10.11, así como jupyter, y gestionar fácilmente las bibliotecas con el uso de poetry.

## Requisitos

1. Docker https://docs.docker.com/engine/install/
2. Git https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
3. Visual Studio Code: https://code.visualstudio.com/download

La instalación de Visual Studio Code (VSC) es opcional, pero te lo recomiendo ampliamente usarlo, sobre todo si tienes experiencia programando. Si tienes poca experiencia programando, evitalo. 

## Instalación

### Clona el repo
1. Elige una carpeta en tu máquina local, donde quieras clonar el repo. Abre tu terminal y usa el siguiente comando: 

`git clone https://github.com/cuauhtemocbe/Diplomado-Ciencia-Datos.git`

El comando creará una carpeta en tu computadora llamada **Diplomado-Ciencia-Datos**.

### Usando VSC (Opción avanzada)
Si quieres desarrollar o ejecutar los notebooks en VSC, sique los siguientes pasos:
1. Abre VSC, y busca en la cintilla de opciones `File > Open Folder` y selecciona la carpeta **Diplomado-Ciencia-Datos** para abrir el repositorio.
2. Instala la extensión **Dev Containers** en VSC.
3. Abre la Paleta de Comandos (Command Palette) en VSC con el siguiente atajo `(Shif + Ctrl + P)`. Una vez abierto escribe el siguiente comando: `Dev Containers: Rebuild and Reopen in Container`, el cual creara y levantará el contendor de Docker.
4. En el explorador (Ctrl + Shift + E) navega a la carpeta notebooks y abre el notebook `0. Validar-ambiente.ipynb`.
5. Corre el notebook, al inicio te pedira que selecciones un kernel, selecciona python 3.10.11

### Usando Jupyter Lab (Opción sencilla)
Aquí usaremos Docker para crear nuestro ambiente y levantar jupyter lab.
1. Dentro de la carpeta **Diplomado-Ciencia-Datos**, abre una terminal, y escribe el siguiente comando `docker compose up -d` para que se construir y levantar el contenedor.
2. Después escribe `docker exec -it diplomado-ciencia-datos bash` para entrar dentro del contenedor y usar la terminal.
3. Ejecuta el siguiente comando, para levantar el servicio de jupyer:
`jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''`

4. Finalmente abre el siguiente link en tu navegador http://localhost:8888/lab/tree/notebooks


   

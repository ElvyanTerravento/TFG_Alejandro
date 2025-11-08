Esta es la aplicación para el TFG de Alejandro Leyva Nieto. Esta aplicación implementa un lector de lenguaje de signos
mediante diversos modelos, entre los cuales podemos elegir cual utilizar.

Para descargar correctamente TODOS los modelos, es necesario utilizar "git lfs", una extensión para poder mover archivos
de mayor tamaño. De este modo, para poder descargar los modelos más grandes debemos seguir estos pasos:

1. Instalamos LFS en la carpeta donde queramos este proyecto:
  - git lfs install

2. Clonamos el repositorio completo
  - git clone https://github.com/Alexleynieto/TFG_Alejandro

3. Cambiamos a la carpeta correspondiente al proyecto y traemos archivos con LFS.
  - cd TFG_Alejandro
  - git lfs fetch --all
  - git lfs checkout

A continuación ya podemos ejecutar este proyecto, ejecutando el siguiente comando:
  - ant clean compile run

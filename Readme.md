Proyecto: Asistente Virtual con IA

Este proyecto consiste en un asistente virtual basado en Inteligencia Artificial que utiliza FastAPI para el backend y Nuxt para el frontend. El asistente emplea transformers y SentenceTransformers para el procesamiento de lenguaje natural.

🚀 Requisitos Previos

1️⃣ Instalar Dependencias Globales

Asegúrate de tener instalados:

Python 3.8+

Node.js y npm

Git

⚙️ Configuración del Backend (FastAPI)

1️⃣ Clonar el repositorio y acceder al directorio

# Clonar el repositorio
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio/backend

2️⃣ Crear y activar un entorno virtual

# En Linux/Mac
python -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate

3️⃣ Instalar dependencias

pip install -r requirements.txt

4️⃣ Ejecutar el servidor FastAPI

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

El backend estará disponible en: http://localhost:8000

Puedes ver la documentación automática de la API en:

Swagger UI: http://localhost:8000/docs

Redoc UI: http://localhost:8000/redoc

🎨 Configuración del Frontend (Nuxt)

1️⃣ Ir al directorio del frontend

cd ../frontend

2️⃣ Instalar dependencias

npm install

4️⃣ Ejecutar Nuxt en modo desarrollo

npm run dev

El frontend estará disponible en: http://localhost:3000

🛠 Despliegue

Para desplegar el backend en un servidor, usa Docker o una plataforma en la nube como AWS, DigitalOcean o Heroku. Para el frontend, puedes usar Vercel, Netlify o un servidor Nginx.

📝 Notas Adicionales

Gestión de dependencias: Recuerda actualizar requirements.txt cuando instales nuevas librerías en Python con:

pip freeze > requirements.txt

Estructura del Proyecto: Asegúrate de que tanto el backend como el frontend sigan una estructura limpia y modular.

📜 Licencia

Este proyecto está bajo la licencia MIT.

💡 ¡Disfruta construyendo tu asistente virtual! 🚀

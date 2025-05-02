# 📚 Catalogue ChatBot

Un chatbot léger développé avec **FastAPI** qui permet de converser sur l'application [catalogue](https://github.com/CorentinLeGuen/catalogue).
Le chatbot se configure avec la liste des Books disponible sur l'API principale.

## ⚙️ Installation

```shell
pip install -r requirements.txt
```

Créez un fichier .env et placez-y votre clé [OpenAI](https://platform.openai.com/) ainsi que votre clé [catalogue-api](https://github.com/CorentinLeGuen/catalogue).

## 🚀 Lancer le projet

### Conteneurisé
```shell
git clone https://github.com/CorentinLeGuen/catalogue-chatbot.git
cd catalogue-chatbot
docker-compose up --build -d
```

### En local avec uvicorn

> Remplacer les accès dans le [.env](.env.example).

```shell
uvicorn app:app --reload --host 0.0.0.0 --port 8012
```
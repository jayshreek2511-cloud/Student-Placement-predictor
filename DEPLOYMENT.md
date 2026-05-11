# Deployment

This project is ready for Python web hosting services such as Render, Railway, or any platform that supports a Procfile.

## Render

1. Push this repository to GitHub.
2. Create a new Render Web Service from the GitHub repository.
3. Render can use `render.yaml` automatically, or set these values manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Health check path: `/health`
4. Add an environment variable named `APP_SECRET_KEY` with a strong random value if your host does not generate it automatically.

The built-in SQLite database is fine for a demo. For a production app with many users, move history and accounts to PostgreSQL.

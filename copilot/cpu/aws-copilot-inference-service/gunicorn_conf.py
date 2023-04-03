workers = 2
worker_class = 'aiohttp.GunicornWebWorker'
bind = 'localhost:8080'
module = 'app:create_gunicorn_app'
timeout = 600

workers = 1
threads = 1
worker_class = "gevent"
bind = "0.0.0.0:8080"
accesslog = 'gunicorn_access.log'
errorlog  = 'gunicorn_error.log'
loglevel = 'info'
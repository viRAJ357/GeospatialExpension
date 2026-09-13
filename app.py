# Re-export the Flask app from api.py so both 'gunicorn app:app' and 'gunicorn api:app' work
from api import app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

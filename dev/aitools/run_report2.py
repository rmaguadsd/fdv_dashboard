#!/usr/bin/env python
import os

# Import the Flask app instance from the webapp module
from fdv_report2_webapp import app

if __name__ == '__main__':
    debug = (os.environ.get('FDV_REPORT2_DEBUG', '1').strip().lower() not in ('0','false','no','off'))
    host = (os.environ.get('FDV_REPORT2_HOST', '0.0.0.0').strip() or '0.0.0.0')
    try:
        port = int(os.environ.get('FDV_REPORT2_PORT', '5057'))
    except Exception:
        port = 5057
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)

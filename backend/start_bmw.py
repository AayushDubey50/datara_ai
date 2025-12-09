#!/usr/bin/env python3

import bmw

if __name__ == '__main__':
    try:
        print('Starting BMW Flask server...')
        print('FiftyOne integration:', bmw.FIFTYONE_AVAILABLE)
        print('Server will be available at: http://localhost:5051')
        print('API endpoints:')
        print('  - GET /api/bmw_grill/info - Dataset information')
        print('  - GET /api/bmw_grill/images - List all images')
        print('  - GET /api/bmw_grill/view/<name> - View specific image')
        if bmw.FIFTYONE_AVAILABLE:
            print('  - GET /api/bmw_grill/fiftyone - Open FiftyOne dataset viewer')
        bmw.app.run(host='0.0.0.0', port=5051, debug=False)
    except KeyboardInterrupt:
        print('Server stopped')
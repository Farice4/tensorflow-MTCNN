from flask import Flask, jsonify, make_response, request, abort, redirect, send_file
from werkzeug.utils import secure_filename
import logging
import os
import base64
import uuid

import face_processor
import train.config as config

app = Flask(__name__)


@app.route('/model_app/tensorflow_mtcnn/detect', methods=['POST'])
def upload():
    try:
        image = request.files['image']
        basepath = ("/tmp/")
        fn = (secure_filename(image.filename)).split('.')
        filename = str(uuid.uuid4()) + fn[0] + '.' + fn[1]
        upload_path = os.path.join(basepath, filename)
        image.save(upload_path)
        face_processor.detection(upload_path, filename)
        return make_response(jsonify(convert_data(filename)), 200)
        #return send_file('../output/predicted_image.png', mimetype='image/png')
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)

def convert_data(filename):
    result = {"type": "img"}
    with open(config.out_path + '%s' % filename, "rb") as img_file:
        img_data = base64.b64encode(img_file.read())
    result['data'] = ("data:image/jpg;base64,%s" % img_data)
    os.remove(config.out_path + '%s' % filename)
    return result

@app.errorhandler(400)
def bad_request(erro):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource no found.'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)

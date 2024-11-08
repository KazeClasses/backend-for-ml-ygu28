import os
from flask import Flask
from flask_cors import CORS
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # Register blueprints
    from . import database, classifier
    app.register_blueprint(database.database_service, url_prefix="/database")
    app.register_blueprint(classifier.classifier_service, url_prefix="/classifier")

    # Load configuration
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Simple route for testing
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    # Enable CORS
    CORS(app)

    return app

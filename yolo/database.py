from flask import Blueprint, request, jsonify, current_app
import psycopg
import psycopg.rows

database_service = Blueprint("database", __name__)

def connect_postgres() -> psycopg.Connection[psycopg.rows.TupleRow]:
    database_client = psycopg.connect(
        host=current_app.config["POSTGRES_HOST"],
        port=current_app.config["POSTGRES_PORT"],
        user=current_app.config["POSTGRES_USER"],
        password=current_app.config["POSTGRES_PASSWORD"],
        dbname=current_app.config["POSTGRES_DB"],
        row_factory=psycopg.rows.dict_row
    )
    # Create the table if it doesn't exist
    with database_client.cursor() as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS classes (id UUID PRIMARY KEY, name TEXT)"
        )
    return database_client

@database_service.route("/get_table", methods=["GET"])
def get_table():
    with connect_postgres() as client:
        cursor = client.cursor()
        cursor.execute("SELECT * FROM classes")
        return jsonify(cursor.fetchall())

@database_service.route("/add_class", methods=["POST"])
def upload_video():
    with connect_postgres() as client:
        cursor = client.cursor()
        data = request.form
        cursor.execute(
            "INSERT INTO classes (id, name) VALUES (gen_random_uuid(), %s)",
            (data.get("class"))
        )
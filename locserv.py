from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return send_file("index.html", as_attachment=False);

@app.route('/<path:subpath>')
def catch_all(subpath):
    return send_file(subpath, as_attachment=False);

if __name__ == '__main__':
    app.run(debug=True)
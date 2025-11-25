from flask import Flask, render_template, request, url_for
from predictor import make_prediction
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    error = None
    plot_exists = False

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if symbol and start_date and end_date:
            result = make_prediction(symbol, start_date, end_date)
            if isinstance(result, dict):
                context = result
                fname = result.get("plot_filename")
                if fname:
                    plot_path = os.path.join(os.path.dirname(__file__), 'static', fname)
                    plot_exists = os.path.isfile(plot_path)
                    context["plot_url"] = url_for('static', filename=fname)
            else:
                error = result
        else:
            error = "Todos os campos são obrigatórios."

    return render_template('index.html', error=error, plot_exists=plot_exists, **context)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
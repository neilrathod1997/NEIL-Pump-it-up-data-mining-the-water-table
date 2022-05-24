"""Flask Demo"""
from flask import Flask, render_template
import temp as model
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template(index.html)


@app.route('/predict', methods=['POST'])
def predict()
    if request.method == 'POST':
        variable = request.form['variable']
        data = pd.read_csv('static/test.csv')
        size = len(data[variable])
        plt.bar(range(size), data[variable])
        imagepath = os.path.join('static', 'image' + '.png')
        plt.savefig(imagepath)
        return render_template('image.html', image = imagepath)

if __name__ == '__main__':
    app.run()

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import cv2
import io
from openpyxl import Workbook

app = Flask(__name__)
CORS(app)  # allow CORS for frontend requests

# HSV color thresholds
color_ranges = {
    'magenta': ([120, 40, 40], [170, 255, 255]),
    'red': ([0, 50, 50], [10, 255, 255]),
    'blue': ([100, 50, 50], [140, 255, 255]),
    'green': ([40, 50, 50], [80, 255, 255]),
    'cyan': ([80, 50, 50], [100, 255, 255]),
    'yellow': ([20, 50, 50], [40, 255, 255])
}

# Fixed pixel boundaries (adjust based on your image layout)
x_min_p, x_max_p = 72, 876
y_min_p, y_max_p = 603, 72

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    color = request.form['color']
    x_max = float(request.form['x_max'])
    y_max = float(request.form['y_max'])

    # Read image
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color mask
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    # Extract data
    data_points = []
    for px in range(x_min_p, x_max_p + 1):
        for py in range(y_max_p, y_min_p + 1):
            if mask[py, px] > 0:
                graph_x = ((px - x_min_p) / (x_max_p - x_min_p)) * x_max
                graph_y = ((y_min_p - py) / (y_min_p - y_max_p)) * y_max
                data_points.append((graph_x, graph_y))

    if not data_points:
        return jsonify({'error': 'No data points detected.'}), 400

    df = pd.DataFrame(data_points, columns=['Time (s)', 'Phy Rate (Mbps)'])
    df = df.groupby('Time (s)').mean().reset_index()

    # Save to Excel in memory
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="extracted_graph_data.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == '__main__':
    app.run(debug=True)

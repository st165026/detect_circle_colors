import os

from flask import Flask, jsonify, request, send_file
import cv2
import numpy as np
from skimage.feature import canny
from scipy.ndimage import binary_fill_holes

app = Flask(__name__)

@app.route('/get_picture', methods=['GET'])
def get_picture():
    filename = request.args.get('name')+".png"
    return send_file(filename, mimetype='image/png')


@app.route('/take_picture', methods=['POST'])
def take_picture():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    # reading the input using the camera
    result, image = cam.read()

    # If image will detected without any error,
    # show result
    if result:
        # saving image in local storage
        cv2.imwrite(request.json["name"] + ".png", image)
    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")

    return jsonify({'url': request.json['name']+".png"}), 200

@app.route('/delete_picture', methods=['DELETE'])
def delete_picture():
    filename = request.args.get('name')+".png"
    try:
        os.remove(filename)
        return jsonify({'message': 'yes'})
    except FileNotFoundError:
        return jsonify({'error': 'image not found'}), 404
    except Exception as e:
        return jsonify({'error': 'other error'}), 500


def circle_edge_clearness(image, x, y, r):
    edge_pixels = []
    theta = np.linspace(0, 2*np.pi, 360)
    for angle in theta:
        xi = int(x + r * np.cos(angle))
        yi = int(y + r * np.sin(angle))
        if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
            edge_pixels.append(image[yi, xi])
    edge_pixels = np.array(edge_pixels)
    edge_std = np.std(edge_pixels)
    return edge_std > 50  # Edge standard deviation threshold

def is_transparent(edge_img, filled_img, threshold=0.05):
    # Calculate the proportion of edge pixels to filled pixels
    edge_count = np.sum(edge_img)
    filled_count = np.sum(filled_img)
    if edge_count / filled_count < threshold:
        return True
    return False


@app.route('/detect_image', methods=['POST'])
def detect_image():
    data = request.json
    image_url = data['url']
    image = cv2.imread(image_url)
    if image is None:
        return jsonify({'error': 'Image could not be loaded'}), 400

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray, sigma=3)  # Canny edge detection
    filled = binary_fill_holes(edges)  # Fill the holes to find complete objects

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=100, param2=50, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        grid_result = [([(False, None)] * 3) for _ in range(3)]
        for (x, y, r) in circles:
            patch = image[y - r:y + r, x - r:x + r]
            edge_patch = edges[y - r:y + r, x - r:x + r]
            filled_patch = filled[y - r:y + r, x - r:x + r]
            if is_transparent(edge_patch, filled_patch):
                circle_color = "transparent"
            else:
                mean_b, mean_g, mean_r = cv2.mean(patch)[:3]
                circle_color = "red" if mean_r > mean_g + 10 and mean_r > mean_b + 10 else "transparent"
            grid_x = x // (image.shape[1] // 3)
            grid_y = y // (image.shape[0] // 3)
            if 0 <= grid_x < 3 and 0 <= grid_y < 3:
                grid_result[grid_y][grid_x] = (True, circle_color)
    else:
        return jsonify({'result': 'No circles detected'}), 200
    return jsonify({'grid': grid_result}), 200





if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
from PIL import Image
import geopandas as gpd


# FLASK APP

app = Flask(__name__)
CORS(app)


# LOAD TRAINED MODEL
model = tf.keras.models.load_model(
    r"C:\Users\breez\OneDrive\Desktop\treeleaf_identification\best_model_finetuned.keras"
)

print("MODEL LOADED SUCCESSFULLY")
print("MODEL INPUT SHAPE:", model.input_shape)


# CLASS NAMES
# To MATCH TRAINING ORDER

leaf_names = [
    'Jackfruit',
    'Mango',
    'Neem',
    'Peepal'
]


# LOAD SHP FILE ONCE AT STARTUP

gdf = gpd.read_file(
    r"E:\Third_Sem\spices_identification\Trees_Waypoints\UOM_Treepoints.shp"
)

print("SHP LOADED. TOTAL POINTS:", len(gdf))


# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# PREDICTION API

@app.route('/predict', methods=['POST'])
def predict():
    try:

        if 'image' not in request.files:
            return jsonify({'error': 'Image file missing'})

        
        # READ AND PREPROCESS IMAGE
        # MODEL HAS Rescaling(1./255) INSIDE
        # SO SEND RAW PIXELS - NO preprocess_input
       

        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))

        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # MODEL PREDICTION

        prediction = model.predict(img_array)[0]

        print("RAW PREDICTION:", prediction)
        print("ALL CLASS SCORES:", {
            leaf_names[i]: round(float(prediction[i]) * 100, 2)
            for i in range(len(leaf_names))
        })

        confidence = float(np.max(prediction) * 100)
        predicted_class = leaf_names[np.argmax(prediction)]

        print("PREDICTED CLASS:", predicted_class)
        print("CONFIDENCE:", confidence)

        # REJECT LOW CONFIDENCE
        if confidence < 50:
            return jsonify({
                'class': 'Uncertain',
                'confidence': round(confidence, 2),
                'points': [],
                'message': 'Low confidence - try a clearer image'
            })

        
        # MATCH SHP POINTS
        # FIRST 4 LETTERS MATCHING

        predicted_short = predicted_class[:4].lower()
        print("MATCHING WITH:", predicted_short)

        matched_points = []

        for _, row in gdf.iterrows():
            name = str(row['Name']).strip()

            if predicted_short in name.lower():
                lon = row.geometry.x
                lat = row.geometry.y
                matched_points.append({
                    'name': name,
                    'lat': lat,
                    'lon': lon
                })

        print("MATCHED POINTS:", len(matched_points))

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence, 2),
            'points': matched_points
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", str(e))
        return jsonify({'error': str(e)})


# RUN FLASK SERVER

if __name__ == '__main__':
    app.run(debug=True)
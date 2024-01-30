from flask import Flask, request, jsonify, render_template  
import joblib

app = Flask(__name__)

# Loading the model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # index.html file with data entry form

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form  # request.form to get data from a form

    # Extracting the necessary features from the input data
    average_score = float(data['average_score'])
    review_total_negative_word_counts = int(data['review_total_negative_word_counts'])
    total_number_of_reviews = int(data['total_number_of_reviews'])
    review_total_positive_word_counts = int(data['review_total_positive_word_counts'])
    total_number_of_reviews_reviewer_has_given = int(data['total_number_of_reviews_reviewer_has_given'])
    distance_to_city_center = float(data['distance_to_city_center'])
    hotel_country_Austria = int(data['hotel_country_Austria'])
    hotel_country_France = int(data['hotel_country_France'])
    hotel_country_Italy = int(data['hotel_country_Italy'])
    hotel_country_Netherlands = int(data['hotel_country_Netherlands'])
    hotel_country_Spain = int(data['hotel_country_Spain'])
    hotel_country_United_Kingdom = int(data['hotel_country_United Kingdom'])
    month = int(data['month'])
    nationality_mean_score = float(data['nationality_mean_score'])
    positive_review_contains_good = int(data['positive_review_contains_good'])
    negative_review_contains_terrible = int(data['negative_review_contains_terrible'])

    features = [
        average_score,
        review_total_negative_word_counts,
        total_number_of_reviews,
        review_total_positive_word_counts,
        total_number_of_reviews_reviewer_has_given,
        distance_to_city_center,
        hotel_country_Austria,
        hotel_country_France,
        hotel_country_Italy,
        hotel_country_Netherlands,
        hotel_country_Spain,
        hotel_country_United_Kingdom,
        month,
        nationality_mean_score,
        positive_review_contains_good,
        negative_review_contains_terrible
    ]


    prediction = model.predict([features])

    # Predicted hotel rating as JSON
    return jsonify({'predicted_rating': prediction[0]})

if __name__ == '__main':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, render_template
from predict import MentalHealthPredictor
import json
import traceback

app = Flask(__name__)
predictor = None

def get_emoji_for_emotion(emotion, sentiment):
    """Return appropriate emoji based on emotion and sentiment"""
    emoji_map = {
        'Anxiety': '😟' if sentiment < 0 else '😌',
        'Depression': '😢' if sentiment < 0 else '🌤️',
        'Study Tips': '📚',
    }
    return emoji_map.get(emotion, '🤔')

def format_response(prediction_result):
    """Format the prediction result for display"""
    emotion = prediction_result['predicted_class']
    sentiment = prediction_result['sentiment']
    confidence = prediction_result['confidence']
    
    emoji = get_emoji_for_emotion(emotion, sentiment)
    
    # Format confidence as percentage
    confidence_pct = f"{confidence * 100:.1f}%"
    
    # Get sentiment description
    sentiment_desc = "negative" if sentiment < -0.2 else "positive" if sentiment > 0.2 else "neutral"
    
    return {
        'response': prediction_result['response'],
        'emotion': {
            'label': emotion,
            'emoji': emoji,
            'confidence': confidence_pct
        },
        'sentiment': {
            'score': sentiment,
            'description': sentiment_desc
        },
        'resources': prediction_result['supportive_resources']
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global predictor
    
    try:
        if predictor is None:
            predictor = MentalHealthPredictor()
        
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({
                'error': 'No message provided'
            }), 400
        
        # Get prediction and format response
        prediction = predictor.predict(user_input)
        formatted_response = format_response(prediction)
        
        return jsonify(formatted_response)
        
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        print(f"Error in chat endpoint: {error_trace}")
        
        return jsonify({
            'error': 'An error occurred',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if predictor is None:
            predictor = MentalHealthPredictor()
        
        # Test prediction with a simple input
        test_result = predictor.predict("test message")
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def initialize_app():
    """Initialize the application and load the model"""
    global predictor
    try:
        predictor = MentalHealthPredictor()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Application will load model on first request")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True)

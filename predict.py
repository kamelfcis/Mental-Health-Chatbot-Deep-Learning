import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_nlp_features(self, text):
        # Perform sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Get POS tags
        pos_tags = blob.tags
        
        # Count different types of words
        noun_count = len([tag for word, tag in pos_tags if tag.startswith('NN')])
        verb_count = len([tag for word, tag in pos_tags if tag.startswith('VB')])
        adj_count = len([tag for word, tag in pos_tags if tag.startswith('JJ')])
        
        return np.array([
            sentiment,
            subjectivity,
            noun_count,
            verb_count,
            adj_count
        ])

class MentalHealthPredictor:
    def __init__(self, model_path='mental_health_model.h5', 
                 tokenizer_path='tokenizer.json',
                 label_encoder_path='label_encoder_classes.npy',
                 max_len=100):
        try:
            # Initialize text preprocessor
            self.preprocessor = TextPreprocessor()
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load tokenizer
            with open(tokenizer_path, 'r') as f:
                tokenizer_json = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)
            
            # Load label encoder classes
            self.label_classes = np.load(label_encoder_path)
            
            self.max_len = max_len
            
            # Load supportive resources
            with open('supportive_resources.json', 'r') as f:
                self.resources = json.load(f)
                
            # Load response dataset
            with open('mental_health_dataset.json', 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
                
        except Exception as e:
            raise Exception(f"Error initializing predictor: {str(e)}")
    
    def get_response(self, emotional_state, user_input, sentiment_score):
        """Get appropriate response based on emotional state, user input, and sentiment"""
        # First try to find a direct match from the dataset
        matching_responses = [
            item['response_en'] 
            for item in self.responses 
            if item['input'].lower() in user_input.lower()
        ]
        
        if matching_responses:
            return matching_responses[0]
        
        # Default responses considering sentiment
        default_responses = {
            'Anxiety': [
                "I understand your anxiety. Let's work through this together. What's causing you the most worry?",
                "It's okay to feel anxious. Have you tried any relaxation techniques that worked for you before?",
                "I hear your anxiety. Would you like to explore some coping strategies?"
            ],
            'Depression': [
                "I'm here to listen and support you. Would you like to share what's been troubling you?",
                "You're not alone in this. What's been on your mind lately?",
                "I understand these feelings can be overwhelming. Let's talk about it."
            ],
            'Study Tips': [
                "Learning can be challenging. What specific aspect would you like help with?",
                "There are many effective study techniques. Would you like to explore some together?",
                "Study stress is common. Let's find strategies that work for you."
            ]
        }
        
        # Get responses for the emotional state or use general responses
        responses = default_responses.get(emotional_state, [
            "I'm here to support you. Would you like to tell me more?",
            "Your feelings are valid. Let's explore them together.",
            "Thank you for sharing. How can I best support you right now?"
        ])
        
        # Choose response based on sentiment
        if sentiment_score < -0.5:
            # More empathetic response for very negative sentiment
            return responses[0]
        elif sentiment_score < 0:
            # Supportive response for slightly negative sentiment
            return responses[1]
        else:
            # Encouraging response for neutral/positive sentiment
            return responses[2]
    
    def predict(self, text):
        """Predict emotional state and provide appropriate response"""
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            nlp_features = self.preprocessor.extract_nlp_features(text)
            
            # Prepare text input
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequence, maxlen=self.max_len)
            
            # Make prediction
            prediction = self.model.predict([padded, nlp_features.reshape(1, -1)])
            predicted_class = self.label_classes[prediction.argmax()]
            confidence = float(prediction.max())
            
            # Get relevant resources
            resources = self.resources.get(predicted_class, [])
            
            # Get appropriate response considering sentiment
            response = self.get_response(
                predicted_class, 
                text, 
                nlp_features[0]  # sentiment score
            )
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'response': response,
                'supportive_resources': resources[:3],  # Return top 3 resources
                'sentiment': float(nlp_features[0])
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

def main():
    """Test the predictor"""
    try:
        predictor = MentalHealthPredictor()
        
        # Test cases
        test_inputs = [
            "I feel really anxious about my upcoming exam",
            "I've been feeling sad and lonely lately",
            "I need help with my study routine"
        ]
        
        for text in test_inputs:
            print(f"\nInput: {text}")
            result = predictor.predict(text)
            print(f"Emotional State: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sentiment Score: {result['sentiment']:.2f}")
            print(f"Response: {result['response']}")
            print("Supportive Resources:")
            for resource in result['supportive_resources']:
                print(f"- {resource}")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

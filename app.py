from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import re

app = Flask(__name__)

# Load dataset
data = pd.read_csv("wellness_dataset.csv")
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Encode target
le = LabelEncoder()
data['burnout'] = le.fit_transform(data['burnout'])

# Text vectorization
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(data['text'])

# Combine numerical features
numerical_features = data[['study_hours', 'sleep_hours', 'screen_time','mood']].values
scaler = StandardScaler()  
numerical_features = scaler.fit_transform(numerical_features)
X = np.hstack((text_features.toarray(), numerical_features))
y = data['burnout']




# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Enhanced advice system
def analyze_text_for_keywords(text):
    """Analyze text for specific keywords and return targeted advice"""
    text_lower = text.lower()
    advice = []

    # Stress and anxiety keywords
    stress_keywords = ['stressed', 'anxious', 'overwhelmed', 'pressure', 'worried', 'nervous', 'panic']
    if any(keyword in text_lower for keyword in stress_keywords):
        advice.append("Try deep breathing exercises: inhale for 4 counts, hold for 4, exhale for 4.")

    # Sleep-related keywords
    sleep_keywords = ['tired', 'exhausted', 'sleepy', 'fatigued', 'drained', 'insomnia']
    if any(keyword in text_lower for keyword in sleep_keywords):
        advice.append("Consider establishing a consistent sleep schedule and creating a relaxing bedtime routine.")

    # Study/work keywords
    study_keywords = ['study', 'work', 'assignment', 'exam', 'deadline', 'busy', 'overloaded']
    if any(keyword in text_lower for keyword in study_keywords):
        advice.append("Break your work into smaller, manageable chunks with regular breaks using the Pomodoro technique.")

    # Emotional keywords
    positive_keywords = ['happy', 'excited', 'confident', 'relaxed', 'peaceful', 'energized']
    if any(keyword in text_lower for keyword in positive_keywords):
        advice.append("Great mindset! Keep nurturing these positive feelings through gratitude journaling.")

    negative_keywords = ['sad', 'depressed', 'hopeless', 'lonely', 'worthless', 'empty']
    if any(keyword in text_lower for keyword in negative_keywords):
        advice.append("Remember that it's okay to ask for help. Consider talking to a trusted friend or counselor.")

    return advice

def get_personalized_recommendations(study_hours, screen_time, sleep_hours, mood, burnout_level):
    """Generate personalized recommendations based on input metrics"""
    recommendations = []

    # Sleep recommendations
    if sleep_hours < 6:
        recommendations.append("🚨 Prioritize sleep! Aim for 7-9 hours nightly. Consider a consistent bedtime routine.")
    elif sleep_hours < 7:
        recommendations.append("💤 You're getting some sleep, but could benefit from more rest. Try winding down earlier.")
    else:
        recommendations.append("✅ Good sleep habits! Keep maintaining your sleep schedule.")
    
    # screentime recommendations    
    if screen_time > 6:
        recommendations.append("📱 Limit screen time, especially before bed. Try the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.")
    elif screen_time > 3:
        recommendations.append("⌚ Moderate screen time. Be mindful of your usage and take breaks regularly.")
    else:
        recommendations.append("✅ Balanced screen time! Keep up the good work.")

    # Study hours recommendations
    if study_hours > 10:
        recommendations.append("⚠️ High study load detected. Consider redistributing tasks or taking a short break.")
    elif study_hours > 8:
        recommendations.append("📚 You're studying a lot. Make sure to include regular breaks and leisure activities.")
    elif study_hours < 4:
        recommendations.append("🎯 If you're not studying enough, try setting small daily goals to build momentum.")

    # Mood-based recommendations
    if mood <= 2:
        recommendations.append("💙 For low mood days, try: listening to uplifting music, calling a friend, or going for a walk.")
    elif mood >= 4:
        recommendations.append("🌟 You're in a good mood! Use this energy for creative activities or helping others.")

    # Burnout-specific recommendations
    if burnout_level == "Burnout":
        recommendations.append("🔴 URGENT: Take immediate action. Stop working, rest, and seek professional help if needed.")
        recommendations.append("📞 Consider talking to a mental health professional or counselor.")
    elif burnout_level == "Moderate":
        recommendations.append("🟡 Monitor your stress levels. Practice mindfulness or meditation daily.")
    else:
        recommendations.append("🟢 Keep up the healthy habits! Consider sharing your wellness tips with others.")

    return recommendations

def get_activity_suggestions(burnout_level):
    """Suggest activities based on burnout level"""
    activities = {
        "Healthy": [
            "🏃‍♂️ Go for a run or hike in nature",
            "🎨 Try a new creative hobby like painting or writing",
            "👥 Plan a social gathering with friends",
            "📖 Read a book for pleasure",
            "🎵 Learn to play a musical instrument"
        ],
        "Moderate": [
            "🧘 Practice yoga or meditation for 10-15 minutes",
            "🌱 Start a small indoor plant garden",
            "📝 Journal your thoughts and feelings",
            "🎬 Watch a feel-good movie or series",
            "☕ Enjoy a cup of tea while listening to calming music"
        ],
        "Burnout": [
            "😴 Take a long nap or rest day",
            "🛀 Have a relaxing bath with essential oils",
            "📱 Limit screen time and social media",
            "🌳 Spend time in nature, even just sitting outside",
            "💤 Focus on restorative activities like gentle stretching"
        ]
    }
    return random.sample(activities[burnout_level], min(3, len(activities[burnout_level])))

def get_motivational_quote():
    """Return a random motivational quote"""
    quotes = [
        "\"The only way to do great work is to love what you do.\" - Steve Jobs",
        "\"Believe you can and you're halfway there.\" - Theodore Roosevelt",
        "\"The future belongs to those who believe in the beauty of their dreams.\" - Eleanor Roosevelt",
        "\"You miss 100% of the shots you don't take.\" - Wayne Gretzky",
        "\"The best way to predict the future is to create it.\" - Peter Drucker",
        "\"Keep your face always toward the sunshine—and shadows will fall behind you.\" - Walt Whitman",
        "\"The only limit to our realization of tomorrow will be our doubts of today.\" - Franklin D. Roosevelt",
        "\"What lies behind us and what lies before us are tiny matters compared to what lies within us.\" - Ralph Waldo Emerson"
    ]
    return random.choice(quotes)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    suggestion = ""
    text_advice = []
    recommendations = []
    activities = []
    quote = ""
    emoji = ""
    timetable = []

    if request.method == "POST":
        text = request.form["text"]
        study = float(request.form["study"])
        screen_time = float(request.form["screen_time"])
        sleep = float(request.form["sleep"])
        mood = float(request.form["mood"])

        text_vec = vectorizer.transform([text]).toarray()
        num_input = scaler.transform([[study, sleep, screen_time, mood]])
        final_input = np.hstack((text_vec, num_input))

        prediction = model.predict(final_input)[0]
        label = le.inverse_transform([prediction])[0]

        # Get emoji based on status
        emoji = {
            "Healthy": "😌 🌱",
            "Moderate": "😐 💛",
            "Burnout": "😰 🔴"
        }[label]

        # Get timetable based on status
        timetable = {
            "Burnout": [
                {"time": "Morning", "activity": "Meditation and rest"},
                {"time": "Afternoon", "activity": "Light exercise"},
                {"time": "Evening", "activity": "Emotional recovery"},
                {"time": "Night", "activity": "Sleep early"}
            ],
            "Moderate": [
                {"time": "Morning", "activity": "Balanced study"},
                {"time": "Afternoon", "activity": "Breaks and hobbies"},
                {"time": "Evening", "activity": "Moderate exercise"},
                {"time": "Night", "activity": "Relaxation"}
            ],
            "Healthy": [
                {"time": "Morning", "activity": "Productive study"},
                {"time": "Afternoon", "activity": "Skill learning"},
                {"time": "Evening", "activity": "Hobbies"},
                {"time": "Night", "activity": "Good sleep"}
            ]
        }[label]

        text_advice = analyze_text_for_keywords(text)
        recommendations = get_personalized_recommendations(study, screen_time,sleep, mood, label)
        activities = get_activity_suggestions(label)
        quote = get_motivational_quote()

        result = label

    return render_template("index.html", result=result, suggestion=suggestion,
                         text_advice=text_advice, recommendations=recommendations,
                         activities=activities, quote=quote, emoji=emoji, timetable=timetable)

if __name__ == "__main__":
    app.run(debug=True)
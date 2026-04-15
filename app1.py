from flask import Flask, request, jsonify
from faqbot import FAQBot
import os

app = Flask(__name__)

# ✅ Load bot مرة واحدة
bot = FAQBot("data.csv")


# =====================================
# CHAT ENDPOINT
# =====================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        question = data.get("question")
        student_id = data.get("student_id")

        # ✅ ناخد التوكن من ال Header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "Authorization header missing"}), 401

        # Bearer token
        token = auth_header.replace("Bearer ", "")

        if not question:
            return jsonify({"error": "Question required"}), 400

        if not student_id:
            return jsonify({"error": "student_id required"}), 400

        bot.token = token

        result = bot.answer(question, student_id=student_id)

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =====================================
# HEALTH CHECK
# =====================================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Guide Bot is working 🚀"
    })


# =====================================
# TEST ENDPOINT (اختياري بس مهم 🔥)
# =====================================
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "message": "API is working fine"
    })


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy  # Corrected import statement
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Update the database URI to connect to MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@hostname/database_name'  
# Replace 'username', 'password', 'hostname', and 'database_name' with your MySQL credentials and database name

db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.String(500), nullable=False)
    bot_response = db.Column(db.String(500), nullable=False)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    bot_response = get_chat_response(msg)
    
    # Save user input and bot response to database
    chat_history = ChatHistory(user_input=msg, bot_response=bot_response)
    db.session.add(chat_history)
    db.session.commit()
    
    return bot_response

def get_chat_response(text):
    # Let's chat for 5 lines
    chat_history_ids = None
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last output tokens from bot
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    db.create_all()  # Create the database tables
    app.run()

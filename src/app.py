from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.agent_graph import create_medibot_agent   
import os


app = Flask(__name__, template_folder="templates")


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


agent = create_medibot_agent()  


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    result = agent.invoke({"input": msg})
    answer = result["answer"]
    reflection = result.get("reflection", "")

    
    formatted_answer = (
        answer.replace("**", "")          # remove Markdown bold markers
              .replace("*", "•")          # convert bullets
              .replace("\n\n", "<br><br>") # paragraph breaks
              .replace("\n", "<br>")      # single line breaks
    )

    formatted_reflection = (
        reflection.replace("**", "")
                  .replace("*", "•")
                  .replace("\n\n", "<br><br>")
                  .replace("\n", "<br>")
    )

    final_html = f"""
    <div style='font-family:Segoe UI, Arial; line-height:1.7; font-size:15px; color:#222;'>
        <div style='margin-bottom:10px;'>{formatted_answer}</div>
        <hr style='border:none; border-top:1px solid #ccc; margin:10px 0;'>
        <div style='color:#555; font-size:0.9em;'><b>🧠 Reflection:</b> {formatted_reflection}</div>
    </div>
    """

    return final_html




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

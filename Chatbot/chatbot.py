from flask import Flask, request, jsonify, render_template_string

# Initialize the Flask application
app = Flask(__name__)

# Sample data for responses
responses = {
    "hi": "Hello! How can I help you today?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "bye": "Goodbye! Have a great day!",
    "what is your name": "I'm Tony's Chatbot, here to assist you.",
    "what can you do": "I can answer simple questions and provide information.",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "what is the weather like": "I can't check the weather, but I hope it's nice where you are!",
    "what is your favorite color": "I like blue, just like the sky!",
    "do you like music": "I don't have ears, but I imagine music is wonderful!",
    "what is the capital of France": "The capital of France is Paris.",
    "who is the president of the United States": "As of 2023, the president is Joe Biden.",
    "what is the largest ocean": "The Pacific Ocean is the largest ocean on Earth.",
    "how many continents are there": "There are seven continents on Earth.",
    "what is the tallest mountain": "Mount Everest is the tallest mountain in the world.",
    "what is the speed of light": "The speed of light is approximately 299,792 kilometers per second.",
    "what is the boiling point of water": "The boiling point of water is 100°C or 212°F at sea level.",
    "what is the freezing point of water": "The freezing point of water is 0°C or 32°F.",
    "what is the largest planet": "Jupiter is the largest planet in our solar system.",
    "what is the smallest planet": "Mercury is the smallest planet in our solar system.",
    "what is the most spoken language": "English is one of the most spoken languages worldwide.",
    "what is the longest river": "The Nile River is often considered the longest river in the world.",
    "what is the largest desert": "The Sahara Desert is the largest hot desert in the world.",
    "what is the most populated country": "China is the most populated country in the world.",
    "what is the currency of Japan": "The currency of Japan is the Yen.",
    "what is the national animal of India": "The Bengal Tiger is the national animal of India.",
    "what is the national flower of Japan": "The Cherry Blossom is the national flower of Japan.",
    "what is the national bird of the USA": "The Bald Eagle is the national bird of the USA.",
    "what is the national sport of Canada": "Ice Hockey is considered the national sport of Canada.",
    "what is the national dish of Italy": "Pasta is often considered a national dish of Italy.",
    "what is the national language of Brazil": "Portuguese is the national language of Brazil.",
    "what is the national anthem of the UK": "God Save the Queen is the national anthem of the UK.",
    "what is the national flag of China": "The national flag of China is red with five yellow stars.",
    "what is the national emblem of Australia": "The national emblem of Australia features the kangaroo and emu.",
    "what is the national tree of Canada": "The Maple Tree is the national tree of Canada.",
    "what is the national fruit of India": "The Mango is the national fruit of India.",
    "what is the national vegetable of Japan": "The Daikon Radish is often considered a national vegetable of Japan.",
    "what is the national drink of Russia": "Vodka is often considered the national drink of Russia.",
    "what is the national dance of Spain": "Flamenco is a traditional dance of Spain.",
    "what is the national costume of Scotland": "The Kilt is a traditional Scottish costume.",
    "what is the national festival of Brazil": "Carnival is a famous festival in Brazil.",
    "what is the national holiday of the USA": "Independence Day is a national holiday in the USA.",
    "what is the national park of South Africa": "Kruger National Park is a famous park in South Africa.",
    "what is the national monument of France": "The Eiffel Tower is a famous monument in France.",
    "what is the national museum of the UK": "The British Museum is a famous museum in the UK.",
    "what is the national library of the USA": "The Library of Congress is the national library of the USA.",
    "what is the national university of Japan": "The University of Tokyo is a prestigious university in Japan.",
    "what is the national airline of Australia": "Qantas is the national airline of Australia.",
    "what is the national railway of India": "Indian Railways is the national railway of India.",
    "what is the national highway of the USA": "The Interstate Highway System is a network of highways in the USA.",
    "what is the national river of China": "The Yangtze River is a major river in China.",
    "what is the national mountain of Japan": "Mount Fuji is a famous mountain in Japan.",
    "what is the national lake of Canada": "Lake Superior is a large lake in Canada.",
    "what is the national sea of the UK": "The North Sea is a sea bordering the UK.",
    "what is the national ocean of the USA": "The Pacific Ocean borders the western USA.",
    "what is the national forest of Brazil": "The Amazon Rainforest is a vast forest in Brazil.",
    "what is the national desert of Australia": "The Great Victoria Desert is a large desert in Australia.",
    "what is the national island of Japan": "Honshu is the largest island of Japan.",
    "what is the national peninsula of India": "The Indian Peninsula is a major landform in India.",
    "what is the national gulf of Mexico": "The Gulf of Mexico is a large gulf bordering Mexico.",
    "what is the national bay of Canada": "Hudson Bay is a large bay in Canada.",
    "what is the national strait of the UK": "The Strait of Dover is a strait between the UK and France.",
    "what is the national channel of the USA": "The English Channel is a channel between the UK and France.",
    "what is the national cape of South Africa": "Cape of Good Hope is a famous cape in South Africa.",
    "what is the national reef of Australia": "The Great Barrier Reef is a famous reef in Australia.",
    "what is the national volcano of Italy": "Mount Vesuvius is a famous volcano in Italy.",
    "what is the national glacier of Argentina": "Perito Moreno Glacier is a famous glacier in Argentina.",
    "what is the national canyon of the USA": "The Grand Canyon is a famous canyon in the USA.",
    "what is the national waterfall of Brazil": "Iguazu Falls is a famous waterfall in Brazil.",
    "what is the national cave of France": "Lascaux Caves are famous caves in France.",
    "what is the national archipelago of Indonesia": "The Indonesian Archipelago is a large group of islands.",
    "what is the national fjord of Norway": "Geirangerfjord is a famous fjord in Norway.",
    "what is the national plateau of Tibet": "The Tibetan Plateau is a large plateau in Tibet.",
    "what is the national valley of Switzerland": "Lauterbrunnen Valley is a famous valley in Switzerland.",
    "what is the national plain of Russia": "The Siberian Plain is a large plain in Russia.",
    "what is the national basin of the USA": "The Great Basin is a large basin in the USA.",
    "what is the national delta of Egypt": "The Nile Delta is a famous delta in Egypt.",
    "what is the national swamp of Florida": "The Everglades is a famous swamp in Florida.",
    "what is the national marsh of Louisiana": "The Atchafalaya Basin is a large marsh in Louisiana.",
    "what is the national bog of Ireland": "The Bog of Allen is a famous bog in Ireland.",
    "what is the national tundra of Alaska": "The Arctic Tundra is a large tundra in Alaska.",
    "what is the national steppe of Mongolia": "The Mongolian Steppe is a large steppe in Mongolia.",
    "what is the national savanna of Africa": "The Serengeti is a famous savanna in Africa.",
    "what is the national prairie of Canada": "The Canadian Prairies are large prairies in Canada.",
    "what is the national rainforest of Brazil": "The Amazon Rainforest is a vast rainforest in Brazil.",
    "what is the national woodland of the UK": "Sherwood Forest is a famous woodland in the UK.",
    "what is the national grove of California": "The Redwood National and State Parks are famous groves in California.",
    "what is the national orchard of Washington": "The Wenatchee Valley is known for its apple orchards.",
    "what is the national vineyard of France": "The Bordeaux region is famous for its vineyards.",
    "what is the national garden of Japan": "The Kenroku-en Garden is a famous garden in Japan.",
    "what is the national park of the USA": "Yellowstone National Park is a famous park in the USA.",
    "default": "I'm sorry, I don't understand that."
}

@app.route("/")
def index():
    return render_template_string(open('index.html').read())

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message").lower()
    print(f"User input: {user_input}")  # Debugging line
    
    # Find a response
    response_text = responses.get(user_input, responses["default"])
    print(f"Response: {response_text}")  # Debugging line
    
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
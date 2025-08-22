# Setup Guide for Board Game RAG System

## What We've Built

You now have a cutting-edge **RAG (Retrieval-Augmented Generation)** system for board game recommendations powered by:

### ✅ **Advanced RAG Architecture**
- **Vector Database**: ChromaDB for semantic similarity search
- **Smart Embeddings**: Sentence-transformers for understanding game descriptions
- **AI Generation**: Google Gemini or OpenAI for natural language responses

### ✅ **Google Gen AI SDK Integration**
- Uses the latest [Google Gen AI SDK](https://github.com/googleapis/python-genai)
- Supports both Gemini and OpenAI providers with easy switching
- Intelligent natural language understanding and generation

### ✅ **Modern API Design**
- **POST `/query`** - Natural language game recommendations with AI responses
- **GET `/search`** - Raw semantic similarity search
- **GET `/stats`** - Database statistics
- **GET `/health`** - System health check

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Setup API Key

**Option A: Google Gemini (Recommended - Free tier available)**

1. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create `.env` file:

```bash
cp .env.example .env
```

3. Edit `.env` file:

```bash
GEMINI_API_KEY=your_actual_gemini_api_key_here
LLM_PROVIDER=gemini
```

**Option B: OpenAI (Alternative)**

```bash
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
```

### 3. Test Your Setup

```bash
python test_providers.py
```

This will verify your API key and test both embedding and text generation.

### 4. Try the Interactive Demo

```bash
python demo_rag.py
```

This will:
- Initialize the RAG system with your game dataset
- Show example queries and AI responses
- Let you ask your own questions

### 5. Start the API Server

```bash
python run_api.py
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## How the RAG System Works

### 1. **Data Processing**
- Loads your `bgg_dataset.csv` with 20,000+ board games
- Creates rich text descriptions combining game info, mechanics, ratings
- Converts descriptions to vector embeddings for semantic search

### 2. **Vector Storage**
- Stores embeddings in ChromaDB for fast similarity search
- Each game becomes a searchable vector in high-dimensional space
- Finds games by meaning, not just keywords

### 3. **Query Processing**
- User asks: *"I want a strategic game for 2-4 players that takes about 90 minutes"*
- System finds semantically similar games using vector search
- AI generates personalized recommendations with explanations

### 4. **AI Response Generation**
- Gemini/GPT takes the relevant games and user query
- Generates natural, conversational recommendations
- Explains why each game fits the user's criteria

## API Usage Examples

### Natural Language Queries (Recommended)

```bash
# Strategic games query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want a strategic game for 2-4 players that takes about 90 minutes",
    "num_results": 5
  }'

# Cooperative games for beginners
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me cooperative games that are good for beginners",
    "num_results": 3
  }'

# Party games for large groups
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best party games for 6+ players?",
    "num_results": 4
  }'
```

### Raw Similarity Search

```bash
# Search without AI response generation
curl "http://localhost:8000/search?q=worker%20placement&limit=3"

# Quick games search
curl "http://localhost:8000/search?q=quick%20game%20under%2030%20minutes&limit=5"
```

### System Information

```bash
# Health check
curl "http://localhost:8000/health"

# Database statistics
curl "http://localhost:8000/stats"
```

## Example API Response

```json
{
  "query": "I want a strategic game for 2-4 players that takes about 90 minutes",
  "response": "Based on your criteria, I'd recommend these excellent strategic games that work great for 2-4 players and take around 90 minutes:\n\n1. **Terraforming Mars** is perfect for your group size and duration. It combines engine building with strategic planning as you compete to terraform Mars. The complexity is just right for strategic gameplay without being overwhelming.\n\n2. **Wingspan** offers beautiful strategic bird collection with multiple paths to victory. It's elegant, engaging, and consistently takes about 75-90 minutes.\n\n3. **Brass: Birmingham** is a fantastic economic strategy game with deep decision-making. It's more complex but incredibly rewarding for strategy game lovers.\n\nAll of these games have excellent ratings and provide the strategic depth you're looking for!",
  "games": [
    {
      "id": "game_1234",
      "score": 0.85,
      "metadata": {
        "name": "Terraforming Mars",
        "year": 2016,
        "min_players": 1,
        "max_players": 5,
        "play_time": 120,
        "rating": 8.4,
        "complexity": 3.2,
        "mechanics": "Card Drafting, Engine Building, Hand Management",
        "domains": "Strategy Games"
      }
    }
  ],
  "metadata": {
    "num_results": 3,
    "provider": "GeminiProvider"
  }
}
```

## Switching Between Providers

You can easily switch between Gemini and OpenAI:

```bash
# Use Gemini (default)
echo "LLM_PROVIDER=gemini" >> .env
echo "GEMINI_API_KEY=your-key-here" >> .env

# Switch to OpenAI
echo "LLM_PROVIDER=openai" >> .env
echo "OPENAI_API_KEY=your-key-here" >> .env
```

The system will automatically detect and use your preferred provider.

## Project Structure

```
ai-boardgame-suggester/
├── data/
│   └── bgg_dataset.csv          # Your board game dataset
├── src/
│   ├── rag_engine.py           # Core RAG implementation
│   ├── rag_api.py              # FastAPI application
│   ├── rag_models.py           # Pydantic models
│   └── llm_providers.py        # LLM provider abstraction
├── chroma_db/                  # Vector database (created automatically)
├── run_api.py                  # API server launcher
├── demo_rag.py                 # Interactive demo
├── test_providers.py           # Test your API keys
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
└── README.md                  # Full documentation
```

## Troubleshooting

### "No LLM API key found"
- Make sure you've copied `.env.example` to `.env`
- Verify your API key is correctly set
- Run `python test_providers.py` to test your setup

### "RAG system not initialized"
- The system needs 30-60 seconds to process the dataset on first startup
- Check the console logs for initialization progress
- Visit `/health` endpoint to check status

### "ChromaDB errors"
- Delete the `chroma_db` folder and restart
- The system will recreate the vector database

### Import/dependency errors
- Make sure your virtual environment is activated
- Run `pip install --force-reinstall -r requirements.txt`

### Performance issues
- First startup is slower (vector database creation)
- Subsequent starts are much faster
- Consider using a smaller dataset for testing

## Advanced Usage

### Custom Queries
The RAG system handles complex, nuanced queries:

```bash
# Comparison queries
"Find a game like Azul but for more players"

# Specific mechanics
"Show me games with dice rolling that aren't too luck-based"

# Mood-based queries
"I love Wingspan but want something with more player interaction"

# Constraint-based
"Gateway games that take less than 45 minutes for 3-5 players"
```

### API Integration
Use in your applications:

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "Strategic games for 2 players",
    "num_results": 3
})

result = response.json()
print(result["response"])  # AI-generated recommendation
for game in result["games"]:
    print(f"- {game['metadata']['name']}")
```

### Production Deployment
- Set environment variables for your deployment platform
- Use a proper WSGI server like gunicorn
- Consider using a managed vector database for scale
- Set up proper logging and monitoring

## What's Revolutionary About This System

### Before (Traditional Recommendation Systems)
- ❌ Keyword-based search only
- ❌ No natural language understanding
- ❌ Static filtering approaches
- ❌ No explanations for recommendations

### Now (RAG-Powered System)
- ✅ **Semantic understanding** - finds games by meaning
- ✅ **Natural language queries** - ask questions naturally
- ✅ **AI-generated explanations** - understand why games are recommended
- ✅ **Context awareness** - considers multiple factors simultaneously
- ✅ **Conversational interface** - like talking to a board game expert

## Next Steps

1. **Get your free Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Run the test script** to verify everything works
3. **Try the interactive demo** to see the AI in action
4. **Explore the API documentation** at `/docs`
5. **Integrate into your applications** using the powerful REST API

Your board game RAG system is now ready to provide intelligent, context-aware recommendations! 🎲🤖

---

**Built with cutting-edge RAG technology for the future of board game discovery**
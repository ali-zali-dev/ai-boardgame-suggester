# AI Board Game Suggester

A Python-based board game recommendation system powered by Google Gemini AI that helps you discover games based on your preferences.

## Features

- **Smart Filtering**: Find games based on player count, duration, complexity, and ratings
- **AI-Powered Queries**: Use natural language to describe what you want (powered by Google Gemini)
- **Rich Dataset**: Powered by BoardGameGeek data with 20,000+ games
- **REST API**: Simple API with just 2 endpoints
- **Fallback Support**: Works even without AI API key using keyword matching

## Quick Start

### 1. Clone and Setup

```bash
cd ai-boardgame-suggester
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Gemini API (Optional)

1. Get your API key from [Google AI Studio](https://ai.google.dev/)
2. Copy `.env.example` to `.env`
3. Add your API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Setup Data

```bash
python setup.py
```

### 4. Start the API

```bash
python run_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Filter-Based Recommendations

**GET** `/recommendations`

Filter games using specific parameters:

```bash
curl "http://localhost:8000/recommendations?num_players=4&complexity_max=2.5&min_rating=7.0"
```

**Parameters:**
- `num_players`: Number of players (1-20)
- `duration_min/max`: Play time in minutes
- `complexity_min/max`: Complexity (1.0-5.0)
- `min_rating`: Minimum BGG rating (0.0-10.0)
- `max_age`: Maximum recommended age
- `n_recommendations`: Number of results (1-50)
- `sort_by`: Sort by "Rating Average", "Complexity Average", "Year Published", or "BGG Rank"

### 2. AI-Powered Natural Language Queries

**GET** `/query`

Describe what you want in plain English:

```bash
curl "http://localhost:8000/query?q=strategic easy game for 3 players"
```

**Example Queries:**
- "I want a strategic but easy game for 3 players"
- "Find me a quick party game for 6 people under 30 minutes"
- "Suggest complex strategy games for 2 players with high ratings"
- "Family-friendly games with dice rolling for 4 players"
- "Cooperative game that takes about an hour"

## Example Response

```json
{
  "games": [
    {
      "name": "Wingspan",
      "year_published": 2019,
      "min_players": 1,
      "max_players": 5,
      "play_time": 70,
      "min_age": 10,
      "rating_average": 8.1,
      "complexity_average": 2.4,
      "bgg_rank": 15,
      "mechanics": "Card Drafting, Dice Rolling, Hand Management",
      "domains": "Strategy Games"
    }
  ],
  "total_found": 10,
  "query_interpretation": "Looking for strategic, easy games for 3 players"
}
```

## How It Works

### AI-Powered Natural Language Processing

When you provide a natural language query like "strategic easy game for 3 players", the system:

1. **Sends to Gemini AI**: Your query is processed by Google's Gemini model
2. **Extracts Parameters**: AI converts your description into specific filters:
   - "strategic" → `domains: ["Strategy Games"]`
   - "easy" → `complexity_max: 2.5`
   - "3 players" → `num_players: 3`
3. **Finds Games**: Uses extracted filters to search the database
4. **Returns Results**: Provides matching games with interpretation

### Fallback System

If no Gemini API key is provided, the system uses keyword-based parsing that still understands basic terms like:
- Player counts: "3 players", "for 4", "6 people"
- Complexity: "easy", "complex", "medium"
- Duration: "quick", "long", specific minutes
- Types: "strategy", "family", "party"

## Project Structure

```
ai-boardgame-suggester/
├── src/
│   ├── api.py                   # FastAPI REST API with Gemini integration
│   ├── api_models.py            # API request/response models
│   ├── data_preprocessor.py     # Data cleaning and preprocessing
│   └── recommendation_engine.py # Recommendation algorithms
├── data/
│   ├── bgg_dataset.csv          # Board game dataset
│   └── processed_games.csv      # Preprocessed data
├── models/                      # Saved models and preprocessors
├── .env                         # Environment variables (API keys)
├── .env.example                 # Environment template
├── requirements.txt             # Python dependencies
├── setup.py                     # Data setup script
├── run_api.py                   # API server startup
├── demo.py                      # Non-interactive demo
└── test_api.py                  # Simple API tests
```

## Testing

### Test the Implementation

```bash
python test_api.py
```

### Test with Demo

```bash
python demo.py
```

### Manual API Testing

1. Start the server: `python run_api.py`
2. Visit: http://localhost:8000/docs
3. Try the interactive API documentation

## Dependencies

- **pandas**: Data processing
- **scikit-learn**: ML algorithms and similarity calculations
- **fastapi**: REST API framework
- **uvicorn**: ASGI server
- **google-genai**: Google Gemini AI integration
- **python-dotenv**: Environment variable management

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `GEMINI_TEMPERATURE` | `0.3` | AI creativity (0.0-1.0) |

### Model Options

- `gemini-1.5-flash`: Fast, efficient (recommended)
- `gemini-1.5-pro`: More capable, slower
- `gemini-2.0-flash-exp`: Experimental latest model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Data source: [BoardGameGeek](https://boardgamegeek.com/)
- AI powered by: [Google Gemini](https://ai.google.dev/)
- Web framework: [FastAPI](https://fastapi.tiangolo.com/)
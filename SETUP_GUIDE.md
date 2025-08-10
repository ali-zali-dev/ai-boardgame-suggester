# Setup Guide for AI Board Game Recommender

## What We've Built

You now have a streamlined AI-powered board game recommendation API with:

### ✅ **2 Simple Endpoints**
1. **GET `/recommendations`** - Filter-based recommendations
2. **GET `/query`** - AI-powered natural language queries

### ✅ **Google Gemini Integration**
- Uses the latest [Google Gen AI SDK](https://github.com/googleapis/python-genai)
- Intelligent parsing of natural language queries
- Fallback to keyword matching when API key not provided

### ✅ **Clean Architecture**
- Removed interactive CLI mode
- Simplified API with only essential endpoints
- Environment-based configuration

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Setup Gemini API (Optional but Recommended)

1. Get your API key from [Google AI Studio](https://ai.google.dev/)
2. Edit `.env` file:

```bash
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.3
```

### 3. Setup Data

```bash
python setup.py
```

### 4. Start API Server

```bash
python run_api.py
```

## Testing the API

### Test Natural Language Queries

```bash
# Strategic easy game for 3 players
curl "http://localhost:8000/query?q=strategic%20easy%20game%20for%203%20players"

# Quick party game for 6 people
curl "http://localhost:8000/query?q=quick%20party%20game%20for%206%20people"

# Complex strategy games with high ratings
curl "http://localhost:8000/query?q=complex%20strategy%20games%20with%20high%20ratings"
```

### Test Filter-Based Recommendations

```bash
# 4 players, easy complexity, high rating
curl "http://localhost:8000/recommendations?num_players=4&complexity_max=2.5&min_rating=7.0"

# Quick games under 45 minutes
curl "http://localhost:8000/recommendations?duration_max=45&min_rating=7.0"

# Strategy games for 2 players
curl "http://localhost:8000/recommendations?num_players=2&min_rating=7.5&n_recommendations=5"
```

## How Natural Language Processing Works

### With Gemini API Key
1. **User Query**: "I want a strategic easy game for 3 players"
2. **Gemini Processing**: AI converts to structured filters:
   ```json
   {
     "num_players": 3,
     "complexity_max": 2.5,
     "domains": ["Strategy Games"],
     "interpretation": "Looking for strategic, easy games for 3 players"
   }
   ```
3. **Database Search**: Uses filters to find matching games
4. **Results**: Returns games with AI interpretation

### Without API Key (Fallback)
- Uses keyword matching for basic understanding
- Still works for simple queries
- Less sophisticated but functional

## Example API Response

```json
{
  "games": [
    {
      "name": "Splendor",
      "year_published": 2014,
      "min_players": 2,
      "max_players": 4,
      "play_time": 30,
      "min_age": 10,
      "rating_average": 7.4,
      "complexity_average": 1.8,
      "bgg_rank": 89,
      "mechanics": "Card Drafting, Set Collection, Engine Building",
      "domains": "Strategy Games"
    }
  ],
  "total_found": 10,
  "query_interpretation": "Looking for strategic, easy games for 3 players"
}
```

## API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Virtual Environment Issues
If you have import issues:
```bash
deactivate
source .venv/bin/activate
pip install --force-reinstall -r requirements.txt
```

### Missing API Key
The system works without a Gemini API key using fallback parsing. To get the full AI experience:
1. Visit https://ai.google.dev/
2. Create an API key
3. Add it to your `.env` file

### Data Setup
If recommendations aren't working:
```bash
python setup.py  # Re-run data preprocessing
```

## What's Different Now

### Before (Old System)
- Interactive command-line interface
- Multiple complex endpoints
- Manual regex parsing
- Cluttered codebase

### After (New System)
- ✅ **2 simple endpoints only**
- ✅ **AI-powered natural language understanding**
- ✅ **Clean, focused codebase**
- ✅ **Environment-based configuration**
- ✅ **Fallback support**

## Next Steps

1. **Get your Gemini API key** for the full AI experience
2. **Test the endpoints** using the examples above
3. **Explore the interactive documentation** at `/docs`
4. **Integrate into your applications** using the simple REST API

Your board game recommendation system is now ready for production use! 🎲
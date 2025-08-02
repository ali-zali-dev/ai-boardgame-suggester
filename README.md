# AI Board Game Suggester

A Python-based board game recommendation system that helps you discover games based on your preferences.

## Features

- **Smart Filtering**: Find games based on player count, duration, complexity, and ratings
- **Similarity Search**: Discover games similar to ones you already love
- **Rich Dataset**: Powered by BoardGameGeek data with thousands of games
- **Interactive Interface**: Easy-to-use command-line interface

## Phase 1: MVP (Minimum Viable Product) ✅

### Completed Features:
1. **Project Setup**: Proper folder structure and dependency management
2. **Data Preprocessing**: 
   - Data cleaning and normalization
   - One-hot encoding for categories and mechanics
   - Feature engineering for better recommendations
3. **Basic Recommendation Logic**:
   - Filtering-based recommendations
   - Input: number of players, duration, complexity preferences
   - Output: top-N matching games with detailed information

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip

### Quick Start

1. **Clone and navigate to the project**:
   ```bash
   cd ai-boardgame-suggester
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the system**:
   ```bash
   python setup.py
   ```

5. **Run the application**:
   ```bash
   python src/main.py
   ```

## Project Structure

```
ai-boardgame-suggester/
├── data/
│   └── bgg_dataset.csv          # Board game dataset
├── src/
│   ├── data_preprocessor.py     # Data cleaning and preprocessing
│   ├── recommendation_engine.py # Recommendation algorithms
│   └── main.py                  # Main application interface
├── models/                      # Saved models and preprocessors
├── requirements.txt             # Python dependencies
├── setup.py                     # Setup script
└── README.md
```

## Usage

### Interactive Mode

Run the main application for an interactive experience:

```bash
python src/main.py
```

The system provides several options:
1. **Get personalized recommendations** - Enter your preferences and get tailored suggestions
2. **Find similar games** - Enter a game name to find similar titles
3. **Browse mechanics** - See popular game mechanics
4. **Browse domains** - See available game categories

### Example Preferences

- **Number of players**: 4
- **Duration**: 60-120 minutes
- **Complexity**: 2-3.5 (on a 1-5 scale)
- **Minimum rating**: 7.0
- **Mechanics**: Card Drafting, Worker Placement
- **Domains**: Strategy Games

### Sample Output

```
=== Found 5 Recommendations ===

1. Wingspan (2019)
   Players: 1-5
   Time: 70 minutes
   Age: 10+
   Rating: 8.1 (Rank #23)
   Complexity: 2.4/5
   Mechanics: Card Drafting, End Game Bonuses, Hand Management...

2. Splendor (2014)
   Players: 2-4
   Time: 30 minutes
   Age: 10+
   Rating: 7.4 (Rank #89)
   Complexity: 1.8/5
   Mechanics: Card Drafting, Set Collection, Engine Building...
```

## How It Works

### 1. Data Preprocessing
- Cleans and normalizes the BoardGameGeek dataset
- Handles missing values and data type conversions
- Creates one-hot encoded features for mechanics and domains
- Normalizes numerical features (ratings, complexity, play time)

### 2. Filtering System
- Filters games based on user criteria:
  - Player count compatibility
  - Duration preferences (min/max)
  - Complexity range
  - Minimum rating threshold
  - Preferred mechanics and domains

### 3. Recommendation Engine
- **Filtering-based**: Returns games matching user criteria
- **Similarity-based**: Uses cosine similarity to find games similar to a reference game
- **Ranking**: Sorts results by rating, popularity, or other metrics

## Dataset

The system uses the BoardGameGeek dataset containing:
- Game names, years, and basic info
- Player count ranges and recommended ages
- Play times and complexity ratings
- User ratings and rankings
- Game mechanics and categories
- Thousands of board games from various genres

## Future Enhancements

- **Collaborative Filtering**: Recommend based on similar users' preferences
- **Content-Based Filtering**: Enhanced similarity using game descriptions
- **Hybrid Approaches**: Combine multiple recommendation strategies
- **Web Interface**: Replace command-line with web-based interface
- **User Profiles**: Save preferences and recommendation history
- **Advanced Filters**: Publisher, designer, year ranges, etc.

## Dependencies

- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms and preprocessing
- `numpy`: Numerical computations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Data provided by BoardGameGeek community
- Inspired by the need to discover great board games efficiently
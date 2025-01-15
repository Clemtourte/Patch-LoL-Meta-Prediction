# League of Legends Meta Prediction using Deep Learning

## Context
This project aims to develop a deep learning model to predict how meta shifts will evolve in League of Legends based on patch notes. It builds upon previous work on retrieving and analyzing League of Legends data, expanding it to include predictive modeling of game meta trends.

## Objective
Develop a deep learning model to predict how meta shifts will evolve in League of Legends based on patch notes.

## Data Sources
- Riot Games patch notes
- League of Legends match data (via Riot Games API)
- Web Scrapping

## Focus Areas
- Analyze the impact of patch changes on champion picks, player strategies, and overall meta trends
- Apply NLP techniques for text analysis of patch notes
- Develop deep learning models for meta predictions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip for installing Python packages
- An API key from Riot Games (create a `.env` file with `API_KEY=your_api_key`)
- API keys or authentication for social media platforms (Twitter, Reddit)

### Steps

1. Create and activate a virtual environment:
   ```
   python -m venv env
   use `env\Scripts\activate.bat` in cmd
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```
   API_KEY=your_riot_api_key
   ```

### Viewing Results / Usage
#### View the database schema
https://dbdiagram.io/d/league_datadb-67879ea86b7fa355c3f52325


#### View in streamlit
To view the results and predictions using the Streamlit app:
```
streamlit run app.py
* Search for User Info: Enter a League of Legends username and tag to get the player's current ranked status.
* Recent Matches: View details about recent ranked matches, including game duration, patch, participants, and their performance stats.
* Expandable Participant Details: Click on a player's name to see more detailed information about their performance in each match.
```

## Contributing
This project is part of a master's thesis. While direct contributions are not accepted, feedback and suggestions are welcome. Please open an issue to discuss any ideas or concerns.

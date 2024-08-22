# DeepLearning Project

## Context
This repository is used to get introductory knowledge to MachineLearning, DeepLearning, Neural Networks concepts and many more starting from zero.
It will be centered around Python and will tend to grow in the future

The main project is about creating a model around the League Of Legends game. Data will either be retrieved through the LOL API or the Leaguepedia API (maybe other various sources).
After the development of a first app that will be focused on retrieving data, I will start working on the machine/deep Learning part.

# League of Legends User Information App

## Description

This application retrieves and displays user information and recent match data from League of Legends using Riot Games' API. It provides details such as ranked information and match statistics, including player performance metrics and game summaries.

## Features

- **Search for User Info**: Enter a League of Legends username and tag to get the player's current ranked status.
- **Recent Matches**: View details about recent ranked matches, including game duration, patch, participants, and their performance stats.
- **Expandable Participant Details**: Click on a player's name to see more detailed information about their performance in each match.

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` for installing Python packages
- An API key from Riot Games (create a `.env` file with `API_KEY=your_api_key`)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Clemtourte/DeepLearning.git
   cd your-repo

2. **Create a Virtual Environment and activate it**
   
   pip install virtualenv
   python -m venv env
   env\Scripte\activate.bat in cmd
   
3. **Install dependencies**

   pip install -r requirements.txt
   
4. **Set up environment variables**

  Create a .env file in the root directory of the project and add your Riot Games API key:
  API_KEY=your_api_key

## Usage

### Running the application

To start the Streamlit application, use the following command:
  streamlit run app.py

## Code Overview
**app.py**: The Streamlit application that provides the user interface for interacting with the Riot Games API and displaying match and user information.
**main.py**: Contains the core logic for interacting with the Riot Games API, including fetching user and match data, processing responses, and saving match data to a JSON file.

## Useful Links 
<li> https://victorzhou.com/blog/intro-to-neural-networks/ </li>
<li> https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/ </li>
<li> https://www.linkedin.com/posts/jean-m-595778118_data-python-projet-activity-7185574695893671937-Mo46?utm_source=share&utm_medium=member_desktop </li>
<li> https://machinelearnia.com/wp-content/uploads/2019/11/Apprendre-le-ML-en-une-semaine.pdf </li>

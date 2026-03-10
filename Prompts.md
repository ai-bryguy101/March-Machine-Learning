Prompts I entered after feeding Claude all of the competition peramitters:

Great I created the git and uploaded these files to it now lets start conceptualizing and planning how we are going to attack this March madness is notoriously random 
so while I want to use a strong model and find good patterns I want to most heavily rely on the outcomes and data of this current season and then enter in a big randomness factor 

Really think about this there is no way to account for all of the little random factors that effect the outcome of games so really what I want to build is a prediction machine for who 
which team wins a head to head matchup based on this seasons statistics and trends then add a weight of typicall trends across the past season and finally a bid dose of randomness for 
March Madness!!!

Let's brainstorm what the framework should be I'm wondering if we should go with ML and algorithms trying to factor in all of the data points to try to best predict OR go with how AI kind of came to be and just use a base model and create a type of AI for this specific case so we just tokenize the teams by their seasonal statistics add attention based on how they performed against all the other teams and feed that into the neural network to predict which token (team) comes out on top then run that through the loss adjust weights until the neural network hopefully finds some kind of pattern humans can't understand to accurately predict  kind of like a real AI? What do you think what approach should we have and what code should we start writing right now?

I want to think more, I'm coming from the perspective that this predicting is pretty much impossible there are too many factors like injuries and random world events that happen that have made it so there has never been a 100% accurate bracket and honestly the most successful ones were just completely random so why do you think traditional ways would work on this? I bet most of ESPNs predictions and modeling has come from what you suggested and they don't seem to work so why not try to tap into the magic of AI maybe scale it down to fit this smaller data or get creative and scale the data up but let the neural network unravel a pattern that makes no sense to us but somehow is weighted in such a way that we get accurate predictions?


Explain what you are thinking more? I want the teams to be embedded by their performance metrics, then the "attention" in this neural network would be how they performed against the other teams they had already played that season so we step through each game every season predicting the winner of matchups learn from that move up a game then add attention based on past matchups. I want the vector embeddings to be very high dimensionally

march-madness-2026/
├── configs/default.yaml        # Hyperparameters
├── data/{raw,processed,submissions}/
├── docs/{ARCHITECTURE,STRATEGY,PROMPTS}.md
├── src/
│   ├── data/preprocessing.py   # ✅ Fixed
│   ├── models/
│   │   ├── team_encoder.py     # ✅ Improved
│   │   ├── game_processor.py   # ✅ Fixed
│   │   ├── attention_matchup.py # 🆕 NEW
│   │   ├── prediction_head.py   # 🆕 NEW
│   │   └── marchnet.py          # 🆕 NEW
│   └── training/pretrain.py     # 🆕 NEW
├── PROJECT_LOG.md
├── README.md
└── requirements.txt            # ✅ Fixed

⏭️ Next Steps

Download the zip and replace your GitHub repo contents
Download Kaggle data into data/raw/
Run preprocessing: python -m src.data.preprocessing data/raw/
Run pre-training: python -m src.training.pretrain --epochs 20
Still need: finetune.py, calibrate.py, generate_submission.py

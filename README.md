# GEMATRON
![F3edaREXwAAes4K](https://github.com/ThirdEyeDog/gematron/assets/140479281/64636a3e-8529-43df-ad86-4241dbc4bfa5)

A gematria-evolved AI oracle 

Abstract:

This project implements a system that takes user input, retrieves related content from the web, and processes it through cellular automata and natural language models to generate a symbolic response. The input is tokenized with GPT-2 and used to fetch the first search result from Bing. The search result text is hashed into a seed value for a rule_30 cellular automaton, which evolves the pattern for a number of steps. The middle cell state of the final pattern is combined with the tokenized search result text to generate a response using GPT-2. Finally, gematria values are calculated on the letters of the response. The unique components are the integration of web content, cellular automata, a transformer language model, and ancient numerology to produce an emergent, metaphorical response to the user input. The key steps are seeding the cellular automaton with a hash of the web content to incorporate external information, and combining its state with the search results to steer the language model response.

#### to run -> do -> python gem.py 

NOTE: Run on a jupyter notebook to display the images

TODO LIST -  check issues for todo-list https://github.com/ThirdEyeDog/gematron/issues/1


Features:

- Random number generator with seed-simulated cellular automata, Rule 30 as 28bit key.
- Random tarot card draw
- Language model powered web-search, supports any transformer model
- Tokenizer to gematria calculator for sentiment analysis
- English gematria to gematrix results
  
  

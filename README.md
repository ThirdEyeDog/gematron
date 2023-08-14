# GEMATRON

A gematria-evolved AI oracle 

Abstract:

This project implements a system that takes user input, retrieves related content from the web, and processes it through cellular automata and natural language models to generate a symbolic response. The input is tokenized with GPT-2 and used to fetch the first search result from Bing. The search result text is hashed into a seed value for a rule_30 cellular automaton, which evolves the pattern for a number of steps. The middle cell state of the final pattern is combined with the tokenized search result text to generate a response using GPT-2. Finally, gematria values are calculated on the letters of the response. The unique components are the integration of web content, cellular automata, a transformer language model, and ancient numerology to produce an emergent, metaphorical response to the user input. The key steps are seeding the cellular automaton with a hash of the web content to incorporate external information, and combining its state with the search results to steer the language model response.

TODO LIST -  check issues for todo-list https://github.com/ThirdEyeDog/gematron/issues/1

This shows the key steps of hashing the web content into a cellular automaton seed, evolving the pattern, extracting the middle cell state, and using it alongside the search result text to generate the final response. The integration of different techniques is what makes this project unique.

```python
# Convert search result text to cellular automaton seed 
hash_object = hashlib.md5(search_result_text.encode())
hex_dig = hash_object.hexdigest()
seed_value = int(hex_dig, 16) % 256
pattern = [int(b) for b in format(seed_value, '08b')]

# Evolve cellular automaton and extract middle cell
pattern = rule_30(pattern) 
ca_response = pattern[len(pattern) // 2]

# Generate response using CA state and search results
combined_seed = f"{ca_response} {gpt2_tokenizer.decode(search_result_tokenized[0])}" 
response_from_gpt2 = generate_gpt2_response(combined_seed)
```



Main Flow:

- Take user input.
- Tokenize it using GPT-2's tokenizer.
- Fetch the web search result of the input text.
- Tokenize the search result.
- Convert the search result into a seed value for cellular automata through MD5 hashing.
- Apply the rule_30 function on the seed pattern.
- Use the cellular automata result and the tokenized search result as a combined seed to generate a response from GPT-2.
- Compute the gematria value of the GPT-2 generated response.
- Output the combined seed based on the cellular automata result and the original user input.


Features:

- Random number generator with seed-simulated cellular automata, Rule 30.
- language model powered web-search, supports any transformer model
- tokenizer to gematria calculator for sentiment analysis 
  

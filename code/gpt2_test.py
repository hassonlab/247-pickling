from transformers import GPT2LMHeadModel, GPT2Tokenizer

CACHE_DIR = '/scratch/gpfs/hgazula/.cache/'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl',
                                          add_prefix_space=True,
                                          cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

lm_model = GPT2LMHeadModel.from_pretrained("gpt2-xl",
                                           output_hidden_states=True,
                                           cache_dir=CACHE_DIR)
lm_model.eval()

sentences = [
    "i'm asking because i wanna measure it you finish the iced tea and this"
]
tokens = tokenizer.tokenize(sentences[0])
ids = tokenizer.convert_tokens_to_ids(tokens)
tok_to_str = tokenizer.convert_tokens_to_string(tokens[18])
print(tokenizer.encode(sentences[0]))
print(tokenizer.decode(ids[18]))

# sentences = ['Hello',
#  'Hello world',
#  'Hello world there',
#  'Hello world there you',
#  'Hello world there you are',
#  'world there you are high'
#  ]

input_ids = tokenizer(sentences, padding=True, return_tensors='pt')

lm_outputs = lm_model(**input_ids)

transformer_hidden_states = lm_outputs[-1]
print(transformer_hidden_states[-1].shape)
print(transformer_hidden_states[-1])

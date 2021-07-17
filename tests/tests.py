import unittest

from code.tfsemb_main import tokenize_transcript
from transformers import GPT2Tokenizer

CACHE_DIR = '/scratch/gpfs/hgazula/.cache/'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl',
                                          add_prefix_space=True,
                                          cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token


def zaid_tokenizer(transcript):
    tokens = []
    with open(transcript, 'r') as fp:
        for line in fp:
            tokens.extend(
                tokenizer.tokenize(line.strip(), add_prefix_space=True))
    return tokens


class Test(unittest.TestCase):
    def test_tokenize_podcast_transcript(self):
        transcript = '/scratch/gpfs/hgazula/247-pickling/data/podcast/podcast-transcription.txt'

        data = tokenize_transcript(transcript)

        tokens = [tokenizer.tokenize(item) for item in data]
        tokens = [item for sublist in tokens for item in sublist]

        self.assertTrue(tokens, zaid_tokenizer(transcript))


if __name__ == '__main__':
    unittest.main()

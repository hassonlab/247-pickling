# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 02:29:07 2023

@author: arnab
"""

import os
import torch
import pickle
import numpy as np
import pandas as pd
import whisper
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)

def load_whisper_model_by_path(model_path, checkpoint):

    processor = WhisperProcessor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path)

    model_path = os.path.join(model_path, f"checkpoint-{checkpoint}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        output_hidden_states=True,
        return_dict=True,
    )

    return model, processor, tokenizer

model_path='/scratch/gpfs/arnab/fine_tune_whisper/data/625/saved_models'
checkpoint=2000 # changeable parameter

model, processor, tokenizer = load_whisper_model_by_path(model_path, checkpoint)
import torch

concat_logits = []

for i in range(0,876):
    x = torch.randn(8, 51865)
    concat_logits.append(x)

print(len(concat_logits))
print(concat_logits[0].shape)

prediction_scores = torch.cat(concat_logits, axis=0)

print(len(prediction_scores))

print(prediction_scores[0].shape)

print(prediction_scores.shape[0])

token_ids = []

for i in range(0,7001):
    x = torch.randn()
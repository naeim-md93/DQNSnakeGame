

def MAELoss(logits, target):
    error = (target - logits).abs()
    return error

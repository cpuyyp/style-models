# README BEFORE RUN
**Before run the notebooks in this folder, must find the source code of the functions below and modify them to work with dict as below**  

These functions are in \<your-env\>/site-packages/transformers/trainer_pt_utils.py. Search for the function names and replace them with the code below.

These modification (to my best knowledge) do not conflict other existing packages. 

```python
def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, dict): ### JOEY EDITTED
        return {k: nested_concat(tensors[k], new_tensors[k], padding_index=padding_index) for k in tensors}
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    elif isinstance(tensors, dict): ### JOEY EDITTED
        return {k: nested_numpify(tensors[k]) for k in tensors}
    return tensors.cpu().numpy()

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, dict): ### JOEY EDITTED
        return {k:nested_detach(tensors[k]) for k in tensors}
    return tensors.detach()

def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    elif isinstance(tensors, dict): ### JOEY EDITTED
        return {k: nested_truncate(tensors[k], limit) for k in tensors}
    return tensors[:limit]
```

# Difference to the other multitask training code
The model and dataloader in this folder uses huggingface trainer to train, while the other one customize the training loop.
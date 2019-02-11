- in v3 we trained model for 10 epochs;
- in v4 we used pre-trained model `model_kiank.h5`; that's the first bug - file
`model.h5` is for updated version of this notebook with 5 layers;
- the second bug - we have to change one line of code - see below:
```python
    # source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)), ndmin=3)
```
To generate resnet18 embeddings, use command:
```
python3 generate_resnet18_embeddings.py <data_path> <file_prefix>
```
It will download dataset from Kaggle, embedd it with resnet18, and save embeddings and labels in separate files at the ```data_path``` folder. Files will be saved with ```file_prefix``` prefixes.
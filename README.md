## Pytorch Implementation of Deep Learning Models/Algorithms

This repositiory is for implementing and training/testing popular model architectures on the `CIFAR10` dataset.



## Requirements

- CUDA Version: 10.2 

```
torch==1.5.0
torchvision==0.6.0
numpy == 1.19.2
```

## Usage

### Training

To train a model, run `train.py`.
If you need to speicfy the model, just use some args.

```
# train alexnet model with using gpu. 50 epochs
$ python train.py --model alexnet --epoch 50 -gpu
```

optional&required arguments

```
--data_dir      default='./data/train',
                help="Directory containing the dataset"
--model         required=True, default='alexnet',
                help="The model you want to train"
--lr            default=0.001,
                help="Learning rate"
--epoch         default=50,
                help="Total training epochs"
--batch_size    default=256,
                help="batch size"
--gpu           action='store_true', default='False',
                help="GPU available"
```


### Evaluate

To evaluate the model, run `evaluate.py`.
If you need to speicfy the model, just use some args.

```
# evaluate alexnet model
$ python evaluate.py --model alexnet --weights ./results/alexnet/best.pth 
```

optional&required arguments

```
--data_dir      default='./data/test',
                help="Directory containing the dataset"
--model         required=True, default='alexnet',
                help="The model you want to test"
--weight        required=True,
                help="The weights file you want to test"
--batch_size    default=256,
                help="batch size"
--gpu           action='store_true', default='False',
                help="GPU available"
```


## Results

|Network|epoch|lr|top1@prec(test)|ModelSize(MB)|
|:---:|:---:|:---:|:---:|:---:|
|AlexNet|50|0.001|74.2578%|266MB|
|VGG|-|-|-|-|-|-|
|ResNet|-|-|-|-|-|-|
|Inception|-|-|-|-|-|-|
|GoogLeNet|-|-|-|-|-|
|-|-|-|-|-|-|
|-|-|-|-|-|-|
|-|-|-|-|-|-|
|-|-|-|-|-|-|




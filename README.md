# Generative Adversarial Networks for celebrity face generation

In this repository, a DCGAN has been defined and trained on a dataset of faces. The main goal here is to get a generator network to generate new images of faces that look as realistic as possible!

The project is broken down into a series of tasks from loading in data to defining and training adversarial networks. At the end of the notebook, the results of the trained Generator has been visualized to see how it performs; the generated samples should look like fairly realistic faces with small amounts of noise.

## Dataset

The main dataset that we will be using to train the adversarial network is [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below. ![image](https://user-images.githubusercontent.com/45627032/180857790-ff020131-7bc0-4ef0-b9c1-da0659554a6b.png)

## Model Definition

A GAN is comprised of two adversarial networks, a discriminator and a generator.

### Discriminator

```
def conv(in_channels, out_channels, kernel_size=4, stride=2, padding = 1, batch_norm= True):
    """
    This function generates convolutional layer that offers optional batch normalization
    """
    layers = []
    transpose_conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                       kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        
        # complete init function
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2)
        self.conv3 = conv(conv_dim*2, conv_dim*4)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, self.conv_dim * 4 * 4 * 4)
        x = self.fc(x)
        
        return x
```

### Generator
```
def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding = 1, batch_norm= True):
    """
    This function generates transposed-convolutional layer that offers optional batch normalization
    """
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels,
                       kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim * 4 * 4 * 4)
        self.t_conv1 = deconv(conv_dim*4, conv_dim*2)
        self.t_conv2 = deconv(conv_dim*2, conv_dim)
        self.t_conv3 = deconv(conv_dim, 3, batch_norm=False)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*4, 4, 4)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = self.t_conv3(x)
        x = F.tanh(x)
        
        return x
```

## Weight Initialization

To help your models converge, we should initialize the weights of the convolutional and linear layers in the model. From reading the [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf), they say:

#### All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

So, we next define a weight initialization function that does just this!

```
def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
    if (hasattr(m, 'weight')) and ((classname.find('Conv') != -1) or (classname.find('Linear') != -1)):
        nn.init.normal(m.weight.data, 0.0, 0.2)
        
    if (hasattr(m, 'bias')) and (m.bias is not None):
        nn.init.constant(m.bias.data, 0.0)
```

## Results

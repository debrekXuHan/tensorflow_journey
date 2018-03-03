import mnist_softmax
import mnist_conv_nn

if __name__ == "__main__":
    print("Prosess MNIST dataset by Softmax Model.")
    mnist_softmax.mnist_softmax()

    print("Prosess MNIST dataset by Convolution Neural Network.")
    mnist_conv_nn.mnist_conv_nn()
    
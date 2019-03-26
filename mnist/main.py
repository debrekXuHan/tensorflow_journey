import argparse
import mnist_softmax
import mnist_conv_nn

if __name__ == "__main__":
    print("Prosess MNIST dataset by Convolution Neural Network.")
    parser = argparse.ArgumentParser(description='Train on MNIST.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    args = parser.parse_args()

    if args.command == "train":
        print("Training process...")
        mnist_conv_nn.mnist_conv_nn_train()
    elif args.command == "test":
        print("Testing process...")
        test_image = "./image/test.png"
        image = mnist_conv_nn.image_trans(test_image)
        mnist_conv_nn.mnist_conv_nn_test(image)
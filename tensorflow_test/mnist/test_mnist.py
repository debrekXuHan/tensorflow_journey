'''
    use python to analyze binary files
'''
import numpy as np
import struct
import matplotlib.pyplot as plt

def loadImageSet(filename):

    binfile = open(filename, 'rb') # read the binary file
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0) # take the first 4 integers

    offset = struct.calcsize('>IIII') # locate at the beginning of image data
    imgNum = head[1]
    width  = head[2]
    height = head[3]

    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B' # fmt format: '>47040000B'
    
    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape into a [60000, 784] array
    
    return imgs, head

def loadLabelSet(filename):

    binfile = open(filename, 'rb') # read the binary file
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0) # take the first 2 integers

    offset = struct.calcsize('>II') # locate at the beginning of label data
    labelNum = head[1]

    numString = '>' + str(labelNum) + 'B' # fmt format: '>60000B'  
    labels = struct.unpack_from(numString, buffers, offset)

    binfile.close()
    labels = np.reshape(labels, [labelNum]) # reshape into a [60000] array
    
    return labels, head

if __name__ == "__main__":
    img_file = 'MNIST_data/train-images.idx3-ubyte'
    label_file = 'MNIST_data/train-labels.idx1-ubyte'

    index = 120

    # print the label of a certain data by index
    labels, labels_head = loadLabelSet(label_file)
    print('labels_head: ', labels_head)
    print(type(labels))
    print(labels[index])

    # show the image of a certain data by index
    print('------------------')
    imgs, imgs_head = loadImageSet(img_file)
    print('imgs_head: ', imgs_head)
    print(type(imgs))
    fig = plt.figure()    
    plt.imshow(np.reshape(imgs[index, :], [28, 28]), cmap = 'binary') # display in black and white   
    plt.show()
    
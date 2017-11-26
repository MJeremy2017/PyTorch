import cPickle

image1 = open('train/1.png', 'rb')

image1 = cPickle.load(image1)


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dic = cPickle.load(fo)
    return dic


unpickle('train/1.png')

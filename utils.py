import tensorflow as tf
import numpy as np

def lrelu(h, alpha=.1):
    """Leaky relu activation function 

    Args
    ----
    h (Tensor) : the weight matrix to perform the activation on 
    alpha (float) : rate of decay when negative

    Returns
    -------
    Tensor
    """
    return tf.maximum(alpha*h, h)


def pix_to_grid(x, y, S=7, im_size=208):
    """Find the grid location of a given x,y coordinate location

    Assumes square image

    Arg
    ---
    x (int) : x pixel location 
    y (int) : y pixel location 
    S (int) : grid subdivisions 
    im_size (int) : number of pixels in on dimension of the image

    Returns
    -------
    i (int, between 0-6) : i grid location 
    j (int, between 0-6) : j grid location 
    """
    alpha_x = alpha_y = S/im_size
    return (np.floor(alpha_x * x).astype(np.int32), 
            np.floor(alpha_y * y).astype(np.int32))


def iou(B1, B2):
    """Find the Intersection of Union

    Arg
    ---
    B1 [x, y, w, h]
    B2 [x, y, w, h]

    Returns
    -------
    float
    """

    _l1 = x1 - w1/2
    _l2 = x2 - w2/2
    _r1 = x1 + w1/2
    _r2 = x2 + w2/2

    _t1 = y1 + h1/2
    _b1 = y1 - h1/2
    _t2 = y2 + h2/2
    _b2 = y2 - h2/2

    x_overlap = np.maximum(
            np.minimum(_r1, _r2) - np.maximum(_l1, _l2),
            0
            )
    y_overlap = np.maximum(
            np.minimum(_b1, _b2) - np.maximum(_t1, _t2),
            0
            )

    overlap_area = x_overlap * y_overlap;
    total_area = w1*h1 + w2*h2 - overlap_area

    return overlap_area / np.maximum(total_area, 1e-10)

def maximum(tensor1, tensor2, axis):
    """Find the maximum value along an axis between two tensors

    Args
    ----
    tensor1 (Tensor) : tensor to perform operation on 
    tensor2 (Tensor) :  tensor to perform operation on 
    axis (int) : axis to perform the operation along

    Returns
    -------
    a tensor of the same shape except the dimension of the axis the operation 
    was performed along is 1. ie input is two tensors of 1 x 7 x 7 x 10
    then output is 1 x 7 x 7 x 1 if axis = 3
    """
    _tensor = tf.concat(
            [tensor1, tensor2],
            axis=axis)

    shape = _tensor.get_shape().as_list()
    shape[shape==None] = -1
    shape[axis] = 1

    return tf.reshape(
            tf.reduce_max(
                _tensor,
                axis=axis),
            shape=shape)

def minimum(tensor1, tensor2, axis):
    """Find the minimum value along an axis between two tensors

    Args
    ----
    tensor1 (Tensor) : tensor to perform operation on 
    tensor2 (Tensor) :  tensor to perform operation on 
    axis (int) : axis to perform the operation along

    Returns
    -------
    a tensor of the same shape except the dimension of the axis the operation 
    was performed along is 1. ie input is two tensors of 1 x 7 x 7 x 10
    then output is 1 x 7 x 7 x 1 if axis = 3
    """
    _tensor = tf.concat(
            [tensor1, tensor2],
            axis=axis)

    shape = _tensor.get_shape().as_list()
    shape[shape==None] = -1
    shape[axis] = 1

    return tf.reshape(
            tf.reduce_min(
                _tensor,
                axis=axis),
            shape=shape)


def tf_iou(x, y, w, h, x_hat, y_hat, w_hat, h_hat):
    """
    Args
    ----
    x (tensor) :
    y (tensor) :
    w (tensor) :
    h (tensor) :
    x_hat (tensor) :
    y_hat (tensor) :
    w_hat (tensor) :
    h_hat (tensor) :

    Returns
    -------

    """
    l = x - w/2
    r = x + w/2
    t = y + h/2
    b = y - h/2

    l_hat = x_hat - w_hat/2
    r_hat = x_hat + w_hat/2
    t_hat = y_hat + h_hat/2
    b_hat = y_hat - h_hat/2

    _x_overlap = []
    _y_overlap = []

    zeros = tf.zeros_like(x, dtype=tf.float32)

    for j in range(2):
        _x_overlap.append( 
                maximum(
                    minimum(
                        slice_and_keep_dims(r, axis=3, index=j), 
                        slice_and_keep_dims(r_hat, axis=3, index=j),
                        axis=3) \
                    - maximum(
                        slice_and_keep_dims(l, axis=3, index=j), 
                        slice_and_keep_dims(l_hat, axis=3, index=j),
                        axis=3),
                    zeros,
                    axis=3
                ))

        _y_overlap.append( 
                maximum(
                    minimum(
                        slice_and_keep_dims(b, axis=3, index=j), 
                        slice_and_keep_dims(b_hat, axis=3, index=j),
                        axis=3) \
                    - maximum( 
                        slice_and_keep_dims(t, axis=3, index=j),
                        slice_and_keep_dims(t_hat, axis=3, index=j),
                        axis=3),
                    zeros,
                    axis=3
                ))

    x_overlap = tf.concat(_x_overlap, axis=3)
    y_overlap = tf.concat(_y_overlap, axis=3)

    overlap_area = x_overlap * y_overlap;
    total_area = w*h + w_hat*h_hat - overlap_area

    
    _epsilon = tf.ones_like(total_area, dtype=tf.float32) * 1e-10
    return overlap_area / maximum(total_area, _epsilon, axis=3)

def list_slice(tensor, indicies, axis):
    """
    """

    slices = []   

    for i in indicies:   
        _slice = slice_and_keep_dims(tensor, index=i, axis=axis)
        slices.append(_slice)

    return tf.concat(slices, axis=3)


def tile_slice(tensor, index, axis, number):
    """
    """
    tensor_slice = slice_and_keep_dims(tensor, index, axis)
    tensor_list = [tensor_slice]*number

    return tf.concat(tensor_list, axis)


def slice_and_keep_dims(tensor, index, axis):
    """Slice a tensor but keep the dimensions the same
    """

    ## Set the shape of the output tensor. 
    # Set any unknown dimensions to -1, so that reshape can infer it correctly. 
    # Set the dimension in the slice direction to be 1, so that overall dimensions are preserved during the operation
    shape = tensor.get_shape().as_list()
    shape[shape==None] = -1
    shape[axis] = 1

    # build the slice operation
    nd = len(shape)
    _slice = [slice(None)]*nd
    _slice[axis] = slice(index,index+1)

    # reshape to the original number of dimensions and return 
    return tf.reshape(tensor[_slice], shape)
    



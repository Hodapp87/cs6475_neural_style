#!/usr/bin/env python2

import caffe
import numpy
import skimage
import scipy.optimize
import cv2

# Initialize caffe to run on GPU 0:
caffe.set_device(0)
caffe.set_mode_gpu()
# The alternative is something like:
# caffe.set_mode_cpu()

# Load style and content images 
#content = caffe.io.load_image("./Roundtanglelake.jpg")
#style   = caffe.io.load_image("./escher_sphere.jpg")
#content = caffe.io.load_image("./brad_pitt.jpg")
#style   = caffe.io.load_image("./picasso_selfport1907.jpg")
content = caffe.io.load_image("13681848113_c41cd968d4_k.jpg")
style   = caffe.io.load_image("1280px-Great_Wave_off_Kanagawa2.jpg")

# Load Caffenet model (source:
# http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel):
# TODO: Where does deploy.prototxt come from?
net = caffe.Net("caffenet/deploy.prototxt",
                "caffenet/bvlc_reference_caffenet.caffemodel",
                caffe.TEST)

# For VGG-19 and the like:
#style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
#content_layers = ["conv4_2"]
style_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
content_layers = ["conv4"]
layers = [l for l in net.blobs if l in content_layers or l in style_layers]

# This file is from:
# https://github.com/BVLC/caffe/tree/master/python/caffe/imagenet
mean_data = numpy.load("caffenet/ilsvrc_2012_mean.npy")

# Normalize input data to the appropriate range for what this neural
# net was trained on:
xform = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
xform.set_mean("data", mean_data.mean(1).mean(1))
xform.set_channel_swap("data", (2,1,0))
xform.set_transpose("data", (2,0,1))
xform.set_raw_scale("data", 255)

# Scale both images to a certain size:
size = 512
style_scale = 1.2
# The below just gets the *longest* edge to 'size' pixels wide.
#content_scaled = skimage.transform.rescale(
#    content, float(size) / max(content.shape[:2]))
#style_scaled = skimage.transform.rescale(
#    style, style_scale * float(size) / max(style.shape[:2]))
#content_scaled = scipy.misc.imresize(
#    content, float(size) / max(content.shape[:2]))
#style_scaled = scipy.misc.imresize(
#    style, style_scale * float(size) / max(style.shape[:2]))

f = float(size) / max(content.shape[:2])
content_scaled = cv2.resize(content, (0, 0), fx=f, fy=f)
f = style_scale * float(size) / max(style.shape[:2])
style_scaled = cv2.resize(style, (0, 0), fx=f, fy=f)

def net_forward(img):
    ch, w, h = img.shape[2], img.shape[0], img.shape[1]
    net.blobs["data"].reshape(1, ch, w, h)
    xform.inputs["data"] = (1, ch, w, h)
    
    net.blobs["data"].data[0] = xform.preprocess("data", img)
    net.forward()

def get_content_repr(img):
    resp = {}
    for layer in layers:
        act = net.blobs[layer].data[0].copy()
        act.shape = (act.shape[0], -1)
        resp[layer] = act
    return(resp)

def get_style_repr(img):
    resp = {}
    for layer in style_layers:
        act = net.blobs[layer].data[0].copy()
        act.shape = (act.shape[0], -1)
        resp[layer] = numpy.dot(act, act.T)
    return(resp)

net_forward(style_scaled)
style_repr = get_style_repr(style_scaled)
net_forward(content_scaled)
content_repr = get_content_repr(content_scaled)

# Make white noise input as the starting image:
numpy.random.seed(12345)
target_img = xform.preprocess(
    "data", numpy.random.randn(*(net.blobs["data"].data.shape[1:])))
#target_img = xform.preprocess("data", content_scaled)

# TODO: What does the below do? ("compute data bounds").  The result
# is a very large list - incidentally equal to the product of the
# dimensions of target_img.
data_min = -xform.mean["data"][:,0,0]
data_max = data_min + xform.raw_scale["data"]
data_bounds = [(data_min[0], data_max[0])]*(target_img.size/3) + \
              [(data_min[1], data_max[1])]*(target_img.size/3) + \
              [(data_min[2], data_max[2])]*(target_img.size/3)

# Ratio between matching the content of the content image, and the
# style of the style image (see figure 3 of the paper):
ratio = 1e3

# Initialize to correct size
# net_forward(content_scaled)

def loss_function(x):
    # Reshape the (flattened) input and feed it into the network:
    net_in = x.reshape(net.blobs["data"].data.shape[1:])
    net.blobs["data"].data[0] = net_in
    net.forward()

    # Get content & style representation of net_in:
    content_repr_tmp = get_content_repr(net_in)
    style_repr_tmp = get_style_repr(net_in)
    
    # Starting at last layer (see self.layers), propagate error back.
    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    for i, layer in enumerate(reversed(layers)):
        next_layer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]

        # Matching paper notation for equations 1 to 7:
        Pl = content_repr[layer]
        Fl = content_repr_tmp[layer]
        Nl = content_repr_tmp[layer].shape[0]
        Ml = content_repr_tmp[layer].shape[1]
        Gl = style_repr_tmp[layer]
        Al = style_repr[layer]
        
        # Content loss:
        w = 1.0 / len(content_layers)
        if layer in content_layers:
            d = Fl - Pl
            # Equations 1 & 2:
            loss += w * (d**2).sum() / 2
            grad += w * (d * (Fl > 0)).reshape(grad.shape)

        # Style loss:
        w = 1.0 / len(style_layers)
        if layer in style_layers:
            q = (Nl * Ml)**-2
            d = Gl - Al
            # Equation 4:
            El = q/4 * (d**2).sum()
            # Equation 6:
            dEl = q * numpy.dot(d, Fl) * (Fl > 0)
            # Equation 5:
            loss += w * El * ratio
            # Equation 7 (ish):
            grad += w * dEl.reshape(grad.shape) * ratio

        # Finally, propagate this error back into the network
        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    # Total Variation Gradient:
    tv_strength = 1e-2
    x_diff = net_in[:, :-1, :-1] - net_in[:, :-1, 1:]
    y_diff = net_in[:, :-1, :-1] - net_in[:, 1:, :-1]
    tv = numpy.zeros(grad.shape)
    tv[:, :-1, :-1] += x_diff + y_diff
    tv[:, :-1, 1:]  -= x_diff
    tv[:, 1:,  :-1] -= y_diff
    grad += tv_strength * tv
            
    # Flatten gradient (as minimize() wants to handle it):
    grad = grad.flatten().astype(numpy.float64)

    return loss, grad

# Now, use a standard optimization routine in SciPy to minimize the
# loss we've defined - by modifying the input image, with the help of
# the gradient that our loss function also returns (hence, we set
# 'jac' to True).  This uses L-BFGS-B, but other optimization methods
# (such as Adam) also work.
res = scipy.optimize.minimize(loss_function,
                              target_img.flatten(),
                              args = (),
                              options = {"maxiter": 1000, "maxcor": 8, "disp": True},
                              method = "L-BFGS-B",
                              jac = True,
                              bounds = data_bounds)
# TODO: Check success?
# TODO: Get answer out of 'res' instead of network?  It'd need
# reshaping.

# Grab final results, and "deprocess" them to undo the preprocessing:
data = net.blobs["data"].data
scipy.misc.imsave("out_water.png",
                  skimage.img_as_ubyte(xform.deprocess("data", data)));

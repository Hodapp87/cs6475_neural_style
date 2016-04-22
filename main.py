#!/usr/bin/env python2

import caffe
import numpy
import skimage
import scipy.linalg.blas
import scipy.optimize

# Initialize caffe to run on GPU 0:
caffe.set_device(0)
caffe.set_mode_gpu()
# The alternative is something like:
# caffe.set_mode_cpu()

# Load style and content images 
#content = caffe.io.load_image("./Roundtanglelake.jpg")
style   = caffe.io.load_image("./escher_sphere.jpg")
content = caffe.io.load_image("./brad_pitt.jpg")
#style   = caffe.io.load_image("./picasso_selfport1907.jpg")

# Load Caffenet model (source:
# http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel):
net = caffe.Net("caffenet/deploy.prototxt",
                "caffenet/bvlc_reference_caffenet.caffemodel",
                caffe.TEST)

style_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
content_layers = ["conv4"]
layers = [l for l in net.blobs if l in content_layers or l in style_layers]

# TODO: Where do deploy.prototxt and ilsvrc_2012_mean.npy come from?
mean_data = numpy.load("caffenet/ilsvrc_2012_mean.npy")

# TODO: What is the purpose of this?
# "all models used are trained on imagenet data"
xform = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
xform.set_mean("data", mean_data.mean(1).mean(1))
xform.set_channel_swap("data", (2,1,0))
xform.set_transpose("data", (2,0,1))
xform.set_raw_scale("data", 255)

# Scale both images to a certain size:
size = 512
style_scale = 1.2
# The below just gets the *longest* edge to 'size' pixels wide.
content_scaled = skimage.transform.rescale(
    content, float(size) / max(content.shape[:2]))
style_scaled = skimage.transform.rescale(
    style, style_scale * float(size) / max(style.shape[:2]))

# Scale neural network & transformer, for this new size of
# 'style_scaled':
# TODO: Does neural-style do this?
dims, x, y = style_scaled.shape[2], style_scaled.shape[0], style_scaled.shape[1]
net.blobs["data"].reshape(1, dims, x, y)
xform.inputs["data"] = (1, dims, x, y)

# Run the style image through the neural network!
# TODO: Why data[0] here?
net.blobs["data"].data[0] = xform.preprocess("data", style_scaled)
net.forward()
style_repr = {}
for layer in style_layers:
    # TODO: Why data[0]?
    act = net.blobs[layer].data[0].copy()
    # TODO: Flattening the latter two (of 3) dimensions, but why?
    act.shape = (act.shape[0], -1)
    # Find Gram matrix (only needed on style layers):
    style_repr[layer] = scipy.linalg.blas.sgemm(1, act, act.T)

# and then do the same thing for the content image:
dims, x, y = content_scaled.shape[2], content_scaled.shape[0], content_scaled.shape[1]
net.blobs["data"].reshape(1, dims, x, y)
xform.inputs["data"] = (1, dims, x, y)
net.blobs["data"].data[0] = xform.preprocess("data", content_scaled)
net.forward()
content_repr = {}
for layer in content_layers:
    # TODO: Why data[0]?
    act = net.blobs[layer].data[0].copy()
    # TODO: Flattening the latter two (of 3) dimensions, but why?
    act.shape = (act.shape[0], -1)
    content_repr[layer] = act

# Make white noise input as the starting image:
numpy.random.seed(12345)
#target_img = xform.preprocess(
#    "data", numpy.random.randn(*(net.blobs["data"].data.shape[1:])))
target_img = xform.preprocess("data", content_scaled)
# TODO: Does the scale of this need changing?  randn isn't in [0,1],
# but the input to xform.preprocess in other usages is.

# TODO: What does the below do? ("compute data bounds").  The result
# is a very large list - incidentally equal to the product of the
# dimensions of target_img.
data_min = -xform.mean["data"][:,0,0]
data_max = data_min + xform.raw_scale["data"]
data_bounds = [(data_min[0], data_max[0])]*(target_img.size/3) + \
              [(data_min[1], data_max[1])]*(target_img.size/3) + \
              [(data_min[2], data_max[2])]*(target_img.size/3)

# TODO: Explain in terms of the paper
ratio = 1e4

def optfn(x):
    # Reshape the (flattened) input and feed it into the network:
    net_in = x.reshape(net.blobs["data"].data.shape[1:])
    net.blobs["data"].data[0] = net_in
    net.forward()

    # Get content & style representation of net_in:
    content_repr_tmp = {}
    for layer in layers:
        act = net.blobs[layer].data[0].copy()
        act.shape = (act.shape[0], -1)
        content_repr_tmp[layer] = act
        
    style_repr_tmp = {}
    for layer in style_layers:
        if layer in content_repr_tmp:
            act = content_repr_tmp[layer]
        else:
            act = net.blobs[layer].data[0].copy()
            act.shape = (act.shape[0], -1)
        # Gram matrix again:
        style_repr_tmp[layer] = scipy.linalg.blas.sgemm(1, act, act.T)
    # style_repr_tmp is incorrect in some subtle way.
    # The state of the neural network differs here somehow.

    # Starting at last layer (see self.layers), propagate error back.
    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    # TODO: Rewrite all this.
    for i, layer in enumerate(reversed(layers)):
        next_layer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]

        # For now, just set 'w' to be even for all style layers:
        w = 1.0 / len(style_layers)
        # style contribution
        if layer in style_layers:
            ## See equation 6 in paper
            c = content_repr_tmp[layer].shape[0]**-2 * content_repr_tmp[layer].shape[1]**-2
            d = style_repr_tmp[layer] - style_repr[layer]
            g = c * scipy.linalg.blas.sgemm(
                1.0, d, content_repr_tmp[layer]) * (content_repr_tmp[layer]>0)
            loss += w * (c/4 * (d**2).sum()) * ratio
            grad += w * g.reshape(grad.shape) * ratio
            
        w = 1.0 / len(content_layers)
        # content contribution
        if layer in content_layers:
            d = content_repr_tmp[layer] - content_repr[layer]
            loss += w * (d**2).sum() / 2
            grad += w * (d * (content_repr_tmp[layer] > 0)).reshape(grad.shape)

        # compute gradient
        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    # Total Variation Gradient,
    # based on https://github.com/kaishengtai/neuralart
    tv_strength = 1e-3
    x_diff = net_in[:, :-1, :-1] - net_in[:, :-1, 1:]
    y_diff = net_in[:, :-1, :-1] - net_in[:, 1:, :-1]
    tv = numpy.zeros(grad.shape)
    tv[:, :-1, :-1] += x_diff + y_diff
    tv[:, :-1, 1:]  -= x_diff
    tv[:, 1:,  :-1] -= y_diff
    grad += tv_strength * tv
            
    # format gradient for minimize() function
    grad = grad.flatten().astype(numpy.float64)

    return loss, grad
   
res = scipy.optimize.minimize(optfn,
                              target_img.flatten(),
                              args = (),
                              options = {"maxiter": 512, "maxcor": 8, "disp": True},
                              method = "L-BFGS-B",
                              jac = True,
                              bounds = data_bounds)
# TODO: Check success?

# Grab final results:
data = net.blobs["data"].data
scipy.misc.imsave("out.png", skimage.img_as_ubyte(xform.deprocess("data", data)));

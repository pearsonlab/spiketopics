"""
Helper functions for dealing with nodes.
"""
from .GammaNode import GammaNode
from .utility_nodes import ProductNode

def check_shapes(par_shapes, pars):
    """
    Check for consistency between par_shapes, a dict with (name, shape)
    pairs, and pars, a dict of (name, value) pairs. 
    """
    for var, shape in par_shapes.iteritems():
        if var not in pars:
            raise ValueError('Argument missing: {}'.format(var))
        elif pars[var].shape != shape:
            raise ValueError('Argument has wrong shape: {}'.format(var))

def initialize_gamma(name, node_shape, **kwargs):
    """
    Initialize a gamma variable.
    """
    par_shapes = ({'prior_shape': node_shape, 'prior_rate': node_shape,
        'post_shape': node_shape, 'post_rate': node_shape })

    check_shapes(par_shapes, kwargs)

    node = GammaNode(kwargs['prior_shape'], kwargs['prior_rate'], 
        kwargs['post_shape'], kwargs['post_rate'], name=name)

    return (node,)

def initialize_gamma_hierarchy(basename, parent_shape, 
    child_shape, **kwargs):
    """
    Initialize a hierarchical gamma variable: 
        lambda ~ Gamma(c, c * m)
        c ~ Gamma
        m ~ Gamma

    basename is the name of the name of the child variable 
    parent_shape and child_shape are the shapes of the parent and
        child variables
    """
    par_shapes = ({'prior_shape_shape': parent_shape, 
        'prior_shape_rate': parent_shape, 
        'prior_mean_shape': parent_shape, 'prior_mean_rate': parent_shape, 
        'post_shape_shape': parent_shape, 'post_shape_rate': parent_shape, 
        'post_mean_shape': parent_shape, 'post_mean_rate': parent_shape, 
        'post_child_shape': child_shape, 'post_child_rate': child_shape})

    check_shapes(par_shapes, kwargs)
    
    shapename = basename + '_shape'
    shape = GammaNode(kwargs['prior_shape_shape'], 
        kwargs['prior_shape_rate'], kwargs['post_shape_shape'], 
        kwargs['post_shape_rate'], name=shapename)

    meanname = basename + '_mean'
    mean = GammaNode(kwargs['prior_mean_shape'], 
        kwargs['prior_mean_rate'], kwargs['post_mean_shape'], 
        kwargs['post_mean_rate'], name=meanname)

    child = GammaNode(shape, ProductNode(shape, mean),
        kwargs['post_child_shape'], kwargs['post_child_rate'], 
        name=basename)

    return (shape, mean, child)


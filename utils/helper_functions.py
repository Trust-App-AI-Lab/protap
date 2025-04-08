import torch
import torch.nn as nn
from einops import repeat, rearrange

def exists(val):
    return val is not None

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i,emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = torch.cat([ x[:, :stop_concat], 
                        emb_layer( to_embedd[:, i] ) 
                      ], dim=-1)
        stop_concat = x.shape[-1]
        
    return x

def contrastive_graph(inputs):
    feats = inputs['feats']
    coors = inputs['coors']
    masks = inputs['mask']
    
    # Preprocessing for the SE3 Transfofrmer.
    feats = repeat(feats, 'b n -> b (n c)', c=1) # Expand the channel.
    masks = repeat(masks, 'b n -> b (n c)', c=1) # Expand the channel.
    
    i = torch.arange(feats.shape[-1], device=feats.device)
    adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))
    
    inputs = {
        "feats" : feats,
        "coors" : coors,
        "mask" : masks,
        "adj_mat" : adj_mat,
    }
    
    return inputs

def masked_mean_pooling(embeddings, masks):
    """
    Perform mean pooling over node embeddings while considering valid nodes.

    Args:
    - embeddings: (batch_size, max_nodes, embedding_dim), node embeddings.
    - masks: (batch_size, max_nodes), binary mask indicating valid nodes.

    Returns:
    - graph_embeddings: (batch_size, embedding_dim), pooled graph representation.
    """
    masks = masks.float().unsqueeze(-1)  # (batch_size, max_nodes, 1) for broadcasting
    sum_embeddings = torch.sum(embeddings * masks, dim=1)  # Sum over valid nodes
    valid_node_count = torch.clamp(masks.sum(dim=1), min=1)  # Avoid division by zero
    
    return sum_embeddings / valid_node_count  # Compute mean while ignoring padding
from typing import Optional
from torch import nn
from torch import Tensor
from layers.RRCF_backbone import RRCF_backbone



class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()

        c_in = configs.num_sensors
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = 2
        n_heads = 8
        d_model = 512
        d_ff = 2048
        dropout = 0.05
        fc_dropout = 0.05
        head_dropout = 0
        individual = 0
        patch_len = 16
        stride = 8
        padding_patch = 'end'
        revin = 1
        affine = 0
        subtract_last = 0

        # model
        self.model = RRCF_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                              max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                              n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                              dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                              attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                              pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                              pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                              subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.model(x)
        x = x.permute(0,2,1)
        return x
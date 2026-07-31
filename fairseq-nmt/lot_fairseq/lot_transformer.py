"""fairseq model registration: a Transformer whose non-causal attentions are LOTFormer.

Registers the ``lot_transformer`` model and the ``lot_transformer_iwslt_de_en``
architecture. A standard fairseq Transformer is built, then the **encoder
self-attention** (and, with ``--lot-convert encoder_cross``, the encoder-decoder
**cross-attention**) modules are replaced by ``LotMultiheadAttention``, copying the
Q/K/V/out projection weights. The causal decoder self-attention is left as softmax.

This targets the classic (argparse) fairseq model-registration API, as used for the
IWSLT'14 De-En Transformer baseline.
"""

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, base_architecture

from .lot_multihead_attention import LotMultiheadAttention


def _convert_to_lot(model, args):
    cfg = dict(
        num_refs=getattr(args, "lot_num_refs", 32),
        sink_eps=getattr(args, "lot_sink_eps", 1.0),
        max_iter=getattr(args, "lot_max_iter", 5),
        learn_z=getattr(args, "lot_learn_z", False),
    )
    # encoder self-attention (bidirectional -> safe for doubly-stochastic attention)
    for layer in model.encoder.layers:
        layer.self_attn = LotMultiheadAttention.from_fairseq(layer.self_attn, **cfg)

    # optionally the non-causal encoder-decoder cross-attention
    if getattr(args, "lot_convert", "encoder") == "encoder_cross":
        for layer in model.decoder.layers:
            layer.encoder_attn = LotMultiheadAttention.from_fairseq(layer.encoder_attn, **cfg)

    return model


@register_model("lot_transformer")
class LotTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument("--lot-num-refs", type=int, default=32,
                            help="LOTFormer pivot-measure size r (landmarks per head)")
        parser.add_argument("--lot-sink-eps", type=float, default=1.0,
                            help="entropic regularization for the Sinkhorn transports")
        parser.add_argument("--lot-max-iter", type=int, default=5,
                            help="number of Sinkhorn iterations")
        parser.add_argument("--lot-learn-z", action="store_true", default=False,
                            help="learn the pivot prior z (otherwise uniform)")
        parser.add_argument("--lot-convert", type=str, default="encoder",
                            choices=["encoder", "encoder_cross"],
                            help="which non-causal attentions to convert to LOTFormer")

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        _convert_to_lot(model, args)
        return model


@register_model_architecture("lot_transformer", "lot_transformer_iwslt_de_en")
def lot_transformer_iwslt_de_en(args):
    # IWSLT'14 De-En Transformer baseline hyperparameters
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
    # LOTFormer defaults
    args.lot_num_refs = getattr(args, "lot_num_refs", 32)
    args.lot_sink_eps = getattr(args, "lot_sink_eps", 1.0)
    args.lot_max_iter = getattr(args, "lot_max_iter", 5)
    args.lot_learn_z = getattr(args, "lot_learn_z", False)
    args.lot_convert = getattr(args, "lot_convert", "encoder")

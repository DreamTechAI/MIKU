"""Flux2's current forward with the legacy PBR pipeline call signature.

The PBR inference path always keeps reference tokens through all
blocks (drop_condition=False). Its weights use the native Flux2 architecture.
Delegating forward avoids obsolete block keyword names and the old single
stream modulation tuple indexing.
"""
from diffusers import Flux2Transformer2DModel


class Flux2ORMTransformer2DModel(Flux2Transformer2DModel):
    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None,
                img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None,
                return_dict=True, condition_length=None, drop_condition=False,
                dual_length=None):
        if drop_condition:
            raise ValueError("ORM inference keeps its PBR reference tokens; drop_condition must be False")
        return super().forward(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            timestep=timestep, img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs, return_dict=return_dict,
        )

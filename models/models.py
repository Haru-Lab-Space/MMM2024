from models.TSF import *
from models.TSF_hard_voting import *
from models.TSF_Resnet_Swin import *
from models.TSF_ViT_Resnet import *
from models.TSF_ViT_Swin import *
from models.Baseline import *
from models.Baseline_hard_voting import *
from models.Baseline_Resnet_Swin import *
from models.Baseline_ViT_Resnet import *
from models.Baseline_ViT_Swin import *
from models.TSF_ViT import *
from models.TSF_Resnet import *
from models.TSF_Swin import *
from models.utils import *
from models.TSF_ViT_Resnet_hard_voting import *
from models.Baseline_ViT_Resnet_hard_voting import *
from models.Baseline_Resnet import *
from models.Baseline_ViT import *
from models.Baseline_Swin import *

def get_model(architecture, img_encoder,encoder_layer,text_embedding, pooling_layer):
    if architecture == "TSF":
        return TSF(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_hard_voting":
        return TSF_hard_voting(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_Resnet_Swin":
        return TSF_Resnet_Swin(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_ViT_Resnet":
        return TSF_ViT_Resnet(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_ViT_Resnet_hard_voting":
        return TSF_ViT_Resnet_hard_voting(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_ViT_Swin":
        return TSF_ViT_Swin(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline":
        return Baseline(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_hard_voting":
        return Baseline_hard_voting(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_Resnet_Swin":
        return Baseline_Resnet_Swin(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_ViT_Resnet":
        return Baseline_ViT_Resnet(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_ViT_Resnet_hard_voting":
        return Baseline_ViT_Resnet_hard_voting(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_ViT_Swin":
        return Baseline_ViT_Swin(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_Swin":
        return TSF_Swin(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_Resnet":
        return TSF_Resnet(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "TSF_ViT":
        return TSF_ViT(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_ViT":
        return Baseline_ViT(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_Resnet":
        return Baseline_Resnet(img_encoder,encoder_layer,text_embedding, pooling_layer)
    elif architecture == "Baseline_Swin":
        return Baseline_Swin(img_encoder,encoder_layer,text_embedding, pooling_layer)

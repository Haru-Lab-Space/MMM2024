from torch import nn
import torch.nn.functional as F
import torch 

class Baseline_hard_voting(nn.Module):
  def __init__(self,image_encoder,text_encoder,text_embedding, pooling_layer):
    super(Baseline_hard_voting,self).__init__()
    #image_encoder
    self.vit_encoder = image_encoder[0]
    self.resnet_encoder = image_encoder[1]
    self.swintransformer_encoder = image_encoder[2]

    #text_encoder & text embedding
    self.text_encoder = text_encoder
    self.text_embedding = text_embedding
    self.pooling_layer = pooling_layer

    # layernorm
    self.layernorm_197 = nn.LayerNorm(197)
    self.layernorm_768 = nn.LayerNorm(768)

    # self.layernorm_1024 = nn.LayerNorm(1024)
    #linear
    self.linear = nn.Linear(2,1)

    # projection
    # self.resnet_projection_2 = nn.Linear(2048,197)
    self.resnet_projection_2 = nn.Linear(2048,197)
    self.resnet_projection_3 = nn.Linear(49,768)

    # Dropout
    self.dropout = nn.Dropout(p=0.2)

    self.swintransformer_projection_1 = nn.Linear(49,197)
    self.final_projection = nn.Linear(197,1)
  def forward(self,img,text):
    batch_size = img.size(0)
    #text_features
    text_embedding = self.text_embedding(text) # torch.Size([16, 197, 1024])

    #image_features
    image_features_0 = self.vit_encoder(img).last_hidden_state #torch.Size([16, 197, 768])
    image_features_1 = self.resnet_encoder(img).last_hidden_state #torch.Size([16, 512, 7, 7])
    image_features_2 = self.swintransformer_encoder(img).last_hidden_state #torch.Size([16, 49, 768])
    dim = image_features_1.size(1)
    #projection_1
    image_features_1 = image_features_1.reshape(batch_size, dim,-1) #torch.Size([16, 512, 49])
    image_features_1 = image_features_1.permute(0,2,1) #torch.Size([16, 49, 512])
    image_features_1 = self.resnet_projection_2(image_features_1) #torch.Size([16, 49, 197])
    image_features_1 = self.layernorm_197(image_features_1)
    image_features_1 = image_features_1.permute(0,2,1) #torch.Size([16, 197, 49])
    image_features_1 = self.resnet_projection_3(image_features_1) #torch.Size([16, 197, 768])
    image_features_1 = self.layernorm_768(image_features_1)
    # image_features_1 = self.dropout(image_features_1)
    #projection_2
    image_features_2 = image_features_2.permute(0,2,1) #torch.Size([16, 768, 49])
    image_features_2 = self.swintransformer_projection_1(image_features_2) #torch.Size([16, 768, 197])
    image_features_2 = self.layernorm_197(image_features_2)
    image_features_2 = image_features_2.permute(0,2,1) #torch.Size([16, 197, 768])

    text_embedding = self.pooling_layer(self.text_encoder(text_embedding).last_hidden_state)


    image_features_0 = self.final_projection(image_features_0.permute(0,2,1)).squeeze(-1)
    image_features_1 = self.final_projection(image_features_1.permute(0,2,1)).squeeze(-1)
    image_features_2 = self.final_projection(image_features_2.permute(0,2,1)).squeeze(-1)

    logits_0 = F.cosine_similarity(text_embedding,image_features_0, dim =-1)
    logits_1 = F.cosine_similarity(text_embedding,image_features_1, dim =-1)
    logits_2 = F.cosine_similarity(text_embedding,image_features_2, dim =-1)
    logits = (logits_0 + logits_1 + logits_2)/3
    return logits
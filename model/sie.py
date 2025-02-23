import enum
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import pytorch_lightning as pl
from transformers import AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.nn import functional as F

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

def get_avg_metrics(all_labels, all_probs, threshold, disease='None', class_names=id2disease):
    # Convert probabilities to predictions using threshold
    all_preds = (all_probs > threshold).astype(float)
    
    # Calculate per-class metrics
    ret = {}
    n_classes = all_labels.shape[1]
    
    # Per-class F1 scores
    for i in range(n_classes):
        class_name = class_names[i] if i < len(class_names) else "control"
        try:
            class_f1 = f1_score(all_labels[:, i], all_preds[:, i])
            class_precision = precision_score(all_labels[:, i], all_preds[:, i])
            class_recall = recall_score(all_labels[:, i], all_preds[:, i])
            class_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except:
            class_f1 = 0.0
            class_precision = 0.0
            class_recall = 0.0
            class_auc = 0.5
            
        ret[f"{class_name}_f1"] = class_f1
        ret[f"{class_name}_precision"] = class_precision
        ret[f"{class_name}_recall"] = class_recall
        ret[f"{class_name}_auc"] = class_auc
    
    # Macro metrics (average of per-class metrics)
    ret["macro_f1"] = np.mean([ret[f"{class_names[i]}_f1"] for i in range(n_classes-1)] + [ret["control_f1"]])
    ret["macro_precision"] = np.mean([ret[f"{class_names[i]}_precision"] for i in range(n_classes-1)] + [ret["control_precision"]])
    ret["macro_recall"] = np.mean([ret[f"{class_names[i]}_recall"] for i in range(n_classes-1)] + [ret["control_recall"]])
    ret["macro_auc"] = np.mean([ret[f"{class_names[i]}_auc"] for i in range(n_classes-1)] + [ret["control_auc"]])
    
    # Micro metrics (calculated globally)
    ret["micro_f1"] = f1_score(all_labels.flatten(), all_preds.flatten())
    ret["micro_precision"] = precision_score(all_labels.flatten(), all_preds.flatten())
    ret["micro_recall"] = recall_score(all_labels.flatten(), all_preds.flatten())
    
    # Sample-wise accuracy (exact matches)
    ret["sample_acc"] = accuracy_score(all_labels, all_preds)
        

    return ret

def masked_logits_loss(logits, labels, masks=None):
    # treat unlabeled samples(-1) as implict negative (0.)
    labels2 = torch.clamp_min(labels, 0.)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none')
    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        return masked_losses.mean()
    else:
        return losses.mean()


class ContrastiveLearning(nn.Module):
    def __init__(self, hidden_size=768, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projection =  nn.Linear(hidden_size, hidden_size)

    def forward(self, original_embeddings, augmented_embeddings):
        # Project embeddings to contrastive space
        z1 = self.projection(original_embeddings)
        z2 = self.projection(augmented_embeddings)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # Calculate loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class LightningInterface(pl.LightningModule):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        self.disease = kwargs['disease']
        self.criterion = masked_logits_loss
        self.validation_step_outputs = []
        self.test_step_outputs = []  # 添加测试输出存储
        
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
        
    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y, label_masks = batch
        logits, contrastive_loss = self(x)  # 直接使用SIE的返回值，包含logits和contrastive_loss
        loss = self.criterion(logits, y)
        
        total_loss = loss + self.contrastive_weight * contrastive_loss
        self.log('train_cls_loss', loss)
        self.log('train_contrast_loss', contrastive_loss)
        self.log('train_loss', total_loss)
        
        return {'loss': total_loss, 'log': {'train_loss': total_loss}}

    def validation_step(self, batch, batch_nb):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        loss = self.criterion(y_hat, y)
        output = {'val_loss': loss, "labels": yy, "probs": yy_hat}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        ret = get_avg_metrics(all_labels, all_probs, self.threshold, self.disease)
        
        # Log all metrics
        tensorboard_logs = {'val_loss': avg_loss}
        for metric_name, metric_value in ret.items():
            tensorboard_logs[f'val_{metric_name}'] = metric_value
            
        self.best_f1 = max(self.best_f1, ret['macro_f1'])
        
        # Print detailed metrics
        print("\nValidation Results:")
        print(f"Macro F1: {ret['macro_f1']:.4f}")
        print(f"Micro F1: {ret['micro_f1']:.4f}")
        print("\nPer-class F1 scores:")
        for disease in id2disease + ['control']:
            print(f"{disease}: {ret[f'{disease}_f1']:.4f}")
        
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        
        self.validation_step_outputs.clear()  # clear outputs

    def on_test_epoch_start(self):
        self.test_step_outputs = []  # 初始化测试输出列表

    def test_step(self, batch, batch_nb):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        if self.setting == 'binary':
            loss = self.criterion(y_hat, y, label_masks)
        else:
            loss = self.criterion(y_hat, y)
        output = {'test_loss': loss, "labels": yy, "probs": yy_hat}
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        ret = get_avg_metrics(all_labels, all_probs, self.threshold, self.disease)
        
        results = {'test_loss': avg_loss}
        for k, v in ret.items():
            results[f"test_{k}"] = v
        self.log_dict(results)
        
        # Print test results
        print("\nTest Results:")
        print(f"Macro F1: {ret['macro_f1']:.4f}")
        print(f"Micro F1: {ret['micro_f1']:.4f}")
        print("\nPer-class F1 scores:")
        for disease in id2disease + ['control']:
            print(f"{disease}: {ret[f'{disease}_f1']:.4f}")
            
        self.test_step_outputs.clear()  # clear outputs

    def on_after_backward(self):
        pass
        # can check gradient
        # global_step = self.global_step
        # if int(global_step) % 100 == 0:
        #     for name, param in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser





class SIE(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=20, freeze=False, pool_type="first", dis_count=8) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        self.symaug_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.symaug_encoder.named_parameters():
                param.requires_grad = False    
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        self.symaug_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        nn.init.xavier_uniform_(self.symaug_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        encoder_layer_sa = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder_sa = nn.TransformerEncoder(encoder_layer_sa, num_layers=num_trans_layers)

        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim+38, dis_count) 
        self.contrastive = ContrastiveLearning(self.hidden_dim)
      
    def forward(self, batch):
        batch_size = len(batch)
        post_embeddings_list = []
        aug_embeddings_list = []
        symptom_embeddings_list = []
        
        for user_feats in batch:
            # 获取所有posts的embeddings
            posts_output = self.post_encoder(
                user_feats["input_ids"], 
                user_feats["attention_mask"]
            )
            aug_output = self.symaug_encoder(
                user_feats["augmented_input_ids"], 
                user_feats["augmented_attention_mask"]
            )
            
            # 获取每个post的表示
            if self.pool_type == "first":
                posts_embeds = posts_output.last_hidden_state[:, 0, :]  
                aug_embeds = aug_output.last_hidden_state[:, 0, :]    
            else:
                posts_embeds = mean_pooling(posts_output.last_hidden_state, user_feats["attention_mask"])
                aug_embeds = mean_pooling(aug_output.last_hidden_state, user_feats["augmented_attention_mask"])
            
            # 添加位置编码
            posts_embeds = posts_embeds + self.pos_emb[:posts_embeds.shape[0]]
            aug_embeds = aug_embeds + self.symaug_emb[:aug_embeds.shape[0]]
            
            post_embeddings_list.append(posts_embeds)
            aug_embeddings_list.append(aug_embeds)
            symptom_embeddings_list.append(user_feats["symp"])
        
        # 对比学习 (post-level)
        all_post_embeddings = torch.cat(post_embeddings_list, dim=0)  
        all_aug_embeddings = torch.cat(aug_embeddings_list, dim=0)   
        contrastive_loss = self.contrastive(all_post_embeddings, all_aug_embeddings)
        
        user_embeddings = []
        user_symptoms = []
        for i in range(batch_size):
            user_posts = post_embeddings_list[i] 
            user_embedding = user_posts.mean(dim=0) 
            user_embeddings.append(user_embedding)
            
            user_symp = torch.tensor(symptom_embeddings_list[i], 
                                   device=user_embedding.device).float()
            user_symp = user_symp.mean(dim=0) 
            user_symptoms.append(user_symp)
        
        user_embeddings = torch.stack(user_embeddings) 
        user_symptoms = torch.stack(user_symptoms) 
       
        # 分类
        combined_features = torch.cat([user_embeddings, user_symptoms], dim=1)
        logits = self.clf(self.dropout(combined_features))
        
        return logits, contrastive_loss



class SIEClassifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="bert-base-uncased", user_encoder="none", 
                 num_trans_layers=2, freeze_word_level=False, pool_type="first", 
                contrastive_weight=0.1, setting='multilabel', **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model_type = model_type
        self.user_encoder = user_encoder
        self.model = SIE(model_type, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        self.contrastive_weight = contrastive_weight
        self.setting = setting  # 添加setting属性
        self.lr = lr
        
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, _ = batch
        logits, contrastive_loss = self(x)
        
        # Classification loss (multi-label)
        cls_loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Combined loss
        total_loss = cls_loss + self.contrastive_weight * contrastive_loss
        
        self.log('train_cls_loss', cls_loss)
        self.log('train_contrast_loss', contrastive_loss)
        self.log('train_loss', total_loss)
        
        return {'loss': total_loss}
    
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=5e-3)
        parser.add_argument("--user_encoder", type=str, default="none")
        parser.add_argument("--pool_type", type=str, default="first")
        parser.add_argument("--num_trans_layers", type=int, default=2)
        parser.add_argument("--freeze_word_level", action="store_true")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
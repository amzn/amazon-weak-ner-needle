"""Provide a General NER Model Class."""
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForTokenClassification, AutoConfig
from transformers.modeling_auto import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    PretrainedConfig,
    OrderedDict,
)
from transformers.modeling_outputs import TokenClassifierOutput
from loss import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from crfutils import ConditionalRandomField
from crfutils import allowed_transitions


def wrapclass(cls):
    """Wrap AutoModelForTokenClassification to support CRF and other functions."""
    class NERAddon(cls):
        def __init__(self, config):
            super().__init__(config)
            self.loss_func = config.loss_func

            self.features_dict = config.features_dict
            self.features_dim = config.features_dim
            self.hidden_size = config.hidden_size
            # list of feature column index
            self.feature_list = sorted(list(self.features_dict.keys()))
            feature_embedding = []
            for feature in self.feature_list:
                feature_embedding.append(
                    nn.Embedding(self.features_dim[feature], self.features_dict[feature]))
            self.feature_embedding = nn.ModuleList(feature_embedding)
            self.use_cnn = config.use_cnn
            if self.use_cnn:
                conv_input_channels = 0
                for feature in self.feature_list:
                    conv_input_channels += self.features_dict[feature]
                cnn_kernels = config.cnn_kernels.split(",")
                conv_layers = []
                for i in cnn_kernels:
                    kernel_size = int(i)
                    conv_layers.append(nn.Conv1d(in_channels=conv_input_channels,
                                                 out_channels=config.cnn_out_channels,
                                                 kernel_size=kernel_size))
                self.conv_layers = nn.ModuleList(conv_layers)
                self.hidden_size += len(self.conv_layers) * config.cnn_out_channels
            else:
                for feature in self.feature_list:
                    self.hidden_size += self.features_dict[feature]

            self.use_crf = config.use_crf
            if self.use_crf:
                inversed_label_map = config.inversed_label_map
                constraints = allowed_transitions("BIO", inversed_label_map)
                self.crf_layer = ConditionalRandomField(config.num_labels, constraints)

            self.init_weights()
            self.return_list = []

        def set_returning(self, return_list):
            """Only triggered in very speicial case (do profiling)."""
            self.return_list = return_list

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            features=None,
            predict_mask=None,
            weights=None,
        ):

            base_model = getattr(self, self.base_model_prefix)
            outputs = base_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            cat_features = [sequence_output]

            feature_embeddings = []

            for i in range(len(self.feature_list)):
                try:
                    one_feature = features[:, i, :]
                    feature_embeddings.append(self.feature_embedding[i](one_feature))
                except Exception:
                    raise RuntimeError("Encoutnered error embedding feature %s embedding_layer %s feature position %s " %
                                       (one_feature, self.feature_embedding[i], i))

            if self.use_cnn:
                # cat feature embedding shape batch size X length X (dimension X number of features)
                cat_feature_embedding = torch.cat(feature_embeddings, dim=2)
                # transpose last two dimension for convolution
                cat_feature_embedding = cat_feature_embedding.transpose(1, 2)

                for i in range(len(self.conv_layers)):
                    conv_feature_embedding = self.conv_layers[i](cat_feature_embedding)
                    kernel_size = self.conv_layers[i].kernel_size[0]
                    conv_feature_embedding_padded = nn.functional.pad(input=conv_feature_embedding, pad=(0, kernel_size - 1))
                    conv_feature_embedding_padded_t = conv_feature_embedding_padded.transpose(1, 2)
                    cat_features.append(conv_feature_embedding_padded_t)
            else:
                cat_features += feature_embeddings

            embeddings = torch.cat(cat_features, dim=2)
            logits = self.classifier(embeddings)

            loss = None
            if labels is not None and not self.use_crf:
                if self.loss_func == "DiceLoss":
                    loss_fct = DiceLoss()
                elif self.loss_func == "FocalLoss":
                    loss_fct = FocalLoss()
                elif self.loss_func == "LabelSmoothingCrossEntropy":
                    loss_fct = LabelSmoothingCrossEntropy()
                else:
                    loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    # import pdb; pdb.set_trace()
                    if self.loss_func == "DiceLoss":
                        predict_loss = predict_mask.view(-1) == 1
                        active_labels = active_labels[predict_loss]
                        active_logits = active_logits[predict_loss, :]
                        # import pdb; pdb.set_trace()
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.use_crf:
                # Format input before using CRF layer
                if labels is not None:
                    predict_mask = (labels >= 0).to(torch.long)
                ones = torch.ones_like(predict_mask.to(torch.long))
                idx = torch.cumsum(ones, 1) - 1
                idx = idx * predict_mask + 9999 * (1 - predict_mask)
                idx = idx.sort()[0]
                c_mask = (idx < 9999).to(torch.long)
                idx = idx * c_mask
                c_logits = logits.gather(1, idx[:, :, None].expand_as(logits))
                tags = self.crf_layer.viterbi_tags(c_logits, c_mask)
                p_logits = torch.zeros_like(logits) - 20000.0
                for i, tag in enumerate(tags):
                    for j, t in enumerate(tag[0]):
                        p_logits[i, idx[i][j], t] = tag[1]

                if labels is not None:
                    c_labels = labels.gather(1, idx)
                    c_labels = c_labels * c_mask
                    if self.loss_func == 'nll':
                        nll = -self.crf_layer(c_logits, c_labels, c_mask)
                        loss = (nll * weights).sum()
                    elif self.loss_func == 'corrected_nll':
                        nll = -self.crf_layer(c_logits, c_labels, c_mask)
                        null = -(1 - (-nll).exp()).log()
                        if torch.isnan(null).any() or torch.isinf(null).any():
                            nl = (1 - (-nll).exp())
                            nl = nl + (nl < 1e-4).to(nl).detach() * (1e-4 - nl).detach()
                            null = - nl.log()

                        loss = (nll * weights + null * (1 - weights)).sum()
                    else:
                        raise NotImplementedError(f"{self.loss_func} is not implemented")
                if "nll" in self.return_list:
                    for i, tag in enumerate(tags):
                        for j, t in enumerate(tag[0]):
                            c_labels[i, j] = t
                    nll = -self.crf_layer(c_logits, c_labels, c_mask)
                    output = (nll,) + outputs[2:]
                else:
                    output = outputs[2:]
                output = (p_logits,) + output

                return ((loss,) + output) if loss is not None else output

            if not return_dict:
                if "nll" in self.return_list:
                    nll = nn.functional.cross_entropy(
                        logits.view(-1, self.num_labels),
                        labels.view(-1),
                        reduction='none'
                    ).view(labels.shape)
                    nll = nll.sum(-1) / (labels >= 0).long().sum(-1)
                    output = (nll,) + outputs[2:]
                else:
                    output = outputs[2:]
                output = (logits,) + output
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return NERAddon


MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (config_class, wrapclass(model_class))
        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items()
    ]
)


class NERModel(AutoModelForTokenClassification):
    """NER Model class: AutoModelForTokenClassification + self-defined loading."""

    @classmethod
    def from_config(cls, config):
        """Load model from config."""
        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model from pretrained_model_name_or_path."""
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )

import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

_tokenizer = _Tokenizer()

import torch


def load_clip_to_cpu(args,zero_shot = False):
    backbone_name = args.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, args.root_dir)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot:
        design_details = {"trainer": 'CoOp',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}

        model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def initialize_keys(num_experts, key_dim, init_method='rand_U'):
    """
    Initialize keys for different domains using various methods.

    Args:
    num_domains (int): Number of domains
    key_dim (int): Dimension of each key
    init_method (str): Initialization method ('rand_U', 'rand_N', 'rand_01', 'rand_O')

    Returns:
    torch.Tensor: Initialized keys
    """
    if init_method == 'rand_U':
        # Uniform distribution U(0,1)
        keys = torch.rand(num_experts, key_dim)

    elif init_method == 'rand_N':
        # Standard normal distribution
        keys = torch.randn(num_experts, key_dim)

    elif init_method == 'rand_01':
        # Random 01 matrix
        keys = torch.randint(0, 2, (num_experts, key_dim)).float()

    elif init_method == 'rand_O':
        # Random orthogonal initialization
        temp = torch.randn(num_experts, key_dim)
        q, _ = torch.linalg.qr(temp)
        keys = q

    else:
        raise ValueError("Invalid initialization method")
    return keys

class PromptLearner(nn.Module):

    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.INPUT_SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        route_emb = args.route_emb
        # random initialization
        if args.CSC:
            # print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            # print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.VPTctxList = nn.ParameterList([nn.Parameter(ctx_vectors)
                                            for _ in range(args.num_experts)])

        for single_para in self.VPTctxList:
            nn.init.normal_(single_para, std=0.02)


        # orthogonal_vectors_ = initialize_keys(args.num_experts, 768,args.init_method)
        orthogonal_vectors_ = torch.empty(args.num_experts, 768)
        nn.init.orthogonal_(orthogonal_vectors_, gain=1.0)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.num_experts = args.num_experts
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.CTP
        self.orthogonal_vectors = orthogonal_vectors_

        clip_model_temp_image = load_clip_to_cpu(args, True)
        self.ZS_image_encoder = clip_model_temp_image.visual

    def forward(self, weighted_prompts):
        ctx = weighted_prompts
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class = len(classnames)

        self.gamma = args.gamma
        self.batch = 0
        self.lamba = args.lamba
        self.orthogonal_vectors = self.prompt_learner.orthogonal_vectors
        self.num_experts = args.num_experts
        self.T = 1.0
        self.fixed_embeddings = 0
        self.ZS_image_encoder = self.prompt_learner.ZS_image_encoder
        self.alpha = args.alpha

    def forward(self, image, weight_experts,label=None,  Training=False):

        self.batch += 1
        logit_scale = self.logit_scale.exp()

        experts = self.prompt_learner.VPTctxList

        image_features, image_tokens = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        weights_tokens = weight_experts
        experts_tensor = torch.stack([param for param in experts])

        weighted_experts_ = (weights_tokens.unsqueeze(-1).unsqueeze(-1) * experts_tensor)
        weighted_experts = weighted_experts_.sum(dim=1)
        tokenized_prompts = self.tokenized_prompts

        logits = []
        for idx, imf_i in enumerate(image_features):
            prompts = self.prompt_learner(weighted_experts[idx])
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if Training:
            #zero-shot logits
            fixed_embeddings = self.fixed_embeddings.cuda()  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features, _ = self.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                tea_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
            tea_prob = F.softmax(tea_logits / self.T, dim=-1)
            kl_loss = -tea_prob * F.log_softmax(logits / self.T,
                                                -1) * self.T * self.T
            kl_loss = kl_loss.sum(1).mean()

            loss_ce = F.cross_entropy(logits, label)
            return loss_ce + self.alpha * kl_loss

        else:
            return logits

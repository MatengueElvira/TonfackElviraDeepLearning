
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]          # indice 0 = normal, indice 1 = pneumonie qui représente notre calsse positive
RIN_ARCHS = ("rin_resnet50", "rin_densenet121", "rin_inception_v3", "rin_inception_resnet_v2")   # initialisés avec RadImageNet
EXTRA_ARCHS = ("resnet50", "inception_v3", "inception_resnet_v2") + RIN_ARCHS                      #  backbone + tête
TL_ARCHS = ("resnet18", "efficientnet_b0", "densenet121") + EXTRA_ARCHS                            # tous les modèles pré-entraînés
PRETRAINED_STATUS = {}                         # arch -> True si les poids pré-entraînés ont vraiment été chargés
NORMS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # normalisation ImageNet classique
         "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])}                    # [0,1]  à  [-1,1] (convention RadImageNet / Inception)


# - Prétraitement -
def pil_to_uint8(img, size):
    # Image quelconque au  niveaux de gris ensuite size * size interpolation bilinéaire enfin tableau uint8 (H, W)
    img = img.convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)            # copie du tableau d'image interpolé
def preprocess_pil(img, size):
    # Renvoie un tenseur float32 (1, 1, size, size) dans [0, 1] exactement ce que le modèle attend.
    arr = pil_to_uint8(img, size)
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0).unsqueeze(0)


#  Architectures LightPneumoNet
class LightCNN(nn.Module):
    # CNN léger inspiré de LightPneumoNet : 4 blocs de 2 convolutions et  max-pooling.
    # Blocs 1  à 2 : noyaux 5*5, pooling 3*3 ; blocs 3-4 : noyaux 3*3, pooling 2*2 ; padding=0.
    def __init__(self, in_ch=1, img_size=224, use_bn=False, dropout=0.2, widths=(16, 32, 64, 128), n_classes=2):
        super().__init__()
        cfg = [(widths[0], 5, 3), (widths[1], 5, 3), (widths[2], 3, 2), (widths[3], 3, 2)]
        layers, c = [], in_ch
        for w, k, p in cfg:
            for _ in range(2):
                layers.append(nn.Conv2d(c, w, k, padding=0, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(w))
                layers.append(nn.ReLU())
                c = w
            layers.append(nn.MaxPool2d(p))
        self.features = nn.Sequential(*layers)
        with torch.no_grad():                    # on déduit la taille du vecteur aplati en faisant passer une image vide
            n_flat = self.features(torch.zeros(1, in_ch, img_size, img_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class XRayNet(nn.Module):
    # Wrapper commun : entrée = image niveaux de gris dans [0,1], forme (B,1,H,W) ; sortie = logits (B,2).
    # La normalisation (moyenne/écart-type) et, si besoin, la duplication en 3 canaux sont INTÉGRÉES au modèle.
    def __init__(self, net, arch, head_name, mean, std, expand3):
        super().__init__()
        self.net = net
        self.arch = arch
        self.head_name = head_name               # nom du sous-module « tête » (classifieur final)
        self.expand3 = expand3                   # True pour les modèles ImageNet (3 canaux attendus)
        self.backbone_frozen = False
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        if self.expand3:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return self.net(x)

    # --- utilitaires pour le fine-tuning en deux temps ---
    def head_parameters(self):
        return list(getattr(self.net, self.head_name).parameters())

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.head_parameters()}
        return [p for p in self.net.parameters() if id(p) not in head_ids]

    def set_backbone_trainable(self, flag):
        for p in self.backbone_parameters():
            p.requires_grad_(flag)
        self.backbone_frozen = not flag

    def train(self, mode=True):
        super().train(mode)
        if mode and self.backbone_frozen:        # backbone gelé => on fige aussi ses statistiques BatchNorm
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self
def _torchvision_model(name, pretrained):
    fn = getattr(torchvision.models, name)
    if pretrained:
        net = fn(weights="DEFAULT")
        PRETRAINED_STATUS[name] = True
        return net
    return fn(weights=None)




# ----------- Modèles « backbone + tête » : ResNet50, DenseNet121, InceptionV3, InceptionResNetV2 -----------
# Chaque modèle existe en deux versions : initialisé avec ImageNet (arch = "resnet50", ...) ou avec RadImageNet
# (arch = "rin_resnet50", ...). Les deux ont EXACTEMENT la même architecture : seule l'initialisation des poids change.
_INCEPTION_V3_BLOCKS = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                        "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
                        "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]    # sans AuxLogits, avgpool, dropout, fc


class TimmFeatures(nn.Module):
    # Enveloppe un modèle timm pour qu'il renvoie la carte de caractéristiques (B, C, H, W) sans la tête.
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_features(x)


class BackboneHead(nn.Module):
    # backbone (extracteur de caractéristiques) -> post (ReLU éventuel) -> pooling global -> tête (Dropout + Linear -> 2)
    def __init__(self, backbone, n_features, dropout=0.2, post=None):
        super().__init__()
        self.backbone = backbone
        self.post = post if post is not None else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, 2))

    def forward(self, x):
        f = self.post(self.backbone(x))
        return self.head(self.pool(f).flatten(1))


def _make_backbone(base, imagenet_weights):
    # Renvoie (backbone, nb_caractéristiques, post, module_cible_pour_charger_des_poids_externes)
    if base == "resnet50":
        net = _torchvision_model("resnet50", imagenet_weights)
        bb = nn.Sequential(*list(net.children())[:9])        # conv1 ... layer4 + avgpool : MÊME découpage que l'exemple officiel RadImageNet
        return bb, 2048, None, bb
    if base == "densenet121":
        net = _torchvision_model("densenet121", imagenet_weights)
        return net.features, 1024, nn.ReLU(), net.features   # torchvision applique un ReLU après `features`
    if base == "inception_v3":
        net = _torchvision_model("inception_v3", imagenet_weights, aux_logits=True, init_weights=False)
        bb = nn.Sequential(*[getattr(net, n) for n in _INCEPTION_V3_BLOCKS])
        return bb, 2048, None, bb
    if base == "inception_resnet_v2":
        import timm                                           # torchvision n'a pas InceptionResNetV2 : on utilise timm
        try:
            m = timm.create_model("inception_resnet_v2", pretrained=imagenet_weights, num_classes=0)
            if imagenet_weights:
                PRETRAINED_STATUS["inception_resnet_v2"] = True
        except Exception as e:
            print(f"[!] Poids ImageNet indisponibles pour inception_resnet_v2 ({type(e).__name__}) -> initialisation ALÉATOIRE.")
            if imagenet_weights:
                PRETRAINED_STATUS["inception_resnet_v2"] = False
            m = timm.create_model("inception_resnet_v2", pretrained=False, num_classes=0)
        return TimmFeatures(m), 1536, None, m
    raise ValueError(f"Backbone inconnu : {base}")


_DROP_KEYS = re.compile(r"(^|\.)(AuxLogits|aux_logits|fc|classifier|classif|head|last_linear|logits)\.")
_PREFIXES = ("module.", "model.", "backbone.", "net.", "encoder.", "features.")


def _strip_prefix(key):
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if key.startswith(p):
                key, changed = key[len(p):], True
    return key


def _extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], (dict, nn.Module)):
                return _extract_state_dict(obj[k])
        return obj
    raise TypeError(f"Format de fichier de poids non reconnu : {type(obj).__name__}")


def load_backbone_weights(target, path):
    # Charge des poids externes (RadImageNet) dans le backbone `target`, SANS jamais le faire en silence si ça ne colle pas.
    # Les fichiers peuvent nommer les couches différemment (ex. « backbone.4.0.conv1.weight » vs « layer1.0.conv1.weight »).
    # Stratégie : (1) correspondance exacte des noms ; sinon (2) correspondance par ORDRE si TOUTES les formes concordent.

    obj = torch.load(path, map_location="cpu", weights_only=True)   # mode sûr : seulement des tenseurs

    sd = _extract_state_dict(obj)
    src = [(k, v) for k, v in sd.items() if torch.is_tensor(v) and not _DROP_KEYS.search(k) and not k.endswith("num_batches_tracked")]
    tgt = [(k, v) for k, v in target.state_dict().items() if not k.endswith("num_batches_tracked")]
    src_by_name = {_strip_prefix(k): v for k, v in src}
    if all(k in src_by_name and src_by_name[k].shape == v.shape for k, v in tgt):
        new, mode = {k: src_by_name[k] for k, _ in tgt}, "noms"
    elif len(src) == len(tgt) and all(a[1].shape == b[1].shape for a, b in zip(src, tgt)):
        new, mode = {tk: sv for (tk, _), (_, sv) in zip(tgt, src)}, "ordre"
    else:
        raise RuntimeError(f"Poids incompatibles avec l'architecture : {len(src)} tenseurs dans {Path(path).name} contre "
                           f"{len(tgt)} attendus, ou formes différentes. Fichier mal assorti ou architecture non identique.")
    target.load_state_dict(new, strict=False)
    return mode, len(new)


def _norm_for(arch, norm=None):
    if norm in NORMS:
        return NORMS[norm]
    if arch.startswith("rin_") or arch in ("inception_v3", "inception_resnet_v2"):
        return NORMS["half"]                      # RadImageNet et les Inception « Google » attendent des pixels dans [-1, 1]
    return NORMS["imagenet"]


def build_model(arch, img_size=224, pretrained=True, gray_mean=0.5, gray_std=0.25, dropout=0.2, rin_path=None, norm=None):
    if arch in EXTRA_ARCHS:
        is_rin = arch.startswith("rin_")
        base = arch[4:] if is_rin else arch
        backbone, n_feat, post, target = _make_backbone(base, imagenet_weights=(pretrained and not is_rin))
        if is_rin and pretrained:
            if not rin_path:
                raise FileNotFoundError(f"Poids RadImageNet introuvables pour {arch} (rin_path manquant).")
            mode, n = load_backbone_weights(target, rin_path)
            PRETRAINED_STATUS[arch] = True
            print(f"  [RadImageNet] {Path(rin_path).name} chargé dans {arch} : {n} tenseurs (correspondance par {mode}).")
        mean, std = _norm_for(arch, norm)
        return XRayNet(BackboneHead(backbone, n_feat, dropout, post), arch, "head", mean, std, expand3=True)
    if arch in ("light", "light_bn"):
        net = LightCNN(1, img_size, use_bn=(arch == "light_bn"), dropout=dropout)
        return XRayNet(net, arch, "classifier", [gray_mean], [gray_std], expand3=False)
    imnet_mean, imnet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if arch == "resnet18":
        net = _torchvision_model("resnet18", pretrained)
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, 2))
        head = "fc"
    elif arch == "efficientnet_b0":
        net = _torchvision_model("efficientnet_b0", pretrained)
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, 2))
        head = "classifier"
    elif arch == "densenet121":
        net = _torchvision_model("densenet121", pretrained)
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.classifier.in_features, 2))
        head = "classifier"
    else:
        raise ValueError(f"Architecture inconnue : {arch}")
    return XRayNet(net, arch, head, imnet_mean, imnet_std, expand3=True)


# Explicabilité : Grad-CAM
def cam_target_layer(model):
    # Dernière couche convolutive : celle qui garde l'information spatiale la plus abstraite.
    if model.arch.startswith("light"):
        return model.net.features[-2]            # dernier ReLU, juste avant le max-pooling final
    if model.arch == "resnet18":
        return model.net.layer4
    if model.arch == "efficientnet_b0":
        return model.net.features
    if model.arch == "densenet121":
        return model.net.features.denseblock4
    base = model.arch[4:] if model.arch.startswith("rin_") else model.arch
    if base == "resnet50":
        return model.net.backbone[7]             # layer4 (index 7 dans conv1, bn1, relu, maxpool, layer1..4, avgpool)
    if base == "densenet121":
        return model.net.backbone.denseblock4
    if base == "inception_v3":
        return model.net.backbone[-1]            # Mixed_7c
    if base == "inception_resnet_v2":
        inner = model.net.backbone.m
        last_conv = getattr(inner, "conv2d_7b", None)
        return last_conv if last_conv is not None else list(inner.children())[-4]
    raise ValueError(model.arch)


def grad_cam(model, x, class_idx=None):
    # x est un tenseur de forme (B, 1, H, W) : B images en niveaux de gris,
    # déjà normalisées dans [0, 1] et places sur le même device que le modèle.
    #
    # Principe du Grad-CAM : on veut savoir QUOI dans l'image a poussé le modèle
    # vers sa décision. Pour cela on regarde la dernière couche convolutive celle
    # qui contient encore une information spatiale soit une carte par filtre d'ou le 1 dans le tenseur .
    #
    # On envoie une image, on récupère le score de la classe prédite, puis on
    # rétropropage ce score jusqu'à cette couche. Le gradient obtenu mesure, pour
    # chaque filtre, à quel point il a contribué à la décision. On fait la moyenne
    # de ce gradient sur toute la carte dimensions H et W cela donne un poids
    # unique par filtre qui résume ce filtre a-t-il tiré la décision vers le haut
    # ou vers le bas et de combien".
    #
    # On pondère ensuite les cartes d'activation par ces poids, on somme sur les
    # filtres, et on garde la partie positive (ReLU) : seules les zones qui ont
    # POUSSÉ la décision nous intéressent. Après interpolation à la taille de
    # l'image d'entrée et normalisation dans [0, 1], on obtient une carte de
    # chaleur où les zones claires correspondent aux régions que le modèle a le
    # plus regardées pour décider. C'est cette carte qu'un radiologue pourrait
    # superposer à la radio pour vérifier que le modèle regarde bien les poumons
    # et non un artefact
    model.eval()
    layer = cam_target_layer(model)
    store = {}

    def fwd_hook(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        x = x.clone().requires_grad_(True)       # garantit un graphe de calcul même si le backbone est gelé
        with torch.enable_grad():
            logits = model(x)
            probs = F.softmax(logits, 1)[:, 1]
            if class_idx is None:
                idx = logits.argmax(1)
            else:
                idx = torch.full((logits.shape[0],), int(class_idx), dtype=torch.long, device=logits.device)
            score = logits.gather(1, idx.view(-1, 1)).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
    finally:
        handle.remove()
    act, grad = store["act"], store["grad"]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), probs.detach().cpu().numpy()


#  Chargement et  prédiction
def load_for_inference(weights_path, config_path, device="cpu"):
    cfg = json.loads(Path(config_path).read_text())
    model = build_model(cfg["arch"], cfg["img_size"], pretrained=False,
                        gray_mean=cfg["gray_mean"], gray_std=cfg["gray_std"], dropout=cfg.get("dropout", 0.2),
                        norm=cfg.get("norm"))      # (mean/std sont de toute façon rechargés avec les poids : ce sont des buffers)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.to(device).eval(), cfg


def predict_pil(model, cfg, img, with_cam=True, device="cpu"):
    # Renvoie un dict : probabilité de pneumonie, étiquette selon le seuil clinique, et carte Grad-CAM (H, W) dans [0,1].
    x = preprocess_pil(img, cfg["img_size"]).to(device)
    if with_cam:
        cams, probs = grad_cam(model, x, class_idx=1)
        cam, p = cams[0], float(probs[0])
    else:
        with torch.no_grad():
            p = float(F.softmax(model(x), 1)[0, 1])
        cam = None
    label = CLASS_NAMES[int(p >= cfg["threshold"])]
    return {"prob_pneumonia": p, "label": label, "threshold": cfg["threshold"], "cam": cam,
            "image_gray": (x[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)}

"""CCTB-Net: Context-Conditioned Temporal Behavior Network and 4-term Joint Loss."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from data_pipeline.schema import NUM_CLASSES


class CausalConv1d(nn.Module):
    """1D convolution with left-only causal padding: output at t depends only on <= t."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class FlowEncoder(nn.Module):
    """Pointwise flow encoder E_net(x_net_num, x_net_cat) -> H_net [B, L, d_net]."""

    def __init__(
        self,
        num_features: int,
        cat_vocab_sizes: Sequence[int],
        d_net: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.GELU(),
        )
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for vocab_sz in cat_vocab_sizes:
            v = max(2, int(vocab_sz))
            dim = min(16, max(4, int(round(v ** 0.4)) * 2))
            self.embeddings.append(nn.Embedding(v, dim, padding_idx=0))
            total_emb_dim += dim

        in_dim = 32 + total_emb_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_net),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_net, d_net),
            nn.LayerNorm(d_net),
        )

    def forward(
        self,
        x_net_num: torch.Tensor,
        x_net_cat: torch.Tensor,
        m_net: torch.Tensor,
    ) -> torch.Tensor:
        h_num = self.num_proj(x_net_num)
        emb_list = [
            emb(x_net_cat[..., i].clamp(min=0, max=emb.num_embeddings - 1))
            for i, emb in enumerate(self.embeddings)
        ]
        h = torch.cat([h_num] + emb_list, dim=-1)
        h_net = self.mlp(h)
        return h_net * m_net.unsqueeze(-1).to(h_net.dtype)


class ContextEncoder(nn.Module):
    """Causal temporal context encoder E_ctx(x_ctx, m_ctx) -> h_ctx [B, d_ctx]."""

    def __init__(self, ctx_features: int, obs_channels: int = 3, d_ctx: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = ctx_features + obs_channels
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_ctx),
            nn.GELU(),
        )
        self.conv1 = CausalConv1d(d_ctx, d_ctx, kernel_size=3, dilation=1, use_weight_norm=False)
        self.conv2 = CausalConv1d(d_ctx, d_ctx, kernel_size=3, dilation=2, use_weight_norm=False)
        self.norm = nn.LayerNorm(d_ctx)
        self.dropout = nn.Dropout(dropout)
        self.attn_score = nn.Linear(d_ctx, 1)

    def forward(self, x_ctx: torch.Tensor, m_ctx: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_ctx, m_ctx.to(x_ctx.dtype)], dim=-1)
        h = self.in_proj(inp)  # [B, T, d_ctx]
        # Causal conv over T dimension
        h_t = h.transpose(1, 2)
        h_t = F.gelu(self.conv1(h_t))
        h_t = self.dropout(F.gelu(self.conv2(h_t))) + h.transpose(1, 2)
        h_seq = self.norm(h_t.transpose(1, 2))  # [B, T, d_ctx]

        # Masked attention pooling (prefer buckets with at least one observed source)
        any_obs = m_ctx.any(dim=-1)  # [B, T]
        scores = self.attn_score(h_seq).squeeze(-1).float()  # [B, T]
        has_any = any_obs.any(dim=-1, keepdim=True)
        eff_mask = torch.where(has_any, any_obs, torch.ones_like(any_obs))
        scores = scores.masked_fill(~eff_mask, -1e9)
        weights = torch.softmax(scores, dim=-1).to(h_seq.dtype)
        h_ctx = torch.sum(h_seq * weights.unsqueeze(-1), dim=1)  # [B, d_ctx]
        return h_ctx


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma = 1 + tanh(Delta_gamma), H_tilde = gamma * H + beta."""

    def __init__(
        self,
        d_ctx: int,
        d_feat: int,
        use_gamma: bool = True,
        use_beta: bool = True,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.net = nn.Sequential(
            nn.Linear(d_ctx, d_feat),
            nn.GELU(),
            nn.Linear(d_feat, 2 * d_feat),
        )
        # Identity initialization: Delta_gamma = 0 -> gamma = 1, beta = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h_net: torch.Tensor,
        h_ctx: torch.Tensor,
        m_net: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_gb = self.net(h_ctx)  # [B, 2 * d_feat]
        delta_gamma, beta = torch.chunk(raw_gb, 2, dim=-1)
        gamma = 1.0 + torch.tanh(delta_gamma) if self.use_gamma else torch.ones_like(delta_gamma)
        if not self.use_beta:
            beta = torch.zeros_like(beta)
        out = gamma.unsqueeze(1) * h_net + beta.unsqueeze(1)
        if m_net is not None:
            out = out * m_net.unsqueeze(-1).to(out.dtype)
        return out, gamma, beta


class TCNResidualBlock(nn.Module):
    """Causal dilated residual block with weight normalization."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, m_net: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, C]
        residual = x
        h = x.transpose(1, 2)
        h = self.dropout(F.gelu(self.conv1(h)))
        h = self.dropout(F.gelu(self.conv2(h)))
        out = self.norm(h.transpose(1, 2) + residual)
        if m_net is not None:
            out = out * m_net.unsqueeze(-1).to(out.dtype)
        return out


class AnomalyBranch(nn.Module):
    """Context-conditioned dynamic normal center c(h_ctx) and continuous deviation score s_anom."""

    def __init__(self, d_in: int, d_ctx: int, d_anom: int = 32, use_ctx: bool = True) -> None:
        super().__init__()
        self.use_ctx = use_ctx
        # Bias-free flow projection prevents trivial constant collapse in Deep SAD
        self.flow_proj = nn.Sequential(
            nn.Linear(d_in, d_in, bias=False),
            nn.GELU(),
            nn.Linear(d_in, d_anom, bias=False),
        )
        if use_ctx:
            self.center_mlp = nn.Sequential(
                nn.Linear(d_ctx, d_anom),
                nn.GELU(),
                nn.Linear(d_anom, d_anom),
            )
        else:
            self.fixed_center = nn.Parameter(torch.randn(1, d_anom) * 0.1)

        self.tau = nn.Parameter(torch.tensor(1.0))
        self.raw_kappa = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor, h_ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        u = self.flow_proj(z).float()  # [B, d_anom]
        if self.use_ctx:
            c = self.center_mlp(h_ctx).float()
        else:
            c = self.fixed_center.expand(z.size(0), -1).float()

        s_anom = torch.sum((u - c) ** 2, dim=-1)  # [B], non-negative squared Euclidean distance
        kappa = F.softplus(self.raw_kappa.float()) + 1e-3
        anom_logit = (torch.log1p(s_anom) - self.tau.float()) / kappa
        p_anom = torch.sigmoid(anom_logit)
        return {
            "u": u,
            "c": c,
            "s_anom": s_anom,
            "anom_logit": anom_logit,
            "p_anom": p_anom,
        }


class CCTBNet(nn.Module):
    """Context-Conditioned Temporal Behavior Network (CCTB-Net)."""

    def __init__(self, cfg: Dict[str, Any], meta: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        self.d_net = int(mcfg.get("d_net", 64))
        self.d_ctx = int(mcfg.get("d_ctx", 32))
        self.d_anom = int(mcfg.get("d_anom", 32))
        self.tcn_channels = int(mcfg.get("tcn_channels", 64))
        self.tcn_kernel = int(mcfg.get("tcn_kernel", 3))
        self.tcn_dilations = list(mcfg.get("tcn_dilations", [1, 2, 4, 8]))
        self.dropout = float(mcfg.get("dropout", 0.1))

        self.use_ctx = bool(mcfg.get("use_ctx", True))
        self.shuffle_ctx = bool(mcfg.get("shuffle_ctx", False))
        self.fusion = str(mcfg.get("fusion", "film")).lower()
        self.film_multilayer = bool(mcfg.get("film_multilayer", False))
        self.use_anomaly_branch = bool(mcfg.get("use_anomaly_branch", True))
        self.use_threat_branch = bool(mcfg.get("use_threat_branch", True))

        self.flow_encoder = FlowEncoder(
            num_features=int(meta["num_features"]),
            cat_vocab_sizes=meta["cat_vocab_sizes"],
            d_net=self.d_net,
            dropout=self.dropout,
        )
        if self.use_ctx:
            self.ctx_encoder = ContextEncoder(
                ctx_features=int(meta["ctx_features"]),
                obs_channels=3,
                d_ctx=self.d_ctx,
                dropout=self.dropout,
            )
            if self.fusion == "film":
                self.film_in = FiLMLayer(
                    d_ctx=self.d_ctx,
                    d_feat=self.d_net,
                    use_gamma=bool(mcfg.get("film_use_gamma", True)),
                    use_beta=bool(mcfg.get("film_use_beta", True)),
                )
                if self.film_multilayer:
                    self.film_blocks = nn.ModuleList(
                        [
                            FiLMLayer(
                                d_ctx=self.d_ctx,
                                d_feat=self.tcn_channels,
                                use_gamma=bool(mcfg.get("film_use_gamma", True)),
                                use_beta=bool(mcfg.get("film_use_beta", True)),
                            )
                            for _ in self.tcn_dilations
                        ]
                    )
            else:
                self.concat_proj = nn.Linear(self.d_net + self.d_ctx, self.d_net)

        self.input_proj = (
            nn.Identity()
            if self.d_net == self.tcn_channels
            else nn.Linear(self.d_net, self.tcn_channels)
        )
        self.tcn_blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    channels=self.tcn_channels,
                    kernel_size=self.tcn_kernel,
                    dilation=int(d),
                    dropout=self.dropout,
                )
                for d in self.tcn_dilations
            ]
        )

        self.anomaly_branch = AnomalyBranch(
            d_in=self.tcn_channels,
            d_ctx=self.d_ctx,
            d_anom=self.d_anom,
            use_ctx=self.use_ctx,
        )

        threat_in_dim = self.tcn_channels + (self.d_ctx if self.use_ctx else 0) + 1
        self.threat_head = nn.Sequential(
            nn.LayerNorm(threat_in_dim),
            nn.Linear(threat_in_dim, self.tcn_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.tcn_channels, NUM_CLASSES),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_net_num = batch["x_net_num"]
        x_net_cat = batch["x_net_cat"]
        m_net = batch["m_net"]
        x_ctx = batch["x_ctx"]
        m_ctx = batch["m_ctx"]

        h_net = self.flow_encoder(x_net_num, x_net_cat, m_net)  # [B, L, d_net]
        if self.use_ctx:
            if self.shuffle_ctx and x_ctx.size(0) > 1:
                perm = torch.randperm(x_ctx.size(0), device=x_ctx.device)
                x_ctx = x_ctx[perm]
                m_ctx = m_ctx[perm]
            h_ctx = self.ctx_encoder(x_ctx, m_ctx)  # [B, d_ctx]
            if self.fusion == "film":
                h_mod, gamma, beta = self.film_in(h_net, h_ctx, m_net=m_net)
            else:
                ctx_exp = h_ctx.unsqueeze(1).expand(-1, h_net.size(1), -1)
                h_mod = self.concat_proj(torch.cat([h_net, ctx_exp], dim=-1))
                h_mod = h_mod * m_net.unsqueeze(-1).to(h_mod.dtype)
                gamma = torch.ones(h_net.size(0), self.d_net, device=h_net.device, dtype=h_net.dtype)
                beta = torch.zeros_like(gamma)
        else:
            h_ctx = torch.zeros(h_net.size(0), self.d_ctx, device=h_net.device, dtype=h_net.dtype)
            h_mod = h_net
            gamma = torch.ones(h_net.size(0), self.d_net, device=h_net.device, dtype=h_net.dtype)
            beta = torch.zeros_like(gamma)

        h_seq = self.input_proj(h_mod)
        for idx, block in enumerate(self.tcn_blocks):
            if self.use_ctx and self.fusion == "film" and self.film_multilayer:
                h_seq, _, _ = self.film_blocks[idx](h_seq, h_ctx, m_net=m_net)
            h_seq = block(h_seq, m_net=m_net)

        # Target flow is always aligned at the final window step L - 1
        z = h_seq[:, -1, :]  # [B, tcn_channels]

        anom_out = self.anomaly_branch(z, h_ctx)
        s_anom = anom_out["s_anom"]
        anom_logit = anom_out["anom_logit"]
        p_anom = anom_out["p_anom"]

        if self.use_threat_branch:
            s_feat = torch.log1p(s_anom.detach()).unsqueeze(-1).to(z.dtype)
            threat_parts = [z, h_ctx, s_feat] if self.use_ctx else [z, s_feat]
            logits = self.threat_head(torch.cat(threat_parts, dim=-1)).float()
        else:
            # Anomaly-only ablation: distribute anomaly logit across attack classes
            b = z.size(0)
            logits = torch.zeros(b, NUM_CLASSES, device=z.device, dtype=torch.float32)
            logits[:, 1:] = (anom_logit.unsqueeze(-1) - float(np.log(NUM_CLASSES - 1)))

        return {
            "logits": logits,
            "s_anom": s_anom,
            "anom_logit": anom_logit,
            "p_anom": p_anom,
            "z": z,
            "h_ctx": h_ctx,
            "gamma": gamma,
            "beta": beta,
            "u": anom_out["u"],
            "c": anom_out["c"],
        }


class CCTBJointLoss(nn.Module):
    """Four-term joint loss: L_cls + L_bin + L_SAD + L_cons."""

    def __init__(self, cfg: Dict[str, Any], train_class_counts: Sequence[int]) -> None:
        super().__init__()
        lcfg = cfg["loss"]
        mcfg = cfg["model"]
        self.use_anomaly_branch = bool(mcfg.get("use_anomaly_branch", True))
        self.use_threat_branch = bool(mcfg.get("use_threat_branch", True))

        self.lambda_cls = float(lcfg.get("lambda_cls", 1.0)) if self.use_threat_branch else 0.0
        self.lambda_bin = float(lcfg.get("lambda_bin", 0.5)) if self.use_anomaly_branch else 0.0
        self.lambda_sad = float(lcfg.get("lambda_sad", 0.1)) if self.use_anomaly_branch else 0.0
        self.lambda_cons = (
            float(lcfg.get("lambda_cons", 0.1))
            if (self.use_anomaly_branch and self.use_threat_branch)
            else 0.0
        )

        self.logit_adjust_tau = float(lcfg.get("logit_adjust_tau", 1.0))
        self.normal_pos_weight = float(lcfg.get("normal_pos_weight", 2.0))
        self.sad_eta = float(lcfg.get("sad_eta", 1.0))
        self.sad_eps = float(lcfg.get("sad_eps", 0.1))

        counts = np.asarray(train_class_counts, dtype=np.float64)
        smoothed = np.maximum(counts, 1.0)
        priors = smoothed / smoothed.sum()
        log_priors = torch.tensor(np.log(priors), dtype=torch.float32)
        self.register_buffer("log_priors", log_priors)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        y_cls: torch.Tensor,
        y_bin: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"].float()
        s_anom = outputs["s_anom"].float()
        anom_logit = outputs["anom_logit"].float()
        y_cls = y_cls.long()
        y_bin = y_bin.float()

        # 1. Threat classification loss (Logit-Adjusted Cross-Entropy)
        if self.lambda_cls > 0:
            adj_logits = logits + self.logit_adjust_tau * self.log_priors.unsqueeze(0)
            loss_cls = F.cross_entropy(adj_logits, y_cls)
        else:
            loss_cls = torch.zeros((), device=logits.device)

        # 2. Binary anomaly loss (normal-weighted BCE to counter inverted imbalance)
        if self.lambda_bin > 0:
            sample_w = torch.where(y_bin < 0.5, self.normal_pos_weight, 1.0)
            loss_bin = F.binary_cross_entropy_with_logits(anom_logit, y_bin, weight=sample_w)
        else:
            loss_bin = torch.zeros((), device=logits.device)

        # 3. Context-conditioned Deep SAD loss
        if self.lambda_sad > 0:
            is_norm = y_bin < 0.5
            is_atk = ~is_norm
            sad_norm = s_anom[is_norm].mean() if is_norm.any() else torch.zeros((), device=logits.device)
            sad_atk = (
                (1.0 / (s_anom[is_atk] + self.sad_eps)).mean()
                if is_atk.any()
                else torch.zeros((), device=logits.device)
            )
            loss_sad = sad_norm + self.sad_eta * sad_atk
        else:
            loss_sad = torch.zeros((), device=logits.device)

        # 4. Dual-branch symmetric consistency loss
        if self.lambda_cons > 0:
            cls_atk_logit = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
            p_cls_atk = torch.sigmoid(cls_atk_logit)
            p_anom = torch.sigmoid(anom_logit)
            loss_cons = 0.5 * (
                F.binary_cross_entropy_with_logits(anom_logit, p_cls_atk.detach())
                + F.binary_cross_entropy_with_logits(cls_atk_logit, p_anom.detach())
            )
        else:
            loss_cons = torch.zeros((), device=logits.device)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_bin * loss_bin
            + self.lambda_sad * loss_sad
            + self.lambda_cons * loss_cons
        )
        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_bin": loss_bin.detach(),
            "loss_sad": loss_sad.detach(),
            "loss_cons": loss_cons.detach(),
        }

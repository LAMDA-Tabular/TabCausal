import math
import numpy as np


class PFNDatasetGenerator:
    """
    Heterogeneous explicit graph generator for offline benchmark data generation.

    Output format:
      - x: (n, f, 2)
      - g: (f, f)
      - mask: (f,) or the padded shape (pad_to_dim,)
      - f: original feature dimension
    """

    def __init__(
        self,
        config_mode="default",
        batch_size=32,
        min_f=4,
        max_f=20,
        seed=0,
    ):
        self.config_mode = config_mode
        self.batch_size = batch_size
        self.min_f = min_f
        self.max_f = max_f
        self.rng = np.random.default_rng(seed)

    # =========================
    # Basic helpers
    # =========================
    def _sample_feature_dim(self):
        return int(self.rng.integers(self.min_f, self.max_f + 1))

    def _sample_obs_int(self):
        # Default mixed/observational split used by the lightweight generator.
        if self.rng.random() < 0.75:
            return 100, 100
        else:
            return 200, 0

    def _make_two_channel(self, x, interv_mask):
        x = x.astype(np.float32)
        if interv_mask is None:
            interv_mask = np.zeros_like(x, dtype=np.float32)
        else:
            interv_mask = interv_mask.astype(np.float32)
        return np.stack([x, interv_mask], axis=-1)

    def _pad_dataset(self, x2, g, f_max):
        n, f, _ = x2.shape
        pad_f = f_max - f

        if pad_f <= 0:
            feature_mask = np.ones(f, dtype=np.float32)
            return x2, g, feature_mask

        x_padded = np.pad(
            x2,
            pad_width=((0, 0), (0, pad_f), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        g_padded = np.pad(
            g,
            pad_width=((0, pad_f), (0, pad_f)),
            mode="constant",
            constant_values=0,
        )

        feature_mask = np.zeros(f_max, dtype=np.float32)
        feature_mask[:f] = 1.0

        return x_padded, g_padded, feature_mask

    def _positive_partition(self, total, parts):
        assert total >= parts >= 1
        if parts == 1:
            return [total]

        cuts = sorted(self.rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
        segs = []
        prev = 0
        for c in cuts + [total]:
            segs.append(c - prev)
            prev = c
        return segs

    # =========================
    # Heterogeneous hyperparameter sampling
    # =========================
    def _softmax_np(self, x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / ex.sum()

    def _sample_meta_gamma(self, max_alpha, max_scale, lower_bound=0, do_round=False):
        alpha = math.exp(self.rng.uniform(0.0, math.log(max_alpha)))
        scale = self.rng.uniform(0.0, max_scale)
        val = lower_bound + self.rng.gamma(shape=alpha, scale=(scale / alpha if alpha > 0 else 1.0))
        return int(round(val)) if do_round else float(val)

    def _sample_meta_beta(self, scale, min_v, max_v):
        b = self.rng.uniform(min_v, max_v)
        k = self.rng.uniform(min_v, max_v)
        return float(scale * self.rng.beta(b, k))

    def _sample_meta_trunc_norm_log_scaled(
        self,
        max_mean,
        min_mean,
        lower_bound=0.0,
        do_round=False,
        min_std=0.01,
        max_std=1.0,
    ):
        log_mean = self.rng.uniform(math.log(min_mean), math.log(max_mean))
        log_std = self.rng.uniform(math.log(min_std), math.log(max_std))
        mean = math.exp(log_mean)
        std = mean * math.exp(log_std)

        for _ in range(32):
            x = self.rng.normal(mean, std)
            if x >= 0.0:
                val = lower_bound + x
                return int(round(val)) if do_round else float(val)

        val = lower_bound + max(0.0, mean)
        return int(round(val)) if do_round else float(val)

    def _sample_meta_choice(self, choice_values, low=-3.0, high=5.0):
        if len(choice_values) == 1:
            return choice_values[0]
        logits = [1.0]
        for _ in range(1, len(choice_values)):
            logits.append(self.rng.uniform(low, high))
        probs = self._softmax_np(np.array(logits, dtype=np.float64))
        idx = int(self.rng.choice(len(choice_values), p=probs))
        return choice_values[idx]

    def _sample_meta_choice_mixed(self, choice_values, low=-5.0, high=6.0):
        if len(choice_values) == 1:
            return choice_values[0]
        logits = [1.0]
        for _ in range(1, len(choice_values)):
            logits.append(self.rng.uniform(low, high))
        probs = self._softmax_np(np.array(logits, dtype=np.float64))
        idx = int(self.rng.choice(len(choice_values), p=probs))
        return choice_values[idx]

    def _sample_old_pfn_mlp_hparams(self, f):
        """
        Use broad causal/MLP-side hyperparameter variation for the heterogeneous family.
        """
        num_layers = self._sample_meta_gamma(max_alpha=2, max_scale=3, lower_bound=2, do_round=True)
        hidden_dim = self._sample_meta_gamma(max_alpha=3, max_scale=100, lower_bound=4, do_round=True)
        dropout_prob = self._sample_meta_beta(scale=0.6, min_v=0.1, max_v=5.0)
        noise_std = self._sample_meta_trunc_norm_log_scaled(
            max_mean=0.3, min_mean=1e-4, lower_bound=0.0, do_round=False
        )
        init_std = self._sample_meta_trunc_norm_log_scaled(
            max_mean=10.0, min_mean=1e-2, lower_bound=0.0, do_round=False
        )
        num_causes = self._sample_meta_gamma(max_alpha=3, max_scale=7, lower_bound=2, do_round=True)

        is_causal = True
        pre_sample_weights = self._sample_meta_choice([True, False])
        y_is_effect = self._sample_meta_choice([True, False])
        block_wise_dropout = self._sample_meta_choice([True, False])
        sort_features = self._sample_meta_choice([True, False])
        in_clique = self._sample_meta_choice([True, False])
        activation_name = self._sample_meta_choice_mixed(["tanh", "identity", "relu"])

        hidden_dim = max(hidden_dim, 1 + 2 * f)

        return {
            "num_layers": int(max(2, num_layers)),
            "prior_mlp_hidden_dim": int(hidden_dim),
            "prior_mlp_dropout_prob": float(min(max(dropout_prob, 0.0), 0.99)),
            "noise_std": float(max(noise_std, 1e-6)),
            "init_std": float(max(init_std, 1e-6)),
            "num_causes": int(max(2, min(num_causes, f))),
            "is_causal": bool(is_causal),
            "pre_sample_weights": bool(pre_sample_weights),
            "y_is_effect": bool(y_is_effect),
            "sampling": "normal",
            "prior_mlp_activations": activation_name,
            "block_wise_dropout": bool(block_wise_dropout),
            "sort_features": bool(sort_features),
            "in_clique": bool(in_clique),
            "pre_sample_causes": True,
            "prior_mlp_scale_weights_sqrt": True,
            "random_feature_rotation": True,
        }

    # =========================
    # Graph construction
    # =========================
    def _get_activation_fn(self, name):
        if name == "tanh":
            return np.tanh
        if name == "identity":
            return lambda x: x
        if name == "relu":
            return lambda x: np.maximum(x, 0.0)
        raise ValueError(name)

    def _sample_weight_matrix(
        self,
        in_dim,
        out_dim,
        init_std,
        dropout_prob,
        block_wise_dropout,
        scale_weights_sqrt=True,
        dense_bias=False,
    ):
        W = np.zeros((in_dim, out_dim), dtype=np.float32)

        if block_wise_dropout:
            n_blocks = int(self.rng.integers(1, max(2, math.ceil(math.sqrt(min(out_dim, in_dim))) + 1)))
            w = max(1, out_dim // n_blocks)
            h = max(1, in_dim // n_blocks)
            keep_prob = (n_blocks * w * h) / max(W.size, 1)
            denom = keep_prob ** (0.5 if scale_weights_sqrt else 1.0)
            std = init_std / max(denom, 1e-6)

            for block in range(n_blocks):
                r0 = w * block
                r1 = min(out_dim, w * (block + 1))
                c0 = h * block
                c1 = min(in_dim, h * (block + 1))
                W[c0:c1, r0:r1] = self.rng.normal(0.0, std, size=(c1 - c0, r1 - r0)).astype(np.float32)
        else:
            keep = max(1.0 - dropout_prob, 1e-3)
            std = init_std / (keep ** (0.5 if scale_weights_sqrt else 1.0))
            W = self.rng.normal(0.0, std, size=(in_dim, out_dim)).astype(np.float32)

            if dropout_prob > 0.0:
                eff_keep = max(keep, 0.50) if dense_bias else keep
                mask = self.rng.binomial(1, eff_keep, size=(in_dim, out_dim)).astype(np.float32)
                W *= mask

        for j in range(out_dim):
            if not np.any(np.abs(W[:, j]) > 1e-12):
                i = int(self.rng.integers(0, in_dim))
                W[i, j] = self.rng.normal(0.0, max(init_std, 1e-4))

        return W.astype(np.float32)

    def _build_graph_bundle(self, f, hp):
        if f < 2:
            raise ValueError("f must be >= 2")

        num_layers = int(max(2, min(hp["num_layers"], f)))
        num_roots = int(max(1, min(hp["num_causes"], f - (num_layers - 1))))
        remaining = f - num_roots
        downstream_layers = max(1, num_layers - 1)

        if remaining < downstream_layers:
            num_layers = remaining + 1
            downstream_layers = num_layers - 1

        widths = [num_roots]
        if remaining > 0:
            widths += self._positive_partition(remaining, downstream_layers)
        else:
            widths += []

        layers = []
        cursor = 0
        for w in widths:
            nodes = np.arange(cursor, cursor + w, dtype=np.int32)
            layers.append(nodes)
            cursor += w

        A = np.zeros((f, f), dtype=np.int32)
        Ws = []
        noise_vecs = []

        for li in range(1, len(layers)):
            prev_nodes = layers[li - 1]
            cur_nodes = layers[li]

            dropout_prob = 0.0 if li == 1 else hp["prior_mlp_dropout_prob"]
            dense_bias = bool(hp["in_clique"])

            W = self._sample_weight_matrix(
                in_dim=len(prev_nodes),
                out_dim=len(cur_nodes),
                init_std=hp["init_std"],
                dropout_prob=dropout_prob,
                block_wise_dropout=hp["block_wise_dropout"],
                scale_weights_sqrt=hp["prior_mlp_scale_weights_sqrt"],
                dense_bias=dense_bias,
            )

            for i_local, i_node in enumerate(prev_nodes):
                for j_local, j_node in enumerate(cur_nodes):
                    if abs(W[i_local, j_local]) > 1e-12:
                        A[i_node, j_node] = 1

            if hp["pre_sample_weights"]:
                noise_vec = np.abs(self.rng.normal(0.0, hp["noise_std"], size=(len(cur_nodes),))).astype(np.float32)
            else:
                noise_vec = np.full((len(cur_nodes),), hp["noise_std"], dtype=np.float32)

            Ws.append(W.astype(np.float32))
            noise_vecs.append(noise_vec)

        return {
            "A": A,
            "layers": layers,
            "Ws": Ws,
            "noise_vecs": noise_vecs,
            "activation": self._get_activation_fn(hp["prior_mlp_activations"]),
            "hp": hp,
        }

    # =========================
    # Sampling values
    # =========================
    def _causes_sampler(self, num_causes):
        means = self.rng.normal(0.0, 1.0, size=(num_causes,))
        stds = np.abs(self.rng.normal(0.0, 1.0, size=(num_causes,)) * means)
        stds = np.where(stds < 1e-6, 1e-6, stds)
        return means.astype(np.float32), stds.astype(np.float32)

    def _sample_root_values(self, n_samples, num_causes, sampling, pre_sample_causes):
        if pre_sample_causes:
            means, stds = self._causes_sampler(num_causes)
        else:
            means = np.zeros((num_causes,), dtype=np.float32)
            stds = np.ones((num_causes,), dtype=np.float32)

        if sampling == "normal":
            return self.rng.normal(
                loc=means[None, :],
                scale=np.abs(stds[None, :]),
                size=(n_samples, num_causes),
            ).astype(np.float32)

        if sampling == "uniform":
            return self.rng.random(size=(n_samples, num_causes)).astype(np.float32)

        if sampling == "mixed":
            out = np.zeros((n_samples, num_causes), dtype=np.float32)
            zipf_p = self.rng.random() * 0.66
            multi_p = self.rng.random() * 0.66
            normal_p = self.rng.random() * 0.66

            for c in range(num_causes):
                r = self.rng.random()
                if r > normal_p:
                    out[:, c] = self.rng.normal(loc=means[c], scale=abs(stds[c]), size=(n_samples,)).astype(np.float32)
                elif r > multi_p:
                    k = int(self.rng.integers(2, 11))
                    x = self.rng.integers(0, k, size=(n_samples,)).astype(np.float32)
                    x = (x - x.mean()) / (x.std() + 1e-6)
                    out[:, c] = x
                else:
                    a = 2.0 + self.rng.random() * 2.0
                    x = np.minimum(self.rng.zipf(a, size=(n_samples,)).astype(np.float32), 10.0)
                    x = x - x.mean()
                    out[:, c] = x
            return out

        raise ValueError(f"Unknown sampling mode: {sampling}")

    def _make_balanced_intervention_targets(self, f, n_int):
        """
        Assign intervention targets as evenly as possible across variables.

        For example, n_int=100 and f=30 gives each variable 3 interventions,
        assigns the remaining 10 interventions to randomly chosen variables,
        then shuffles the target sequence.
        """
        if n_int <= 0:
            return np.zeros((0,), dtype=np.int32)

        base = n_int // f
        rem = n_int % f

        targets = np.repeat(np.arange(f, dtype=np.int32), base)

        if rem > 0:
            extra = self.rng.choice(np.arange(f, dtype=np.int32), size=rem, replace=False)
            targets = np.concatenate([targets, extra], axis=0)

        assert len(targets) == n_int
        self.rng.shuffle(targets)
        return targets.astype(np.int32)

    def _simulate_dataset(self, graph_bundle, n_obs, n_int):
        A = graph_bundle["A"]
        layers = graph_bundle["layers"]
        Ws = graph_bundle["Ws"]
        noise_vecs = graph_bundle["noise_vecs"]
        act = graph_bundle["activation"]
        hp = graph_bundle["hp"]

        f = A.shape[0]
        n_total = n_obs + n_int
        X = np.zeros((n_total, f), dtype=np.float32)
        I = np.zeros((n_total, f), dtype=np.float32)

        # Balanced single-target intervention schedule.
        if n_int > 0:
            int_targets = self._make_balanced_intervention_targets(f=f, n_int=n_int)
            int_values = self.rng.normal(0.0, 2.0, size=(n_int,)).astype(np.float32)
            I[np.arange(n_obs, n_obs + n_int), int_targets] = 1.0
        else:
            int_targets = np.zeros((0,), dtype=np.int32)
            int_values = np.zeros((0,), dtype=np.float32)

        root_nodes = layers[0]
        root_vals = self._sample_root_values(
            n_samples=n_total,
            num_causes=len(root_nodes),
            sampling=hp["sampling"],
            pre_sample_causes=hp["pre_sample_causes"],
        )
        X[:, root_nodes] = root_vals

        if n_int > 0:
            for idx_local in range(n_int):
                row = n_obs + idx_local
                t = int_targets[idx_local]
                if t in root_nodes:
                    X[row, t] = int_values[idx_local]

        for li in range(1, len(layers)):
            prev_nodes = layers[li - 1]
            cur_nodes = layers[li]
            W = Ws[li - 1]
            noise_vec = noise_vecs[li - 1]

            prev = X[:, prev_nodes]
            prev_act = act(prev)
            cur = prev_act @ W + self.rng.normal(
                0.0, noise_vec[None, :], size=(n_total, len(cur_nodes))
            ).astype(np.float32)

            if n_int > 0:
                for idx_local in range(n_int):
                    row = n_obs + idx_local
                    t = int_targets[idx_local]
                    where = np.where(cur_nodes == t)[0]
                    if len(where) > 0:
                        cur[row, where[0]] = int_values[idx_local]

            cur = np.clip(cur, -1e6, 1e6).astype(np.float32)
            X[:, cur_nodes] = cur

        if hp["random_feature_rotation"] and f > 1:
            if hp["sort_features"]:
                shift = int(self.rng.integers(0, f))
                perm = np.roll(np.arange(f), shift)
            else:
                perm = self.rng.permutation(f)
            X = X[:, perm]
            I = I[:, perm]
            A = A[perm][:, perm]

        return A.astype(np.int32), X.astype(np.float32), I.astype(np.float32)

    # =========================
    # Public API
    # =========================
    def generate_batch(self):
        datasets = []
        for _ in range(self.batch_size):
            f = self._sample_feature_dim()
            n_obs, n_int = self._sample_obs_int()

            hp = self._sample_old_pfn_mlp_hparams(f)
            graph_bundle = self._build_graph_bundle(f, hp)
            g, x, interv = self._simulate_dataset(graph_bundle, n_obs=n_obs, n_int=n_int)

            x2 = self._make_two_channel(x, interv)
            datasets.append((f, x2, g))

        f_max = max(f for f, _, _ in datasets)
        xs, gs, masks = [], [], []
        for f, x2, g in datasets:
            x_pad, g_pad, mask_pad = self._pad_dataset(x2, g, f_max)
            xs.append(x_pad)
            gs.append(g_pad)
            masks.append(mask_pad)

        return {
            "x": np.stack(xs),
            "g": np.stack(gs),
            "mask": np.stack(masks),
        }

    def generate_single_test(self, f, n_obs=None, n_int=None, n_samples=None, pad_to_dim=None):
        """
        Public single-graph generation helper.

        Prefer n_obs/n_int. For compatibility, n_samples maps to
        n_obs=n_samples and n_int=0.
        """
        if n_samples is not None:
            if n_obs is not None or n_int is not None:
                raise ValueError("Use either n_samples or (n_obs, n_int), not both.")
            n_obs = int(n_samples)
            n_int = 0
        else:
            if n_obs is None or n_int is None:
                raise ValueError("Please provide either n_samples or both n_obs and n_int.")

        hp = self._sample_old_pfn_mlp_hparams(f)
        graph_bundle = self._build_graph_bundle(f, hp)
        g, x, interv = self._simulate_dataset(graph_bundle, n_obs=n_obs, n_int=n_int)
        x2 = self._make_two_channel(x, interv)

        if pad_to_dim is None:
            mask = np.ones(f, dtype=np.float32)
            return {
                "x": x2,
                "g": g,
                "mask": mask,
                "f": f
            }
        else:
            x_padded, g_padded, mask = self._pad_dataset(x2, g, pad_to_dim)
            return {
                "x": x_padded,
                "g": g_padded,
                "mask": mask,
                "f": f
            }

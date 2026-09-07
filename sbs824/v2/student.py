"""Shared permutation-invariant v2 Intent Student implemented in Flax."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class StudentArchitecture:
    edge_hidden: int = 64
    value_hidden: int = 64
    output_hidden: int = 64

    def manifest(self) -> dict:
        return asdict(self)


class IntentStudent(nn.Module):
    """Map one ego feature and an unordered sensed-neighbor set to Intent."""

    architecture: StudentArchitecture = StudentArchitecture()

    @nn.compact
    def __call__(self, self_features, edge_features, edge_mask,
                 *, return_logits: bool = False):
        self_features = jnp.asarray(self_features)
        edge_features = jnp.asarray(edge_features)
        edge_mask = jnp.asarray(edge_mask, dtype=bool)

        edge = nn.silu(nn.Dense(
            self.architecture.edge_hidden, name="edge_dense_1")(
                edge_features))
        edge = nn.silu(nn.Dense(
            self.architecture.edge_hidden, name="edge_dense_2")(edge))
        logits = nn.Dense(1, name="attention_logit")(edge)[..., 0]
        values = nn.silu(nn.Dense(
            self.architecture.value_hidden, name="edge_value")(edge))

        masked_logits = jnp.where(edge_mask, logits, -1.0e30)
        maximum = jnp.max(masked_logits, axis=-1, keepdims=True)
        unnormalized = jnp.where(
            edge_mask, jnp.exp(masked_logits - maximum), 0.0)
        weights = unnormalized / jnp.maximum(
            jnp.sum(unnormalized, axis=-1, keepdims=True), 1.0e-12)
        aggregate = jnp.sum(weights[..., None] * values, axis=-2)

        hidden = jnp.concatenate((self_features, aggregate), axis=-1)
        hidden = nn.silu(nn.Dense(
            self.architecture.output_hidden, name="output_dense_1")(hidden))
        hidden = nn.silu(nn.Dense(
            self.architecture.output_hidden, name="output_dense_2")(hidden))
        logits = nn.Dense(2, name="intent_output")(hidden)
        return logits if return_logits else jnp.tanh(logits)

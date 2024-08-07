#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Page Managers."""

from typing import Optional
from flax import linen as nn
import jax.numpy as jnp

import common_types
import queue
import jax

Array = common_types.Array
DType = common_types.DType

AxisNames = common_types.AxisNames

class PageManager(nn.Module):

  max_num_sequences: int
  max_target_length: int
  page_size: int
  num_pages: int
  max_pages_per_sequence: int = None

  def setup(self):
    """Initialize page manager."""
    self.max_pages_per_sequence = self.max_target_length // self.page_size
    _ = self.variable(
        "cache",
        "page_indices",
        nn.with_logical_partitioning(jnp.arange(self.num_pages, dtype=jnp.int32), ("num_pages",)),
        (self.max_num_sequences, 1),
        jnp.int32,
    )
    _ = self.variable(
        "cache",
        "seq_lengths",
        nn.with_logical_partitioning(jnp.zeros, ("max_num_sequences",)),
        (self.max_num_sequences, 1),
        jnp.int32,
    )
    _ = self.variable(
        "cache",
        "seq_pages",
        nn.with_logical_partitioning(jnp.zeros, ("max_num_sequences",)),
        (self.max_num_sequences, 1),
        jnp.int32,
    )
    _ = self.variable(
        "cache",
        "seq_page_cursor",
        nn.with_logical_partitioning(jnp.zeros, ("max_num_sequences",)),
        (self.max_num_sequences, 1),
        jnp.int32,
    )
    _ = self.variable(
        "cache",
        "seq_page_indices",
        nn.with_logical_partitioning(jnp.zeros, ("max_num_sequences", "max_pages_per_sequence")),
        (self.max_num_sequences, self.max_pages_per_sequence),
        jnp.int32,
    )

  @property
  def page_indices(self) -> nn.Variable:
    return self.variable["cache"]["page_indices"]

  @property
  def seq_lengths(self) -> nn.Variable:
    return self.variable["cache"]["seq_lengths"]

  @property
  def seq_pages(self) -> nn.Variable:
    return self.variable["cache"]["seq_pages"]

  @property
  def seq_page_cursor(self) -> nn.Variable:
    return self.variable["cache"]["seq_page_cursor"]

  @property
  def seq_page_indices(self) -> nn.Variable:
    return self.variable["cache"]["seq_page_indices"]

  def release_slot(self, slot: int) -> None:
    """Release sequence slot and the pages assigned to the slot."""
    for i in range(self.seq_pages.value[slot]):
      page_idx = self.seq_page_indices.value[slot][i]
      self.page_indices.value.at[page_idx].set(0)
    self.seq_lengths.value = jax.lax.dynamic_update_index_in_dim(self.seq_lengths.value, 0, slot, axis=0)
    self.seq_pages.value = jax.lax.dynamic_update_index_in_dim(self.seq_pages.value, 0, slot, axis=0)
    self.seq_page_cursor.value = jax.lax.dynamic_update_index_in_dim(self.seq_page_cursor.value, 0, slot, axis=0)
    empty_page_indices = jnp.zeros(shape=(self.max_pages_per_sequence,), dtype=jnp.int32)
    self.seq_page_indices.value = jax.lax.dynamic_update_index_in_dim(
      self.seq_page_indices.value, empty_page_indices, slot, axis=0
    )

  def assign_prefill_step_pages(self, slot: int, true_length: int) -> None:
    """Allocate pages for sequence slot prefill."""
    seq_pages = jnp.ceil(true_length / self.page_size).astype(jnp.int32)
    seq_page_cursor = (true_length - 1) % self.page_size
    self.seq_lengths.value = jax.lax.dynamic_update_index_in_dim(self.seq_lengths.value, true_length, slot, axis=0)
    self.seq_pages.value = jax.lax.dynamic_update_index_in_dim(self.seq_pages.value, seq_pages, slot, axis=0)
    self.seq_page_cursor.value = jax.lax.dynamic_update_index_in_dim(
      self.seq_page_cursor.value, seq_page_cursor, slot, axis=0
    )
    for i in range(seq_pages):
      assert jnp.count_nonzero(self.page_indices.value[1:]) != self.num_pages-1, "All pages are in use."
      page_idx = jnp.where((self.page_indices.value[1:]==0), size=1)[0]
      self.page_indices.value.at[page_idx].set(1)
      self.seq_page_indices.value = self.seq_page_indices.value.at[slot, i].set(page_idx)

  def assign_decode_step_pages(self) -> None:
    """Allocate pages for decode step."""
    self.seq_lengths.value += jax.lax.cond(self.seq_lengths.value > 0, 1, 0)
    current_seq_pages = self.seq_pages.value
    self.seq_pages.value = jnp.ceil(self.seq_lengths / self.page_size).astype(jnp.int32)
    self.seq_page_cursor.value = (self.seq_lengths.value - 1) % self.page_size
    seq_new_pages = self.seq_pages.value - current_seq_pages
    new_pages = jnp.count_nonzero(seq_new_pages)
    if new_pages:
      updating_slots = jnp.where((seq_new_pages>0), size=self.max_num_sequences)
      for i in range(new_pages):
        assert jnp.count_nonzero(self.page_indices.value[1:]) != self.num_pages-1, "All pages are in use."
        slot = updating_slots[i]
        page_idx = jnp.where((self.page_indices.value[1:]==0), size=1)[0]
        self.page_indices.value.at[page_idx].set(1)
        self.seq_page_indices.value = self.seq_page_indices.value.at[slot, self.seq_pages.value[slot]-1].set(page_idx)

  def update_prefill_step_pages(
      self, key_pages: nn.Variable, value_pages: nn.Variable, key: Array, value: Array, slot: int
  ) -> None:
    """Update pages for prefill step."""
    seq_pages = self.seq_pages[slot]
    key = jnp.transpose(key, axes=(2, 0, 1, 3))
    value = jnp.transpose(value, axes=(2, 0, 1, 3))
    for i in range(seq_pages):
      assigned_page_idx = self.seq_page_indices[slot][i]
      key_block = jax.lax.dynamic_slice_in_dim(key, (i-1) * self.page_size, self.page_size, axis=2)
      value_block = jax.lax.dynamic_slice_in_dim(value, (i-1) * self.page_size, self.page_size, axis=2)
      key_pages.value = jax.lax.dynamic_update_slice_in_dim(key_pages.value, key_block, assigned_page_idx, axis=1)
      value_pages.value = jax.lax.dynamic_update_slice_in_dim(value_pages.value, value_block, assigned_page_idx, axis=1)

  def update_decode_step_pages(self, key_pages: nn.Variable, value_pages: nn.Variable, key: Array, value: Array) -> None:
    """Update pages for decode step."""
    key = jnp.transpose(key, axes=(2, 0, 1, 3))
    value = jnp.transpose(value, axes=(2, 0, 1, 3))
    assigned_page_idx = self.seq_pages - 1
    assigned_page_cursor = self.seq_page_cursor
    key_pages.value = key_pages.value.at[:,assigned_page_idx,assigned_page_cursor,:].set(key)
    value_pages.value = value_pages.value.at[:,assigned_page_idx,assigned_page_cursor,:].set(value)



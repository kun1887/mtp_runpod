from typing import Any, Mapping, Optional

import numpy as np
import torch.utils.data
from itertools import combinations

from torchtune.data._messages import Message
from torchtune.data._utils import truncate
from torchtune.modules.tokenizers import BaseTokenizer, ModelTokenizer
from torchtune.modules.transforms import Transform

from models.tokenizer import vertex_tokenizer


class SiblingDiscoveryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        num_parents: int = 5,
        num_children_per_parent: int = 500,
        training_size: int = 50_000,
        validation_size: int = 10_000,
        test_size: int = 10_000,
        split: str = "train",
        sequence_length: int = 128,
        single_example: bool = False,
        seed: int = 42,
    ):
        # Save and restore outside random state to avoid side effects
        outside_seed = torch.random.get_rng_state()
        torch.manual_seed(seed)

        super(SiblingDiscoveryDataset).__init__()
        self.tokenizer = tokenizer
        self.num_parents = num_parents
        self.num_children_per_parent = num_children_per_parent
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.split = split
        self.sequence_length = sequence_length
        self.seed = seed
        self.single_example = single_example
        if self.single_example:
            self.sequence_length = 5
        assert num_parents < 900
        assert num_children_per_parent < 9000
        assert num_parents * (num_children_per_parent**2) < 10_000_000 # to avoid running out of memory

        self.parents_with_children = self.generate_graph()
        # generate all sibling-sibling-parent triplets
        self.all_triplets = []
        for parent, children in self.parents_with_children.items():
            for i in range(len(children)):
                for j in range(len(children)):
                    if i != j:
                        self.all_triplets.append((children[i], children[j], parent))

        self.permutation = torch.randperm(len(self.all_triplets))
        self.index_splits = {}
        self.index_splits["train"] = self.permutation[:self.training_size]
        i = self.training_size
        self.index_splits["validation"] = self.permutation[i:i + self.validation_size]
        i += self.validation_size
        self.index_splits["test"] = self.permutation[i:i + self.test_size]

        training_triplets = [self.all_triplets[i] for i in self.index_splits["train"]]
        inverted_training_triplets = [(c2, c1, p) for (c1, c2, p) in training_triplets]
        self.training_triplets = set(training_triplets + inverted_training_triplets)

        self.delimiter = self.tokenizer.special_tokens[', ']
        self.bos_token = self.tokenizer.bos_id
        self.eos_token = self.tokenizer.eos_id

        torch.random.set_rng_state(outside_seed)

    def generate_graph(self):
        # sample parents from interval [100, 999]
        parents = torch.randperm(900)[:self.num_parents] + 100
        children = torch.randperm(9000)[:self.num_children_per_parent * self.num_parents] + 1000
        children = children.reshape(self.num_parents, self.num_children_per_parent)
        parents_with_children = {p.item(): children[i].tolist() for i, p in enumerate(parents)}
        return parents_with_children

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        num_triples = 1 if self.single_example else self.sequence_length // 4
        indices = self.index_splits[self.split]

        # sample a number of triplets from the precomputed list
        triplet_indices = indices[torch.randperm(len(indices))[:num_triples]]

        seq = [self.bos_token]
        for idx in triplet_indices:
            child1, child2, parent = self.all_triplets[idx]
            seq.extend([child1, child2, parent, self.delimiter])

        seq = seq[:-1]  # remove last delimiter
        seq.append(self.eos_token)

        attention_mask = [1] * (len(seq)-1)

        return {
            "tokens": seq[:-1],
            "attention_mask": attention_mask,
            "labels": seq[1:],
        }

    def __len__(self):
        if self.split == "train":
            return 1_000_000
        elif self.split == "validation":
            return 1_000
        elif self.split == "test":
            return 1_000
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def split_generated_data(self, generated_sequences: list[list[int]], only_first_item=False):
        generated_data = []
        delimiter = self.delimiter
        for seq in generated_sequences:
            current_subseq = []
            # split by delimiter
            for token in seq:
                if token == delimiter:
                    current_subseq = []
                else:
                    current_subseq.append(token)
                    if len(current_subseq) == 3:
                        generated_data.append(current_subseq)
                        current_subseq = []
                        if only_first_item:
                            break
        return generated_data


    def creativity_score(self, generated_data: list[list[int]], num_items: Optional[int] = None):
        if num_items is None:
            num_items = len(generated_data)

        if len(generated_data) == 0:
            return {
                "creativity_score": 0.0,
                "uniqueness_score": 0.0,
                "novelty_score": 0.0,
                "validity_score": 0.0,
            }

        # only accept triplets
        triplets = [tuple(s) for s in generated_data if len(s) == 3]

        # uniqueness
        unique_triplets = set(triplets)
        uniqueness_score = len(unique_triplets) / len(triplets) if len(triplets) > 0 else 0.0

        # not in training data
        novel_triplets = unique_triplets - self.training_triplets
        novelty_score = len(novel_triplets) / len(unique_triplets) if len(unique_triplets) > 0 else 0.0

        # check if triplets are valid
        valid_triplets = 0
        for triplet in novel_triplets:
            child1, child2, parent = triplet
            if not parent in self.parents_with_children:
                continue
            allowed_children = self.parents_with_children[parent]
            valid = (child1 in allowed_children) and (child2 in allowed_children)
            if valid:
                valid_triplets += 1
        validity_score = valid_triplets / len(novel_triplets) if len(novel_triplets) > 0 else 0.0

        creativity_score = valid_triplets / num_items
        return {
            "creativity_score": creativity_score,
            "uniqueness_score": uniqueness_score,
            "novelty_score": novelty_score,
            "validity_score": validity_score,
        }


def sibling_discovery_dataset(tokenizer: ModelTokenizer, **kwargs):
    ds = SiblingDiscoveryDataset(tokenizer=tokenizer, **kwargs)
    return ds


class TriangleDiscoveryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        num_vertices: int = 999,
        deg: int = 3,
        tri: int = 10,
        training_size: int = 15_000,
        validation_size: int = 1_000,
        test_size: int = 1_000,
        split: str = "train",
        sequence_length: int = 128,
        single_example: bool = False,
        seed: int = 42,
    ):
        # Save and restore outside random state to avoid side effects
        outside_seed = torch.random.get_rng_state()
        torch.manual_seed(seed)

        super(SiblingDiscoveryDataset).__init__()
        self.tokenizer = tokenizer
        self.num_vertices = num_vertices
        self.deg = deg
        self.tri = tri
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.split = split
        self.sequence_length = sequence_length
        self.seed = seed
        self.single_example = single_example
        if self.single_example:
            self.sequence_length = 10
        assert num_vertices < 9000

        self.vertices = (torch.randperm(9000)[:self.num_vertices] + 1000).tolist()
        self.neighbors, self.edges, self.triangles = self.generate_graph()
        self.edges = list(self.edges)

        num_train_triangles = len(self.triangles) // 3
        num_used_triangles = num_train_triangles + self.validation_size + self.test_size
        assert num_used_triangles < len(self.triangles)

        fixed_triangle_set = list(self.triangles)
        fixed_triangle_set = torch.tensor(fixed_triangle_set)[torch.randperm(len(fixed_triangle_set))]
        fixed_triangle_set = [tuple(t.tolist()) for t in fixed_triangle_set]
        self.triangles_splits = {}
        self.triangles_splits["train"] = fixed_triangle_set[:num_train_triangles]
        self.training_triangles = set(self.triangles_splits["train"])
        i = num_train_triangles
        self.triangles_splits["validation"] = fixed_triangle_set[i:i + self.validation_size]
        i += self.validation_size
        self.triangles_splits["test"] = fixed_triangle_set[i:i + self.test_size]

        self.triangle_tok = self.tokenizer.special_tokens['tri: ']
        self.edge_tok = self.tokenizer.special_tokens['edge: ']
        self.delimiter = self.tokenizer.special_tokens[', ']
        self.bos_token = self.tokenizer.bos_id
        self.eos_token = self.tokenizer.eos_id

        torch.random.set_rng_state(outside_seed)

    def generate_graph(self):
        neighbor_dict = {v: set() for v in self.vertices}

        # iterate through vertices and connect them to deg random other vertices
        for v in self.vertices:
            low_degree_vertices = [u for u in neighbor_dict if len(neighbor_dict[u]) <= self.deg * 1.2 and u != v]
            connections = torch.randperm(len(low_degree_vertices))[:self.deg].tolist()
            for c in connections:
                neighbor_dict[v].add(low_degree_vertices[c])
                neighbor_dict[low_degree_vertices[c]].add(v)

        all_triangles = set()
        # now iterate through vertices and try to form triangles
        for v in self.vertices:
            neighbors = list(neighbor_dict[v])
            triangles_v = set()
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and n2 in neighbor_dict[n1]:
                        triangle = tuple(torch.tensor(sorted((v, n1, n2))).tolist())
                        triangles_v.add(triangle)
                        all_triangles.add(triangle)

            for _ in range(self.tri):
                if len(neighbors) < 2:
                    break
                #if len(triangles) >= self.tri:
                #    break
                choices = torch.randperm(len(neighbors))[:2]
                n1 = neighbors[choices[0]]
                n2 = neighbors[choices[1]]
                if n2 not in neighbor_dict[n1]:
                    neighbor_dict[n1].add(n2)
                    neighbor_dict[n2].add(n1)
                triangle = tuple(torch.tensor(sorted((v, n1, n2))).tolist())
                triangles_v.add(triangle)
                all_triangles.add(triangle)

        all_edges = set()
        for v, neighbors in neighbor_dict.items():
            for n in neighbors:
                edge = tuple(torch.tensor(sorted((v, n))).tolist())
                all_edges.add(edge)

        print(f"Generated graph with {len(neighbor_dict)} vertices, {len(all_edges)} edges, an average degree of {np.mean([len(v) for v in neighbor_dict.values()])}, and {len(all_triangles)} triangles")
        return neighbor_dict, all_edges, all_triangles

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        seq = [self.bos_token]
        new_tokens = []

        triangle_range = len(self.triangles_splits[self.split])
        while len(seq) + len(new_tokens) < self.sequence_length:
            seq.extend(new_tokens)
            generate_triangle = (self.split == "validation") or (self.split == "test") or (torch.rand(1).item() < 1/3)
            if generate_triangle:
                # sample random triangle from the precomputed list
                triangle = self.triangles_splits[self.split][torch.randint(0, triangle_range, (1,)).item()]
                # shuffle triangle vertices
                triangle = torch.tensor(triangle)[torch.randperm(3)].tolist()
                new_tokens = [self.triangle_tok,
                              triangle[0], triangle[1], self.delimiter,
                              triangle[1], triangle[2], self.delimiter,
                              triangle[2], triangle[0]]
            else:
                # generate random edge
                edge = self.edges[torch.randint(0, len(self.edges), (1,)).item()]
                edge = torch.tensor(edge)[torch.randperm(2)].tolist()
                new_tokens = [self.edge_tok,
                              edge[0], edge[1], self.delimiter,
                              edge[1], edge[0]]
            if self.single_example:
                break
        if len(seq) <= 1:
            seq.extend(new_tokens)

        seq.append(self.eos_token)

        attention_mask = [1] * (len(seq) - 1)

        return {
            "tokens": seq[:-1],
            "attention_mask": attention_mask,
            "labels": seq[1:],
        }

    def __len__(self):
        if self.split == "train":
            return 1_000_000
        elif self.split == "validation":
            return 1_000
        elif self.split == "test":
            return 1_000
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def split_generated_data(self, generated_sequences: list[list[int]], only_first_item=False):
        generated_data = []
        tri = self.triangle_tok
        edge = self.edge_tok
        for seq in generated_sequences:
            current_subseq = []
            for token in seq:
                if token == tri or token == edge:
                    current_subseq = []
                else:
                    current_subseq.append(token)
                    if len(current_subseq) == 8:
                        generated_data.append(current_subseq)
                        current_subseq = []
                        if only_first_item:
                            break

        return generated_data

    def creativity_score(self, generated_data: list[list[int]], num_items: Optional[int] = None):
        if num_items is None:
            num_items = len(generated_data)

        if len(generated_data) == 0:
            return {
                "creativity_score": 0.0,
                "uniqueness_score": 0.0,
                "novelty_score": 0.0,
                "validity_score": 0.0,
            }

        # only accept lists with 8 numbers ('a', 'b', ',', 'b', 'c', ',', 'c', 'a')
        tuplets = [tuple(s) for s in generated_data if len(s) == 8]

        # uniqueness
        unique_tuplets = set(tuplets)
        uniqueness_score = len(unique_tuplets) / len(tuplets) if len(tuplets) > 0 else 0.0

        # convert to triangles
        generated_triangles = []
        for tup in unique_tuplets:
            if not tup[0] == tup[7] and tup[1] == tup[3] and tup[4] == tup[6]:
                continue

            if not tup[2] == self.delimiter and tup[5] == self.delimiter:
                continue

            triangle = tuple(sorted((tup[0], tup[1], tup[4])))
            generated_triangles.append(triangle)
        unique_triangles = set(generated_triangles)

        # not in training data
        novel_triangles = unique_triangles - self.training_triangles
        novelty_score = len(novel_triangles) / len(unique_triangles) if len(unique_triangles) > 0 else 0.0

        # check if triangles are valid
        valid_triangles = 0
        for triangle in novel_triangles:
            if triangle in self.triangles:
                valid_triangles += 1
        validity_score = valid_triangles / len(novel_triangles) if len(novel_triangles) > 0 else 0.0

        creativity_score = valid_triangles / num_items
        return {
            "creativity_score": creativity_score,
            "uniqueness_score": uniqueness_score,
            "novelty_score": novelty_score,
            "validity_score": validity_score,
        }


def triangle_discovery_dataset(tokenizer: ModelTokenizer, **kwargs):
    ds = TriangleDiscoveryDataset(tokenizer=tokenizer, **kwargs)
    return ds


class CircleConstructionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer: ModelTokenizer,
                 circle_size: int = 9,
                 num_vertices: int = 15,
                 training_size: int = 10_000,
                 validation_size: int = 1_000,
                 test_size: int = 1_000,
                 split: str = "train",
                 sequence_length: int = 128,
                 single_example: bool = False,
                 seed: int = 42):
        # Save and restore outside random state to avoid side effects
        outside_seed = torch.random.get_rng_state()
        torch.manual_seed(seed)

        super(CircleConstructionDataset).__init__()
        self.tokenizer = tokenizer
        self.circle_size = circle_size
        self.num_vertices = num_vertices
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.split = split
        self.sequence_length = sequence_length
        self.seed = seed
        self.single_example = single_example
        if self.single_example:
            self.sequence_length = 3 * circle_size + 1

        assert num_vertices < 90
        assert circle_size < num_vertices

        self.vertices = torch.randperm(90)[:self.num_vertices] + 10
        self.bos_token = self.tokenizer.bos_id
        self.eos_token = self.tokenizer.eos_id
        self.delimiter = self.tokenizer.special_tokens[', ']
        self.circle_token = self.tokenizer.special_tokens['circle: ']

        # get all choices of (num_vertices choose circle_size)
        self.circle_splits = {}
        self.circle_splits["train"] = [self.make_canonical_circle(self.vertices[torch.randperm(self.num_vertices)[:self.circle_size]].tolist())
                                       for _ in range(self.training_size)]
        self.training_circles = set(self.circle_splits["train"])
        self.circle_splits["validation"] = [self.make_canonical_circle(self.vertices[torch.randperm(self.num_vertices)[:self.circle_size]].tolist())
                                            for _ in range(self.validation_size)]
        self.circle_splits["test"] = [self.make_canonical_circle(self.vertices[torch.randperm(self.num_vertices)[:self.circle_size]].tolist())
                                      for _ in range(self.test_size)]

        self.delimiter = self.tokenizer.special_tokens[', ']
        self.circle_token = self.tokenizer.special_tokens['circle: ']
        self.bos_token = self.tokenizer.bos_id

        torch.random.set_rng_state(outside_seed)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        seq = [self.bos_token]
        new_tokens = []

        circle_range = len(self.circle_splits[self.split])
        while len(seq) + len(new_tokens) < self.sequence_length:
            seq.extend(new_tokens)
            # sample random circle from the precomputed list
            circle = self.circle_splits[self.split][torch.randint(0, circle_range, (1,)).item()]
            pairs = [(circle[i], circle[(i+1) % len(circle)]) for i in range(len(circle))]
            # shuffle circle vertices
            shuffled_pairs = [pairs[i] for i in torch.randperm(len(pairs))]
            new_tokens = [self.circle_token]
            for (v1, v2) in shuffled_pairs:
                new_tokens.extend([v1, v2, self.delimiter])
            new_tokens = new_tokens[:-1]  # remove last delimiter
            if self.single_example:
                break
        if len(seq) <= 1:
            seq.extend(new_tokens)

        seq.append(self.eos_token)
        attention_mask = [1] * (len(seq) - 1)

        return {
            "tokens": seq[:-1],
            "attention_mask": attention_mask,
            "labels": seq[1:],
        }

    def __len__(self):
        if self.split == "train":
            return 1_000_000
        elif self.split == "validation":
            return 1_000
        elif self.split == "test":
            return 1_000
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def make_canonical_circle(self, circle: list[int]) -> tuple[int]:
        # sort circle so that it starts with the smallest vertex
        min_index = circle.index(min(circle))
        circle = circle[min_index:] + circle[:min_index]
        return tuple(circle)

    def split_generated_data(self, generated_sequences: list[list[int]], only_first_item=False):
        generated_data = []
        circle = self.circle_token
        for seq in generated_sequences:
            current_subseq = []
            for token in seq:
                if token == circle:
                    current_subseq = [token]
                else:
                    current_subseq.append(token)
                    if len(current_subseq) == 3 * self.circle_size:
                        generated_data.append(current_subseq)
                        current_subseq = []
                        if only_first_item:
                            break
        return generated_data

    def creativity_score(self, generated_data: list[list[int]], num_items: Optional[int] = None):
        if num_items is None:
            num_items = len(generated_data)

        if len(generated_data) == 0:
            return {
                "creativity_score": 0.0,
                "uniqueness_score": 0.0,
                "novelty_score": 0.0,
                "validity_score": 0.0,
            }

        # only accept lists with 3 * circle_size
        circles = [s for s in generated_data if len(s) == 3 * self.circle_size]

        circle_pairs = [[(circle[i+1], circle[i+2]) for i in range(0, len(circle), 3)] for circle in circles]

        actual_circles = []
        for circle_pair in circle_pairs:
            initial_vertex = circle_pair[0][0]
            current_pair = circle_pair[0]
            c = [initial_vertex]
            origins = [pair[0] for pair in circle_pair]
            for _ in range(len(circle_pair)):
                current_vertex = current_pair[1]
                if current_vertex not in origins:
                    current_vertex = None
                    break
                next_index = origins.index(current_vertex)
                current_pair = circle_pair[next_index]
                c.append(current_vertex)
            if current_vertex != initial_vertex:
                continue
            c = c[:-1]
            # now sort circle so that it starts with the smallest vertex
            c = self.make_canonical_circle(c)
            actual_circles.append(c)
        validity_score = len(actual_circles) / len(circles) if len(circles) > 0 else 0.0

        unique_circles = set(actual_circles)
        uniqueness_score = len(unique_circles) / len(actual_circles) if len(actual_circles) > 0 else 0.0

        novel_circles = unique_circles - self.training_circles
        novelty_score = len(novel_circles) / len(unique_circles) if len(unique_circles) > 0 else 0.0

        creativity_score = len(novel_circles) / num_items
        return {
            "creativity_score": creativity_score,
            "uniqueness_score": uniqueness_score,
            "novelty_score": novelty_score,
            "validity_score": validity_score,
        }


def circle_construction_dataset(tokenizer: ModelTokenizer, **kwargs):
    ds = CircleConstructionDataset(tokenizer=tokenizer, **kwargs)
    return ds


class LineConstructionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer: ModelTokenizer,
                 line_size: int = 9,
                 num_vertices: int = 15,
                 training_size: int = 10_000,
                 validation_size: int = 1_000,
                 test_size: int = 1_000,
                 split: str = "train",
                 sequence_length: int = 128,
                 single_example: bool = False,
                 seed: int = 42):
        # Save and restore outside random state to avoid side effects
        outside_seed = torch.random.get_rng_state()
        torch.manual_seed(seed)

        super(CircleConstructionDataset).__init__()
        self.tokenizer = tokenizer
        self.line_size = line_size
        self.num_vertices = num_vertices
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.split = split
        self.sequence_length = sequence_length
        self.seed = seed
        self.single_example = single_example
        if self.single_example:
            self.sequence_length = 3 * (line_size-1) + 1

        assert num_vertices < 90
        assert line_size < num_vertices

        self.vertices = torch.randperm(90)[:self.num_vertices] + 10
        self.bos_token = self.tokenizer.bos_id
        self.delimiter = self.tokenizer.special_tokens[', ']
        self.line_token = self.tokenizer.special_tokens['line: ']

        # get all choices of (num_vertices choose circle_size)
        self.line_splits = {}
        self.line_splits["train"] = [tuple(self.vertices[torch.randperm(self.num_vertices)[:self.line_size]].tolist())
                                     for _ in range(self.training_size)]
        self.training_circles = set(self.line_splits["train"])
        self.line_splits["validation"] = [tuple(self.vertices[torch.randperm(self.num_vertices)[:self.line_size]].tolist())
                                          for _ in range(self.validation_size)]
        self.line_splits["test"] = [tuple(self.vertices[torch.randperm(self.num_vertices)[:self.line_size]].tolist())
                                    for _ in range(self.test_size)]

        self.delimiter = self.tokenizer.special_tokens[', ']
        self.line_token = self.tokenizer.special_tokens['line: ']
        self.bos_token = self.tokenizer.bos_id
        self.eos_token = self.tokenizer.eos_id

        torch.random.set_rng_state(outside_seed)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        seq = [self.bos_token]
        new_tokens = []

        line_range = len(self.line_splits[self.split])
        while len(seq) + len(new_tokens) < self.sequence_length:
            seq.extend(new_tokens)
            # sample random circle from the precomputed list
            line = self.line_splits[self.split][torch.randint(0, line_range, (1,)).item()]
            pairs = [(line[i], line[i+1]) for i in range(len(line)-1)]
            # shuffle line vertices
            shuffled_pairs = [pairs[i] for i in torch.randperm(len(pairs))]
            new_tokens = [self.line_token]
            for (v1, v2) in shuffled_pairs:
                new_tokens.extend([v1, v2, self.delimiter])
            new_tokens = new_tokens[:-1]  # remove last delimiter
            if self.single_example:
                break
        if len(seq) <= 1:
            seq.extend(new_tokens)

        seq.append(self.eos_token)
        attention_mask = [1] * (len(seq) - 1)

        return {
            "tokens": seq[:-1],
            "attention_mask": attention_mask,
            "labels": seq[1:],
        }

    def __len__(self):
        if self.split == "train":
            return 1_000_000
        elif self.split == "validation":
            return 1_000
        elif self.split == "test":
            return 1_000
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def split_generated_data(self, generated_sequences: list[list[int]], only_first_item=False):
        generated_data = []
        line = self.line_token
        for seq in generated_sequences:
            current_subseq = []
            for token in seq:
                if token == line:
                    current_subseq = [token]
                else:
                    current_subseq.append(token)

                if len(current_subseq) == 3 * (self.line_size-1):
                    generated_data.append(current_subseq)
                    current_subseq = []
                    if only_first_item:
                        break

        return generated_data

    def creativity_score(self, generated_data: list[list[int]], num_items: Optional[int] = None):
        if num_items is None:
            num_items = len(generated_data)

        if len(generated_data) == 0:
            return {
                "creativity_score": 0.0,
                "uniqueness_score": 0.0,
                "novelty_score": 0.0,
                "validity_score": 0.0,
            }

        # only accept lists with 3 * (line_size - 1)
        lines = [s for s in generated_data if len(s) == 3 * (self.line_size-1)]

        line_pairs = [[(line[i+1], line[i+2]) for i in range(0, len(line), 3)] for line in lines]

        actual_lines = []
        for line_pair in line_pairs:
            origins = [pair[0] for pair in line_pair]
            targets = [pair[1] for pair in line_pair]
            # find start of line
            start = [v for v in origins if v not in targets]
            if len(start) != 1:
                continue
            initial_vertex = start[0]
            start_idx = origins.index(initial_vertex)
            current_pair = line_pair[start_idx]
            c = [initial_vertex]
            for _ in range(len(line_pair) - 1):
                current_vertex = current_pair[1]
                if current_vertex not in origins:
                    current_vertex = None
                    break
                next_index = origins.index(current_vertex)
                current_pair = line_pair[next_index]
                c.append(current_vertex)
            if current_vertex is None:
                continue
            c.append(current_pair[1])
            # now sort circle so that it starts with the smallest vertex
            actual_lines.append(tuple(c))
        validity_score = len(actual_lines) / len(lines) if len(lines) > 0 else 0.0

        unique_lines = set(actual_lines)
        uniqueness_score = len(unique_lines) / len(actual_lines) if len(actual_lines) > 0 else 0.0

        novel_lines = unique_lines - self.training_circles
        creativity_score = len(novel_lines) / num_items
        novelty_score = len(novel_lines) / len(unique_lines) if len(unique_lines) > 0 else 0.0

        return {
            "creativity_score": creativity_score,
            "uniqueness_score": uniqueness_score,
            "novelty_score": novelty_score,
            "validity_score": validity_score,
        }


def line_construction_dataset(tokenizer: ModelTokenizer, **kwargs):
    ds = LineConstructionDataset(tokenizer=tokenizer, **kwargs)
    return ds



if __name__ == "__main__":
    tokenizer = vertex_tokenizer(max_seq_len=128)

    print("Line construction dataset")
    ds = LineConstructionDataset(tokenizer=tokenizer, split="train", single_example=True)
    generated_lines = []
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_lines.append(item['labels'])
    ds.split = "validation"
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_lines.append(item['labels'])
    split_data = ds.split_generated_data(generated_lines)
    creativity_score = ds.creativity_score(split_data)
    print(f"Creativity score: {creativity_score}")  # should be 1/2

    print("Circle construction dataset")
    ds = CircleConstructionDataset(tokenizer=tokenizer, split="train", single_example=True)
    generated_data = []
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_data.append(item['labels'])
    ds.split = "validation"
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_data.append(item['labels'])
    split_data = ds.split_generated_data(generated_data)
    creativity_score = ds.creativity_score(split_data)
    print(f"Creativity score: {creativity_score}")  # should be 1/2

    print("Triangle discovery dataset")
    ds = TriangleDiscoveryDataset(tokenizer=tokenizer, split="train", single_example=True)
    generated_data = []
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_data.append(item['labels'])
    ds.split = "validation"
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_data.append(item['labels'])
    split_data = ds.split_generated_data(generated_data, only_first_item=True)
    creativity_score = ds.creativity_score(split_data)
    print(f"Creativity score: {creativity_score}")  # should be slightly less than 2/3

    print("Sibling discovery dataset")
    ds = SiblingDiscoveryDataset(tokenizer=tokenizer, split="train", single_example=True)
    generated_data = []
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_data.append(item['labels'])
    ds.split = "validation"
    for i in range(20):
        item = ds[i]
        tokens = item['tokens']
        generated_data.append(item['labels'])
    split_data = ds.split_generated_data(generated_data, only_first_item=True)
    creativity_score = ds.creativity_score(split_data)
    print(f"Creativity score: {creativity_score}")  # should be slightly less than 1/2
    pass


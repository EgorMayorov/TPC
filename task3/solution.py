import codecs
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set, Dict, Callable, Any, TypeVar, Optional, Tuple, Union, NamedTuple
from enum import Enum
import math

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import EncodingFast
from torch import LongTensor, BoolTensor, Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    EvalPrediction,
    BertTokenizerFast,
    BertModel,
    BertForTokenClassification
)
from transformers.integrations import TensorBoardCallback
from transformers.modeling_utils import unwrap_model
from torch.nn.functional import pad, one_hot, softmax


class DatasetType(str, Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


class TypedSpan(NamedTuple):
    start: int
    end: int
    type: str


@dataclass
class Example:
    text_id: int
    example_start: int
    input_ids: LongTensor
    start_offset: LongTensor
    end_offset: LongTensor
    target_label_ids: Optional[LongTensor]
    source_text: str


class NERDataset(Dataset[Example]):
    def __init__(self, examples: Iterable[Example]):
        self._examples = list(examples)
    def __getitem__(self, index) -> Example:
        return self._examples[index]
    def __len__(self):
        return len(self._examples)


class BatchedExamples:
    text_ids: Tuple[int, ...]
    example_starts: Tuple[int, ...]
    input_ids: LongTensor
    start_offset: LongTensor
    end_offset: LongTensor
    padding_mask: BoolTensor


class Solution:
    def get_dataset_files(
            self,
            dataset_dir: Path,
            dataset_type: DatasetType,
            *,
            exclude_filenames: Set[str] = None
    ) -> Tuple[List[Path], List[Path]]:
        if exclude_filenames is None:
            exclude_filenames = set()
        dataset_dir = dataset_dir.joinpath(dataset_type.value)
        if not dataset_dir.exists():
            raise RuntimeError(f'Dataset directory {dataset_dir} does not exist!')
        if not dataset_dir.is_dir():
            raise RuntimeError(f'Provided path {dataset_dir} is not a directory!')
        def is_not_excluded(file: Path) -> bool:
            return file.with_suffix('').name not in exclude_filenames
        return sorted(filter(is_not_excluded, dataset_dir.glob('*.txt'))), sorted(filter(is_not_excluded, dataset_dir.glob('*.ann')))

    def read_annotation(self, annotation_file: Path) -> Set[TypedSpan]:
        collected_annotations: Set[TypedSpan] = set()
        with codecs.open(annotation_file, 'r', 'utf-8') as f:
            for line in f:
                if line.startswith('T'):
                    _, span_info, value = line.strip().split('\t')
                    if ';' not in span_info:
                        category, start, end = span_info.split(' ')
                        collected_annotations.add(TypedSpan(int(start), int(end), category))
        return collected_annotations

    def collect_categories(self, annotation_files: Iterable[Path]) -> Set[str]:
        all_annotations = list(map(self.read_annotation, annotation_files))
        all_categories: Set[str] = set()
        for document_annotations in all_annotations:
            all_categories.update(map(lambda span: span.type, document_annotations))
        return all_categories

    def invert(self, d):
        return {v: k for k, v in d.items()}

    def convert_to_examples(
            self,
            text_id: int,
            encoding: EncodingFast,
            category_mapping: Dict[str, int],
            no_entity_category: str,
            *,
            max_length: int = 128,
            entities: Optional[Set[TypedSpan]] = None,
            source_str: str
    ) -> Iterable[Example]:
        """Encodes entities and splits encoded text into chunks."""
        sequence_length = len(encoding.ids)
        offset = torch.tensor(encoding.offsets, dtype=torch.long)
        target_label_ids: Optional[LongTensor] = None
        if entities is not None:
            token_start_mapping = {}
            token_end_mapping = {}
            for token_idx, (token_start, token_end) in enumerate(encoding.offsets):
                token_start_mapping[token_start] = token_idx
                token_end_mapping[token_end] = token_idx
            no_entity_category_id = category_mapping[no_entity_category]
            category_id_mapping = self.invert(category_mapping)
            text_length = len(encoding.ids)
            target_label_ids = torch.full((text_length, ), fill_value=no_entity_category_id, dtype=torch.long).long()
            for start, end, category in entities:
                try:
                    token_start = token_start_mapping[start]
                except KeyError:
                    if start + 1 in token_start_mapping:
                        token_start = token_start_mapping[start + 1]
                    elif start - 1 in token_start_mapping:
                        token_start = token_start_mapping[start - 1]
                    else:
                        continue
                try:
                    token_end = token_end_mapping[end]
                except KeyError:
                    if end + 1 in token_end_mapping:
                        token_end = token_end_mapping[end + 1]
                    elif end - 1 in token_end_mapping:
                        token_end = token_end_mapping[end - 1]
                    else:
                        continue
                if target_label_ids[token_start] != no_entity_category_id:
                    from_category = category_id_mapping[target_label_ids[token_start].item()]
                else:
                    for i in range(token_start, token_end + 1):
                        target_label_ids[i] = category_mapping[category]
        chunk_start = 0
        while chunk_start < sequence_length:
            chunk_end = min(chunk_start + max_length, sequence_length)
            ex = Example(
                text_id,
                chunk_start,
                torch.tensor(encoding.ids[chunk_start:chunk_end], dtype=torch.long).long(),
                offset[chunk_start:chunk_end, 0],
                offset[chunk_start:chunk_end, 1],
                target_label_ids[chunk_start:chunk_end] if target_label_ids is not None else None,
                source_str
            )
            yield ex
            chunk_start = chunk_end

    def read_text(self, text_file: Path) -> str:
        with codecs.open(text_file, 'r', 'utf-8') as f:
            return f.read()

    def read_nerel(
            self,
            dataset_dir: Path,
            dataset_type: DatasetType,
            tokenizer: Callable[[List[str]], List[EncodingFast]],
            category_mapping: Dict[str, int],
            no_entity_category: str,
            *,
            exclude_filenames: Set[str] = None
    ) -> Iterable[Example]:
        text_files, annotation_files = self.get_dataset_files(dataset_dir, dataset_type, exclude_filenames=exclude_filenames)
        all_annotations = list(map(self.read_annotation, annotation_files))
        all_texts = list(map(self.read_text, text_files))
        encodings = tokenizer(all_texts)
        for text_id, (encoding, entities, text) in enumerate(zip(encodings, all_annotations, all_texts)):
            if text_id < 2048:
                yield from self.convert_to_examples(text_id, encoding, category_mapping, entities=entities, no_entity_category=no_entity_category, source_str=text)

    def compute_metrics(
            self,
            evaluation_results: EvalPrediction,
            category_id_mapping: Dict[int, str],
            no_entity_category_id: int,
            short_output = False
    ) -> Dict[str, float]:
        predictions = np.argmax(evaluation_results.predictions, axis=-1)
        padding_mask = label_mask = (evaluation_results.label_ids != -100)
        label_ids = evaluation_results.label_ids[label_mask]
        predictions = predictions[label_mask]
        unique_label_ids = set(np.unique(label_ids[label_ids != no_entity_category_id]))
        labels = sorted(category_id_mapping.keys())
        f1_category_scores = f1_score(label_ids, predictions, average=None, labels=labels, zero_division=0)
        recall_category_scores = recall_score(label_ids, predictions, average=None, labels=labels, zero_division=0)
        precision_category_scores = precision_score(label_ids, predictions, average=None, labels=labels, zero_division=0)
        results: Dict[str, float] = {}
        sum_f1 = 0
        sum_recall = 0
        sum_precision = 0
        for category_id, (f1, recall, precision) in enumerate(zip(f1_category_scores, recall_category_scores, precision_category_scores)):
            if category_id == no_entity_category_id:
                continue
            category = category_id_mapping[category_id]
            if not short_output:
                results[f'F1_{category}'] = f1
                results[f'Recall_{category}'] = recall
                results[f'Precision_{category}'] = precision
            sum_f1 += f1
            sum_recall += recall
            sum_precision += precision
        num_categories = len(category_id_mapping) - 1
        results['F1_macro'] = sum_f1 / num_categories
        results['Recall_macro'] = sum_recall / num_categories
        results['Precision_macro'] = sum_precision / num_categories
        return results

    def collate_examples(
            self,
            examples: Iterable[Example],
            *,
            padding_id: int = -100,
            pad_length: Optional[int] = None
    ) -> Dict[str, Union[BatchedExamples, Optional[LongTensor]]]:
        all_text_ids: List[int] = []
        all_example_starts: List[int] = []
        all_input_ids: List[LongTensor] = []
        all_padding_masks: List[BoolTensor] = []
        all_start_offsets: List[LongTensor] = []
        all_end_offsets: List[LongTensor] = []
        target_label_ids: Optional[List[LongTensor]] = None
        no_target_label_ids: Optional[bool] = None
        for example in examples:
            all_text_ids.append(example.text_id)
            all_example_starts.append(example.example_start)
            all_input_ids.append(example.input_ids)
            all_start_offsets.append(example.start_offset)
            all_end_offsets.append(example.end_offset)
            all_padding_masks.append(torch.ones_like(example.input_ids, dtype=torch.bool).bool())
            if no_target_label_ids is None:
                no_target_label_ids = (example.target_label_ids is None)
                if not no_target_label_ids:
                    target_label_ids: List[LongTensor] = []
            if (example.target_label_ids is None) != no_target_label_ids:
                raise RuntimeError('Inconsistent examples at collate_examples!')
            if example.target_label_ids is not None:
                target_label_ids.append(example.target_label_ids)
        padded_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=padding_id).long()
        padded_start_offsets = pad_sequence(all_start_offsets, batch_first=True, padding_value=0).long()
        padded_end_offsets = pad_sequence(all_end_offsets, batch_first=True, padding_value=-100).long()
        padded_padding_masks = pad_sequence(all_padding_masks, batch_first=True, padding_value=False).bool()
        padded_labels = pad_sequence(target_label_ids, batch_first=True, padding_value=-100).long() if not no_target_label_ids else None
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_padding_masks,
            'labels': padded_labels
        }

    def main(self):
        dataset_dir = self.dataset_dir
        no_entity_category = self.no_entity_category
        batch_size = 128
        category_id_mapping = self.category_id_mapping
        category_mapping = self.category_mapping
        no_entity_category_id = self.no_entity_category_id

        tokenizer = BertTokenizerFast.from_pretrained("cointegrated/rubert-tiny")
        model = BertForTokenClassification.from_pretrained("cointegrated/rubert-tiny", num_labels=len(category_mapping.keys()))
        # tokenizer = BertTokenizerFast.from_pretrained(self.pre_trained_model_dir)
        # model = BertForTokenClassification.from_pretrained(self.pre_trained_model_dir, num_labels=len(category_mapping.keys()))

        def tokenize(texts: List[str]) -> List[EncodingFast]:
            batch_encoding = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True, return_token_type_ids=False)
            return batch_encoding.encodings

        train_dataset = NERDataset(self.read_nerel(dataset_dir, DatasetType.TRAIN, tokenize, category_mapping, no_entity_category))
        dev_dataset = NERDataset(self.read_nerel(dataset_dir, DatasetType.DEV, tokenize, category_mapping, no_entity_category))

        training_args = TrainingArguments(
            output_dir=self.trained_model_dir,
            learning_rate=8 * 1e-5,
            weight_decay=1e-5,
            optim="adafactor",
            full_determinism=False,
            per_device_train_batch_size=batch_size,
            num_train_epochs=100,
            evaluation_strategy="steps",
            eval_steps=150,
        )

        pad_token_id = 0

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=partial(self.collate_examples, padding_id=pad_token_id, pad_length=128),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=partial(
                self.compute_metrics,
                category_id_mapping=category_id_mapping,
                no_entity_category_id=no_entity_category_id
            ),
            callbacks=[TensorBoardCallback()]
        )

        # trainer.save_model(self.pre_trained_model_dir) # сохранение модели
        # tokenizer.save_pretrained(self.pre_trained_model_dir) # сохранение токенайзера

        trainer.train()

        trained_model = unwrap_model(trainer.model_wrapped)
        # torch.save(trained_model, Path(training_args.output_dir).joinpath("final_model.pt"))      # No need
        trainer.save_model(self.trained_model_dir) # сохранение модели
        tokenizer.save_pretrained(self.trained_model_dir) # сохранение токенайзера
        metrics = trainer.evaluate()
        print(metrics)

    def __init__(self):
        self.no_entity_category = 'NO_ENTITY'
        self.category_id_mapping = {0: 'AGE', 1: 'AWARD', 2: 'CITY',
                                    3: 'COUNTRY', 4: 'CRIME', 5: 'DATE',
                                    6: 'DISEASE', 7: 'DISTRICT', 8: 'EVENT',
                                    9: 'FACILITY', 10: 'FAMILY', 11: 'IDEOLOGY',
                                    12: 'LANGUAGE', 13: 'LAW', 14: 'LOCATION',
                                    15: 'MONEY', 16: 'NATIONALITY', 17: 'NO_ENTITY',
                                    18: 'NUMBER', 19: 'ORDINAL', 20: 'ORGANIZATION',
                                    21: 'PENALTY', 22: 'PERCENT', 23: 'PERSON',
                                    24: 'PRODUCT', 25: 'PROFESSION', 26: 'RELIGION',
                                    27: 'STATE_OR_PROVINCE', 28: 'TIME', 29: 'WORK_OF_ART'}
        self.category_mapping = {'AGE': 0, 'AWARD': 1, 'CITY': 2,
                                 'COUNTRY': 3, 'CRIME': 4, 'DATE': 5,
                                 'DISEASE': 6, 'DISTRICT': 7, 'EVENT': 8,
                                 'FACILITY': 9, 'FAMILY': 10, 'IDEOLOGY': 11,
                                 'LANGUAGE': 12, 'LAW': 13, 'LOCATION': 14,
                                 'MONEY': 15, 'NATIONALITY': 16, 'NO_ENTITY': 17,
                                 'NUMBER': 18, 'ORDINAL': 19, 'ORGANIZATION': 20,
                                 'PENALTY': 21, 'PERCENT': 22, 'PERSON': 23,
                                 'PRODUCT': 24, 'PROFESSION': 25, 'RELIGION': 26,
                                 'STATE_OR_PROVINCE': 27, 'TIME': 28, 'WORK_OF_ART': 29}
        self.no_entity_category_id = list(self.category_id_mapping.keys())[list(self.category_id_mapping.values()).index(self.no_entity_category)]
        # self.dataset_dir = Path("/content/drive/MyDrive/data")
        # self.pre_trained_model_dir = "/content/drive/MyDrive/TPC/task3/not_trained_model"
        # self.trained_model_dir = "/content/drive/MyDrive/TPC/task3/trained_model"
        self.dataset_dir = Path("./data")
        self.pre_trained_model_dir = "./not_trained_model"
        self.trained_model_dir = "./trained_model"

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        no_entity_category = self.no_entity_category
        batch_size = 128
        category_id_mapping = self.category_id_mapping
        category_mapping = self.category_mapping
        no_entity_category_id = self.no_entity_category_id

        new_tokenizer = BertTokenizerFast.from_pretrained(self.trained_model_dir)
        new_model = BertForTokenClassification.from_pretrained(self.trained_model_dir, local_files_only=True)
        tokenized_texts = new_tokenizer(texts, return_offsets_mapping=True)

        tokenized_texts_tensors = [torch.Tensor(t) for t in tokenized_texts.input_ids]
        padded_input_ids = pad_sequence(tokenized_texts_tensors, batch_first=True).long()

        padded_output_ids = []
        for i in range(0, padded_input_ids.shape[1], batch_size):
            batch = padded_input_ids[:, i : min(i + batch_size, padded_input_ids.shape[1])]
            padded_output_ids.append(torch.argmax(new_model(batch).logits, dim=-1))
        padded_output_ids = torch.cat(padded_output_ids, dim=1)

        pre_output = []
        for i in range(len(padded_output_ids)):
            span_set = []
            for j in range(len(tokenized_texts.offset_mapping[i])):
                if (tokenized_texts.offset_mapping[i][j][1] != 0 and padded_output_ids[i][j] != no_entity_category_id):
                    span_set.append((int(tokenized_texts.offset_mapping[i][j][0]), int(tokenized_texts.offset_mapping[i][j][1]), category_id_mapping[int(padded_output_ids[i][j])]))
            pre_output.append(span_set)
        output = []
        for token_list in pre_output:
            text_set = set()
            text_len = len(token_list)
            i = 1
            category = token_list[0][2]
            start = token_list[0][0]
            end = token_list[0][1]
            while i < text_len:
                if (category == token_list[i][2] and (token_list[i][0] - end == 0 or token_list[i][0] - end == 1)):
                    end = token_list[i][1]
                else:
                    text_set.add((start, end, category))
                    category = token_list[i][2]
                    start = token_list[i][0]
                    end = token_list[i][1]
                i += 1
            output.append(text_set)
        return output


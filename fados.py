import os

import torch
import numpy as np
from tqdm import tqdm
from music21 import corpus
from torch.utils.data import TensorDataset
from DatasetManager.helpers import standard_name, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, \
    standard_note, OUT_OF_RANGE, REST_SYMBOL


from DatasetManager.music_dataset import MusicDataset


class FadosDataset(MusicDataset):

    def __init__(self,
                 voice_ids,
                 sequences_size=8,
                 subdivision=4,
                 cache_dir=None):
        super(FadosDataset, self).__init__(cache_dir=cache_dir)

        self.num_voices = len(voice_ids)
        self.sequences_size = sequences_size
        self.subdivision = subdivision

    def iterator_gen(self):
        return (corpus.parse(filename) for filename in os.listdir())

    def compute_index_dicts(self):
        print('Computing index dicts')
        self.index2note_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.note2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            note_set.add(REST_SYMBOL)

        # get all notes: used for computing pitch ranges
        for chorale in tqdm(self.iterator_gen()):
            for part_id, part in enumerate(chorale.parts[:self.num_voices]):
                for n in part.flat.notesAndRests:
                    note_sets[part_id].add(standard_name(n))

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2note_dicts,
                                                    self.note2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})

    def is_valid(self, fado):
        # We only consider 4-part chorales
        if not len(fado.parts) == 4:
            return False
        # todo contains chord
        return True

    def compute_voice_ranges(self):
        assert self.index2note_dicts is not None
        assert self.note2index_dicts is not None
        self.voice_ranges = []
        print('Computing voice ranges')
        for voice_index, note2index in tqdm(enumerate(self.note2index_dicts)):
            notes = [
                standard_note(note_string)
                for note_string in note2index
            ]
            midi_pitches = [
                n.pitch.midi
                for n in notes
                if n.isNote
            ]
            min_midi, max_midi = min(midi_pitches), max(midi_pitches)
            self.voice_ranges.append((min_midi, max_midi))

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        # todo check on chorale with Chord
        print('Making tensor dataset')
        self.compute_index_dicts()
        self.compute_voice_ranges()
        one_tick = 1 / self.subdivision
        chorale_tensor_dataset = []
        metadata_tensor_dataset = []
        for chorale_id, chorale in tqdm(enumerate(self.iterator_gen())):

            # precompute all possible transpositions and corresponding metadatas
            chorale_transpositions = {}
            metadatas_transpositions = {}

            # main loop
            for offsetStart in np.arange(
                    chorale.flat.lowestOffset -
                    (self.sequences_size - one_tick),
                    chorale.flat.highestOffset,
                    one_tick):
                offsetEnd = offsetStart + self.sequences_size
                current_subseq_ranges = self.voice_range_in_subsequence(
                    chorale,
                    offsetStart=offsetStart,
                    offsetEnd=offsetEnd)

                transposition = self.min_max_transposition(current_subseq_ranges)
                min_transposition_subsequence, max_transposition_subsequence = transposition

                for semi_tone in range(min_transposition_subsequence,
                                       max_transposition_subsequence + 1):
                    start_tick = int(offsetStart * self.subdivision)
                    end_tick = int(offsetEnd * self.subdivision)

                    try:
                        # compute transpositions lazily
                        if semi_tone not in chorale_transpositions:
                            (chorale_tensor,
                             metadata_tensor) = self.transposed_score_and_metadata_tensors(
                                chorale,
                                semi_tone=semi_tone)
                            chorale_transpositions.update(
                                {semi_tone:
                                     chorale_tensor})
                            metadatas_transpositions.update(
                                {semi_tone:
                                     metadata_tensor})
                        else:
                            chorale_tensor = chorale_transpositions[semi_tone]
                            metadata_tensor = metadatas_transpositions[semi_tone]

                        local_chorale_tensor = self.extract_score_tensor_with_padding(
                            chorale_tensor,
                            start_tick, end_tick)
                        local_metadata_tensor = self.extract_metadata_with_padding(
                            metadata_tensor,
                            start_tick, end_tick)

                        # append and add batch dimension
                        # cast to int
                        chorale_tensor_dataset.append(
                            local_chorale_tensor[None, :, :].int())
                        metadata_tensor_dataset.append(
                            local_metadata_tensor[None, :, :, :].int())
                    except KeyError:
                        # some problems may occur with the key analyzer
                        print(f'KeyError with chorale {chorale_id}')

        chorale_tensor_dataset = torch.cat(chorale_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)

        dataset = TensorDataset(chorale_tensor_dataset,
                                metadata_tensor_dataset)

        print(f'Sizes: {chorale_tensor_dataset.size()}, {metadata_tensor_dataset.size()}')
        return dataset


if __name__ == "__main__":


    pass
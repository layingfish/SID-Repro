

from logging import getLogger
from datasets import Dataset


class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):


        return len(self.user2id)

    @property
    def n_items(self):


        return len(self.item2id)

    @property
    def n_interactions(self):


        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):


        return self.n_interactions / self.n_users

    @property
    def user2id(self):


        return self.id_mapping['user2id']

    @property
    def item2id(self):


        return self.id_mapping['item2id']

    def _download_and_process_raw(self):


        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):


        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}
        for user in self.all_item_seqs:
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def _sliding_train(self, min_hist=2, max_hist=50):


        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}

        for user, seq in self.all_item_seqs.items():
            L = len(seq)


            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(seq)


            if L > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(seq[:-1])


            if L > 2:


                hi_max = min(L - 3, max_hist)

                if hi_max >= min_hist:
                    for t in range(min_hist, hi_max + 1):
                        datasets['train']['user'].append(user)
                        datasets['train']['item_seq'].append(seq[:t + 1])


        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def split(self):


        if self.split_data is not None:
            return self.split_data


        strategy = self.config.get('split', 'leave_one_out')


        val = self.config.get('train_sliding', False)
        slide = val if isinstance(val, bool) else str(val).lower() == "true"

        if strategy in ['leave_one_out', 'last_out']:
            if slide:
                datasets = self._sliding_train(
                    min_hist=self.config.get('min_hist_len', 2),
                    max_hist=self.config.get('max_hist_len', 50)
                )
            else:
                datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)

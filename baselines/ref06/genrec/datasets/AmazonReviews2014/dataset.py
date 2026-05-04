

import os
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Optional

from genrec.dataset import AbstractDataset
from genrec.utils import download_file, clean_text


class AmazonReviews2014(AbstractDataset):


    def __init__(self, config: dict):
        super(AmazonReviews2014, self).__init__(config)

        self.category = config['category']
        self._check_available_category()
        self.log(
            f'[DATASET] Amazon Reviews 2014 for category: {self.category}'
        )

        self.cache_dir = os.path.join(
            config['cache_dir'], 'AmazonReviews2014', self.category
        )
        self._download_and_process_raw()

    def _check_available_category(self):


        available_categories = [
            'Books',
            'Electronics',
            'Movies_and_TV',
            'CDs_and_Vinyl',
            'Clothing_Shoes_and_Jewelry',
            'Home_and_Kitchen',
            'Kindle_Store',
            'Sports_and_Outdoors',
            'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care',
            'Toys_and_Games',
            'Video_Games',
            'Tools_and_Home_Improvement',
            'Beauty',
            'Apps_for_Android',
            'Office_Products',
            'Pet_Supplies',
            'Automotive',
            'Grocery_and_Gourmet_Food',
            'Patio_Lawn_and_Garden',
            'Baby',
            'Digital_Music',
            'Musical_Instruments',
            'Amazon_Instant_Video'
        ]
        assert self.category in available_categories, \
            f'Category "{self.category}" not available. ' \
            f'Available categories: {available_categories}'

    def _download_raw(self, path: str, type: str = 'reviews') -> str:


        url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{type}_{self.category}{"_5" if type == "reviews" else ""}.json.gz'
        base_name = os.path.basename(url)
        local_filepath = os.path.join(path, base_name)
        if not os.path.exists(local_filepath):
            download_file(url, local_filepath)
        return local_filepath

    def _parse_gz(self, path: str):


        import json
        import ast
        with gzip.open(path, 'rt', encoding='utf-8') as g:
            for line in g:
                if not line:
                    continue
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    try:
                        obj = ast.literal_eval(s)
                        yield obj
                    except Exception:

                        continue

    def _load_reviews(self, path: str) -> list:


        self.log('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        return reviews

    def _get_item_seqs(self, reviews: list[tuple]) -> dict:


        item_seqs = defaultdict(list)
        for data in reviews:
            user, item, time = data
            item_seqs[user].append((item, time))


        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [_[0] for _ in item_time]
        return item_seqs

    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:


        self.log('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            iids = []
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids
        return self.all_item_seqs, self.id_mapping

    def _process_reviews(self,
        input_path: str,
        output_path: str
    ) -> tuple[dict, dict]:


        seq_file = os.path.join(output_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping

        self.log('[DATASET] Processing reviews...')


        reviews = self._load_reviews(input_path)
        item_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)


        self.log('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        return all_item_seqs, id_mapping

    def _load_metadata(
        self,
        path: str,
        item2id: dict
    ) -> dict:


        self.log('[DATASET] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        return data

    def _sent_process(self, raw: str) -> str:


        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw)
        elif len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    cleaned = clean_text(v).strip()
                    if cleaned:

                        if cleaned.endswith('.'):
                            cleaned = cleaned[:-1]
                        sentence += cleaned
                        sentence += ', '
            sentence = sentence[:-2]
        elif isinstance(raw, list):
            items = []
            for v1 in raw:
                cleaned = clean_text(v1).strip()
                if cleaned:

                    if cleaned.endswith('.'):
                        cleaned = cleaned[:-1]
                    items.append(cleaned)
            sentence = ', '.join(items)
        else:
            sentence = clean_text(raw).strip()

            if sentence.endswith('.'):
                sentence = sentence[:-1]

        return sentence

    def _extract_meta_sentences(
        self,
        metadata: dict
    ) -> dict:


        self.log('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_parts = []
            keys = set(meta.keys())


            features_mapping = {
                'title': 'Title',
                'price': 'Price',
                'brand': 'Brand',
                'categories': 'Category',
                'feature': 'Features',
                'description': 'Description'
            }

            for feature, label in features_mapping.items():
                if feature in keys and meta[feature]:
                    processed_value = self._sent_process(meta[feature])
                    if processed_value.strip():

                        if processed_value.endswith('.'):
                            meta_parts.append(f"{label}: {processed_value}")
                        else:
                            meta_parts.append(f"{label}: {processed_value}.")


            meta_sentence = ' '.join(meta_parts)
            item2meta[item] = meta_sentence
        return item2meta

    def _process_meta(
        self,
        input_path: str,
        output_path: str
    ) -> Optional[dict]:


        process_mode = self.config['metadata']
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.log('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)

        self.log(f'[DATASET] Processing metadata, mode: {process_mode}')

        if process_mode == 'none':

            return None

        item2meta = self._load_metadata(
            path=input_path,
            item2id=self.item2id
        )
        if process_mode == 'raw':
            pass
        elif process_mode == 'sentence':

            item2meta = self._extract_meta_sentences(metadata=item2meta)
        else:
            raise NotImplementedError('Metadata processing type not implemented.')

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    def _download_and_process_raw(self):


        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        with self.accelerator.main_process_first():
            reviews_localpath = self._download_raw(
                path=raw_data_path,
                type='reviews'
            )
            meta_localpath = self._download_raw(
                path=raw_data_path,
                type='meta'
            )


        np.random.seed(12345)


        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        self.all_item_seqs, self.id_mapping = self._process_reviews(
            input_path=reviews_localpath,
            output_path=processed_data_path
        )

        self.item2meta = self._process_meta(
            input_path=meta_localpath,
            output_path=processed_data_path
        )

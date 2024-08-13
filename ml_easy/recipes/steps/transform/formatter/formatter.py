from abc import ABC, abstractmethod
from typing import Dict

import nltk  # type: ignore
import regex as re
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet as wn  # type: ignore
from pydantic import BaseModel


class TextCleanerConfig(BaseModel):
    regex_patterns: Dict[str, str]


class TextFormatterConfig(BaseModel):
    cleaner: TextCleanerConfig


class TextCleaner(ABC):
    def __init__(self, conf: TextCleanerConfig) -> None:
        self.conf = conf

    def __call__(self, text: str):
        return self.clean(text)

    @abstractmethod
    def clean(self, text: str) -> str:
        pass


class AvsCleaner(TextCleaner):
    def __init__(self, config_settings: TextCleanerConfig) -> None:
        super().__init__(config_settings)

    def clean(self, text: str):
        for pattern, replacement in self.conf.regex_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text


class LemmatizerStrategy(ABC):

    def __call__(self, text: str, *args, **kwargs):
        return self.lemmatize(text)

    @abstractmethod
    def lemmatize(self, text: str) -> str:
        pass


class AvsLemmatizer(LemmatizerStrategy):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    def __init__(self):
        self.wl = WordNetLemmatizer()

    def lemmatize(self, text: str) -> str:
        word_pos_tags = nltk.pos_tag(word_tokenize(text))
        tokens = [
            self.wl.lemmatize(word, self.__get_wordnet_pos(pos_tag))
            for idx, (word, pos_tag) in enumerate(word_pos_tags)
        ]
        return ' '.join(tokens)

    @classmethod
    def __get_wordnet_pos(cls, tag):
        """This is a helper function to map NTLK position tags"""
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN

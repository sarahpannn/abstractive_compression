import os

from datasets import load_dataset

class GenericKnowledgeBase(object):
    def __init__(self, dataset_name, subset=None):
        self.dataset_name = dataset_name
        
        if subset is not None:
            self.dataset = load_dataset(dataset_name, subset)
        else:
            self.dataset = load_dataset(dataset_name)

    def get_train_split(self):
        return
    
    def get_test_split(self):
        return
    
    def get_val_split(self):
        return
    
    def get_chunks(self):
        return
    

class GenericQuestionDataset(object):
    def __init__(self, dataset_name, subset=None):
        self.dataset_name = dataset_name

        if subset is not None:
            self.dataset = load_dataset(dataset_name, subset)
        else:
            self.dataset = load_dataset(dataset_name)
    
    def get_train_split(self):
        return
    
    def get_test_split(self):
        return
    
    def get_val_split(self):
        return
    
    def get_questions(self):
        return
    

class WikiText(GenericKnowledgeBase):
    def __init__(self, subset=None):
        super().__init__('wikitext', subset)
    
    def get_train_split(self):

        return self.dataset['train']
    
    def get_test_split(self):
        return self.dataset['test']
    
    def get_val_split(self):
        return self.dataset['validation']
    
    @staticmethod
    def get_chunks(data):
        return [x['text'] for x in data]


class TriviaQA(GenericQuestionDataset):
    def __init__(self, subset=None):
        super().__init__('trivia_qa', subset)
    
    def get_train_split(self):
        return self.dataset['train']
    
    def get_test_split(self):
        return self.dataset['test']
    
    def get_val_split(self):
        return self.dataset['validation']
    
    @staticmethod
    def get_questions(data):
        return [x['question'] for x in data]
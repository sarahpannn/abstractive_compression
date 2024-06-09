import os
import argparse
import faiss
import datasets
from datasets import Dataset, concatenate_datasets
import openai
from openai import OpenAI
import sentence_transformers
from sentence_transformers import SentenceTransformer

from utils.raw_datasets import WikiText, TriviaQA

def parse_args():
    parser = argparse.ArgumentParser(description='Main')

    parser.add_argument(
        '--knowledge_base',
        type=str,
        default='wikitext',
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='paraphrase-MiniLM-L6-v2',)
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--question_path',
        type=str,
        default='mandarjoshi/trivia_qa',
    )
    parser.add_argument(
        '--embedding_save_path',
        type=str,
        default='.'
    )
    parser.add_argument(
        '--re_embed',
        action='store_true',
    )
    parser.add_argument(
        '--oai_key',
        type=str,
    )
    parser.add_argument(
        '--oai_model',
        type=str,
        default='gpt-3.5-turbo',
    )
    parser.add_argument(
        '--summarization_prompt',
        type=str,
        default='Compress the information in the retrieved documents into a 2-sentence summary that could be used to answer the question. Do not include information that is not present in the retrieved documents, just summarize the retrieved information in ways that might be relevant to the question. Do not answer the question.'
    )
    parser.add_argument(
        '--question_split',
        type=float,
        default=0.2,
    )

    return parser.parse_args()


def check_compatiblity(args):
    if args.knowledge_base not in ['wikitext']: 
        raise ValueError('Knowledge base not yet supported')
    if args.question_path not in ['mandarjoshi/trivia_qa']: 
        raise ValueError('Question path not yet supported')


def encode_data(args, model, data):
    if args.re_embed or not os.path.exists(args.embedding_save_path + 'embeddings.arrow'):
        embeddings = model.encode(data)
        datasets.save_to_disk(args.embedding_save_path, embeddings)
    else:
        embeddings = datasets.load_from_disk(args.embedding_save_path + 'embeddings.arrow')
    return embeddings


def main():
    args = parse_args()

    check_compatiblity(args)

    model = SentenceTransformer(args.embedding_model)
        
    # prepare kb
    print('Preparing knowledge base')
    kb = WikiText(subset='wikitext-103-raw-v1')
    kb = concatenate_datasets([kb.get_train_split(), kb.get_test_split(), kb.get_val_split()])
    chunks = WikiText.get_chunks(kb)
    kb_embeddings = encode_data(args, model, chunks)

    # prepare questions
    print('Preparing questions')
    questions = TriviaQA(subset='rc')
    questions = questions.select(range(int(args.question_split * len(questions)))) # TODO: fix to be random
    questions = TriviaQA.get_questions(questions.get_test_split())
    question_embeddings = encode_data(args, model, questions)

    # Create the index
    print('Creating index')
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)

    # Get the top k results
    _, I = index.search(question_embeddings, args.top_k)

    prompts = []
    for i in range(len(questions)):
        summary = ''
        for j in I.tolist()[i]:
            summary += chunks[j]
        qa_prompt = f"{args.summarization_prompt} \nQuestion: {questions[i]} \nRetrieved documents: {summary} \nCompressed documents:"
        prompts.append(qa_prompt)

    client = OpenAI(args.oai_key)

    completions = [client.chat.completions.create(model=args.oai_model, messages=[{'role': 'user', 'content': prompt}]) for prompt in prompts]
    completions = [completion.choices[0].message['content'] for completion in completions]

    # save completions
    print('Saving completions')
    # list to dataset
    completions = Dataset.from_dict({'text': completion for completion in completions})
    datasets.save_to_disk('completions', completions)


if __name__ == '__main__':
    main()
   


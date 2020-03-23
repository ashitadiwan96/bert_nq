from __future__ import absolute_import, division, print_function
from flask import Flask, render_template
from flask_socketio import SocketIO

import argparse
import logging
import os
import random
import glob
import timeit
import pdb

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from transformers import AdamW


from utils_newsqa import (read_newsqa_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended,NewsqaExample)

# The follwing import is the official NewsQA evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_newsqa_evaluate import EVAL_OPTS, main as evaluate_on_newsqa

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
   # 'albert': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
}

model_path = '/network/home/diwanash/bert-joint-baseline'
model = torch.load(model_path+'/bert_joint.ckpt')
#print('model loaded')
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
tokenizer = tokenizer_class.from_pretrained(model_path)
config = config_class.from_pretrained(model_path)
model = model_class.from_pretrained(model_path,
                                        from_tf=bool('.ckpt' in model_path),
                                        config=config,
                                        cache_dir= None)


def read_chat_examples(text, question_text):
    """Read a SQuAD json file into a list of NewsqaExample."""
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    paragraph_text = text
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    example = NewsqaExample(
            qas_id=1,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text='',
            start_position=None,
            end_position=None,
            is_impossible=False)
    examples.append(example)
    return examples

def evaluate(model, tokenizer, prefix, text, question_text):
    dataset, examples, features = load_and_cache_examples(tokenizer, evaluate=True, output_examples=True, group="test", text=text, question_text=question_text)
    n_gpu = torch.cuda.device_count()
    model_type = 'bert'
    #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_batch_size = 6 * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    questions = []
    for example in examples:
        print(example.question_text)
        questions.append(example.question_text)
    # multi-gpu evaluate
    #if n_gpu > 1:
       # model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    model.to(device)
    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(dataset))
    print("  Batch size = %d", eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1]
                      }
            if model_type != 'distilbert':
                inputs['token_type_ids'] = None if model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
            example_indices = batch[3]
            if model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time

    #print("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    #output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    #output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    #if args.version_2_with_negative:
    #    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    #else:
    #    output_null_log_odds_file = None

    pred =  write_predictions(examples, features, all_results, 20,
                        30, True, '1.json',
                        '2.json', '3.json', False,
                        False, 0.0)
    answers = []
    for (p,a) in pred.items():
       answers.append(a)
    
    return answers

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(tokenizer, evaluate, output_examples, group, text, question_text):

    # Load data features from cache or dataset file
    input_file = 'chat_sample.json'
        

    print("Creating features from dataset file at %s", input_file)
    #examples = read_newsqa_examples(input_file=input_file,
    #                                            is_training=not evaluate,
    #                                            version_2_with_negative=False, group=group)
    examples = read_chat_examples(text, question_text)
    features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=128,
                                                doc_stride=128,
                                                max_query_length=64,
                                                is_training=False,
                                                cls_token_segment_id=0,
                                                pad_token_segment_id=0,
                                                cls_token_at_end=False,
                                                sequence_a_is_doc=False)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)


    if output_examples:
        return dataset, examples, features
    return dataset


def getTextFromHTMLLink(url):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    rows = soup.find_all('p')
    str_cells = str(rows)
    cleantext = BeautifulSoup(str_cells, "lxml").get_text()
    return cleantext



#text="NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted a wealthy businessman facing the death sentence for the killing of a teen in a case dubbed \"the house of horrors.\"\n\n\n\nMoninder Singh Pandher was sentenced to death by a lower court in February.\n\n\n\nThe teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years.\n\n\n\nThe Allahabad high court has acquitted Moninder Singh Pandher, his lawyer Sikandar B. Kochar told CNN.\n\n\n\nPandher and his domestic employee Surinder Koli were sentenced to death in February by a lower court for the rape and murder of the 14-year-old.\n\n\n\nThe high court upheld Koli's death sentence, Kochar said.\n\n\n\nThe two were arrested two years ago after body parts packed in plastic bags were found near their home in Noida, a New Delhi suburb. Their home was later dubbed a \"house of horrors\" by the Indian media.\n\n\n\nPandher was not named a main suspect by investigators initially, but was summoned as co-accused during the trial, Kochar said.\n\n\n\nKochar said his client was in Australia when the teen was raped and killed.\n\n\n\nPandher faces trial in the remaining 18 killings and could remain in custody, the attorney said."
#question_text="When was Pandher sentenced to death?"
#result = evaluate(model, tokenizer, prefix=1,text=text, question_text=question_text)
#print(result)





global readSource
readSource  = 0
global text
@app.route('/')
def sessions():
    return render_template('session.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
   # question_text="When was Pandher sentenced to death?"
   # result = evaluate(model, tokenizer, prefix=1,text=text, question_text=question_text)
   # print(result)
   # json["ans"] =
    global readSource
    global text
    if json["question"] == 'User connected':
        socketio.emit('my response', json, callback=messageReceived)
        readSource = 0
    elif json['question'] == 'source':
        if readSource == 0:
            if json['source_type'] == 'link':
                text = getTextFromHTMLLink(json['source'])
            elif json['source_type'] == 'text':
                text = json['source']
            else:
                source_file = 'netanyahu.txt'
                with open(source_file, 'r') as f:
                    text =  ' '.join(f.readlines())
            readSource = 1

    else:
#        if readSource == 0:
 #           if json['source_type'] == 'link':
  #              text = getTextFromHTMLLink(json['source'])
   #         elif json['source_type'] == 'text':
    #            text = json['source']
     #       else:
      #          source_file = 'netanyahu.txt'
       #         with open(source_file, 'r') as f:
        #            text =  ' '.join(f.readlines())
         #   readSource = 1
        question_text = json['question']
        print('readsource is',readSource)
        print("text is", text)
        result = evaluate(model, tokenizer, prefix=1,text=text, question_text=question_text)
        json["answer"] = result
        print(json)
        print('ans appended')
        socketio.emit('my response', json, callback=messageReceived)

if __name__ == '__main__':
    #socketio.run(app, debug=False)
    model_path = 'tune_bert_try1/tune_bert_1/checkpoint-10000/pytorch_model.bin'
    model = torch.load(model_path)
    print('model loaded')
    socketio.run(app,debug=False)

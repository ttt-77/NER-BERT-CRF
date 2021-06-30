import numpy as np
import os
import torch
import argparse
import time
from model import BERT_CRF_NER
from preprocess import dataLoader
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
def f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id=3

    num_proposed = len(y_pred[y_pred>ignore_id])
    num_correct = (np.logical_and(y_true==y_pred, y_true>ignore_id)).sum()
    num_gold = len(y_true[y_true>ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    return precision, recall, f1

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate(model, predict_dataloader,  epoch_th, dataset_name,device):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total=0
    correct=0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            # _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f minutes for evaluation' \
        % (epoch_th, 100.*test_acc, 100.*precision, 100.*recall, 100.*f1, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1

def train(args,model):
    #%%
    if args.load_checkpoint and os.path.exists(args.output_dir+'ner_bert_crf_checkpoint.pt'):
        checkpoint = torch.load(args.output_dir+'ner_bert_crf_checkpoint.pt', map_location='cpu')
        start_epoch = checkpoint['epoch']+1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict=checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain NER_BERT_CRF model, epoch:',checkpoint['epoch'],'valid acc:',
                checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_prev = 0
        valid_f1_prev = 0

    model.to(args.device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
            and not any(nd in n for nd in new_param)], 'weight_decay': args.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
            and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions','hidden2label.weight')] \
            , 'lr':args.lr0_crf_fc, 'weight_decay': args.weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
            , 'lr':args.lr0_crf_fc, 'weight_decay': 0.0}
    ]
    total_train_steps = int(args.train_examples_len / args.batch_size / args.gradient_accumulation_steps * args.epochs)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate0, warmup=args.warmup_proportion, t_total=total_train_steps)
    
    
    global_step_th = int(args.train_examples_len / args.batch_size / args.gradient_accumulation_steps * start_epoch)
    for epoch in range(start_epoch, args.epochs):
        tr_loss = 0
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(args.train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

            if args.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

            neg_log_likelihood.backward()

            tr_loss += neg_log_likelihood.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate0 * warmup_linear(global_step_th/total_train_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

            print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(args.train_dataloader), neg_log_likelihood.item()))

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start)/60.0))
        valid_acc, valid_f1 = evaluate(model, args.dev_dataloader, epoch, 'Valid_set',args.device)

        # Save a checkpoint
        if valid_f1 > valid_f1_prev:
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                'valid_f1': valid_f1, 'max_seq_length': args.max_seq_length, 'lower_case': args.do_lower_case},
                        os.path.join(args.output_dir, 'ner_bert_crf_checkpoint.pt'))
            valid_f1_prev = valid_f1

def test(args,model):
    #%%
    '''
    Test_set prediction using the best epoch of NER_BERT_CRF model
    '''
    checkpoint = torch.load(args.output_dir+'ner_bert_crf_checkpoint.pt', map_location='cpu')
    epoch = checkpoint['epoch']
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict=checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print('Loaded the pretrain  NER_BERT_CRF  model, epoch:',checkpoint['epoch'],'valid acc:',
        checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

    model.to(args.device)
    #evaluate(model, train_dataloader, batch_size, total_train_epochs-1, 'Train_set')
    evaluate(model, args.test_dataloader, epoch, 'Test_set', args.device)
    # print('Total spend:',(time.time()-train_start)/60.0)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument('-data_dir', type=str, default='',
                        help='Path of data file')
    parser.add_argument('--do_eval', type=bool, default=True,
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict', type=bool, default=True,
                        help='Whether to run the model in inference mode on the test set.')
    parser.add_argument('--load_checkpoint', type=bool, default=True,
                        help='Whether load checkpoint file before train model.')
    parser.add_argument('--max_seq_length', type=int, default=180,
                        help='The maximum length of one sentence.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of episode to train ')
    parser.add_argument('--weight_decay_finetune', type=float, default=1e-5,
                        help='weight_decay_finetune')
    parser.add_argument('--weight_decay_crf_fc', type=float, default=5e-6,
                        help='weight_decay_crf_fc')
    parser.add_argument('--learning_rate0', type=float, default=5e-5,
                        help='learning rate (default:5e-5)')
    parser.add_argument('--lr0_crf_fc', type=float, default=8e-5,
                        help='lr0_crf_fc (default:8e-5)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='warmup_proportion')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Path of output file')
    parser.add_argument('--bert_model_scale', type=str, default='bert-base-cased',
                        help='Which Bert model will be used')
    parser.add_argument('--do_lower_case', type=bool, default=False,
                        help='Whether do lower case in tokenize')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Which device to train')
    args = parser.parse_args()

    start_label_id, stop_label_id, args.train_dataloader, args.dev_dataloader, args.test_dataloader,label_list_len ,args.train_examples_len = dataLoader(args.bert_model_scale, args.do_lower_case, args.data_dir, args.max_seq_length, args.batch_size)
    bert_model = BertModel.from_pretrained(args.bert_model_scale)
    model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, label_list_len, args.max_seq_length, args.batch_size, args.device)
    print("Start Training\n")
    train(args, model)
    print("Start Testing\n")
    test(args,model)
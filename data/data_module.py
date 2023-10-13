import torch
import json
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler



class FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, config,dataset, templates, tokenizer, add_special_tokens=True,val=False):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.config = config
        self.val = val
        if self.config.n_ways==3:
            self.lable_mapping = {"entailment":0,"SUPPORTS":0,"SUPPORT":0,"SUPPORTED":0,"true":0,
            "REFUTES":2,"REFUTED":2,"CONTRADICT":2,"not_entailment":2,"false":2,"CONTRADICT":2,"NEI":1}

     
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates

        example = self.dataset[key] # 第key条数据

        if "idx" not in example:
            example["idx"] = example["id"]
        
        gen_evid = example["gold_evidence_text"]
        example["premise"] = gen_evid
        assert example["premise"]!=""

        example["hypothesis"] = example["claim"]
        example["label_str"] = example["label"] 
        
        v_label = None
        idx = None
        verification_input_ids = None
        answer_choices_ids = None

        example["label"] = self.lable_mapping[example["label"]] # supoort:0, NEI:1, refute:2
        input_str, _ = template.apply(example) # 把前提和假设代入模板
    
        answer_choices1 = template.get_answer_choices_list(example) # 模板给出的回答选项，eg. ['True', 'Inconclusive', 'False']
        assert len(answer_choices1) == self.config.n_ways

        if not self.val:
            all_inputs = [input_str]
            example["t_claim"] = "it is true that "+example["claim"]
            example["f_claim"] = "it is false that "+example["claim"]
            example["n_claim"] = "it is unclear that "+example["claim"]

            example["hypothesis"] = example["t_claim"]
            input_strt, _ = template.apply(example)
            all_inputs.append(input_strt)

            example["hypothesis"] = example["f_claim"]
            input_strf, _ = template.apply(example)
            all_inputs.append(input_strf)

            example["hypothesis"] = example["n_claim"]
            input_strn, _ = template.apply(example)
            all_inputs.append(input_strn)

            input_str = all_inputs
            verification_input_ids = [
                        self.tokenizer(
                            candidate, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                        ).input_ids.squeeze(0)
                        for candidate in all_inputs
                    ]
            tid = int(example["idx"])*10
            idx = [tid,tid+1,tid+2,tid+3]

        if not self.val:
            con_l = {0:[0,0,2,2],1:[1,1,1,0],2:[2,2,0,2]}
            answer_choices_ids = [[
                    self.tokenizer(
                        answer_choice, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                    ).input_ids.squeeze(0)
                    for answer_choice in answer_choices1
                ] for _ in range(4)]
            v_label = [example["label"],con_l[example["label"]][1],con_l[example["label"]][2],con_l[example["label"]][3]]

        if verification_input_ids is None:
            # 使用分词器将input_str转换为token IDs
            # return_tensors="pt": 这告诉分词器返回PyTorch张量 
            # truncation=True: 如果输入字符串的token数量超过分词器的最大长度，它将被截断。
            # add_special_tokens=self.add_special_tokens: 这确定是否应添加特殊的开始和结束token（例如，对于BERT模型，它们分别是[CLS]和[SEP]）
            # .input_ids: 从分词器返回的结果中提取token IDs，squeeze(0)用于移除批次维度，将形状从(1, sequence_length)更改为(sequence_length,)
            verification_input_ids = self.tokenizer(
                input_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                ).input_ids.squeeze(0)

        if v_label is None: # gold label的mapping id
            v_label = torch.LongTensor([example["label"]])
        
        if idx is None: # 这条数据的id
            idx = int(example["idx"])
            idx = torch.LongTensor([idx])
      
        if answer_choices_ids is None:
            # 每个选择字串都张量化的列表
            answer_choices_ids = [
                self.tokenizer(
                    answer_choice, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                ).input_ids.squeeze(0)
                for answer_choice in answer_choices1
            ]

       
        return answer_choices_ids, verification_input_ids,v_label, idx


def create_collate_fn(pad_token_id, pretrain,val=False):
    def collate_fn(batch):
        if not pretrain:
            answer_choices_ids_t, verification_input_ids_t,v_labels_t,idx_t = zip(*batch)
            
        if not val:
            answer_choices_ids = [xx for x in answer_choices_ids_t for xx in x ]
            verification_input_ids = [xx for x in verification_input_ids_t for xx in x ] # 有4个input，很相似但不同，应该是论文中的异构输入
            v_labels = [torch.tensor([xx]) for x in v_labels_t for xx in x ]
            idx = [torch.tensor([xx]) for x in idx_t for xx in x ]

        else:
            answer_choices_ids = answer_choices_ids_t
            verification_input_ids = verification_input_ids_t
            v_labels = v_labels_t
            idx = idx_t
        
        output_batch = {}

        if not pretrain:

            flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices] # len=3*8=24
            num_choice = [len(list_choices) for list_choices in answer_choices_ids]

            if verification_input_ids[0] is not None:
                verification_input_ids = torch.nn.utils.rnn.pad_sequence(verification_input_ids, batch_first=True, padding_value=pad_token_id)
            flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
            )
            answer_choices_ids = torch.reshape(flat_answer_choices_ids,(len(answer_choices_ids), max(num_choice), -1))

            idx = torch.cat(idx)       
            v_labels = torch.cat(v_labels)
            output_batch.update(
                {
                    "verification_input_ids": verification_input_ids,
                    "answer_choices_ids": answer_choices_ids,
                    "v_labels": v_labels,
                    "idx": idx
                }
            )

        return output_batch

    return collate_fn



class FinetuneDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.config.few_shot:
            _ = self.dataset_reader.read_few_shot_dataset()
    
    def load_jsonl(self,filename):
        d_list = []
        with open(filename, encoding='utf-8', mode='r') as in_f:
            print("Load Jsonl:", filename)
            for line in in_f:
                item = json.loads(line.strip())
                d_list.append(item)

        return d_list

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.config.few_shot:
            self.train_dataset1 = self.dataset_reader.read_few_shot_dataset()
        else:
            self.train_dataset1 = self.dataset_reader.read_orig_dataset("train")
        self.dev_dataset1 = self.dataset_reader.read_orig_dataset("validation") # {'id': 3, 'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.', 'gold_evidence_text': 'Rare Variants Create Synthetic Genome-Wide Associations: We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.Rare Variants Create Synthetic Genome-Wide Associations: We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.Rare Variants Create Synthetic Genome-Wide Associations: In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.', 'label': 'SUPPORT'}

        val_flag=False if self.config.stage==2 else True
        self.train_dataset = FinetuneDatasetWithTemplate(
            self.config,self.train_dataset1, self.dataset_reader.get_train_template(), self.tokenizer,val=val_flag
        )

        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.config,self.dev_dataset1, self.dataset_reader.get_eval_template(), self.tokenizer,val=True
        )

        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")

    def train_dataloader(self):
        val_flag=False if self.config.stage==2 else True
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False,val=val_flag),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        # sampler = DistributedSampler(self.dev_dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            # sampler=sampler, 
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False, val=True),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
            # num_workers=0
        )
    


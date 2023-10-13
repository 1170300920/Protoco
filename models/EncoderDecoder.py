import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning import LightningModule
from utils.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler


class EncoderDecoder(LightningModule):
    """
    Encoder Decoder
    """

    def __init__(self, config, tokenizer, transformer, dataset_reader):
        """
        :param config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = transformer
        self.dataset_reader = dataset_reader

        self.use_deepspeed = self.config.compute_strategy.startswith("deepspeed")
        self.use_ddp = self.config.compute_strategy.startswith("ddp")
        self.load_model() # 从checkpoint load 权重

        self._last_global_step_saved = -1

    def training_step(self, batch, batch_idx):

        if self.config.mc_loss > 0 or self.config.unlikely_loss > 0:

            verification_input_ids, choices_ids, v_labels = batch["verification_input_ids"], batch["answer_choices_ids"], batch["v_labels"]
            idxs = batch["idx"]

            num_choices = self.config.n_ways

            if self.config.stage==2:
                with torch.no_grad(): # e.g. idxs=tensor([270, 271, 272, 273], device='cuda:1')
                    pred_output = self.predict(batch,if_inner=True)[0]
                    claim_choice_score = pred_output['choices_scores'] # [[0.5366212725639343, 1.7223026752471924, 0.34041303396224976]]
                    claim_idx = pred_output['idx'].tolist() # [0]
                    claim_choice_score = torch.tensor(claim_choice_score).to(verification_input_ids[0].device)
                    claim_choice_score = claim_choice_score.unsqueeze(0) # tensor([[[0.5366, 1.7223, 0.3404]]], device='cuda:1')
                    claim_pred = pred_output['prediction'] # [2]
                    con_l = {0:[0,0,2,2],1:[1,1,1,0],2:[2,2,0,2]} # 这是啥？？？ 应该是原命题标签对应的 肯定、否定、不确定 命题的 约束label
                    pre_labels = {idxs[x].item():claim_pred[xi] for xi,x in enumerate(claim_idx) } # {270: 2} predict里面筛选整除10的索引，这里是选出来的索引对应的预测值
                    
                    for kk in claim_idx:
                        k = idxs[kk].item() # k=270
                        if not self.config.zero_shot:
                            pre_labels[k] = v_labels[kk].item() # 设为真实label
                        pre_labels[k+1] = con_l[pre_labels[k]][1]
                        pre_labels[k+2] = con_l[pre_labels[k]][2]
                        pre_labels[k+3] = con_l[pre_labels[k]][3]
                    # 原来的v_labels e.g tensor([1, 1, 1, 0], device='cuda:1')
                    for ii,iidx in enumerate(idxs):
                        v_labels[ii]=pre_labels[iidx.cpu().item()]

                if self.config.zero_shot:    
                    select_idx = (idxs%10!=0).nonzero().squeeze(1)
                    verification_input_ids = torch.index_select(verification_input_ids,0,select_idx)
                    choices_ids = torch.index_select(choices_ids,0,select_idx)
                    v_labels = torch.index_select(v_labels,0,select_idx)

            input_ids_ = verification_input_ids
            labels_ = v_labels
            
            
            def get_encode_states(input_ids_1,nc=self.config.n_ways):                 
                attention_mask = (input_ids_1 != self.tokenizer.pad_token_id).float()
                encoder_hidden_states_ = self.model.encoder(input_ids=input_ids_1, attention_mask=attention_mask)
                encoder_hidden_states1 = encoder_hidden_states_[0] # torch.Size([1, 159, 2048])
                encoder_hidden_states = encoder_hidden_states1.unsqueeze(dim=1).repeat(1, nc, 1, 1).flatten(0, 1) # torch.Size([3, 159, 2048]) .flatten(0, 1)是将0,1维度合并
                attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, nc, 1).flatten(0, 1)
                return attention_mask,encoder_hidden_states
            
            def get_decode_outputs(attention_mask_,flat_ids_,encoder_hidden_states_):
                # tensor([[2163,    1],                         tensor([[   0, 2163],
                # [3836,    1],                         ===>    [   0, 3836],
                # [ 465,    1]], device='cuda:1')               [   0,  465]], device='cuda:1')
                decoder_input_ids = torch.cat([torch.zeros_like(flat_ids_[:, :1]), flat_ids_[:, :-1]], dim=1)
                decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
                
                model_output = self.model(
                    attention_mask=attention_mask_,
                    encoder_outputs=[encoder_hidden_states_],
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                return model_output
            
            attention_masks,encoder_hidden_states = get_encode_states(input_ids_,num_choices)

            flat_choices_ids,lm_targets,lm_target1s,flat_ids = [],[],[],[]
            choices_ids = [choices_ids] if not isinstance(choices_ids,list) else choices_ids
            for choices_ids_sub in choices_ids:
                # tensor([[2163,    1],[3836,    1],[ 465,    1]], device='cuda:1')
                flat_choices_ids_ = choices_ids_sub.flatten(0, 1) # torch.Size([3, 2])
                # tensor([[2163,    1], [3836,    1], [ 465,    1]], device='cuda:1')
                lm_target_ = flat_choices_ids_ - 100 * (flat_choices_ids_ == self.tokenizer.pad_token_id).long()
                flat_ids_ = flat_choices_ids_
                lm_target1_ = lm_target_
                flat_choices_ids.append(flat_choices_ids_)
                lm_targets.append(lm_target_)
                lm_target1s.append(lm_target1_)
                flat_ids.append(flat_ids_)


            attention_masks = [attention_masks] if not isinstance(attention_masks,list) else attention_masks
            encoder_hidden_states = [encoder_hidden_states] if not isinstance(encoder_hidden_states,list) else encoder_hidden_states
            num_choices = [num_choices] if not isinstance(num_choices,list) else num_choices
            labels_ = [labels_] if not isinstance(labels_,list) else labels_
            model_outputs = []
            loss = 0.0
            for attention_mask,encoder_hidden_state,flat_id,flat_choices_id,lm_target,choices_id,nc,lm_target1,labels_1 in \
                zip(attention_masks,encoder_hidden_states,flat_ids,flat_choices_ids,lm_targets,choices_ids,num_choices,lm_target1s,labels_):

                model_output = get_decode_outputs(attention_mask,flat_id,encoder_hidden_state)
                model_outputs.append(model_output)
                bs1 = labels_1.size()[0] 
                # lm_target.shape = torch.Size([3, 2])  lm_target1.shape = torch.Size([3, 2])
                # model_output.logits.shape = torch.Size([3, 2, 32128])
                # flat_choices_id.shape = torch.Size([3, 2]) 
                # F.cross_entropy(model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1).shape = torch.Size([6, 32128])
                # lm_target.flatten(0, 1).shape = torch.Size([6])
                # tensor([[2163,    1], [3836,    1],   [ 465,    1]], device='cuda:1')
                choices_scores = (
                    torch.reshape(F.cross_entropy(model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                    ,(bs1, nc, -1))
                    .sum(dim=-1)
                )
                if self.config.length_norm > 0:
                    choices_scores = choices_scores / torch.pow(
                        (choices_id != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                    )
                tensorboard_logs = {}

                labels_1_t = labels_1.squeeze(1) if len(labels_1.size())==2 else labels_1
                # 计算模型输出的 logits 与 lm_target 之间的交叉熵损失，并重新整形它
                # torch.reshape(model_output.logits,(bs1, nc, *model_output.logits.size()[1:]))[range(bs1), labels_1_t].shape = torch.Size([1, 2, 32128])
                # torch.reshape(lm_target1,(bs1, nc, -1))[range(bs1), labels_1_t].shape = torch.Size([1, 2])
                lm_loss = F.cross_entropy(
                    torch.reshape(model_output.logits,(bs1, nc, *model_output.logits.size()[1:]))[range(bs1), labels_1_t].flatten(
                        0, 1
                    ), # [2, 32128]?
                    torch.reshape(lm_target1,(bs1, nc, -1))[range(bs1), labels_1_t].flatten(0, 1), # [2] # tensor([3836,    1], device='cuda:1')
                )

                tensorboard_logs = {"lm_loss": lm_loss.item()}
                if self.config.mc_loss > 0 and nc==self.config.n_ways:
                    labels_1_t = labels_1.squeeze(1) if len(labels_1.size())==2 else labels_1
                    # choices_scores.shape = torch.Size([1, 3])
                    # labels_1_t.shape = torch.Size([1])
                    mc_loss = F.cross_entropy(-choices_scores, labels_1_t)
                    tensorboard_logs["mc_loss"] = mc_loss.item()
                else:
                    mc_loss = 0.0

                if self.config.unlikely_loss > 0:
                    # model_output.logits[:,:flat_choices_id.size()[-1],:].shape=torch.Size([3, 2, 32128])
                    # lm_target1.shape=torch.Size([3, 2])
                    cand_loglikely = -torch.reshape(F.cross_entropy(
                        model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1), lm_target1.flatten(0, 1), reduction="none"
                    ),(bs1, nc, -1))
                    # 对于lm_target1中小于0的位置，加入一个大的负值，使得这些位置的log likelihood极小
                    cand_loglikely += torch.reshape((lm_target1 < 0),(bs1, nc, -1)) * -100
                    labels_1_t = labels_1.squeeze(1) if len(labels_1.size())==2 else labels_1
                    # 对于真实标签的位置，设置一个大的负值
                    cand_loglikely[range(bs1), labels_1_t] = -100
                    # 计算unlikely_loss：对log likelihood取指数得到概率，然后计算1 minus这个概率，并取对数
                    # 最后，取所有不是-100的位置的平均值
                    unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
                    tensorboard_logs["unlikely_loss"] = unlikely_loss.item()
                else:
                    unlikely_loss = 0.0

                tmp = lm_loss + mc_loss * self.config.mc_loss + unlikely_loss * self.config.unlikely_loss
                loss += tmp
            tensorboard_logs["loss"] = loss.item()

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            self.log_dict(tensorboard_logs)

        if self.global_step % self.config.save_step_interval == 0:
            self.save_model()

        return loss

    def predict(self, batch,if_inner=False):

        verification_input_ids, choices_ids, v_labels = batch["verification_input_ids"], batch["answer_choices_ids"], batch["v_labels"]
        idxs = batch["idx"]
        num_choices = self.config.n_ways

        if if_inner:
            # 筛选出idxs中可以被10整除的所有索引，并使用这些索引来从三个tensor中选择对应的行
            select_idx = (idxs%10==0).nonzero().squeeze(1)
            verification_input_ids = torch.index_select(verification_input_ids,0,select_idx)
            choices_ids = torch.index_select(choices_ids,0,select_idx)
            v_labels = torch.index_select(v_labels,0,select_idx)

        input_ids_ = verification_input_ids
        labels_ = v_labels
        
        # 获取输入input_ids_1的编码器隐藏状态，并将其扩展重复为nc倍
        def get_encode_states(input_ids_1,nc=self.config.n_ways):
            # 根据输入的 token ID 判断哪些不是填充（pad）标记，并生成注意力掩码
            attention_mask = (input_ids_1 != self.tokenizer.pad_token_id).float()
            # 使用模型的编码器对输入的 token ID 进行编码
            encoder_hidden_states_ = self.model.encoder(input_ids=input_ids_1, attention_mask=attention_mask)
            # 获取编码器的隐藏状态（通常，模型的输出可能是一个元组，所以这里取第一个元素）
            encoder_hidden_states1 = encoder_hidden_states_[0]
            # 扩展隐藏状态维度，并将其重复 `nc` 次，然后将前两个维度展平
            encoder_hidden_states = encoder_hidden_states1.unsqueeze(dim=1).repeat(1, nc, 1, 1).flatten(0, 1)
            # 扩展注意力掩码的维度，并将其重复 `nc` 次，然后将前两个维度展平
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, nc, 1).flatten(0, 1)
            return attention_mask,encoder_hidden_states
        
        def get_decode_outputs(attention_mask_,flat_ids_,encoder_hidden_states_):
            # 生成 decoder 的输入。这基本上是将原始的 flat_ids_ 向右移动一位，
            # 并在最前面加上零。这常常用于语言建模任务，使得模型可以预测下一个 token。
            decoder_input_ids = torch.cat([torch.zeros_like(flat_ids_[:, :1]), flat_ids_[:, :-1]], dim=1)
            # 生成 decoder 的注意力掩码。这里假设每个位置都是有效的（非 pad），
            # 所以生成了一个全部为1的掩码。
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            # 使用模型进行前向传播，获得输出。模型可能是一个seq2seq结构，例如BART或T5，
            # 这里同时提供了编码器的输出和解码器的输入
            model_output = self.model(
                attention_mask=attention_mask_,
                encoder_outputs=[encoder_hidden_states_],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            return model_output
        # 输入经过编码器的隐藏状态并扩展nc次
        attention_masks,encoder_hidden_states = get_encode_states(input_ids_,num_choices)

        input_idx_lists=[]
        input_idx_lists.append([k for k in idxs])

        flat_choices_ids,lm_targets,lm_target1s,flat_ids = [],[],[],[]

        choices_ids = [choices_ids] if not isinstance(choices_ids,list) else choices_ids
        for choices_ids_sub in choices_ids:
            # 展平前两个维度
            flat_choices_ids_ = choices_ids_sub.flatten(0, 1)
            # 为 lm_target_ 创建目标值，其中，如果原始值是 pad_token_id，则减去 100，否则保持不变
            lm_target_ = flat_choices_ids_ - 100 * (flat_choices_ids_ == self.tokenizer.pad_token_id).long()
            flat_ids_ = flat_choices_ids_
            lm_target1_ = lm_target_
            # 原始值和复制值都加入各列表，也不知道在干啥
            flat_choices_ids.append(flat_choices_ids_)
            lm_targets.append(lm_target_)
            lm_target1s.append(lm_target1_)
            flat_ids.append(flat_ids_)

        attention_masks = [attention_masks] if not isinstance(attention_masks,list) else attention_masks
        encoder_hidden_states = [encoder_hidden_states] if not isinstance(encoder_hidden_states,list) else encoder_hidden_states
        num_choices = [num_choices] if not isinstance(num_choices,list) else num_choices
        labels_ = [labels_] if not isinstance(labels_,list) else labels_
        model_outputs = []
        batch_output = []
        for attention_mask,encoder_hidden_state,flat_id,flat_choices_id,lm_target,choices_id,nc,lm_target1,labels_1,idx_sub in \
            zip(attention_masks,encoder_hidden_states,flat_ids,flat_choices_ids,lm_targets,choices_ids,num_choices,lm_target1s,labels_,input_idx_lists):

            model_output = get_decode_outputs(attention_mask,flat_id,encoder_hidden_state)
            model_outputs.append(model_output)
            # batch size
            bs1 = labels_1.size()[0]
            # 计算模型输出的 logits 与 lm_target 之间的交叉熵损失，并重新整形它
            choices_scores = (
                torch.reshape(F.cross_entropy(model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                ,(bs1, nc, -1))
                .sum(dim=-1)
            )
            # 如果设置了长度标准化，则进行标准化处理
            if self.config.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_id != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                )
            choices_scores_o = choices_scores.tolist()
            # 获取最小得分和相应的预测
            pred_score, prediction = choices_scores.min(dim=1)
            # 获取真实标签的得分
            score_gt = choices_scores[range(bs1), labels_1]
            # 将真实标签的得分替换为最大得分
            choices_scores[range(bs1), labels_1] = choices_scores.max(dim=-1)[0]
            # 获取更改后的最小得分
            score_cand = choices_scores.min(dim=-1)[0]

            batch_output_sub = {
                "prediction": prediction.tolist(),
                "pred_scores": pred_score.tolist(),
                "labels": labels_1.tolist(),
                "idx": [k.item() for k in idx_sub],
                "log.score_gt": score_gt.tolist(),
                "log.score_cand": score_cand.tolist(),
            }
            if if_inner:
                batch_output_sub = {
                "prediction": prediction.tolist(),
                "choices_scores": choices_scores_o,
                "labels": labels_1.tolist(),
                "idx": select_idx,
            }
            batch_output.append(batch_output_sub)
        return batch_output

    def validation_step(self, batch, batch_idx):
        batch_output = self.predict(batch)
        return batch_output

    def validation_epoch_end(self, outputs):
        # exchange outputs between processes
        if self.use_deepspeed or self.use_ddp:
            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, outputs)
            if dist.get_rank() == 0:
                if isinstance(outputs[0],list):
                    tmp = []
                    for batch_output in outputs:
                        for outputs1 in gathered_outputs:
                            tmp+=outputs1
                    outputs = tmp #这是在干嘛，使用val的时候把output扩张了平方倍？？
                else:
                    outputs = [batch_output for outputs in gathered_outputs for batch_output in outputs] #origs

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            # let rank 0 collect all outputs
            # now output should be two types
            if isinstance(outputs[0],list):
                accumulated = {key: [] for key in outputs[0][0].keys()}
                for batch_output1 in outputs:
                    for batch_output in batch_output1:
                        for key, value in batch_output.items():
                            accumulated[key].extend(value)
            else:
                accumulated = {key: [] for key in outputs[0].keys()}
                for batch_output in outputs:
                    for key, value in batch_output.items():
                        accumulated[key].extend(value)


            # multi-process may yield dupliated examples in the last batch
            valid_mask = []
            idx_set = set()
            for ii,idx in enumerate(accumulated["idx"]):
                valid_mask.append(idx not in idx_set)
                idx_set.add(idx)
            for key, values in accumulated.items():
                accumulated[key] = [v for v, m in zip(values, valid_mask) if m]
            # valid_mask 把之前复制的又去重了，暂时看不出为什么
            # compute and log results
            result_fname=self.config.test_pred_file
            metrics = self.dataset_reader.compute_metric(accumulated,result_fname)
            # self.log_dict(metrics)
        else:
            metrics = {}

        # self.save_model()
        # if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
        #     self.log_dict(metrics)
        #     result_fname=self.config.test_pred_file
        #     with open(result_fname,'w') as outfile:
        #         print(f"Writing to {result_fname}")
        #         for key,value in accumulated.items():
        #             outfile.write(f"{key}:{value}\n")
        # current_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1

        # print(f"Starting validation_epoch_end in Rank {current_rank}")

        # if not (self.use_deepspeed or self.use_ddp) or current_rank == 0:
        #     self.log_dict(metrics)
        #     # result_fname = self.config.test_pred_file
        #     # with open(result_fname, 'w') as outfile:
        #     #     for key, value in accumulated.items():
        #     #         outfile.write(f"{key}:{value}\n")
        #     # print(f"File written in Rank {current_rank}")

        # print(f"Ending validation_epoch_end in Rank {current_rank}")

        return metrics

    def configure_optimizers(self):
        optimizer, self.trainable_param_names = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_end(self):
        self.save_model(finish=True)

    def load_model(self):
        if self.config.load_weight != "":
            trainable_states = torch.load(self.config.load_weight, map_location=torch.device("cpu"))
            load_result = self.model.load_state_dict(trainable_states, strict=False)
            assert (
                len(load_result.unexpected_keys) == 0
            ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def save_model(self, finish=False):
        if self.config.save_model and (finish or self._last_global_step_saved != self.global_step):
            if finish:
                model_fname = os.path.join(self.config.exp_dir, "finish.pt")
            else:
                model_fname = os.path.join(self.config.exp_dir, f"global_step{self.global_step}.pt")

            if self.use_deepspeed or self.use_ddp:
                torch.distributed.barrier()
                if dist.get_rank() == 0:
                    trainable_states = {
                        param_name: param_weight.cpu()
                        for param_name, param_weight in self.model.state_dict().items()
                        if param_name in self.trainable_param_names
                    }
                    torch.save(trainable_states, model_fname)
                    print(model_fname)
            else:
                trainable_states = {
                    param_name: param_weight.cpu()
                    for param_name, param_weight in self.model.state_dict().items()
                    if param_name in self.trainable_param_names
                }
                torch.save(trainable_states, model_fname)

            self._last_global_step_saved = self.global_step

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        pass

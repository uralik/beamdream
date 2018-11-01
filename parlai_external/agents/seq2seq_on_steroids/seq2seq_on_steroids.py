import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from parlai.core.torch_agent import Beam, TorchAgent, Output
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.core.utils import padded_tensor, round_sigfigs, set_namedtuple_defaults, NEAR_INF, argsort
from parlai.agents.seq2seq.modules import opt_to_kwargs
from .modules import SteroidSeq2seq
import random
#import ipdb
from collections import namedtuple
from .beam_on_steroids import Beam
import time

Output = namedtuple('Output', ['text', 'text_candidates', 'human', 'greedy', 'beam10', 'iterbeam10', 'parbeam10', 'parbeam10rank', 'reranked_candidates'])
set_namedtuple_defaults(Output, default=None)

Batch = namedtuple('Batch', ['text_vec', 'text_lengths', 'label_vec',
                             'label_lengths', 'labels', 'valid_indices',
                             'candidates', 'candidate_vecs', 'image',
                             'memory_vecs', 'injected_pred', 'injected_type',
                             'old_text', 'injected_pred_vecs'])
set_namedtuple_defaults(Batch, default=None)

class SteroidSeq2seqAgent(Seq2seqAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        super().add_cmdline_args(argparser)

        agent = argparser.add_argument_group('SteroidSeq2seq args')
        agent.add_argument('--search-type-during-eval', type=str, default='greedy',
                choices=['human', 'greedy', 'beam', 'blockbeam'])

        agent.add_argument('--howtorank', type=str, choices=['ranker', 'beam'],
                default='ranker')
        agent.add_argument('--cand-type', type=str, choices=['none', 'current_labels', 
            'history', 'all'], default='none', 
            help='Candidates used to train ranker part, history assumes injected preds')
        agent.add_argument('--margin', type=float, default=1.0, help='Margin for RLoss')
        agent.add_argument('--input-dropout', type=float, default=0.1, help='Change '
                'input token to UNK with probability given here')
        agent.add_argument('--num-rank-cand', default=1, type=int, 
                help='This is only used when we do current-labels cand-type')
        agent.add_argument('--lmweight', default=1.0, type=float,
                help='weight for LM loss')
        agent.add_argument('--rankweight', default=1.0, type=float,
                help='weight for Rank loss')
        agent.add_argument('--rankhiddensize', default=512, type=int,
                help='Hidden size of all layers in ranker')
        agent.add_argument('--ranknl', default=2, type=int,
                help='Number of linear layers in ranker')
        agent.add_argument('--ranklossreduce', type=str, choices=['sum', 'elementwise_mean'],
                default='elementwise_mean', help='reduce type for the loss')
        agent.add_argument('--rankloss', type=str, default='margin', choices=['ce', 'margin'], help='The loss which we use in the optimization criterion')
        agent.add_argument('--rank-activation', type=str, default='ReLU',
                help='Ranker activation function, should be nn.*')
        agent.add_argument('--strict-load', type='bool', default=True)
        agent.add_argument('--iter-cand', type=int, default=1)
        agent.add_argument('--min-hamming-dist', type=int, default=1)
        agent.add_argument('--count-overlaps', type='bool', default=False)

        # not used right now
        agent.add_argument('--dump-all-preds', type='bool', default=False)


        TorchAgent.add_cmdline_args(argparser)
        SteroidSeq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent


    def _init_model(self, states=None):
        """Initialize model, override to change model setup."""
        opt = self.opt

        kwargs = opt_to_kwargs(opt)
        self.model = SteroidSeq2seq(
            len(self.dict), opt['embeddingsize'], opt['hiddensize'],
            padding_idx=self.NULL_IDX, start_idx=self.START_IDX,
            longest_label=states.get('longest_label', 1),
            **kwargs)

        if (opt.get('dict_tokenizer') == 'bpe' and
                opt['embedding_type'] != 'random'):
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.decoder.lt.weight,
                                  opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(self.model.encoder.lt.weight,
                                      opt['embedding_type'], log=False)
        if opt['embedding_type'].endswith('fixed'):
            print('Seq2seq: fixing embedding weights.')
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                self.model.decoder.e2s.weight.requires_grad = False
        self.id = 'SteroidSeq2seq'
        self.metrics['rank_loss'] = 0.0
        self.metrics['total_batches'] = 0.0
        self.metrics['overlap'] = 0
        self.overlap_count = {'predicted':0, 'ranked0':0, 'ranked1':0, 'ranked2':0,
                'ranked3':0, 'ranked4':0}  # for word overlap numerator
        self.num_predicted_count = 0  # for word overlap denominator
        self.injpred_selected_count = 0
        self.pred_count = 0
        self.iter_cand = opt['iter_cand']
        self.count_overlaps = opt['count_overlaps']
        self.howtorank = opt['howtorank']
        self.min_hamming_dist = opt['min_hamming_dist']

        if opt['cand_type'] == 'all':
            self.cand_type = ['current_labels', 'history']
        else:
            self.cand_type = [opt['cand_type']]
        

        self.model.post_ranker = nn.ModuleList()

        rank_hidden = self.opt.get('rankhiddensize', 512)
        rank_activation = getattr(nn, self.opt['rank_activation'])
        self.model.post_ranker.append(nn.Linear(self.opt['hiddensize'], rank_hidden))
        self.model.post_ranker.append(rank_activation())

        for i in range(self.opt.get('ranknl', 2) - 2):
            self.model.post_ranker.append(nn.Linear(rank_hidden, rank_hidden))
            self.model.post_ranker.append(rank_activation())
        self.model.post_ranker.append(nn.Linear(rank_hidden, 1))

        if states:
            # set loaded states if applicable
            self.model.load_state_dict(states['model'], strict=self.opt['strict_load'])

        if self.use_cuda:
            self.model.cuda()

        if self.opt['rankloss'] == 'margin':
            self.rank_criterion = nn.MultiMarginLoss(margin=self.opt['margin'], reduction=self.opt['ranklossreduce'])
        elif self.opt['rankloss'] == 'ce':
            self.rank_criterion = nn.CrossEntropyLoss(reduction=self.opt['ranklossreduce'])
        self.inject = self.opt['dump_all_preds']

        assert self.opt.get('person_tokens', False) is True, 'We extract past labels using person tokens'
        
    # we need this for mturk eval where only shared agents are used
    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['inject'] = self.inject
        shared['iter_cand'] = self.iter_cand
        shared['count_overlaps'] = self.count_overlaps
        shared['cand_type'] = self.cand_type
        shared['howtorank'] = self.howtorank
        shared['min_hamming_dist'] = self.min_hamming_dist

        return shared

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        
        if shared:
            # set up shared properties
            self.inject = shared['inject']
            self.iter_cand = shared['iter_cand']
            self.count_overlaps = shared['count_overlaps']
            self.cand_type = shared['cand_type']
            self.howtorank = shared['howtorank']
            self.min_hamming_dist = shared['min_hamming_dist']
        else:
            pass

        
   
    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['rank_loss'] = 0.0
        self.metrics['total_batches'] = 0.0

    def _add_input_dropout(self, batch):
        unk_tok_textvec = torch.Tensor(size=batch.text_vec.size()).fill_(self.dict[self.dict.unk_token]).to(batch.text_vec.device).long()
        probs = torch.Tensor(size=batch.text_vec.size()).uniform_(0, 1).to(batch.text_vec.device)
        for i in range(batch.text_vec.size(0)):
            probs[i][batch.text_lengths[i]:] = 0
        dropped_input = torch.where(probs > (1-self.opt['input_dropout']), unk_tok_textvec, batch.text_vec)
        return dropped_input

    def match_batch(self, batch_reply, valid_inds, output=None):
        """Match sub-batch of predictions to the original batch indices.
        THIS EXTENDS TO NEW OUTPUT
        Batches may be only partially filled (i.e when completing the remainder
        at the end of the validation or test set), or we may want to sort by
        e.g the length of the input sequences if using pack_padded_sequence.

        This matches rows back with their original row in the batch for
        calculating metrics like accuracy.

        If output is None (model choosing not to provide any predictions), we
        will just return the batch of replies.

        Otherwise, output should be a parlai.core.torch_agent.Output object.
        This is a namedtuple, which can provide text predictions and/or
        text_candidates predictions. If you would like to map additional
        fields into the batch_reply, you can override this method as well as
        providing your own namedtuple with additional fields.

        :param batch_reply: Full-batchsize list of message dictionaries to put
            responses into.
        :param valid_inds: Original indices of the predictions.
        :param output: Output namedtuple which contains sub-batchsize list of
            text outputs from model. May be None (default) if model chooses not
            to answer. This method will check for ``text`` and
            ``text_candidates`` fields.
        """
        if output is None:
            return batch_reply
        if output.text is not None:
            for i, response in zip(valid_inds, output.text):
                batch_reply[i]['text'] = response
        if output.text_candidates is not None:
            for i, cands in zip(valid_inds, output.text_candidates):
                batch_reply[i]['text_candidates'] = cands
        if output.human is not None:
            for i, text in zip(valid_inds, output.human):
                batch_reply[i]['human'] = text
        if output.greedy is not None:
            for i, text in zip(valid_inds, output.greedy):
                batch_reply[i]['greedy'] = text
        if output.beam10 is not None:
            for i, text in zip(valid_inds, output.beam10):
                batch_reply[i]['beam10'] = text
        if output.iterbeam10 is not None:
            for i, text in zip(valid_inds, output.iterbeam10):
                batch_reply[i]['iterbeam10'] = text
        if output.parbeam10 is not None:
            for i, text in zip(valid_inds, output.parbeam10):
                batch_reply[i]['parbeam10'] = text
        if output.parbeam10rank is not None:
            for i, text in zip(valid_inds, output.parbeam10rank):
                batch_reply[i]['parbeam10rank'] = text
        if output.reranked_candidates is not None:
            for i, text in zip(valid_inds, output.reranked_candidates):
                batch_reply[i]['reranked_candidates'] = text


        return batch_reply

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        #ipdb.set_trace()
        # helps with memory usage
        self._init_cuda_buffer(self.model, self.criterion, batchsize,
                               self.truncate or 180)
        self.model.train()
        self.zero_grad()
        try:
            dropped_input = self._add_input_dropout(batch)
            out = self.model(dropped_input, batch.label_vec)

            # generated response
            scores = out[0] 
            _, preds = scores.max(2)

            score_view = scores.view(-1, scores.size(-1))
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            loss *= self.opt['lmweight']
            self.metrics['total_batches'] += 1

            rank_loss = 0
            rank_cands = []
            #  now we can train against different candidates
            if 'none' not in self.cand_type:
                # this is our correct candidate (target)
                encoder_states = out[2]
                _, hidden_target, cells_target = self.model._decode_forced(batch.label_vec, encoder_states, with_cells=True)
                target_hidden = out[3]
                target_ranker_input = self.model._get_ranker_input(cells_target, batch.label_lengths)
                target_ranker_output = self.model.forward_post_ranker(target_ranker_input)
                rank_cands.append(target_ranker_output)

            if 'current_labels' in self.cand_type:
                # this is the case when we add random candidates
                for i in range(self.opt['num_rank_cand']):
                    shuffled_targets = torch.cat([batch.label_vec[i:], batch.label_vec[:i]], dim=0)
                    shifted_lengths = batch.label_lengths[i:] + batch.label_lengths[:i]
                    scores, hidden_cand, cells_cand = self.model._decode_forced(shuffled_targets, encoder_states, with_cells=True)
                    ranker_input_cand = self.model._get_ranker_input(cells_cand, shifted_lengths)
                    ranker_output_cand = self.model.forward_post_ranker(ranker_input_cand)
                    rank_cands.append(ranker_output_cand)
            
            if 'history' in self.cand_type:
                # history cands, we take it from batch.injected_pred, i.e. from input
                # type of the history is controlled via the dataset file
                assert batch.injected_pred is not None, 'history cands work only in case of injected dataset'
                injected_pred_padded, lengths = padded_tensor(batch.injected_pred_vecs, use_cuda=True)

                scores, hidden_cand, cells_cand = self.model._decode_forced(injected_pred_padded, encoder_states, with_cells=True)
                ranker_input_cand = self.model._get_ranker_input(cells_cand, lengths)
                ranker_output_cand = self.model.forward_post_ranker(ranker_input_cand)
                rank_cands.append(ranker_output_cand)


            if 'none' not in self.cand_type:
                p_y_given_x = torch.cat(rank_cands , dim=1)
                rank_targets = torch.Tensor(batchsize).fill_(0).long().to(p_y_given_x.device)
                rank_loss = self.rank_criterion(p_y_given_x, rank_targets)
                rank_loss = self.opt['rankweight'] * rank_loss
                self.metrics['rank_loss'] = rank_loss
                

            loss = loss + rank_loss
            loss.backward()
            self.update_params()

        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
            else:
                raise e

    
    @staticmethod
    def beam_search(model, batch, encoder_states, beam_size, start=1, end=2,
                    pad=0, min_length=3, min_n_best=5, max_ts=40, block_ngram=0, iter_cands=1, min_hamming_dist=1):
        """ Beam search given the model and Batch
        This function uses model with the following reqs:
        - model.encoder takes input returns tuple (enc_out, enc_hidden, attn_mask)
        - model.decoder takes decoder params and returns decoder outputs after attn
        - model.output takes decoder outputs and returns distr over dictionary

        Function arguments:
        model : nn.Module, here defined in modules.py
        batch : Batch structure with input and labels
        beam_size : Size of each beam during the search
        start : start of sequence token
        end : end of sequence token
        pad : padding token
        min_length : minimum length of the decoded sequence
        min_n_best : minimum number of completed hypothesis generated from each beam
        max_ts: the maximum length of the decoded sequence

        Return:
        beam_preds_scores : list of tuples (prediction, score) for each sample in Batch
        n_best_preds_scores : list of n_best list of tuples (prediction, score) for
                              each sample from Batch
        beams : list of Beam instances defined in Beam class, can be used for any
                following postprocessing, e.g. dot logging.
        """
        batch_size = len(batch.text_lengths)
        if iter_cands > 1:
            assert batch_size == 1, 'bs 1 for now'
            batch_size = iter_cands
            assert encoder_states is None, "we expand encoder_states here"
            encoder_states = model.encoder(batch.text_vec.expand([iter_cands, -1]))
            original_encoder_states = model.encoder(batch.text_vec)
        else:
            if encoder_states is None:
                encoder_states = model.encoder(batch.text_vec)
            original_encoder_states = encoder_states
        # expand to iter_cands here
        enc_out = encoder_states[0]
        enc_hidden = encoder_states[1]
        attn_mask = encoder_states[2]

        current_device = encoder_states[0][0].device

        beams = [Beam(beam_size, min_length=min_length, padding_token=pad,
                      bos_token=start, eos_token=end, min_n_best=min_n_best,
                      block_ngram=block_ngram) for i in range(batch_size)]
        decoder_input = torch.Tensor([start]).detach().expand(
            batch_size, 1).long().to(current_device)
        # repeat encoder_outputs, hiddens, attn_mask
        decoder_input = decoder_input.repeat(
            1, beam_size).view(beam_size * batch_size, -1)
        enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            batch_size * beam_size, -1, enc_out.size(-1))
        attn_mask = encoder_states[2].repeat(
            1, beam_size).view(attn_mask.size(0) * beam_size, -1)
        repeated_hiddens = []
        if isinstance(enc_hidden, tuple):  # LSTM
            for i in range(len(enc_hidden)):
                repeated_hiddens.append(
                    enc_hidden[i].unsqueeze(2).repeat(1, 1, beam_size, 1))
            num_layers = enc_hidden[0].size(0)
            hidden_size = enc_hidden[0].size(-1)
            enc_hidden = tuple([repeated_hiddens[i].view(
                num_layers, batch_size *
                beam_size, hidden_size) for i in range(len(repeated_hiddens))])
        else:  # GRU
            num_layers = enc_hidden.size(0)
            hidden_size = enc_hidden.size(-1)
            enc_hidden = enc_hidden.unsqueeze(2).repeat(1, 1, beam_size, 1).view(
                num_layers, batch_size * beam_size, hidden_size)

        hidden = enc_hidden
        for ts in range(max_ts):
            if all((b.done() for b in beams)):
                break
            #import ipdb; ipdb.set_trace()
            output, hidden = model.decoder(
                decoder_input.to(current_device), hidden, (enc_out, attn_mask))
            score = model.output(output)
            # score contains softmax scores for batch_size * beam_size samples
            score = score.view(batch_size, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                partial_hyps = []
                if iter_cands > 1:
                    for j in range(i):
                        if len(beams[j].outputs) > 1:
                            partial_hyps.extend(beams[j].partial_hyps)
                b.advance(score[i], partial_hyps, min_hamming_dist)
            decoder_input = torch.cat(
                [b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            permute_hidden_idx = torch.cat(
                [beam_size * i +
                    b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
            # permute decoder hiddens with respect to chosen hypothesis now
            if isinstance(hidden, tuple):  # LSTM
                for i in range(len(hidden)):
                    hidden[i].data.copy_(hidden[i].data.index_select(
                        dim=1, index=permute_hidden_idx.to(current_device)))
            else:  # GRU
                hidden.data.copy_(hidden.data.index_select(
                    dim=1, index=permute_hidden_idx.to(current_device)))
        for b in beams:
            b.check_finished()

        beam_preds_scores = [list(b.get_top_hyp()) for b in beams]
        for pair in beam_preds_scores:
            pair[0] = Beam.get_pretty_hypothesis(pair[0])

        n_best_beams = [b.get_rescored_finished(
            n_best=min_n_best) for b in beams]
        n_best_beam_preds_scores = []
        for i, beamhyp in enumerate(n_best_beams):
            this_beam = []
            for hyp in beamhyp:
                pred = beams[i].get_pretty_hypothesis(
                    beams[i].get_hyp_from_finished(hyp))
                score = hyp.score
                this_beam.append((pred, score))
            n_best_beam_preds_scores.append(this_beam)

        return beam_preds_scores, n_best_beam_preds_scores, beams, original_encoder_states

    def rerank_candidates(self, candidates, encoder_states):
        """
        Candidates reranking based on the post ranker in the self.model
        encoder_states are assumed from the whole minibatch
        candidates is a list of lists, i.e. internal list is a list of candidates
        for the sample in the mini-batch
        This reranking is used only with minibatch = 1
        """
        assert encoder_states[0].size(0) == 1, 'Batch size 1 is only supported here'
        rank_scores = []
        with torch.no_grad():
            cands_vecs, lengths = padded_tensor(candidates, use_cuda=self.use_cuda) # (cand_num, max_length)
            exp_enc_states = []
            exp_enc_states.append(encoder_states[0].expand(len(candidates), -1, -1))
            if isinstance(encoder_states[1], tuple):
                exp_enc_states.append((encoder_states[1][0].expand(-1, len(candidates), -1).contiguous(),
                                    encoder_states[1][1].expand(-1, len(candidates), -1).contiguous()))
            else:
                exp_enc_states.append(encoder_states[1].expand(-1, len(candidates), -1))
            exp_enc_states.append(encoder_states[2].expand(len(candidates), -1))
            scores, hidden_cand, cells_cand = self.model._decode_forced(cands_vecs, exp_enc_states, with_cells=True)
            ranker_input_cand = self.model._get_ranker_input(cells_cand, lengths)
            ranker_output_cand = self.model.forward_post_ranker(ranker_input_cand)

        rank_scores = ranker_output_cand.view(-1)
        topcand = candidates[torch.argmax(rank_scores)]
        return [topcand], rank_scores

    #  following functions used for debugging
    def write_cands(self, inp, cands, bscores, rscores):
        with open('./rank_output.log', 'a') as f:
            f.write('\n\n\n')
            f.write('input: {}\n'.format(self._v2t(inp)))
            f.write('\n')
            for i, c in enumerate(cands):
                f.write('BS:{:.5f}\t RS:{:.5f}\t {}\n'.format(bscores[i], rscores[i], self._v2t(c)))
            f.write('\n')
            f.write('beam best: {}\n'.format(self._v2t(cands[torch.argmax(torch.Tensor(bscores))])))
            f.write('rank best: {}\n'.format(self._v2t(cands[torch.argmax(rscores)])))

    def write_cands_inline(self, cands):
        with open('/misc/vlgscratch4/ChoGroup/kulikov/data-parlai/local-runs/original-refactored/beam_nbest_all.txt', 'a') as f:
            f.write('{}\n'.format('\t'.join(cands)))

    def print_cands(self, inp, cands, bscores, rscores):
        if rscores is None:
            rscores = torch.zeros(len(bscores))
        print('\n\n\n')
        print('input: {}'.format(self._v2t(inp)))
        print('\n')
        for i, c in enumerate(cands):
            print('BS:{:.5f}\t RS:{:.5f}\t {}'.format(bscores[i], rscores[i], self._v2t(c)))
        print('\n')
        print('beam best: {}'.format(self._v2t(cands[torch.argmax(torch.Tensor(bscores))])))
        print('rank best: {}'.format(self._v2t(cands[torch.argmax(rscores)])))

    def get_reranked_cands_instr(self, cands, bscores, rscores):
        if rscores is None:
            rscores = torch.zeros(len(bscores))
        output = []
        for i, c in enumerate(cands):
            output.append(('{:.5f}\t{:.5f}\t{}'.format(bscores[i], rscores[i], self._v2t(c))))
        return output

    def make_preds_for_inject(self, batch):
        assert self.inject == True
        assert len(batch.labels) == 1, 'only single batch is assumed here to keep order'
        # human
        human = batch.labels

        # greedy
        out = self.model(batch.text_vec, ys=None)
        scores, _ = out[0], out[1]
        _, preds = scores.max(2)
        greedy = [self._v2t(p) for p in preds]
        encoder_states = out[2]

        # beam
        out = SteroidSeq2seqAgent.beam_search(self.model, batch, encoder_states, self.beam_size, 
                    start=self.START_IDX, end=self.END_IDX, pad=self.NULL_IDX,
                    min_length=self.beam_min_length, min_n_best=self.beam_min_n_best, beam_block_hypos=[], block_ngram=self.beam_block_ngram)
        beam10 = [self._v2t(out[0][0][0][1:])]

        output = Output(beam10, cand_choices, human, greedy, beam10)
        return output

    def get_valid_ppl(self, batch, encoder_states=None):
        assert batch.label_vec is not None, 'We need labels to compute PPL'
        if encoder_states is None:
            encoder_states = self.model._encode(batch.text_vec)
        scores, hidden = self.model._decode_forced(batch.label_vec, encoder_states)
        f_scores = scores  # forced scores
        _, f_preds = f_scores.max(2)  # forced preds
        score_view = f_scores.view(-1, f_scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        correct = ((batch.label_vec == f_preds) * notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens

    def compute_log_prob(self, predictions, encoder_states):
        scores, hidden = self.model._decode_forced(predictions, encoder_states)
        import ipdb; ipdb.set_trace()

    def _build_cands(self, batch):
        """Augment with injected pred here
        """
        if not batch.candidates:
            return None, None
        cand_inds = [i for i in range(len(batch.candidates))
                     if batch.candidates[i]]
        cands = []
        for i in cand_inds:
            with_history = batch.candidate_vecs[i]
            if batch.injected_pred is not None:
                with_history.append(batch.injected_pred_vecs[i])
            cands.append(with_history)
        lengths = []
        for i, c in enumerate(cands):
            cands[i], length = padded_tensor(c, use_cuda=self.use_cuda)
            lengths.append(length)
        return cands, cand_inds, lengths

    def get_num_common(self, t1, t2):
        l1 = t1.split()
        l2 = t2.split()
        return len(list(set(l1).intersection(l2)))

    def count_overlap(self, batch, preds, cands):
        text_preds = [self._v2t(p) for p in preds] # batch size
        pred_vs_inj_count = [self.get_num_common(t1,t2) for t1,t2 in zip(text_preds, batch.injected_pred)] 
        topranked = []
        tcands = list(map(list, zip(*cands)))
        for i in range(5):
            topranked.append([self.get_num_common(t1,t2) for t1,t2 in zip(tcands[i], batch.injected_pred)])
        self.overlap_count['predicted'] += sum(pred_vs_inj_count)
        for i in range(5):
            self.overlap_count['ranked{}'.format(i)] += sum(topranked[i])

        self.num_predicted_count += sum([len(t.split()) for t in text_preds])

        pred_selected = [1 if cands[i][0] == batch.injected_pred[i] else 0 for i in range(len(cands))]
        self.injpred_selected_count += sum(pred_selected)
        self.pred_count += len(pred_selected)


    def eval_step(self, batch):
        self.model.eval()
        cand_scores = None
        cand_choices = None
        reranked_beam_candidates = None

        if self.inject == False:
            if self.beam_size == 1:  # greedy search
                out = self.model(batch.text_vec, ys=None)
                scores, _ = out[0], out[1]
                _, preds = scores.max(2)
                encoder_states = out[2]

                preds = preds.cpu()
                if batch.label_vec is not None:
                    self.get_valid_ppl(batch, encoder_states)

            elif self.beam_size > 1:  # beam search or iter-beam based on iter_cand (for iterbeam only batchsize 1)
                beam_candidates = []
                beam_candidates_scores = []
                encoder_states = None
                out = SteroidSeq2seqAgent.beam_search(self.model, batch, encoder_states, self.beam_size, 
                        start=self.START_IDX, end=self.END_IDX, pad=self.NULL_IDX, 
                        min_length=self.beam_min_length, min_n_best=self.beam_min_n_best,
                        block_ngram=self.beam_block_ngram, iter_cands=self.iter_cand, min_hamming_dist=self.min_hamming_dist)


                beam_preds_scores, nbest_beam_preds_scores, beams, encoder_states = out
                beam_candidates = [i[0].tolist()[1:] for i in beam_preds_scores]  # here results from 'minibatch'
                beam_candidates_scores = [i[1].item() for i in beam_preds_scores]

                #self.write_cands_inline([self._v2t(c[0]) for c in nbest_beam_preds_scores[0]])
                #self.write_cands_inline([self._v2t(c[0]) for c in beam_preds_scores if len(c[0]) > 3])
                
                if self.iter_cand > 1:
                    if self.howtorank == 'ranker':
                        preds, rank_scores = self.rerank_candidates(beam_candidates, encoder_states)  # this is only one best pred with bs=1
                        reranked_beam_candidates = [x for _,x in sorted(zip(rank_scores.tolist(), beam_candidates), key = lambda x: x[0])]
                        reranked_beam_candidates = [self.get_reranked_cands_instr(beam_candidates, beam_candidates_scores, rank_scores)]

                    elif self.howtorank == 'beam':
                        reranked_beam_candidates = [x for _,x in sorted(zip(beam_candidates_scores, beam_candidates), key=lambda x: x[0])]
                        reranked_beam_candidates = reranked_beam_candidates[::-1]
                        preds = [reranked_beam_candidates[0]]
                        reranked_beam_candidates = [self.get_reranked_cands_instr(beam_candidates, beam_candidates_scores, None)]

                else:
                    preds = beam_candidates
                    reranked_beam_candidates = [self.get_reranked_cands_instr([c[0] for c in nbest_beam_preds_scores[0]], [c[1] for c in nbest_beam_preds_scores[0]], None)]  # this is only for bs=1 case!


                if self.beam_dot_log is True:
                    for i, b in enumerate(beams):
                        dot_graph = b.get_beam_dot(dictionary=self.dict, n_best=3)
                        image_name = self._v2t(batch.text_vec[i, -20:]).replace(' ', 
                                '-').replace('__null__', '')
                        dot_graph.write_png(os.path.join(
                            self.beam_dot_dir, "{}.png".format(image_name)))


            if batch.candidates:
                cand_params = self._build_cands(batch)
                cand_scores = self.model._rank(cand_params, encoder_states)
            else:
                cand_scores = None
            if cand_scores is not None:
                candidates_withinjection = batch.candidates
                if batch.injected_pred is not None:
                    for i in range(len(candidates_withinjection)):
                        candidates_withinjection[i].append(batch.injected_pred[i])
                cand_preds = cand_scores.sort(1, True)[1]
                cand_choices = self._pick_cands(cand_preds, cand_params[1],
                                                candidates_withinjection)
            
            pred_text = [self._v2t(t) for t in preds]
            if self.count_overlaps is True:
                self.count_overlap(batch, preds, cand_choices)

            output = Output(pred_text, cand_choices, reranked_candidates=reranked_beam_candidates)

        # not used in this work
        elif self.inject == True:
            output = self.make_preds_for_inject(batch)

        return output


    def report(self):
        m = super().report()
        if self.metrics['total_batches'] > 0:
            m['rank_loss'] = self.metrics['rank_loss']
        if self.num_predicted_count > 0:
            m['overlap_prediction'] = self.overlap_count['predicted'] / self.num_predicted_count
            for i in range(5):
                m['overlap_ranked{}'.format(i)] = self.overlap_count['ranked{}'.format(i)] / self.num_predicted_count
            m['injected_ranked0'] = self.injpred_selected_count / self.pred_count
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)

        return m

    def batchify(self, obs_batch, sort=True,
                 is_valid=lambda obs: 'text_vec' in obs or 'image' in obs):
        """Create a batch of valid observations from an unchecked batch.

        A valid observation is one that passes the lambda provided to the
        function, which defaults to checking if the preprocessed 'text_vec'
        field is present which would have been set by this agent's 'vectorize'
        function.

        Returns a namedtuple Batch. See original definition above for in-depth
        explanation of each field.

        If you want to include additonal fields in the batch, you can subclass
        this function and return your own "Batch" namedtuple: copy the Batch
        namedtuple at the top of this class, and then add whatever additional
        fields that you want to be able to access. You can then call
        super().batchify(...) to set up the original fields and then set up the
        additional fields in your subclass and return that batch instead.

        :param obs_batch: List of vectorized observations
        :param sort:      Default False, orders the observations by length of
                          vectors. Set to true when using
                          torch.nn.utils.rnn.pack_padded_sequence.
                          Uses the text vectors if available, otherwise uses
                          the label vectors if available.
        :param is_valid:  Function that checks if 'text_vec' is in the
                          observation, determines if an observation is valid
        """
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens = None, None
        if any('text_vec' in ex for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = padded_tensor(_xs, self.NULL_IDX, self.use_cuda)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = (labels_avail or
                             any('eval_labels_vec' in ex for ex in exs))

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = padded_tensor(label_vecs, self.NULL_IDX, self.use_cuda)
            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens,
                    descending=True
                )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        # MEMORIES
        mems = None
        if any('memory_vecs' in ex for ex in exs):
            mems = [ex.get('memory_vecs', None) for ex in exs]

        inj_preds = None
        # PRED
        if any('injected_pred' in ex for ex in exs):
            inj_preds = [ex.get('injected_pred', None) for ex in exs]

        injected_pred_vecs = None
        if any('injected_pred_vecs' in ex for ex in exs):
           injected_pred_vecs = [ex.get('injected_pred_vecs', None) for ex in exs]

        inj_types = None
        # PRED TYPE
        if any('injected_type' in ex for ex in exs):
            inj_types = [ex.get('injected_type', None) for ex in exs]

        # OLD TEXT
        old_texts = None
        if any('old_text' in ex for ex in exs):
            old_texts = [ex.get('old_text', None) for ex in exs]


        return Batch(text_vec=xs, text_lengths=x_lens, label_vec=ys,
                     label_lengths=y_lens, labels=labels,
                     valid_indices=valid_inds, candidates=cands,
                     candidate_vecs=cand_vecs, image=imgs, memory_vecs=mems,
                     injected_pred=inj_preds, injected_type=inj_types,
                     old_text=old_texts, injected_pred_vecs=injected_pred_vecs)


    def vectorize(self, *args, **kwargs):
        """Override vectorize for seq2seq."""
        obs = super().vectorize(*args, **kwargs)
        if 'injected_pred' in obs:
            obs['injected_pred_vecs'] = self._vectorize_text(obs['injected_pred'], 
                    False, True, kwargs['truncate'], False)

        return obs

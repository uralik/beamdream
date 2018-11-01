from parlai.agents.seq2seq.modules import Seq2seq, pad
import torch.nn as nn
import torch
import time

class SteroidSeq2seq(Seq2seq):

    def forward_post_ranker(self, ranker_input):
        x = ranker_input
        for layer in self.post_ranker:
            x = layer(x)
        return x

    # we do this to keep all the cell states if necessary
    def _decode_forced(self, ys, encoder_states, with_cells=False):
        """Decode with teacher forcing."""
        bsz = ys.size(0)
        seqlen = ys.size(1)

        #ipdb.set_trace()
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])

        # input to model is START + each target except the last
        y_in = ys.narrow(1, 0, seqlen - 1)
        xs = torch.cat([self._starts(bsz), y_in], 1)

        scores = []
        cells = []
        if self.attn_type == 'none':
            # do the whole thing in one go
            output, hidden = self.decoder(xs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
        else:
            # need to feed in one token at a time so we can do attention
            # TODO: do we need to do this? actually shouldn't need to since we
            # don't do input feeding
            for i in range(seqlen):
                xi = xs.select(1, i).unsqueeze(1)
                output, hidden = self.decoder(xi, hidden, attn_params)
                score = self.output(output)
                scores.append(score)
                cells.append(hidden)

        scores = torch.cat(scores, 1)
        if with_cells == True:
            return scores, hidden, cells
        else:
            return scores, hidden

    def _decode(self, encoder_states, maxlen):
        """Decode maxlen tokens."""
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])
        bsz = encoder_states[0].size(0)

        xs = self._starts(bsz)  # input start token

        scores = []
        for _ in range(maxlen):
            # generate at most longest_label tokens
            output, hidden = self.decoder(xs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
            xs = score.max(2)[1]  # next input is current predicted output

        scores = torch.cat(scores, 1)
        return scores, hidden


    def forward(self, xs, ys=None, cand_params=None, prev_enc=None,
                maxlen=None):
        """Get output predictions from the model.

        :param xs:          (bsz x seqlen) LongTensor input to the encoder
        :param ys:          expected output from the decoder. used for teacher
                            forcing to calculate loss.
        :param cand_params: set of candidates to rank, and indices to match
                            candidates with their appropriate xs.
        :param prev_enc:    if you know you'll pass in the same xs multiple
                            times, you can pass in the encoder output from the
                            last forward pass to skip recalcuating the same
                            encoder output.
        :param maxlen:      max number of tokens to decode. if not set, will
                            use the length of the longest label this model
                            has seen. ignored when ys is not None.

        :returns: scores, candidate scores, and encoder states
            scores contains the model's predicted token scores.
                (bsz x seqlen x num_features)
            candidate scores are the score the model assigned to each candidate
                (bsz x num_cands)
            encoder states are the (output, hidden, attn_mask) states from the
                encoder. feed this back in to skip encoding on the next call.
        """
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_states = self._encode(xs, prev_enc)

        # rank candidates if they are available
        cand_scores = self._rank(cand_params, encoder_states)
        
        if ys is not None:
            # use teacher forcing
            scores, hidden = self._decode_forced(ys, encoder_states)
        else:
            scores, hidden = self._decode(encoder_states, maxlen or self.longest_label)

        return scores, cand_scores, encoder_states, hidden


    def _get_ranker_input(self, hidden, lengths=None):
        """Here we assume only LSTM in case of cell
        right now only LSTM cell is here
        """
        h = [i[0] for i in hidden]
        c = [i[1] for i in hidden]
        cells = torch.stack(c, 0)
        last_ts = (torch.LongTensor(lengths)-1).unsqueeze(0).unsqueeze(2).expand_as(cells).to(cells.device)
        last_cells = cells.gather(0, last_ts)[0]
        last_layer_cells = last_cells[-1]

        return last_layer_cells


    def _rank(self, cand_params, encoder_states):
        """Rank each cand by the post ranker part."""
        if cand_params is None:
            return None

        cands, cand_inds, lengths = cand_params
        if cands is None:
            return None
        encoder_states = self._align_inds(encoder_states, cand_inds)

        cand_scores = []
        for batch_idx in range(len(cands)):
            # we do one set of candidates at a time
            curr_cs = cands[batch_idx]
            num_cands = curr_cs.size(0)

            # select just the one hidden state
            cur_enc_states = self._extract_cur(
                encoder_states, batch_idx, num_cands)

            score, hidden, cells = self._decode_forced(curr_cs, cur_enc_states, with_cells=True)
            ranker_input = self._get_ranker_input(cells, lengths[batch_idx])
            ranker_output = self.forward_post_ranker(ranker_input)

            cand_scores.append(ranker_output)

        max_len = max(len(c) for c in cand_scores)
        cand_scores = torch.cat(
            [pad(c, max_len, pad=self.NULL_IDX).unsqueeze(0)
             for c in cand_scores], 0)
        return cand_scores
